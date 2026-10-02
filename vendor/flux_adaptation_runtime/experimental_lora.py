"""Experimental actual-FLUX projection LoRA helpers; no automatic model load.

CPU toy validation is separate from real FLUX/GPU validation. The cpu backend
keeps frozen weights on CPU with an explicit input-gradient backward, rather
than using unsafe inference forward hooks. nf4/int8 require bitsandbytes CUDA.
"""
from __future__ import annotations
import math, re, types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

TARGET=re.compile(r'\.(q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)$')
HEAD_PREFIXES=('emb_in.action','emb_in.action_cond','final_layer.action','final_layer.action_cond')

class FrozenCPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias):
        if weight.device.type!='cpu' and weight.device!=x.device:raise ValueError('Frozen base must remain CPU or on the fixed input device')
        ctx.weight=weight
        ctx.weight_version=weight._version
        ctx.input_dtype=x.dtype
        # CUDA copies are temporary and not retained for autograd input gradients.
        return F.linear(x.to(weight.dtype),weight.to(x.device),None if bias is None else bias.to(x.device))

    @staticmethod
    def backward(ctx,gradient):
        if ctx.weight._version!=ctx.weight_version:raise RuntimeError('Frozen base mutated between forward and backward')
        if not ctx.needs_input_grad[0]:return None,None,None
        weight=ctx.weight.to(gradient.device)
        result=gradient.to(weight.dtype).matmul(weight)
        return result.to(ctx.input_dtype),None,None

class FrozenCPUBase(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.in_features=source.in_features;self.out_features=source.out_features
        self.register_buffer('weight',source.weight.detach().cpu().clone())
        self.register_buffer('bias',None if source.bias is None else source.bias.detach().cpu().clone())

    def _apply(self,*args,**kwargs):
        # Outer model.to(cuda) moves adapters and resident modules, not CPU bases.
        return self

    def forward(self,x):return FrozenCPUFunction.apply(x,self.weight,self.bias)

def place_frozen_base_budget(dit,gib):
    """Explicit one-time immutable placement, before any forward/backward."""
    if gib<0:raise ValueError('Negative resident base budget')
    budget=int(gib*2**30);used=0;rows=[]
    bases=sorted(((name,m) for name,m in dit.named_modules() if isinstance(m,FrozenCPUBase)),
                 key=lambda pair:pair[1].weight.numel()*pair[1].weight.element_size(),reverse=True)
    for name,base in bases:
        if base.weight.device.type!='cpu':raise ValueError('Placement must occur exactly once from CPU bases')
        if base.weight.dtype!=torch.bfloat16:raise ValueError('Real resident base placement requires original BF16 weights')
        size=sum(b.numel()*b.element_size() for b in base.buffers())
        if used+size<=budget:
            base.weight=base.weight.to('cuda')
            if base.bias is not None:base.bias=base.bias.to('cuda')
            used+=size;rows.append(name)
    return {'requested_budget_bytes':budget,'resident_base_bytes':used,'resident_base_modules':rows,
            'placement':'Immutable frozen buffers moved once before graph creation; no migration hooks'}

class ProjectionLoRA(nn.Module):
    def __init__(self,base,rank=32,alpha=32):
        super().__init__();self.base=base;self.rank=rank;self.scale=alpha/rank
        self.in_features=base.in_features;self.out_features=base.out_features
        self.A=nn.Parameter(torch.empty(rank,self.in_features,dtype=torch.float32))
        self.B=nn.Parameter(torch.zeros(self.out_features,rank,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self._requires_bf16_input=False

    def forward(self,x):
        original=self.base(x)
        delta=F.linear(F.linear(x.to(self.A.dtype),self.A),self.B)*self.scale
        return original+delta.to(original.dtype)

def replace_projections(dit,*,backend='cpu',rank=32,alpha=32):
    """Operate on a strictly-loaded CPU training DiT, never prepared inference.

    Caller owns placement, native cached loss, optimizer, normalizers, grading
    and adapter serialization. Existing inference hooks must be removed first.
    """
    if backend not in ('cpu','nf4','int8'):raise ValueError(backend)
    if any(p.device.type!='cpu' for p in dit.parameters()):raise ValueError('Start with CPU model; avoid double GPU base copies')
    dit.requires_grad_(False)
    modules=[(name,m) for name,m in dit.named_modules() if isinstance(m,nn.Linear) and TARGET.search('.'+name)]
    rows=[]
    for name,source in modules:
        if backend=='cpu':base=FrozenCPUBase(source)
        else:
            import bitsandbytes as bnb
            if backend=='nf4':
                base=bnb.nn.Linear4bit(source.in_features,source.out_features,bias=source.bias is not None,
                    compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
            else:
                base=bnb.nn.Linear8bitLt(source.in_features,source.out_features,bias=source.bias is not None,has_fp16_weights=False)
            base.load_state_dict(source.state_dict())
            base.requires_grad_(False)
            base=base.to('cuda')
        wrapper=ProjectionLoRA(base,rank,alpha)
        parent_name,leaf=name.rsplit('.',1);setattr(dit.get_submodule(parent_name),leaf,wrapper)
        rows.append({'module':name,'frozen_parameters':source.weight.numel(),
                     'adapter_parameters':rank*(source.in_features+source.out_features)})
    # Full heads remain normal floating modules; direct weight access is safe.
    for name,m in dit.named_modules():
        if name in HEAD_PREFIXES:m.float().requires_grad_(True)
    if not rows:raise ValueError('No native FLUX projection targets found')
    return rows

def install_block_checkpointing(dit):
    """Preserve parameter names while wrapping native ModeBlock forward calls."""
    names=[]
    for name,block in dit.named_modules():
        if type(block).__name__ not in ('ModeBlock','SingleStreamBlock'):continue
        if hasattr(block,'_adapt_original_forward'):raise ValueError('Checkpointing already installed')
        block._adapt_original_forward=block.forward
        def wrapped(self,*args,**kwargs):
            if self.training and torch.is_grad_enabled():
                return checkpoint(self._adapt_original_forward,*args,use_reentrant=False,**kwargs)
            return self._adapt_original_forward(*args,**kwargs)
        block.forward=types.MethodType(wrapped,block);names.append(name)
    return names

def adapter_and_head_state(dit):
    """Small state only; frozen/quantized base is rebuilt from pinned original."""
    return {name:value.detach().cpu().clone() for name,value in dit.named_parameters() if value.requires_grad}

def load_adapter_and_head_state(dit,state):
    parameters=dict(dit.named_parameters())
    expected={name for name,value in parameters.items() if value.requires_grad}
    if set(state)!=expected:raise ValueError('Adapter/head keys differ from reconstructed model')
    with torch.no_grad():
        for name,value in state.items():
            if value.shape!=parameters[name].shape:raise ValueError(f'Shape mismatch: {name}')
            parameters[name].copy_(value.to(parameters[name]))

def merge_into_unwrapped_dit(dit,state,*,rank,alpha):
    """Merge onto strictly restored original BF16 CPU DiT before placement.

    This is original base + learned low-rank updates, not quantized bases.
    Quantized-training adaptation should be evaluated again after BF16 merge.
    """
    parameters=dict(dit.named_parameters());copied=[];merged=[]
    with torch.no_grad():
        for name,value in state.items():
            if name.endswith('.A'):
                module_name=name[:-2];module=dit.get_submodule(module_name)
                if not isinstance(module,nn.Linear):raise TypeError(module_name)
                bname=module_name+'.B'
                if bname not in state:raise ValueError(f'Missing {bname}')
                A=value.float();B=state[bname].float()
                if A.shape[0]!=rank or B.shape[1]!=rank:raise ValueError('Rank mismatch')
                delta=(B@A)*(alpha/rank)
                if delta.shape!=module.weight.shape:raise ValueError(module_name)
                module.weight.copy_((module.weight.float()+delta.to(module.weight.device)).to(module.weight.dtype))
                merged.append(module_name)
            elif not name.endswith('.B'):
                if name not in parameters or not name.startswith(HEAD_PREFIXES):raise ValueError(f'Unexpected head key {name}')
                if value.shape!=parameters[name].shape:raise ValueError(name)
                parameters[name].copy_(value.to(parameters[name]));copied.append(name)
    return {'merged_projections':merged,'copied_heads':copied}
