"""Restore trained FLUX LoRA into the original CPU policy before GPU placement.

Default inference retains original BF16 weights, FP32 adapters and FP32 robot
heads under BF16 autocast, matching training arithmetic. An optional merged
BF16 mode reports rounding. Both are separate from frozen zero-shot experiments.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime/flux-so101-work'))
import local_runtime_fast as runtime

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024), b''):h.update(part)
    return h.hexdigest()

def load_adapted_policy(folder, prompts, *, resident_gib=12., progress=None, restoration='unmerged'):
    import torch
    from safetensors.torch import load_file
    from flux_action.config import PolicyConfig
    from flux_action.inference import so101
    if restoration not in ('merged','unmerged'):raise ValueError(restoration)
    folder = Path(folder)
    adapter = folder/'adapter.safetensors'
    metadata = json.loads((folder/'adapter-metadata.json').read_text())
    config = json.loads((folder/'policy-config.json').read_text())
    expected_base='0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01'
    if metadata['original_model_sha256']!=expected_base:
        raise ValueError('Adapter was trained against a different checkpoint')
    for name, key in [('adapter.safetensors','adapter_sha256'),
                      ('policy-config.json','config_sha256'),
                      ('normalization.json','normalization_sha256')]:
        if digest(folder/name)!=metadata[key]:raise ValueError(f'Adapter package hash mismatch: {name}')
    alpha = float(metadata['alpha']); rank = int(metadata['rank'])
    if not (rank>0 and alpha>0):raise ValueError('Invalid adapter rank/alpha')
    state = load_file(str(adapter), device='cpu')
    original_loader = so101.load_policy
    restored = {}

    class ProjectionInference(torch.nn.Module):
        def __init__(self, base, a, b):
            super().__init__()
            self.base=base.requires_grad_(False)
            self.A=torch.nn.Parameter(a.float().clone(),requires_grad=False)
            self.B=torch.nn.Parameter(b.float().clone(),requires_grad=False)
            self.in_features=base.in_features;self.out_features=base.out_features
            self._requires_bf16_input=False
        def forward(self,x):
            original=self.base(x)
            delta=torch.nn.functional.linear(
                torch.nn.functional.linear(x.to(self.A.dtype),self.A),self.B)*(alpha/rank)
            return original+delta.to(original.dtype)

    def cpu_load(*args, **kwargs):
        policy = original_loader(*args, **kwargs)
        original_config = json.loads(json.dumps(policy.config.to_dict()))
        # Only the simulator training normalizers and documented loss weights
        # may differ; preserve inference embodiment, timing and sampler.
        allowed = {'state_normalization','action_normalization',
                   'video_loss_weight','action_loss_weight',
                   'caption_dropout','augment','camera_dropout'}
        changed = {k for k in original_config if original_config[k]!=config[k]}
        if changed-allowed:
            raise ValueError(f'Unexpected policy contract changes: {changed-allowed}')
        policy.config = PolicyConfig(**config)
        parameters = dict(policy.dit.named_parameters())
        projection_names = {k[:-2] for k in state if k.endswith('.A')}
        expected_projections={name[:-7] for name in parameters
            if name.endswith('.weight') and re.search(r'\.(q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)\.weight$', '.'+name)}
        head_prefixes=('emb_in.action.','emb_in.action_cond.','final_layer.action.')
        expected_keys={name+suffix for name in expected_projections for suffix in ['.A','.B']}
        expected_keys.update(name for name in parameters if name.startswith(head_prefixes))
        if set(state)!=expected_keys:
            raise ValueError(f'Adapter tensor keys differ: missing={expected_keys-set(state)}, extra={set(state)-expected_keys}')
        consumed = set()
        max_rounding = 0.
        with torch.no_grad():
            for name in sorted(projection_names):
                a = state[name+'.A']; b = state[name+'.B']
                p = parameters[name+'.weight']
                if a.shape!=(rank,p.shape[1]) or b.shape!=(p.shape[0],rank):
                    raise ValueError(f'Rank mismatch: {name}')
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError(f'Nonfinite projection adapter: {name}')
                if restoration=='merged':
                    merged = p.float()+(b.float()@a.float())*(alpha/rank)
                    if not torch.isfinite(merged).all():raise ValueError(f'Nonfinite merge: {name}')
                    rounded = merged.to(p.dtype)
                    max_rounding=max(max_rounding,float((rounded.float()-merged).abs().max()))
                    p.copy_(rounded)
                else:
                    parent,leaf=name.rsplit('.',1)
                    owner=policy.dit.get_submodule(parent)
                    setattr(owner,leaf,ProjectionInference(policy.dit.get_submodule(name),a,b))
                consumed.update([name+'.A',name+'.B'])
            prefixes=('emb_in.action.','emb_in.action_cond.',
                      'final_layer.action.','final_layer.action_cond.')
            for name, value in state.items():
                if name in consumed:continue
                if not name.startswith(prefixes):raise ValueError(f'Unexpected trained tensor: {name}')
                p = parameters[name]
                if p.shape!=value.shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid head parameter: {name}')
                if restoration=='merged':p.copy_(value.to(p.dtype))
                else:p.data=value.detach().float().clone()
                consumed.add(name)
        if set(state)!=consumed:raise ValueError('Unconsumed adapter tensors')
        restored.update(projection_count=len(projection_names),trainable_tensors=len(state),
                        max_projection_bf16_rounding=max_rounding,
                        changed_config_fields=sorted(changed),mode=restoration)
        return policy

    with patch.object(so101,'load_policy',cpu_load):
        policy, result = runtime.load_staged_policy(prompts,resident_gib=resident_gib,progress=progress)
    if metadata['original_model_sha256']!=result['sha256']['model.safetensors']:
        raise ValueError('Adapter base checkpoint hash differs from loaded FLUX checkpoint')
    result['trained_adapter']={
        'directory':str(folder),'sha256':digest(adapter),
        'metadata_sha256':digest(folder/'adapter-metadata.json'),
        'config_sha256':digest(folder/'policy-config.json'),
        'metadata':metadata,'restoration':restored,
        'inference_precision':('Original BF16 base plus FP32 adapters and full heads under BF16 autocast'
            if restoration=='unmerged' else 'Original BF16 base with FP32-computed, BF16-rounded LoRA merge and heads'),
        'warning':('Unmerged restoration preserves training modules and precision; task success requires a separate closed-loop test.'
            if restoration=='unmerged' else 'Merged training and inference arithmetic differ by documented BF16 rounding; task success requires a separate closed-loop test.')}
    return policy,result

def predict_adapted(policy,batch):
    import torch
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return runtime.predict(policy,batch)
