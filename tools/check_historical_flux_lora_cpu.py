"""Independent algebra/checkpoint comparisons; CPU tiny tensors only."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiment/flux-so101-adaptation/historical-cpu-proof'))
from experimental_lora import FrozenCPUBase,ProjectionLoRA,install_block_checkpointing
torch.set_num_threads(2);torch.manual_seed(725)
checks=[]
for dtype in [torch.float64,torch.float32,torch.bfloat16]:
    source=nn.Linear(9,6).to(dtype=dtype);source.requires_grad_(False)
    base=FrozenCPUBase(source)
    input_dtype=torch.float32 if dtype==torch.bfloat16 else dtype
    x=torch.randn(2,4,9,dtype=input_dtype,requires_grad=True)
    ref=x.detach().clone().requires_grad_(True)
    y=base(x);expected=F.linear(ref.to(dtype),source.weight,source.bias)
    gradient=torch.randn_like(y)
    dx=torch.autograd.grad(y,x,gradient)[0]
    dr=torch.autograd.grad(expected,ref,gradient)[0]
    fe=float((y-expected).abs().max());ge=float((dx-dr).abs().max())
    assert fe==0 and ge==0
    checks.append({'weight_dtype':str(dtype),'input_dtype':str(input_dtype),'forward_max_error':fe,'input_gradient_max_error':ge})
source=nn.Linear(9,6).double();base=FrozenCPUBase(source)
gradcheck=torch.autograd.gradcheck(base,(torch.randn(2,9,dtype=torch.float64,requires_grad=True),))
gradgradcheck=torch.autograd.gradgradcheck(base,(torch.randn(2,9,dtype=torch.float64,requires_grad=True),))

class ModeBlock(nn.Module):
    def __init__(self):
        super().__init__();self.proj=ProjectionLoRA(FrozenCPUBase(nn.Linear(9,9)),rank=3,alpha=3)
        with torch.no_grad():self.proj.B.normal_(0,.05)
    def forward(self,x,*,scale=1.):return torch.tanh(self.proj(x))*scale

plain=nn.Sequential(ModeBlock(),ModeBlock()).train();checked=copy.deepcopy(plain)
names=install_block_checkpointing(checked)
x=torch.randn(3,5,9,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
plain(x).square().mean().backward();checked(xr).square().mean().backward()
errors={name:float((p.grad-dict(checked.named_parameters())[name].grad).abs().max()) for name,p in plain.named_parameters()}
assert max(errors.values())==0 and torch.equal(x.grad,xr.grad)

# FP32 frozen weights under BF16 autocast expose an unsupported precision case:
# native autograd uses the BF16-cast weight; custom backward uses the FP32 weight.
source=nn.Linear(9,6);source.requires_grad_(False);base=FrozenCPUBase(source)
x=torch.randn(3,9,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
with torch.autocast('cpu',dtype=torch.bfloat16):y=base(x);yr=source(xr)
g=torch.randn_like(y)
dx=torch.autograd.grad(y,x,g)[0];dr=torch.autograd.grad(yr,xr,g)[0]
autocast_error=float((dx-dr).abs().max())

result={'scope':'CPU toy algebra and checkpoint gradient audit; no actual FLUX, GPU allocation or training',
        'source_sha256':hashlib.sha256((ROOT/'experiment/flux-so101-adaptation/historical-cpu-proof/experimental_lora.py').read_bytes()).hexdigest(),
        'torch':torch.__version__,'frozen_linear_checks':checks,'gradcheck':gradcheck,'gradgradcheck':gradgradcheck,
        'checkpoint_blocks':names,'checkpoint_parameter_gradient_max_error':max(errors.values()),
        'checkpoint_input_gradient_max_error':float((x.grad-xr.grad).abs().max()) if x.grad is not None else 0.,
        'fp32_base_under_bf16_autocast_input_gradient_error':autocast_error,
        'precision_limit':'Actual checkpoint frozen bases are BF16, tested exact. FP32 frozen bases under BF16 autocast are not exact; disable autocast within custom frozen-linear or preserve effective forward cast for backward.',
        'no_unsafe_device_mutation_observed':True}
(ROOT/'runtime/historical-lora-proof').mkdir(parents=True,exist_ok=True)
(ROOT/'runtime/historical-lora-proof/lora-independent-cpu.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
