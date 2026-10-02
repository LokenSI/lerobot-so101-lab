"""Independent CPU audit of current partial-residency helper, no GPU calls."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'runtime/flux-so101-adaptation'))
from experimental_lora import FrozenCPUBase,ProjectionLoRA,place_frozen_base_budget
torch.set_num_threads(2);torch.manual_seed(835)
base=FrozenCPUBase(nn.Linear(7,5).bfloat16())
x=torch.randn(2,7,requires_grad=True);y=base(x)
with torch.no_grad():base.weight.add_(.25)
mutation=False
try:y.float().sum().backward()
except RuntimeError as error:mutation='mutated between forward and backward' in str(error)
assert mutation

layer=ProjectionLoRA(FrozenCPUBase(nn.Linear(7,7).bfloat16()),rank=3,alpha=3)
with torch.no_grad():layer.B.normal_(0,.04)
reference=copy.deepcopy(layer)
x=torch.randn(3,4,7,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
base_sha=hashlib.sha256(layer.base.weight.view(torch.uint8).numpy().tobytes()).hexdigest()
reference(xr).float().square().mean().backward()
with torch.autograd.graph.save_on_cpu(pin_memory=False):
    loss=checkpoint(layer,x,use_reentrant=False).float().square().mean()
loss.backward()
paramerror=max(float((p.grad-dict(reference.named_parameters())[name].grad).abs().max()) for name,p in layer.named_parameters())
inputerror=float((x.grad-xr.grad).abs().max())
assert paramerror==0 and inputerror==0
new_sha=hashlib.sha256(layer.base.weight.view(torch.uint8).numpy().tobytes()).hexdigest()
assert base_sha==new_sha
# Zero budget tests validation/metadata without executing any CUDA movement.
placement=place_frozen_base_budget(layer,0.)
assert placement['resident_base_bytes']==0 and not placement['resident_base_modules']
pointer=layer.base.weight.data_ptr();dtype=layer.base.weight.dtype
layer.to(dtype=torch.float64)
assert layer.base.weight.data_ptr()==pointer and layer.base.weight.dtype==dtype
result={'scope':'CPU-only current helper mutation/checkpoint/save_on_cpu/zero-budget/_apply audit; actualCUDAplacement not tested',
    'helper_sha256':hashlib.sha256((ROOT/'runtime/flux-so101-adaptation/experimental_lora.py').read_bytes()).hexdigest(),
    'mutation_guard':mutation,'checkpoint_parameter_gradient_error':paramerror,'checkpoint_input_gradient_error':inputerror,
    'frozen_buffer_hash_unchanged':base_sha==new_sha,'outer_to_preserves_frozen_base_buffer':True,'zero_budget_metadata':placement,
    'review_notes':['ctx.weight directreference bypasses savedtensor CPU hooks for frozenweights only',
                    'Callplacement onlyafterencoder caching/release andbefore anytraining graph',
                    'BF16frozenbase required byplacement; FP32base underBF16autocast remains unsupported',
                    'Start8GiBresidentbudget conservative;10GiB requiresactualfullwindowpeak/globalfreeVRAM check',
                    'Do not mutatebuffers via.data or replace them whilegraph alive; versionguard onlycatchesordinaryinplace modifications']}
(Path(__file__).parent/'current-residency-cpu-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
