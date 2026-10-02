"""Actual safetensors + original loader merge using tiny mocked CPU policy."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
from flux_action.config import PolicyConfig
from flux_action.inference import so101
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'runtime/flux-so101-adaptation'))
import flux_adapted_inference as integration
from experimental_lora import FrozenCPUBase,ProjectionLoRA
torch.set_num_threads(2);torch.manual_seed(935)
OUT=Path(__file__).parent/'tiny-adapted-loader'
OUT.mkdir(exist_ok=True)
cfg=PolicyConfig(action_dim=6)
BASE_HASH='0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01'

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.ModuleDict({name:nn.Linear(8,8,bias=False) for name in ['q_proj','k_proj','v_proj','attn_out','mlp_in','mlp_out']})])
        self.emb_in=nn.ModuleDict({'action':nn.Linear(6,8,bias=False),'action_cond':nn.Linear(12,8,bias=False)})
        self.final_layer=nn.ModuleDict({'action':nn.Linear(8,6,bias=False),'action_cond':nn.Linear(8,12,bias=False)})

template=Tiny().bfloat16()
state={}
for name,layer in template.named_modules():
    if name.startswith('blocks.') and isinstance(layer,nn.Linear):
        state[name+'.A']=torch.randn(2,8)*.04
        state[name+'.B']=torch.randn(8,2)*.03
for name,p in template.named_parameters():
    if name.startswith(('emb_in.action.','emb_in.action_cond.','final_layer.action.')):state[name]=p.float()+.07

def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def prepare(tensors=state,contract=None):
    save_file(tensors,str(OUT/'adapter.safetensors'))
    config=json.loads(json.dumps(cfg.to_dict()))
    config['state_normalization']={'q01':[0.]*6,'q99':[1.]*6}
    config['action_normalization']={'q01':[-1.]*6,'q99':[1.]*6}
    if contract:config.update(contract)
    (OUT/'policy-config.json').write_text(json.dumps(config))
    (OUT/'normalization.json').write_text(json.dumps({'statistics':{'state':config['state_normalization'],'action':config['action_normalization']}}))
    metadata={'rank':2,'alpha':2,'backend':'cpu','original_model_sha256':BASE_HASH,
        'adapter_sha256':h(OUT/'adapter.safetensors'),'config_sha256':h(OUT/'policy-config.json'),
        'normalization_sha256':h(OUT/'normalization.json')}
    (OUT/'adapter-metadata.json').write_text(json.dumps(metadata))

def base_loader(*args,**kwargs):return SimpleNamespace(config=copy.deepcopy(cfg),dit=copy.deepcopy(template))
def placement(prompts,**kwargs):
    policy=so101.load_policy('mock-cpu-original')
    assert all(p.device.type=='cpu' for p in policy.dit.parameters())
    return policy,{'sha256':{'model.safetensors':BASE_HASH}}
def load(mode='unmerged'):
    with patch.object(so101,'load_policy',base_loader),patch.object(integration.runtime,'load_staged_policy',placement):
        return integration.load_adapted_policy(OUT,['tiny-test'],restoration=mode)

result={'scope':'CPU tiny mocked policy, actual safetensors and actual adapted loader; no real pretrained weights or GPU',
    'loader_sha256':h(ROOT/'scripts/flux_adapted_inference.py')}
result['restorations']={}
prepare()
for mode in ['merged','unmerged']:
    try:
        policy,report=load(mode)
        errors=[];forward_errors=[]
        for name,p in template.named_parameters():
            if name.startswith('blocks.'):
                prefix=name[:-7]
                restored=policy.dit.get_submodule(prefix)
                if mode=='merged':
                    expected=(p.float()+state[prefix+'.B']@state[prefix+'.A']).bfloat16()
                    errors.append(float((restored.weight-expected).detach().float().abs().max()))
                else:
                    errors.append(float((restored.base.weight-p).detach().float().abs().max()))
                    errors.append(float((restored.A-state[prefix+'.A']).detach().abs().max()))
                    errors.append(float((restored.B-state[prefix+'.B']).detach().abs().max()))
                    training=ProjectionLoRA(FrozenCPUBase(template.get_submodule(prefix)),rank=2,alpha=2)
                    with torch.no_grad():training.A.copy_(state[prefix+'.A']);training.B.copy_(state[prefix+'.B'])
                    x=torch.randn(2,4,8)
                    with torch.autocast('cpu',dtype=torch.bfloat16):
                        actual=restored(x);expected=training(x)
                    forward_errors.append(float((actual-expected).detach().float().abs().max()))
            else:
                got=dict(policy.dit.named_parameters())[name]
                expected=(state[name].bfloat16() if mode=='merged' else state[name].float()) if name in state else p
                errors.append(float((got-expected).detach().float().abs().max()))
                if mode=='unmerged' and name in state:assert got.dtype==torch.float32
        assert max(errors)==0 and max(forward_errors or [0.])==0
        result['restorations'][mode]={'passed':True,'weight_head_max_error':max(errors),
            'unmerged_training_forward_max_error':max(forward_errors or [0.]),
            'restoration':report['trained_adapter']['restoration']}
    except Exception as e:result['restorations'][mode]={'passed':False,'error':f'{type(e).__name__}: {e}'}
negative={}
for name,tensors,contract in [
    ('missing_projection', {k:v for k,v in state.items() if not k.startswith('blocks.0.q_proj.')},None),
    ('missing_head', {k:v for k,v in state.items() if k!='emb_in.action.weight'},None),
    ('changed_sampler',state,{'sampler':'unrecognized'}),
    ('unexpected_head_key',{**state,'final_layer.unexpected.weight':torch.ones(1)},None),
]:
    prepare(tensors,contract)
    try:load('unmerged');negative[name]={'rejected':False}
    except Exception as e:negative[name]={'rejected':True,'error':f'{type(e).__name__}: {e}'}
result['negative_controls']=negative
prepare()
(Path(__file__).parent/'adapted-loader-independent-cpu.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
