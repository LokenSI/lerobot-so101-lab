"""Independent CPU-only FLUX data/cache audit; no encoders or policy loaded."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'runtime/flux-so101-adaptation/train-broad-v1'
NATIVE=ROOT/'runtime/flux-so101-adaptation/native320x240-v1'
torch.set_num_threads(2)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def units(q):
    x=np.array(q,dtype=np.float64,copy=True)
    x[:,:5]=np.rad2deg(x[:,:5]);x[:,5]=(x[:,5]+.17453)/(1.74533+.17453)*100
    return x.astype(np.float32)
def norm(x,stats,clip):
    lo=np.asarray(stats['q01'],dtype=np.float32);hi=np.asarray(stats['q99'],dtype=np.float32)
    span=hi-lo;span=np.where(span>1e-6,span,1.).astype(np.float32)
    return np.clip(2*(x-lo)/span-1,-clip,clip)

plan=json.loads((FOLDER/'frozen-plan.json').read_text())
config=json.loads((FOLDER/'policy-config.json').read_text())
normalizers=json.loads((FOLDER/'normalization.json').read_text())
stats=normalizers['statistics'];checks={};episodes={};numeric_rows=[]
def check(name,ok,**details):checks[name]={'passed':bool(ok),**details}
def near(name,a,b,tol=2e-5):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    check(name,err<=tol,max_error=err,tolerance=tol)
check('split_disjoint_complete',set(plan['training_seeds']).isdisjoint(plan['validation_seeds']) and set(plan['training_seeds'])|set(plan['validation_seeds'])==set(range(20)))
check('validation_excluded_statistics',normalizers['validation_seeds']==[0,7,14] and set(r['seed'] for r in normalizers['training_episodes'])==set(plan['training_seeds']))
check('statistics_config_identical',config['state_normalization']==stats['state'] and config['action_normalization']==stats['action'])
check('model_pinned_hash_recorded',plan['original_model_sha256']=='0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01')
for name,digest in plan['source_sha256'].items():check('frozen_training_source_'+name,sha(FOLDER/name)==digest)
allstates=[];alldeltas=[]
for seed in range(20):
    path=NATIVE/f'seed-{seed:03d}/episode.npz'
    nativehash=sha(path)
    check(f'native_seed{seed}_hash',nativehash==plan['episode_hashes'][str(seed)])
    with np.load(path,allow_pickle=False) as data:
        s=data['states'];a=data['commands'];times=data['time_s']
    with zipfile.ZipFile(path) as archive:
        for name in ['images_scene.npy','images_wrist.npy']:
            with archive.open(name) as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                check(f'seed{seed}_{name}_header',shape==(525,256,256,3) and dtype==np.uint8 and not fortran)
    with np.load(ROOT/f'runs/so101-teaser-dataset/seed-{seed:03d}/episode.npz',allow_pickle=False) as original:
        expected_s=units(original['states']);expected_a=units(original['actions'])
    near(f'seed{seed}_native_measured_state_units',s,expected_s,1e-6)
    near(f'seed{seed}_native_issued_command_units',a,expected_a,1e-6)
    check(f'seed{seed}_numeric_finite',np.isfinite(s).all() and np.isfinite(a).all() and np.isfinite(times).all())
    near(f'seed{seed}_30hz_time',np.diff(times),np.full(len(times)-1,1/30),1e-10)
    episodes[seed]=(s,a)
    if seed in plan['training_seeds']:
        original_path=ROOT/f'runs/so101-teaser-dataset/seed-{seed:03d}/episode.npz'
        recorded=next(r['source_sha256'] for r in normalizers['training_episodes'] if r['seed']==seed)
        check(f'seed{seed}_normalization_source_hash',sha(original_path)==recorded)
        allstates.append(s);delta=np.diff(a,axis=0);delta[:,-1]=a[1:,-1];alldeltas.append(delta)
for name,values in [('state',allstates),('action',alldeltas)]:
    combined=np.concatenate(values)
    for quantile,p in [('q01',.01),('q99',.99)]:near(f'train_only_{name}_{quantile}',np.quantile(combined,p,axis=0),stats[name][quantile],1e-12)

print('Verified all20nativeepisodes/train-onlystatistics; loadingCPUcache',flush=True)
bundle=torch.load(FOLDER/'cached-windows.pt',map_location='cpu',weights_only=True)
check('cached_window_counts',len(bundle['train'])==255 and len(bundle['val'])==45)
trainseeds=set();valseeds=set();state_error=action_error=0.;latent_shapes=set();context_shapes=set()
for split,rows in bundle.items():
    expected_seeds=plan['validation_seeds'] if split=='val' else plan['training_seeds']
    seen=set()
    for row in rows:
        seed,start=int(row['seed']),int(row['start']);seen.add((seed,start))
        (valseeds if split=='val' else trainseeds).add(seed)
        assert seed in expected_seeds and start in plan['window_action_starts']
        s,a=episodes[seed]
        # State/images include t; command history stops at t-1. First past
        # delta channel row is explicitly all zeros, including gripper.
        measured=norm(s[start-7:start+1],stats['state'],config['normalization_clip'])
        previous=a[start-8:start]
        deltas=np.diff(previous,axis=0);deltas[:,-1]=previous[1:,-1]
        normalized_previous=np.concatenate([np.zeros((1,6),dtype=np.float32),norm(deltas,stats['action'],config['normalization_clip'])])
        state=np.concatenate([normalized_previous,measured],axis=1)[None]
        future=a[start:start+42]
        target=future-np.concatenate([a[start-1:start],future[:-1]])
        target[:,-1]=future[:,-1]
        target=norm(target,stats['action'],config['normalization_clip'])[None]
        state_error=max(state_error,float(np.max(np.abs(row['state'].numpy()-state))))
        action_error=max(action_error,float(np.max(np.abs(row['actions'].numpy()-target))))
        check(f'cache_{split}_{seed}_{start}_shapes',tuple(row['state'].shape)==(1,8,12) and tuple(row['actions'].shape)==(1,42,6) and row['idx']==[0])
        tensors=[row['state'],row['actions'],row['latents'],*row['ctxs']]
        check(f'cache_{split}_{seed}_{start}_finite_cpu',all(t.device.type=='cpu' and torch.isfinite(t).all().item() for t in tensors))
        latent_shapes.add(tuple(row['latents'].shape));context_shapes.add(tuple(row['ctxs'][0].shape))
        indices=np.arange(start-7,start+43)
        assert len(indices)==50 and indices[0]>=0 and indices[-1]<len(s)
    check(f'{split}_episode_window_coverage',seen=={(seed,start) for seed in expected_seeds for start in plan['window_action_starts']})
check('no_validation_optimizer_windows',trainseeds==set(plan['training_seeds']) and valseeds=={0,7,14} and trainseeds.isdisjoint(valseeds))
check('cached_state_history_normalization',state_error<2e-5,max_error=state_error)
check('cached_future_deltas_absolute_gripper',action_error<2e-5,max_error=action_error)
check('latent_shapes',latent_shapes=={(1,96,13,8,16)},observed=[list(s) for s in latent_shapes])
check('context_shapes',all(s[0]==1 and s[1]==320 for s in context_shapes),observed=[list(s) for s in context_shapes])
out={'scope':'CPU-only numeric/provenance/cache audit; no originalmodel/encoder loaded, GPUoperations or latentdecode',
     'checker_sha256':sha(Path(__file__)),'checks':checks,'passed':all(c['passed'] for c in checks.values()),
     'cache_sha256':sha(FOLDER/'cached-windows.pt'),'frozen_plan_sha256':sha(FOLDER/'frozen-plan.json'),
     'window_counts':{'train':255,'val':45},'train_only_normalization':True,
     'latent_provenance_note':'Frozen actualencoder-cache producer source, exactnativeepisodehashes, finite/shapes verified. No GPU decoder used to independently reproduce latent pixels.'}
(FOLDER/'independent-cache-cpu-audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':out['passed'],'window_counts':out['window_counts'],'state_max_error':state_error,'action_max_error':action_error,'failed_checks':[k for k,v in checks.items() if not v['passed']]},indent=2))
raise SystemExit(0 if out['passed'] else 1)
