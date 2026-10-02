"""Real FLUX7B cached-encoder LoRA smoke training; explicit caller execution.

Produces a separately normalized simulator-domain adapter. Never executes a
robot controller, edits pretrained weights, uploads or claims task success.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from safetensors.torch import save_file
from flux_action.inference import so101
from flux_action.models.text_encoder import load_text_encoder
from flux_action.models.video_vae import load_video_vae
from flux_action.policy import PreparedWindows
from experimental_lora import replace_projections,install_block_checkpointing,adapter_and_head_state
from prepare_numeric import ROOT,to_units,window_arrays

PROMPT='Grasp the red block and place it in the gray bin.'

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def train_only_quantiles():
    states=[];targets=[];rows=[]
    for seed in range(20):
        if seed in (0,7,14):continue
        path=ROOT/f'runs/so101-teaser-dataset/seed-{seed:03d}/episode.npz'
        with np.load(path) as src:
            s=to_units(src['states']);a=to_units(src['actions'])
        delta=np.diff(a,axis=0);delta[:,-1]=a[1:,-1]
        states.append(s);targets.append(delta)
        rows.append({'seed':seed,'source_sha256':file_hash(path)})
    stats={name:{'q01':np.quantile(np.concatenate(values),.01,axis=0).tolist(),
                 'q99':np.quantile(np.concatenate(values),.99,axis=0).tolist()}
           for name,values in [('state',states),('action',targets)]}
    return {'statistics':stats,'scope':'New native simulator-domain normalization; training episodes only; no legacy-coordinate mapping',
        'units':['degrees']*5+['gripper percentage'],'action_targets':'Consecutive command deltas, absolute gripper; first artificial zero-delta row excluded per episode',
        'training_episodes':rows,'validation_seeds':[0,7,14],'control_hz':30,'camera_projection':'320x240 rendered then each view resized256x256'}

def copy_prepared(prepared,device):
    return PreparedWindows(idx=list(prepared.idx),latents=prepared.latents.detach().clone().to(device),
        ctxs=[x.detach().clone().to(device) for x in prepared.ctxs],state=prepared.state.detach().clone().to(device),
        actions=prepared.actions.detach().clone().to(device))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',default='150')
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=int,default=8)
    p.add_argument('--backend',choices=('cpu','nf4','int8'),default='cpu')
    p.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new experiment output directory')
    a.output.mkdir(parents=True)
    for script in (Path(__file__),Path(__file__).with_name('experimental_lora.py')):
        (a.output/script.name).write_bytes(script.read_bytes())
    report={'scope':'Actual pretrained FLUX7B native flow-loss training feasibility; not a robot success test',
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'completed':False,'phases':[]}
    def event(name,**kw):
        row={'phase':name,'elapsed_seconds':time.perf_counter()-started,**kw}
        report['phases'].append(row);print(json.dumps(row),flush=True)
        (a.output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n')
    started=time.perf_counter()
    try:
        torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
        torch.cuda.reset_peak_memory_stats()
        event('load_cpu_encoders')
        base=ROOT/'models/flux-action/base';package=ROOT/'models/flux-action/so101'
        encoder=load_text_encoder(str(base/'text_encoder'),compile_model=False)
        vae=load_video_vae(str(base/'video_vae.safetensors'),compile_model=False)
        real_constructor=so101.FluxActionPolicy
        with patch.object(so101,'FluxActionPolicy',lambda config,**kw:real_constructor(config,video_vae=vae,text_encoder=encoder,**kw)):
            policy=so101.load_policy(package)
        event('loaded_real_cpu_policy',parameters=sum(p.numel() for p in policy.dit.parameters()))
        normalizers=train_only_quantiles()
        (a.output/'normalization.json').write_text(json.dumps(normalizers,indent=2)+'\n')
        policy.config.state_normalization=normalizers['statistics']['state']
        policy.config.action_normalization=normalizers['statistics']['action']
        policy.config.caption_dropout=0.;policy.config.augment=False;policy.config.camera_dropout={}
        (a.output/'policy-config.json').write_text(json.dumps(policy.config.to_dict(),indent=2)+'\n')
        # First parameter determines policy.device. Only this tiny head moves during frozen caching.
        policy.dit.emb_in.to('cuda')
        if policy.device.type!='cuda':raise RuntimeError('Cache preparation requires CUDA first-parameter device')
        policy.set_text_encoder_offload(True)
        event('cache_text')
        policy._context(PROMPT,torch.device('cuda'))
        policy._context('',torch.device('cuda'))
        policy.video_vae.module.to('cuda');policy.eval()
        with np.load(a.episode) as episode:
            states=episode['states'].copy();commands=episode['commands'].copy()
            scene=episode['images_scene'].copy();wrist=episode['images_wrist'].copy()
        caches=[]
        for t in map(int,a.starts.split(',')):
            arrays=window_arrays(states,commands,t);indices=arrays['image_indices']
            batch={'state':torch.from_numpy(arrays['state'])[None],
                'command_history':torch.from_numpy(arrays['command_history'])[None],
                'action':torch.from_numpy(arrays['action'])[None],'task':[PROMPT],
                'images.scene':torch.from_numpy(scene[indices].transpose(0,3,1,2).copy())[None],
                'images.wrist':torch.from_numpy(wrist[indices].transpose(0,3,1,2).copy())[None]}
            event('encode_window',action_start=t)
            prepared=policy.prepare(batch)
            cpu=copy_prepared(prepared,'cpu');caches.append(cpu)
            torch.save({'idx':cpu.idx,'latents':cpu.latents,'ctxs':cpu.ctxs,'state':cpu.state,'actions':cpu.actions},a.output/f'cached-window-{t}.pt')
            event('cached_window',action_start=t,latent_shape=list(cpu.latents.shape),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del scene,wrist,prepared,batch
        policy.video_vae.module.cpu();policy.text_encoder.cpu();policy.dit.cpu();policy.reset()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        event('install_actual_lora',backend=a.backend,rank=a.rank)
        rows=replace_projections(policy.dit,backend=a.backend,rank=a.rank,alpha=a.alpha)
        blocks=install_block_checkpointing(policy.dit)
        policy.dit.to('cuda');policy.train()
        # Embodiment-conditioning output is never in the loss; do not optimize dead heads.
        policy.dit.final_layer['action_cond'].requires_grad_(False)
        trainable=[p for p in policy.dit.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.)
        initial={k:v.clone() for k,v in adapter_and_head_state(policy.dit).items()}
        report.update(trainable_parameters=sum(p.numel() for p in trainable),projection_count=len(rows),
            checkpointed_blocks=blocks,original_model_sha256=file_hash(package/'model.safetensors'),
            native_episode_sha256=file_hash(a.episode),training_scope='Small overfit feasibility on explicitly selected expert windows',
            precision='Original BF16 frozen base, FP32 adapters/full heads, BF16 autocast' if a.backend=='cpu' else 'Quantized frozen projections with FP32 adapters/heads and BF16 compute')
        event('begin_real_gradient_updates',trainable_parameters=report['trainable_parameters'],resident_gpu_bytes=torch.cuda.memory_allocated())
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            prepared=copy_prepared(caches[step%len(caches)],'cuda')
            with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):
                loss,info=policy({},prepared=prepared)
            if not torch.isfinite(loss):raise ValueError('Nonfinite real FLUX loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            event('optimizer_update',step=step+1,loss=float(loss.detach()),grad_norm=float(norm),
                video_mse=float(info['video_mse']),action_mse=float(info['action_mse']),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),resident_gpu_bytes=torch.cuda.memory_allocated())
        final=adapter_and_head_state(policy.dit)
        changed=[name for name in final if not torch.equal(initial[name],final[name])]
        if not any(name.endswith('.B') for name in changed):raise ValueError('No actual LoRA B update occurred')
        save_file(final,str(a.output/'adapter.safetensors'))
        metadata={'format':'Native dit-relative trainable heads plus projection .A/.B tensors',
            'rank':a.rank,'alpha':a.alpha,'backend':a.backend,'original_model_sha256':report['original_model_sha256'],
            'adapter_sha256':file_hash(a.output/'adapter.safetensors'),'config_sha256':file_hash(a.output/'policy-config.json'),
            'normalization_sha256':file_hash(a.output/'normalization.json'),'merge':'W <- original BF16 W + alpha/rank*(B@A); full heads copied by same name',
            'calibration':'Native simulator degrees and gripper percentage; no pretrained legacy offset/sign map',
            'sampling':'Original checkpoint settings retained','claimed_robot_success':False,
            'excluded_dead_head':'final_layer.action_cond (not part of native training loss)',
            'source_revision':'e2dd1d8dbc5977b54315d61f7548c63c043d6d4f'}
        (a.output/'adapter-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        report.update(completed=True,changed_trainable_tensors=changed,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        event('saved_actual_adapter',changed_tensors=len(changed))
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        event('failed');raise

if __name__=='__main__':main()
