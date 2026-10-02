"""Real FLUX adaptation on fixed17/3 expert split and native frozen encodings.

Default stops at checkpoint128 for development rollout, with a frozen400-update
budget. Resume preserves optimizer/RNG. No hardware, publishing, or task claim.
"""
from __future__ import annotations
import argparse,json,random,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from flux_action.inference import so101
from flux_action.config import PolicyConfig
from flux_action.models.text_encoder import load_text_encoder
from flux_action.models.video_vae import load_video_vae
from flux_action.policy import PreparedWindows
from experimental_lora import replace_projections,install_block_checkpointing,place_frozen_base_budget,adapter_and_head_state,load_adapter_and_head_state
from train_smoke import ROOT,PROMPT,copy_prepared,train_only_quantiles,file_hash
from prepare_numeric import window_arrays

VAL={0,7,14};STARTS=list(range(8,457,32))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native',type=Path,default=Path(__file__).parent/'native320x240-v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-at',type=int,default=128)
    p.add_argument('--planned-steps',type=int,default=400)
    p.add_argument('--resume',type=Path)
    p.add_argument('--resident-base-gib',type=float,default=10.)
    a=p.parse_args()
    if not 0<a.stop_at<=a.planned_steps:raise ValueError('Invalid frozen update budget')
    if a.resume is None and a.output.exists():raise FileExistsError('New experiment needs a new output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    for script in (Path(__file__),Path(__file__).with_name('experimental_lora.py'),Path(__file__).with_name('train_smoke.py')):
        snapshot=a.output/script.name
        if not snapshot.exists():snapshot.write_bytes(script.read_bytes())
    started=time.perf_counter();report={'scope':'Actual FLUX7B task adaptation; offline losses do not establish robot task success','events':[],'completed_requested_stage':False}
    def event(phase,**kw):
        row={'phase':phase,'elapsed_seconds':time.perf_counter()-started,**kw};report['events'].append(row)
        (a.output/('training-report-resume.json' if a.resume else 'training-report.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        package=ROOT/'models/flux-action/so101';base=ROOT/'models/flux-action/base'
        norm=train_only_quantiles();stats=norm['statistics']
        cache_path=a.output/'cached-windows.pt'
        if a.resume:
            event('load_original_cpu_policy_for_resume')
            constructor=so101.FluxActionPolicy
            with patch.object(so101,'FluxActionPolicy',lambda config,**kw:constructor(config,video_vae=nn.Identity(),text_encoder=nn.Identity(),**kw)):
                policy=so101.load_policy(package)
            policy.config=PolicyConfig(**json.loads((a.output/'policy-config.json').read_text()))
            frozen=json.loads((a.output/'frozen-plan.json').read_text())
            if frozen['planned_optimizer_updates']!=a.planned_steps:raise ValueError('Frozen budget changed on resume')
            if json.loads((a.output/'normalization.json').read_text())!=norm:raise ValueError('Training normalizers/data changed on resume')
            bundle=torch.load(cache_path,map_location='cpu',weights_only=True)
        else:
            event('load_cpu_real_encoders_and_policy')
            encoder=load_text_encoder(str(base/'text_encoder'),compile_model=False)
            vae=load_video_vae(str(base/'video_vae.safetensors'),compile_model=False)
            constructor=so101.FluxActionPolicy
            with patch.object(so101,'FluxActionPolicy',lambda config,**kw:constructor(config,video_vae=vae,text_encoder=encoder,**kw)):
                policy=so101.load_policy(package)
            policy.config.state_normalization=stats['state'];policy.config.action_normalization=stats['action']
            policy.config.caption_dropout=0.;policy.config.augment=False;policy.config.camera_dropout={}
            (a.output/'normalization.json').write_text(json.dumps(norm,indent=2)+'\n')
            (a.output/'policy-config.json').write_text(json.dumps(policy.config.to_dict(),indent=2)+'\n')
            frozen={'task_instruction':PROMPT,'training_seeds':[s for s in range(20) if s not in VAL],'validation_seeds':sorted(VAL),
                'window_action_starts':STARTS,'planned_optimizer_updates':a.planned_steps,'pilot_stop_at':a.stop_at,
                'rank':8,'alpha':8,'lr_adapters':1e-4,'lr_heads':5e-4,'optimizer':'AdamW','weight_decay':0.,
                'normalization':'Train-only native simulator degrees + gripper percentage; no legacy calibration map',
                'camera_projection':'320x240 native render, resized256perview','EMA':False,
                'original_model_sha256':file_hash(package/'model.safetensors'),'episode_hashes':{},
                'source_sha256':{script.name:file_hash(script) for script in (Path(__file__),Path(__file__).with_name('experimental_lora.py'))}}
            for seed in range(20):
                episode=a.native/f'seed-{seed:03d}/episode.npz'
                rr=json.loads(episode.with_name('report.json').read_text())
                if not rr['success'] or rr['replay_max_state_error_rad']>=1e-8:raise ValueError(f'Expert replay failed at seed {seed}')
                actual=file_hash(episode)
                if actual!=rr['native_episode_sha256']:raise ValueError('Expert native episode changed')
                frozen['episode_hashes'][str(seed)]=actual
            (a.output/'frozen-plan.json').write_text(json.dumps(frozen,indent=2)+'\n')
            policy.dit.emb_in.to('cuda');assert policy.device.type=='cuda'
            policy.set_text_encoder_offload(True);event('cache_real_text')
            policy._context(PROMPT,torch.device('cuda'));policy.video_vae.module.to('cuda');policy.eval()
            bundle={'train':[],'val':[]};shared_ctx=None
            for seed in range(20):
                with np.load(a.native/f'seed-{seed:03d}/episode.npz') as ep:
                    states=ep['states'].copy();commands=ep['commands'].copy();scene=ep['images_scene'].copy();wrist=ep['images_wrist'].copy()
                split='val' if seed in VAL else 'train'
                for start in STARTS:
                    ar=window_arrays(states,commands,start);ix=ar['image_indices']
                    batch={'state':torch.from_numpy(ar['state'])[None],'command_history':torch.from_numpy(ar['command_history'])[None],
                        'action':torch.from_numpy(ar['action'])[None],'task':[PROMPT],
                        'images.scene':torch.from_numpy(scene[ix].transpose(0,3,1,2).copy())[None],
                        'images.wrist':torch.from_numpy(wrist[ix].transpose(0,3,1,2).copy())[None]}
                    cpu=copy_prepared(policy.prepare(batch),'cpu')
                    if shared_ctx is None:shared_ctx=cpu.ctxs[0]
                    bundle[split].append({'seed':seed,'start':start,'idx':cpu.idx,'latents':cpu.latents,
                        'ctxs':[shared_ctx],'state':cpu.state,'actions':cpu.actions})
                event('cached_episode',seed=seed,split=split,windows=len(STARTS),peak_gpu_bytes=torch.cuda.max_memory_allocated())
            torch.save(bundle,cache_path)
            del scene,wrist,batch,cpu
            policy.video_vae.module.cpu();policy.text_encoder.cpu();policy.dit.cpu();policy.reset()
            # These modules are never invoked after caching; release actual encoder host storage.
            policy.frozen.video_vae=nn.Identity();policy.frozen.text_encoder=nn.Identity();del encoder,vae
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        event('install_real_adapters')
        replace_projections(policy.dit,backend='cpu',rank=8,alpha=8);install_block_checkpointing(policy.dit)
        policy.dit.final_layer['action_cond'].requires_grad_(False)
        if a.resume:load_adapter_and_head_state(policy.dit,load_file(str(a.resume/'adapter.safetensors')))
        policy.dit.to('cuda');placement=place_frozen_base_budget(policy.dit,a.resident_base_gib)
        heads=[];adapters=[]
        for name,param in policy.dit.named_parameters():
            if param.requires_grad:(adapters if name.endswith(('.A','.B')) else heads).append(param)
        opt=torch.optim.AdamW([{'params':adapters,'lr':1e-4},{'params':heads,'lr':5e-4}],weight_decay=0.)
        begin=0
        if a.resume:
            state=torch.load(a.resume/'optimizer-state.pt',map_location='cpu',weights_only=True)
            opt.load_state_dict(state['optimizer']);begin=state['step'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        schedule=random.Random(42).choices(range(len(bundle['train'])),k=a.planned_steps)
        report.update(frozen_plan=frozen,placement=placement,train_windows=len(bundle['train']),validation_windows=len(bundle['val']),begin_step=begin)
        event('ready_for_real_updates',resident_gpu_bytes=torch.cuda.memory_allocated())
        def prepared(row):return PreparedWindows(**{k:row[k] for k in ('idx','latents','ctxs','state','actions')})
        def validation(step):
            policy.eval();values=[]
            # Fixed seeds retain the same flow-noise realization for each held-out window.
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for index,row in enumerate(bundle['val']):
                    torch.manual_seed(10000+index);loss,info=policy({},prepared=copy_prepared(prepared(row),'cuda'))
                    values.append({'seed':row['seed'],'start':row['start'],'loss':float(loss),'action_mse':float(info['action_mse']),'video_mse':float(info['video_mse'])})
            torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
            (a.output/f'validation-step-{step:04d}.json').write_text(json.dumps({'scope':'Held-out native flow loss only, not task success','windows':values,
                'mean_loss':float(np.mean([v['loss'] for v in values])),'mean_action_mse':float(np.mean([v['action_mse'] for v in values]))},indent=2)+'\n')
            policy.train();event('heldout_flow_validation',step=step,mean_loss=float(np.mean([v['loss'] for v in values])))
        if begin==0:validation(0)
        def save(step):
            ck=a.output/f'checkpoint-{step:04d}';ck.mkdir(exist_ok=False)
            save_file(adapter_and_head_state(policy.dit),str(ck/'adapter.safetensors'))
            for name in ('policy-config.json','normalization.json'):(ck/name).write_bytes((a.output/name).read_bytes())
            metadata={'rank':8,'alpha':8,'task_instruction':PROMPT,'backend':'original-bf16-fixed-placement','original_model_sha256':frozen['original_model_sha256'],
                'adapter_sha256':file_hash(ck/'adapter.safetensors'),'config_sha256':file_hash(ck/'policy-config.json'),
                'normalization_sha256':file_hash(ck/'normalization.json'),'optimizer_updates':step,
                'format':'Native dit-relative FP32 trainable heads plus projection .A/.B','EMA':False,'claimed_robot_success':False,
                'sampling':'Original checkpoint sampler/history/canvas retained','calibration':'Native simulator degrees/gripper percentage, no legacy offset/sign mapping'}
            (ck/'adapter-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            torch.save({'step':step,'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},ck/'optimizer-state.pt')
            event('checkpoint_saved',step=step,path=str(ck),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        for step in range(begin,a.stop_at):
            policy.train();opt.zero_grad(set_to_none=True);row=bundle['train'][schedule[step]]
            with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):
                loss,info=policy({},prepared=copy_prepared(prepared(row),'cuda'))
            if not torch.isfinite(loss):raise ValueError('Nonfinite actual FLUX loss')
            loss.backward();normgrad=torch.nn.utils.clip_grad_norm_(adapters+heads,1.,error_if_nonfinite=True);opt.step()
            if step<4 or (step+1)%8==0:
                torch.cuda.synchronize();event('optimizer_update',step=step+1,seed=row['seed'],start=row['start'],loss=float(loss.detach()),
                    action_mse=float(info['action_mse']),grad_norm=float(normgrad),peak_gpu_bytes=torch.cuda.max_memory_allocated())
            if (step+1)%128==0 or step+1==a.stop_at:
                save(step+1);validation(step+1)
        report.update(completed_requested_stage=True,optimizer_updates=a.stop_at,full_budget_finished=a.stop_at==a.planned_steps)
        event('stage_complete')
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());event('failed');raise

if __name__=='__main__':main()
