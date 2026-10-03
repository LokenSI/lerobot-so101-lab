"""Paired-instruction checkpoint evaluation with only RGB/joint/language policy inputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
import office_environment as office

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--policy-module',default='office_smolvla_adapter');p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[20000,20001])
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--tasks',nargs='*')
    p.add_argument('--wording',choices=['train','validation'],default='validation')
    p.add_argument('--development-training-scenes',action='store_true')
    p.add_argument('--image-size',type=int,default=256);p.add_argument('--chunk-execution',type=int,default=8)
    p.add_argument('--max-steps',type=int);p.add_argument('--host-monitor',type=Path);p.add_argument('--policy-rng-seed-base',type=int,default=500000)
    p.add_argument('--retain-predictions',action='store_true',help='Save full fresh chunks, actual input hashes and first three RGB/state observations')
    p.add_argument('--policy-rng-seed-fixed',type=int);p.add_argument('--concurrency-note',default='not specified');a=p.parse_args()
    if a.split=='test' and min(a.seeds)<20000:raise ValueError('Sealed test starts at seed20000')
    if a.development_training_scenes and a.split!='validation':raise ValueError('Training-scene diagnostic is development only')
    if a.split=='validation' and min(a.seeds)<10000 and not a.development_training_scenes:raise ValueError('Training scenes require explicit --development-training-scenes')
    if a.output.exists():raise FileExistsError('Evaluation refuses to overwrite retained attempts')
    a.output.mkdir(parents=True);module=importlib.import_module(a.policy_module)
    load_status={'scope':'checkpoint load/evaluation attempt','complete':False,'checkpoint':str(a.checkpoint),'adapter':a.policy_module}
    (a.output/'load-status.json').write_text(json.dumps(load_status,indent=2))
    gpu_samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=2)
                gpu_samples.append(int(value.splitlines()[0]))
            except (OSError,ValueError,subprocess.SubprocessError):pass
            stop.wait(.5)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:policy=module.make_policy(str(a.checkpoint),a.device)
    except Exception as exc:
        stop.set();worker.join(timeout=3);load_status['error']=str(exc);load_status['whole_device_peak_mib']=max(gpu_samples) if gpu_samples else None
        (a.output/'load-status.json').write_text(json.dumps(load_status,indent=2));raise
    if policy.metadata.get('action_units')!='radians':raise ValueError('Require explicit six-joint simulator radians adapter')
    report={'scope':'new MuJoCo checkpoint evaluation; no physical robot/domain-transfer claim','policy':policy.metadata,
            'policy_adapter_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'source_hashes':office.source_hashes(),
            'clock_mode':'simulation pauses for inference; video excludes pauses','split':a.split,'seeds':a.seeds,'episodes':[],
            'complete':False,'successes':0,'paired_initial_state_checks':[],
            'policy_rng':'same torch seed for each paired instruction within a scene','policy_rng_seed_base':a.policy_rng_seed_base}
    report['wording']=a.wording
    report['development_overfit_control']=a.development_training_scenes
    report['policy_rng_seed_fixed']=a.policy_rng_seed_fixed
    report['concurrency_note']=a.concurrency_note
    report['chunk_execution']=a.chunk_execution
    report['execution_device']=a.device
    report['render_backend']=os.environ.get('MUJOCO_GL','default')
    report['cpu_threads']=os.environ.get('OMP_NUM_THREADS','default')
    report['prediction_retention']=a.retain_predictions
    report['task_successes']=0
    report['supervisor']='Existing simulator clips actuator targets to authored joint ranges; raw actions and violations are retained; no IK/manual correction'
    def check_admission():
        if a.host_monitor and a.host_monitor.exists():
            try:observed=json.loads(a.host_monitor.read_text())
            except (OSError,ValueError):return
            if observed.get('threshold_14gib_exceeded') or observed.get('threshold_48gib_ram_exceeded'):
                raise MemoryError('Whole-host GPU/RAM admission failed; stopping owned evaluation')
    try:
        check_admission()
        for seed in a.seeds:
            initial_hash=None
            for task in office.instructions(seed,a.wording):
                if a.tasks and task['id'] not in a.tasks:continue
                import torch
                policy_seed=a.policy_rng_seed_fixed if a.policy_rng_seed_fixed is not None else a.policy_rng_seed_base+seed
                torch.manual_seed(policy_seed)
                if hasattr(policy,'set_rng_seed'):policy.set_rng_seed(policy_seed)
                env=office.make_env(seed,a.image_size,a.image_size,True);policy.reset()
                out=a.output/f'seed-{seed}-{task["id"]}';out.mkdir()
                arrays={k:[] for k in ['states','actions','applied_actions','qpos_after','qvel_after','timestamps','inference_seconds']}
                import imageio.v2 as imageio
                from PIL import Image,ImageDraw
                writer=imageio.get_writer(str(out/'rollout.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=1)
                h=hashlib.sha256(env.data.qpos.tobytes()).hexdigest()
                if initial_hash is None:initial_hash=h
                if initial_hash!=h:raise ValueError('Paired scene reset is not identical')
                queue=[];clipped=0;started=time.perf_counter();predictions=[];prediction_rows=[];early_inputs=[]
                def save_predictions():
                    if not a.retain_predictions or not predictions:return
                    np.savez_compressed(out/'predictions.npz',ticks=np.asarray([r['tick'] for r in prediction_rows]),chunks=np.stack(predictions))
                    (out/'prediction-inputs.json').write_text(json.dumps(prediction_rows,indent=2))
                    if early_inputs:
                        np.savez_compressed(out/'first-inputs.npz',ticks=np.asarray([r[0] for r in early_inputs]),states=np.stack([r[1] for r in early_inputs]),scene=np.stack([r[2] for r in early_inputs]),wrist=np.stack([r[3] for r in early_inputs]))
                try:
                    for tick in range(a.max_steps or 585*len(task['goals'])):
                        if tick%8==0:check_admission()
                        state=env.data.qpos[env.qadr].astype(np.float32).copy();latency=0.
                        if not queue:
                            obs={'state':state,'scene':env.render('scene'),'wrist':env.render('wrist')}
                            before=time.perf_counter();chunk=np.asarray(policy.predict(obs,task['instruction']))
                            latency=time.perf_counter()-before
                            if chunk.ndim!=2 or chunk.shape[1]!=6 or not np.isfinite(chunk).all():raise ValueError('Invalid policy action chunk')
                            if a.retain_predictions:
                                predictions.append(chunk.copy());prediction_rows.append({'tick':tick,'input_sha256':getattr(policy,'last_input_sha256',{}),
                                    'server_synchronized_seconds':getattr(policy,'last_server_inference_seconds',None),'end_to_end_seconds':latency})
                                if len(early_inputs)<3:early_inputs.append((tick,state.copy(),obs['scene'].copy(),obs['wrist'].copy()))
                            queue=list(chunk[:a.chunk_execution])
                            if not queue:raise ValueError('Empty policy output')
                        action=np.asarray(queue.pop(0));lo,hi=env.model.actuator_ctrlrange.T
                        clipped+=int(np.any(action<lo-1e-6)|np.any(action>hi+1e-6))
                        arrays['states'].append(state);arrays['actions'].append(action);arrays['timestamps'].append(float(env.data.time))
                        arrays['inference_seconds'].append(latency);env.step(office.sim.q_to_deg(action))
                        arrays['applied_actions'].append(env.data.ctrl.copy())
                        arrays['qpos_after'].append(env.data.qpos.copy());arrays['qvel_after'].append(env.data.qvel.copy())
                        frame=Image.fromarray(env.render('overview'));draw=ImageDraw.Draw(frame)
                        draw.rectangle((0,0,a.image_size,44),fill=(0,0,0))
                        draw.text((4,4),str(policy.metadata['model'])+' CHECKPOINT | simulation',fill='white')
                        draw.text((4,20),task['instruction'][:40],fill='yellow');writer.append_data(np.asarray(frame))
                    graded=office.grade(env,task['goals']);strict=graded['success'] and clipped==0
                    np.savez_compressed(out/'trajectory.npz',**{k:np.asarray(v) for k,v in arrays.items()})
                    save_predictions()
                    row={'seed':seed,'task':task,'scene':env.scene_spec,'policy':policy.metadata,'wording':a.wording,
                         'wording_sha256':hashlib.sha256(task['instruction'].encode()).hexdigest(),
                         'development_overfit_control':a.development_training_scenes and seed<10000,'policy_rng_seed':policy_seed,'initial_state_sha256':h,'grader':graded,'success':strict,
                         'task_success':graded['success'],'strict_zero_raw_limit_events_success':strict,'chunk_execution':a.chunk_execution,
                         'supervisor':report['supervisor'],
                         'command_limit_events':clipped,'steps':len(arrays['states']),'wall_seconds':time.perf_counter()-started,
                         'fresh_inference_p50_s':float(np.median([x for x in arrays['inference_seconds'] if x>0])),
                         'fresh_inference_p95_s':float(np.percentile([x for x in arrays['inference_seconds'] if x>0],95)),
                         'trajectory_sha256':hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()}
                    if prediction_rows:
                        server=[r['server_synchronized_seconds'] for r in prediction_rows if r['server_synchronized_seconds'] is not None]
                        row['server_synchronized_inference_p50_s']=float(np.median(server)) if server else None
                        row['server_synchronized_inference_p95_s']=float(np.percentile(server,95)) if server else None
                        row['prediction_trace_sha256']=hashlib.sha256((out/'predictions.npz').read_bytes()).hexdigest()
                        row['command_limit_metric_scope']='policy-decoded commands before common actuator supervisor; native decoder bounds differ by model'
                    (out/'episode.json').write_text(json.dumps(row,indent=2));(out/'grading-trace.json').write_text(json.dumps(env.grading_trace))
                    report['episodes'].append(row);report['successes']=sum(r['success'] for r in report['episodes'])
                    report['task_successes']=sum(r['task_success'] for r in report['episodes'])
                    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'seed':seed,'task':task['id'],'success':strict}),flush=True)
                except Exception as exc:
                    np.savez_compressed(out/'trajectory.npz',**{k:np.asarray(v) for k,v in arrays.items()})
                    save_predictions()
                    row={'seed':seed,'task':task,'scene':env.scene_spec,'policy':policy.metadata,'initial_state_sha256':h,
                         'grader':{'success':False,'reason':'incomplete rollout'},'success':False,'error':str(exc),
                         'steps':len(arrays['qpos_after']),'command_limit_events':clipped,
                         'trajectory_sha256':hashlib.sha256((out/'trajectory.npz').read_bytes()).hexdigest()}
                    (out/'episode.json').write_text(json.dumps(row,indent=2));(out/'grading-trace.json').write_text(json.dumps(env.grading_trace))
                    report['episodes'].append(row);raise
                finally:writer.close();env.close()
            report['paired_initial_state_checks'].append({'seed':seed,'initial_state_sha256':initial_hash,'identical':True})
        report['complete']=True
    except Exception as exc:
        report['error']=str(exc);raise
    finally:
        stop.set();worker.join(timeout=3)
        report['whole_device_peak_mib']=max(gpu_samples) if gpu_samples else None
        report['host_memory_admission']='requires external Windows whole-system monitor; WSL process memory is insufficient'
        if hasattr(policy,'close'):policy.close()
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
