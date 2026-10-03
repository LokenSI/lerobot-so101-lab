"""Bounded remote stock training launcher; dry-run is the default.

No provisioning or credentials. Run only on an already authorized cloud VM.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

PINS={'smolvla':'6e1fa4faf2a42927d463591aebaa0f62c2e654b7','groot':'51d4c89f72fda44cbf77285c6a8114b52676b8a1'}
JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=PINS,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-model',required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--max-wall-seconds',type=int,default=2700)
    p.add_argument('--resume-config',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--checkpoint-barrier',action='store_true');p.add_argument('--save-total-limit',type=int)
    p.add_argument('--checkpoint-export-scope',choices=('full','inference_until_final'),default='full');a=p.parse_args()
    if not 1<=a.steps<=4000 or not 1<=a.save_every<=a.steps or not 1<=a.max_wall_seconds<=7200:raise ValueError('Bounded recipe exceeded')
    if a.checkpoint_export_scope!='full':
        if a.model!='groot' or not a.checkpoint_barrier:raise ValueError('Scoped export requires GR00T checkpoint barrier')
        if a.steps%a.save_every:raise ValueError('Final full checkpoint requires steps divisible by save interval')
    a.source=a.source.resolve();a.dataset=a.dataset.resolve();a.output=a.output.resolve();a.package=a.package.resolve()
    pin=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    if pin!=PINS[a.model]:raise ValueError('Source revision differs from frozen package')
    info_path=a.dataset/'meta/info.json';info=json.loads(info_path.read_text())
    contract_path=a.dataset/'meta/office-contract.json';contract=json.loads(contract_path.read_text())
    if contract.get('state_action_units')!='radians_all_six' or contract.get('joint_order')!=JOINTS or contract.get('camera_order')!=['scene','wrist'] or contract.get('split')!='train':raise ValueError('Office training contract differs or held-out data supplied')
    if info['fps']!=30 or info.get('total_episodes',0)<2:raise ValueError('Need nonempty 30Hz office dataset')
    for key in ['observation.state','action']:
        if info['features'][key]['shape']!=[6]:raise ValueError('State/action shape must be6')
    for key in ['observation.images.scene','observation.images.wrist']:
        if key not in info['features']:raise ValueError('Missing office camera '+key)
    python=str(a.source/'.venv/bin/python')
    if a.model=='groot':
        if info['codebase_version']!='v2.1':raise ValueError('Stock GR00T requires v2.1')
        modality=a.dataset/'meta/modality.json'
        if json.loads(modality.read_text())!=json.loads((a.package/'office_modality.json').read_text()):raise ValueError('Modality differs')
        limit=a.save_total_limit if a.save_total_limit is not None else (2 if a.checkpoint_barrier else a.steps//a.save_every+3)
        if limit<1:raise ValueError('Invalid retention count')
        launcher=str(Path(__file__).with_name('cloud_train_groot_barrier.py')) if a.checkpoint_barrier else 'gr00t/experiment/launch_finetune.py'
        effective=info['total_frames']-15*info['total_episodes']
        shards=min((effective+1023)//1024,info['total_episodes'])
        command=[python,launcher,'--base-model-path',a.base_model,
            '--dataset-path',str(a.dataset),'--embodiment-tag','NEW_EMBODIMENT','--modality-config-path',str(a.package/'office_groot_config.py'),
            '--num-gpus','1','--output-dir',str(a.output),'--max-steps',str(a.steps),'--save-steps',str(a.save_every),
            '--save-total-limit',str(limit),'--global-batch-size','8','--dataloader-num-workers','4',
            '--episode-sampling-rate','1.0','--shard-size','1024','--num-shards-per-epoch',str(shards),
            '--state-dropout-prob','0.1','--learning-rate','0.0001','--warmup-ratio','0.05',
            '--no-tune-llm','--no-tune-visual','--tune-projector','--tune-diffusion-model','--no-use-wandb','--no-save-only-model']
        if a.resume_config:command.append('--resume-from-checkpoint')
    else:
        if info['codebase_version']!='v3.0':raise ValueError('Pinned LeRobot requires v3.0')
        command=[python,'-m','lerobot.scripts.lerobot_train',
            '--dataset.repo_id=local/office-train','--dataset.root='+str(a.dataset),'--dataset.video_backend=pyav',
            '--policy.path='+a.base_model,'--policy.device=cuda','--policy.push_to_hub=false',
            '--policy.input_features=null',
            '--policy.n_action_steps=8','--policy.train_expert_only=true','--policy.train_state_proj=true','--policy.freeze_vision_encoder=true',
            '--policy.scheduler_warmup_steps='+str(min(50,max(1,a.steps//10))),'--policy.scheduler_decay_steps='+str(a.steps),
            '--batch_size=16','--num_workers=4','--steps='+str(a.steps),'--save_checkpoint=true','--save_freq='+str(a.save_every),
            '--env_eval_freq=0','--eval_steps=0','--log_freq=10','--wandb.enable=false','--seed=42','--output_dir='+str(a.output)]
        if a.resume_config:command=[python,'-m','lerobot.scripts.lerobot_train','--resume=true','--config_path='+str(a.resume_config)]
    plan={'model':a.model,'source_revision':pin,'command':command,'steps':a.steps,'max_wall_seconds':a.max_wall_seconds,
          'checkpoint_export_barrier':a.checkpoint_barrier,
          'checkpoint_export_scope':a.checkpoint_export_scope,'final_full_checkpoint_step':a.steps if a.checkpoint_barrier else None,
          'dataset_info_sha256':digest(info_path),'dataset_root':str(a.dataset),'state_action_units':'all six channels radians',
          'dataset_contract_sha256':digest(contract_path),
          'camera_order':['scene','wrist'],'checkpoint_claim':'Training progress only; no task-success claim without independent closed-loop evaluation',
          'base_model':a.base_model,'started_unix':time.time(),'dry_run':not a.execute}
    a.output.parent.mkdir(parents=True,exist_ok=True);plan_path=a.output.parent/(a.output.name+'-launch-manifest.json')
    if a.resume_config:
        if a.model=='smolvla':
            saved=json.loads(a.resume_config.read_text())
            if saved['steps']!=a.steps or saved['save_freq']!=a.save_every:raise ValueError('Resume must retain saved step and checkpoint schedule')
        plan_path=plan_path.with_name(plan_path.stem+'-resume-'+str(time.time_ns())+'.json')
    elif plan_path.exists():raise FileExistsError('Unique output/launch manifest required')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:return
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',HF_HUB_OFFLINE='1')
    env.update(OFFICE_CHECKPOINT_EXPORT_SCOPE=a.checkpoint_export_scope,OFFICE_FINAL_STEP=str(a.steps))
    # Model snapshots should be downloaded and pinned before billed training starts.
    log_path=a.output.parent/(a.output.name+'-stdout-'+str(int(time.time()))+'.log')
    started=time.monotonic()
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,cwd=a.source,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=a.max_wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
            plan['wall_time_cap_reached']=True
    plan.update(exit_code=code,elapsed_seconds=time.monotonic()-started,stdout=str(log_path))
    result=plan_path.with_name(plan_path.stem+'-result-'+str(int(time.time()))+'.json');result.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2));sys.exit(code if code>=0 else 1)
if __name__=='__main__':main()
