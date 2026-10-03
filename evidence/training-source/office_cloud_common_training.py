"""Execute one authorized common-six stock stage with an owned native process guard."""
import argparse,datetime,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path

SOURCE={'smolvla':Path('/home/ubuntu/office-action-training/sources/lerobot'),'groot':Path('/home/ubuntu/office-action-training/sources/Isaac-GR00T')}
PACKAGE=Path('/home/ubuntu/workspace/action-training/package')
ROOT=Path('/home/ubuntu/workspace/action-training/groot-eval/common-training-v1')
DEADLINE=datetime.datetime(2026,10,3,14,52,tzinfo=datetime.timezone.utc)

def sample():
    gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()[0])
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    return {'gpu_mib':gpu,'ram_mib':(values['MemTotal']-values['MemAvailable'])/1024}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=SOURCE,required=True);p.add_argument('--stage',choices=['smoke','main'],required=True);p.add_argument('--attempt',choices=['v1','v2'],default='v1');a=p.parse_args()
    recipe_path=ROOT/'recipe.json';recipe=json.loads(recipe_path.read_text());model=recipe['models'][a.model];settings=model[a.stage]
    report_root=ROOT if a.attempt=='v1' else ROOT/'groot-worker1-v2'
    if a.attempt=='v2' and a.model!='groot':raise ValueError('v2 is the explicit GR00T shard/worker scheduling repair')
    report_root.mkdir(exist_ok=True)
    run=Path('/data/action-training/runs')/f'common-six-{a.attempt}-{a.model}-{a.stage}'
    report_path=report_root/f'{a.model}-{a.stage}-execution-report.json'
    if report_path.exists() or run.exists():raise FileExistsError('Do not overwrite earlier training stages')
    if a.stage=='main':
        gate=json.loads((report_root/f'{a.model}-smoke-reload.json').read_text())
        commands=gate.get('commands_absolute_radians',[])
        if not gate['passed'] or not commands or any(len(row)!=6 or not all(math.isfinite(v) for v in row) for row in commands):raise ValueError('Actual finite native smoke reload gate missing')
    dataset=ROOT/('lerobot-v3' if a.model=='smolvla' else 'groot-v2')/'train'
    initial=sample()
    if initial['gpu_mib']>=71680 or initial['ram_mib']>=102400 or datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise MemoryError('Prelaunch resource/deadline refusal')
    source=SOURCE[a.model];python=source/'.venv/bin/python'
    preflight=[str(python),str(PACKAGE/'scripts/cloud_train_run.py'),'--model',a.model,'--source',str(source),'--dataset',str(dataset),
      '--base-model',model['base_model_snapshot'],'--package',str(PACKAGE/'training'),'--output',str(run),
      '--steps',str(settings['updates']),'--save-every',str(settings['save_every']),'--max-wall-seconds',str(settings['max_wall_seconds'])]
    if a.model=='groot':preflight.extend(['--save-total-limit','1'])
    subprocess.run(preflight,check=True,cwd=source)
    manifest=run.parent/(run.name+'-launch-manifest.json');plan=json.loads(manifest.read_text())
    if plan['source_revision']!=model['source_pin']:raise ValueError('Source pin differs')
    native_command=plan['command'].copy()
    scheduling_fix=None
    if a.attempt=='v2':
        index=native_command.index('--dataloader-num-workers')+1
        scheduling_fix={'stock_workers':native_command[index],'actual_workers':1,'reason':'three epoch shards with four workers leaves one empty schedule and native IndexError; single worker receives all three shards','data_or_model_modified':False}
        native_command[index]='1'
    report={'scope':'actual fresh-base common-six stock training stage; no robot task-success claim','model':a.model,'stage':a.stage,'attempt':a.attempt,'scheduling_fix':scheduling_fix,
      'recipe_sha256':hashlib.sha256(recipe_path.read_bytes()).hexdigest(),'stock_launch_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'stock_preflight_dry_run':True,'actual_native_command':native_command,'native_execution_started':False,'complete':False,
      'output':str(run),'samples':[],'deadline_utc':DEADLINE.isoformat()}
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',WANDB_DISABLED='true')
    started=time.monotonic();proc=None
    try:
        with (report_root/f'{a.model}-{a.stage}-stdout.log').open('w') as log:
            proc=subprocess.Popen(native_command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(native_execution_started=True,owned_pid=proc.pid)
            while proc.poll() is None:
                measured=sample();report['samples'].append(measured)
                if measured['gpu_mib']>=71680 or measured['ram_mib']>=102400:raise MemoryError('Whole cloud GPU/RAM guard crossing')
                if time.monotonic()-started>=settings['max_wall_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('Stage wall/deadline cap')
                time.sleep(.5)
            report.update(exit_code=proc.returncode,complete=proc.returncode==0)
            if proc.returncode:raise RuntimeError('Native trainer exited with failure')
    except Exception as error:
        report['error']=str(error)
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['gpu_peak_mib']=max((row['gpu_mib'] for row in report['samples']),default=initial['gpu_mib'])
        report['ram_peak_mib']=max((row['ram_mib'] for row in report['samples']),default=initial['ram_mib'])
        report_path.write_text(json.dumps(report,indent=2));print(json.dumps({key:value for key,value in report.items() if key!='samples'}),flush=True)

if __name__=='__main__':main()
