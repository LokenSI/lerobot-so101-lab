"""Resource-guarded actual native common-six checkpoint cold reload gate."""
import argparse,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
from office_cloud_common_training import ROOT,SOURCE,PACKAGE,sample,DEADLINE,datetime

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=SOURCE,required=True);p.add_argument('--stage',choices=['smoke','main'],required=True);p.add_argument('--attempt',choices=['v1','v2'],default='v1');a=p.parse_args()
    report_root=ROOT if a.attempt=='v1' else ROOT/'groot-worker1-v2'
    run=Path('/data/action-training/runs')/f'common-six-{a.attempt}-{a.model}-{a.stage}';steps=20 if a.stage=='smoke' else 2000
    checkpoint=run/f'checkpoint-{steps}' if a.model=='groot' else run/'checkpoints'/f'{steps:06d}'/'pretrained_model'
    output=report_root/f'{a.model}-{a.stage}-reload.json';guard=report_root/f'{a.model}-{a.stage}-reload-guard.json'
    if output.exists() or guard.exists():raise FileExistsError('Keep native reload evidence')
    episode=ROOT.parent/'common-training-original-episode.npz'
    if hashlib.sha256(episode.read_bytes()).hexdigest()!='66eb600934eaaed68ae702ff53f8c9806b1c3d56b2177a6981c9a51d6c4dc6a2':raise ValueError('Original reload observation episode differs')
    first=sample()
    if first['gpu_mib']>=71680 or first['ram_mib']>=102400:raise MemoryError('Cold reload admission failed')
    command=[str(SOURCE[a.model]/'.venv/bin/python'),str(PACKAGE/'scripts/cloud_train_reload_probe.py'),'--model',a.model,'--checkpoint',str(checkpoint),'--episode',str(episode),'--task','Pick up the red cube and put it in the left tray.','--output',str(output)]
    env=os.environ.copy();env.update(PYTHONPATH=str(PACKAGE/'scripts'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
    samples=[];started=time.monotonic();error=None
    with (report_root/f'{a.model}-{a.stage}-reload-stdout.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=SOURCE[a.model],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                row=sample();samples.append(row)
                if row['gpu_mib']>=71680 or row['ram_mib']>=102400:raise MemoryError('Cold reload resource guard crossing')
                if time.monotonic()-started>300 or datetime.datetime.now(datetime.timezone.utc)>=DEADLINE:raise TimeoutError('Cold reload deadline')
                time.sleep(.5)
            if proc.returncode:raise RuntimeError('Native reload process failed')
            result=json.loads(output.read_text());commands=result['commands_absolute_radians']
            if not result['passed'] or not commands or any(len(row)!=6 or not all(math.isfinite(v) for v in row) for row in commands):raise ValueError('Reload returned invalid/nonfinite actions')
        except Exception as exc:
            error=str(exc)
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
        finally:
            report={'scope':'actual native saved cold reload, no physical task success','passed':error is None,'error':error,'owned_pid':proc.pid,'command':command,'elapsed_seconds':time.monotonic()-started,
              'gpu_peak_mib':max((row['gpu_mib'] for row in samples),default=first['gpu_mib']),'ram_peak_mib':max((row['ram_mib'] for row in samples),default=first['ram_mib']),
              'samples':len(samples),'original_episode_sha256':hashlib.sha256(episode.read_bytes()).hexdigest()}
            guard.write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
