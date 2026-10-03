"""Actual checkpoint save/reload inference gate, never a robot success claim."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from cloud_train_policy import CloudPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['groot','smolvla'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    start=time.monotonic();report={'scope':'Real saved-checkpoint inference only; not task success','model':a.model,'checkpoint':str(a.checkpoint),'task':a.task}
    try:
        policy=CloudPolicy(a.model,a.checkpoint,a.device)
        with np.load(a.episode) as ep:
            scene=ep['images_scene'][0];wrist=ep['images_wrist'][0];state=ep['states'][0]
        policy.reset();commands=policy.predict(scene,wrist,state,a.task)
        report.update(passed=True,shape=list(commands.shape),commands_absolute_radians=commands.tolist(),elapsed_seconds=time.monotonic()-start)
        report['checkpoint_files']={str(p.relative_to(a.checkpoint)):p.stat().st_size for p in a.checkpoint.rglob('*') if p.is_file()}
    except Exception as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error));raise
    finally:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
