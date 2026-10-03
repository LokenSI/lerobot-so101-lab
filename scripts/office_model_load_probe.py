"""Bounded local cold-load/actual inference measurement, without a robot-success claim."""
import argparse,hashlib,importlib,json,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--policy-module',required=True)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);status={'scope':'local explicit BF16 constructor-load optimization and five native inference calls; no robot or simultaneous-renderer success','complete':False}
    policy=None;start=time.monotonic()
    try:
        import torch
        torch.manual_seed(510000);torch.cuda.reset_peak_memory_stats()
        module=importlib.import_module(a.policy_module);policy=module.make_policy(str(a.checkpoint),'cuda')
        status.update(policy=policy.metadata,cold_load_seconds=time.monotonic()-start,cold_load_owned_cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        torch.cuda.reset_peak_memory_stats();policy.reset();rows=[];chunks=[]
        with np.load(a.episode) as episode:
            for index in [0,8,16,24,32]:
                observation={'state':episode['states'][index].astype(np.float32),'scene':episode['images_scene'][index],'wrist':episode['images_wrist'][index]}
                torch.cuda.synchronize();before=time.perf_counter();commands=policy.predict(observation,'Pick up the red cube and put it in the left tray.');torch.cuda.synchronize()
                if commands.ndim!=2 or commands.shape[1]!=6 or not np.isfinite(commands).all():raise ValueError('Invalid/nonfinite native predictions')
                count=min(8,len(commands),len(episode['actions'])-index)
                rows.append({'frame':index,'shape':list(commands.shape),'synchronized_seconds':time.perf_counter()-before,
                  'first8_expert_mae_radians':float(np.abs(commands[:count]-episode['actions'][index:index+count]).mean()),
                  'input_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in observation.items()}});chunks.append(commands.copy())
        np.savez_compressed(a.output/'predictions.npz',frames=np.asarray([row['frame'] for row in rows]),chunks=np.stack(chunks))
        status.update(complete=True,rows=rows,inference_owned_cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,elapsed_seconds=time.monotonic()-start,
                      predictions_sha256=hashlib.sha256((a.output/'predictions.npz').read_bytes()).hexdigest(),episode_sha256=hashlib.sha256(a.episode.read_bytes()).hexdigest())
    except Exception as error:status['error']=str(error);raise
    finally:(a.output/'report.json').write_text(json.dumps(status,indent=2));print(json.dumps({key:value for key,value in status.items() if key!='policy'}),flush=True)

if __name__=='__main__':main()
