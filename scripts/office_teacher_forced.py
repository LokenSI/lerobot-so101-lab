"""Retain full learned chunks on shared original train-scene observations; no task score."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from office_comparison_adapter import ComparisonPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refuse to overwrite teacher-forced diagnostics')
    a.output.mkdir(parents=True);contract=json.loads(a.contract.read_text());spec=contract['teacher_forcing']
    root=Path(__file__).resolve().parents[1];source=root/spec['source_npz']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_npz_sha256']:raise ValueError('Teacher-forced input episode differs from frozen contract')
    with np.load(source) as archive:
        states=archive['states'];actions=archive['actions'];scene=archive['images_scene'];wrist=archive['images_wrist']
    task=json.loads(source.with_name('episode.json').read_text())['task']['instruction']
    policy=None;rows=[];chunks=[];status={'scope':'teacher-forced training-scene diagnostic; no rollout or robot success',
       'complete':False,'contract_sha256':hashlib.sha256(a.contract.read_bytes()).hexdigest(),'source_npz_sha256':spec['source_npz_sha256'],
       'anchor_order':spec['frame_indices'],'rng_reset_scope':'once before sequential anchor requests','rows':rows}
    try:
        policy=ComparisonPolicy(a.contract,'cpu');policy.set_rng_seed(spec['torch_rng_reset_once_before_anchor_sequence']);policy.reset();status['policy']=policy.metadata
        for index in spec['frame_indices']:
            observation={'state':states[index].astype(np.float32).copy(),'scene':scene[index].copy(),'wrist':wrist[index].copy()}
            before=time.perf_counter();chunk=policy.predict(observation,task);transport=time.perf_counter()-before
            if chunk.ndim!=2 or chunk.shape[1]!=6 or chunk.shape[0]<8 or not np.isfinite(chunk).all():raise ValueError('Invalid native predicted chunk')
            expected=actions[index:index+8].astype(np.float32);error=np.abs(chunk[:8]-expected)
            row={'frame':index,'input_sha256':policy.last_input_sha256,'server_synchronized_seconds':policy.last_server_inference_seconds,
                 'end_to_end_seconds':transport,'per_joint_first8_mae_rad':error.mean(0).tolist(),'first8_mae_rad':float(error.mean()),
                 'max_first8_error_rad':float(error.max()),'first_command_rad':chunk[0].tolist(),'expert_first_command_rad':expected[0].tolist(),'chunk_shape':list(chunk.shape)}
            rows.append(row);chunks.append(chunk);print(json.dumps(row),flush=True)
        np.savez_compressed(a.output/'full-chunks.npz',frames=np.asarray(spec['frame_indices']),chunks=np.stack(chunks),
                            expert_first8=np.stack([actions[index:index+8] for index in spec['frame_indices']]))
        status['full_chunks_sha256']=hashlib.sha256((a.output/'full-chunks.npz').read_bytes()).hexdigest();status['complete']=True
    except Exception as exc:status['error']=str(exc);raise
    finally:
        if policy is not None:policy.close()
        (a.output/'teacher-forced-report.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
