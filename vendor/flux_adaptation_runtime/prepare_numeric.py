"""CPU-only preparation audit; never loads FLUX, renders or trains."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def to_units(q):
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., :5] = np.rad2deg(out[..., :5])
    out[..., 5] = (out[..., 5] + 0.17453) / (1.74533 + 0.17453) * 100
    return out.astype(np.float32)

def window_arrays(states, commands, start):
    """Exact e2dd1d8 standalone WindowDataset.window_at history alignment."""
    if start < 8 or start + 42 >= len(states):
        raise ValueError('Need complete preceding commands and endpoint image within one episode')
    future = commands[start:start + 42]
    previous = commands[start - 1]
    delta = np.diff(np.concatenate([previous[None], future]), axis=0)
    delta[:, -1] = future[:, -1]
    return dict(state=states[start - 7:start + 1],
                command_history=commands[start - 8:start], action=future,
                unnormalized_flow_targets=delta,
                image_indices=np.arange(start - 7, start + 43))

def model_inventory(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    tensors = {k:v for k,v in header.items() if k != '__metadata__'}
    return {'path':str(path.relative_to(ROOT)), 'parameters':sum(int(np.prod(v['shape'])) for v in tensors.values()),
            'stored_tensor_bytes':sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values()),
            'dtype_tensor_counts':{d:sum(v['dtype']==d for v in tensors.values()) for d in sorted({v['dtype'] for v in tensors.values()})},
            'scope':'Header-only count, not a measured training memory requirement. Encoders, gradients, optimizer and activations are excluded.'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=ROOT/'runs/so101-teaser-dataset')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'prepared-numeric')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(a.dataset.glob('seed-*/episode.npz')):
        report=json.loads(path.with_name('report.json').read_text())
        with np.load(path) as src:
            states=to_units(src['states']);commands=to_units(src['actions'])
            assert len(states)==len(commands) and states.shape[1:]==(6,)
            assert np.isfinite(states).all() and np.isfinite(commands).all()
            assert src['images_front'].shape[1:]==(96,96,3)
        # Explicit episode-level split for this independent FLUX preparation.
        seed=int(report['seed']); split='val' if seed in (0,7,14) else 'train'
        output=a.output/f'seed-{seed:03d}';output.mkdir(exist_ok=True)
        np.savez_compressed(output/'numeric.npz',states=states,commands=commands)
        starts=list(range(8,len(states)-42))
        assert len(starts)>0
        arrays=window_arrays(states,commands,starts[0])
        np.savez_compressed(output/'first-window-audit.npz',**arrays)
        # Verify delta integration exactly reproduces issued commands, gripper stays absolute.
        restored=arrays['unnormalized_flow_targets'].copy()
        restored[:,:5]=commands[starts[0]-1,:5]+np.cumsum(restored[:,:5],axis=0)
        err=float(np.max(np.abs(restored-arrays['action'])))
        assert err<1e-4
        rows.append({'seed':seed,'split':split,'frames':len(states),'valid_action_starts':len(starts),
                     'source_episode_sha256':digest(path),'source_report_sha256':digest(path.with_name('report.json')),
                     'source_episode':str(path.relative_to(ROOT)), 'recorded_success':bool(report['success']),
                     'max_delta_reconstruction_error_units':err,'rgb_status':'Original 96 px intentionally NOT resized/copied; native 256 px replay required'})
    assert len(rows)==20, f'Expected20demos, found{len(rows)}'
    result={'status':'CPU numeric preparation only; no training or native256 render performed',
        'task':'Grasp the red block and place it in the gray bin.',
        'joint_names':JOINTS,'units':['degrees']*5+['gripper percentage'],
        'calibration':'Simulator q_to_deg: five arm radians -> degrees; gripper -0.17453..1.74533rad ->0..100. Must match any trained adapter rollout.',
        'flow_target_conversion':'Five consecutive issued-command deltas; absolute gripper. Do not delta-transform data before passing to official processor.',
        'history_alignment':{'state':'[t-7,t] inclusive pre-action observations','command_history':'[t-8,t-1] inclusive previous issued commands',
            'action':'[t,t+41] inclusive issued targets','images':'[t-7,t+42] inclusive, 50 native frames'},
        'normalization':'Retain checkpoint-owned quantiles unless a separately frozen sim-domain normalization is selected before training. No quantiles changed here.',
        'base_checkpoint_header':model_inventory(ROOT/'models/flux-action/so101/model.safetensors'),
        'baseline_sha256':digest(ROOT/'scripts/so101_pick_place_baseline.py'),
        'official_loader_sha256':digest(ROOT/'runtime/flux-action-src/src/flux_action/training/data.py'),
        'episode_count':len(rows),'train_episodes':sum(r['split']=='train' for r in rows),'validation_episodes':sum(r['split']=='val' for r in rows),
        'episodes':rows}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))

if __name__=='__main__':main()
