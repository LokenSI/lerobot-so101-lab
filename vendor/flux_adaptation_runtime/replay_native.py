"""Re-render expert demonstrations with256px cameras, never upsample96px.

Requires existing MuJoCo WSL environment; no FLUX model is loaded. This is
scripted expert replay for adaptation data, never a FLUX policy rollout.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(Path(__file__).parent))
from prepare_numeric import to_units, digest

def main():
    import mujoco
    import so101_pick_place_baseline as base
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',default=','.join(str(i) for i in range(20)))
    p.add_argument('--output',type=Path,default=ROOT/'datasets/flux-so101-native-v1')
    p.add_argument('--width',type=int,default=320)
    p.add_argument('--height',type=int,default=240)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in map(int,a.seeds.split(',')):
        source=ROOT/f'runs/so101-teaser-dataset/seed-{seed:03d}/episode.npz'
        with np.load(source) as src:
            states=src['states'].copy();commands=src['actions'].copy()
        env=base.make_env(seed,width=a.width,height=a.height,offscreen=True)
        out=a.output/f'seed-{seed:03d}';out.mkdir(exist_ok=True)
        maxerr=0.;images_scene=[];images_wrist=[];times=[];actual_states=[]
        try:
            initial_qpos=env.data.qpos.copy()
            for state,command in zip(states,commands,strict=True):
                maxerr=max(maxerr,float(np.max(np.abs(env.data.qpos[env.qadr]-state))))
                if maxerr>=1e-8:
                    raise ValueError(f'Physics replay diverged before action at seed{seed}: {maxerr}rad')
                times.append(float(env.data.time));actual_states.append(env.data.qpos[env.qadr].copy())
                images_scene.append(np.asarray(Image.fromarray(env.render('scene')).resize((256,256),Image.Resampling.BILINEAR)))
                images_wrist.append(np.asarray(Image.fromarray(env.render('wrist')).resize((256,256),Image.Resampling.BILINEAR)))
                env.step(base.sim.q_to_deg(command))
            grade=base.task_success(env)
            if not grade['success']:raise ValueError(f'Expert replay no longer meets strict grader: {grade}')
            # Complete replay only; no saved-object-pose teleports at any control step.
            np.savez_compressed(out/'episode.npz',images_scene=np.asarray(images_scene,dtype=np.uint8),
                images_wrist=np.asarray(images_wrist,dtype=np.uint8),states=to_units(actual_states),
                commands=to_units(commands),time_s=np.asarray(times),initial_qpos=initial_qpos)
            mujoco.mj_saveLastXML(str(out/'scene.xml'),env.model)
            row={'seed':seed,'frames':len(states),'success':grade['success'],
                'replay_max_state_error_rad':maxerr,'grade':grade,'source_episode_sha256':digest(source),
                'native_episode_sha256':digest(out/'episode.npz'),'baseline_sha256':digest(ROOT/'scripts/so101_pick_place_baseline.py'),
                'control':'Scripted expert action replay, not FLUX','camera_render':f'Native {a.width}x{a.height} physical camera projection, then bilinear resize to 256x256 per view',
                'alignment':'Measured state + images immediately before corresponding command',
                'object_pose_written_after_reset':False,'attachments':False,'mujoco_version':mujoco.__version__}
            (out/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
            print(f'seed{seed}:strictsuccess={grade["success"]},maxreplayerror={maxerr}',flush=True)
        finally:env.close()
    (a.output/'summary.json').write_text(json.dumps({'scope':'Native camera expert replays for potential FLUX adaptation; no FLUX training or policy results','runs':rows},indent=2)+'\n')

if __name__=='__main__':main()
