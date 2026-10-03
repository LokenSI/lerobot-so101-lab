"""Generate real contact-physics expert episodes; never a learned-policy success."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import office_environment as office

def run(seed,split,task,out,size,render):
    out.mkdir(parents=True,exist_ok=False)
    env=office.make_env(seed,width=size,height=size,offscreen=render)
    initial=env.data.qpos.copy();started=time.perf_counter()
    arrays={k:[] for k in ('states','actions','qpos_before','qvel_before','qpos_after','qvel_after','timestamps','phases')}
    if render:arrays.update(images_scene=[],images_wrist=[])
    try:
        for action,phase in office.expert_actions(env,task['goals']):
            arrays['states'].append(env.data.qpos[env.qadr].copy());arrays['actions'].append(action)
            arrays['qpos_before'].append(env.data.qpos.copy());arrays['qvel_before'].append(env.data.qvel.copy())
            arrays['timestamps'].append(float(env.data.time));arrays['phases'].append(phase)
            if render:
                for camera in ('scene','wrist'):arrays[f'images_{camera}'].append(env.render(camera))
            env.step(office.sim.q_to_deg(action))
            arrays['qpos_after'].append(env.data.qpos.copy());arrays['qvel_after'].append(env.data.qvel.copy())
        result=office.grade(env,task['goals'])
        path=out/'episode.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in arrays.items()})
        row={'seed':seed,'split':split,'task':task,'scene':env.scene_spec,'initial_qpos':initial.tolist(),
             'initial_state_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),
             'success':result['success'],'grader':result,'controller':'privileged scripted IK expert','learned_policy':False,
             'object_attachment':False,'object_pose_write_after_reset':False,'fps':office.sim.FPS,'state_action_units':['radian']*6,
             'observations_rendered':render,'camera_size':size,'frames':len(arrays['states']),'physics_dt':office.sim.PHYSICS_DT,
             'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_hashes':office.source_hashes(),'wall_seconds':time.perf_counter()-started}
        (out/'episode.json').write_text(json.dumps(row,indent=2));(out/'grading-trace.json').write_text(json.dumps(env.grading_trace))
        print(json.dumps({'seed':seed,'task':task['id'],'success':row['success'],'wall_seconds':row['wall_seconds']}),flush=True)
        return row
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=office.OUTPUT/'dataset')
    p.add_argument('--train-seeds',type=int,nargs='*',default=[1000,1001,1002]);p.add_argument('--val-seeds',type=int,nargs='*',default=[10000])
    p.add_argument('--image-size',type=int,default=256);p.add_argument('--no-render',action='store_true');p.add_argument('--tasks',nargs='*')
    a=p.parse_args();assert not set(a.train_seeds)&set(a.val_seeds),'Seeds must not leak across splits'
    if any(s<0 or s>=10000 for s in a.train_seeds):raise ValueError('Train seeds must be below10000; sealed tests start20000')
    if any(s<10000 or s>=20000 for s in a.val_seeds):raise ValueError('Validation seeds must be10000..19999')
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for split,seeds in [('train',a.train_seeds),('validation',a.val_seeds)]:
        for seed in seeds:
            for task in office.instructions(seed,split):
                if a.tasks and task['id'] not in a.tasks:continue
                out=a.output/split/f'seed-{seed}'/task['id']
                if (out/'episode.json').exists():row=json.loads((out/'episode.json').read_text())
                else:row=run(seed,split,task,out,a.image_size,not a.no_render)
                rows.append({'path':str(out.relative_to(a.output)),'success':row['success'],'seed':seed,'split':split,'task':task,'sha256':row['dataset_sha256'],'rendered':row['observations_rendered']})
                (a.output/'manifest.json').write_text(json.dumps({'format':'office-expert-v1','episodes':rows,'accepted':sum(r['success'] and r['rendered'] for r in rows),'scripted_expert_only':True,'checkpoint_evaluations':0},indent=2))
    print(json.dumps({'episodes':len(rows),'successful_expert_physics':sum(r['success'] for r in rows),'training_ready':sum(r['success'] and r['rendered'] for r in rows)}))
if __name__=='__main__':main()
