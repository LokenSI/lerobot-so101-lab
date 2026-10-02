"""Prospective trained FLUX guidance ablation; no scripted pickup corrections.

Explicit measured observations and previous actually issued commands enter FLUX.
Object pose is used only for reset, grading and recorded diagnostics.
"""
from __future__ import annotations
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback
import sys
import numpy as np
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'runtime/flux-so101-work'))
import so101_pick_place_baseline as baseline
import local_runtime_fast as local_runtime
from flux_adapted_inference import load_adapted_policy, predict_adapted

MEDIAN = {
    'state': {'q01':[-37.969,-99.316,-45.78,22.987,-68.712,.45],
              'q99':[35.288,43.076,90.308,95.704,17.753,40.324]},
    'action': {'q01':[-1.652,-3.235,-3.147,-2.198,-1.708,0.],
               'q99':[1.701,3.487,3.307,2.072,1.699,40.733]}}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def policy_rgb(native_rgb):
    # Match the recorded expert replay's uint8 PIL preprocessing exactly.
    # Preserve the native 320x240 virtual-camera projection before resizing.
    return np.asarray(Image.fromarray(native_rgb).resize((256,256),Image.Resampling.BILINEAR))

def transform(values, mode, inverse=False):
    out = np.array(values, dtype=np.float64, copy=True)
    if mode == 'legacy_candidate':
        out[...,1] = 90 - out[...,1]
        out[...,2] += -90 if inverse else 90
        out[...,4] += 2.789136 if inverse else -2.789136
    return out

def safe_command(env, command):
    # Construct unbounded radians first so gripper clipping is measured too.
    raw_q = np.deg2rad(np.asarray(command,dtype=np.float64))
    raw_q[-1] = baseline.sim.GRIPPER_CLOSED + float(command[-1])/100*(baseline.sim.GRIPPER_OPEN-baseline.sim.GRIPPER_CLOSED)
    lo,hi = env.model.actuator_ctrlrange.T
    issued_q = np.clip(raw_q,lo,hi)
    issued = baseline.sim.q_to_deg(issued_q)
    return issued, bool(np.any(np.abs(raw_q-issued_q)>1e-9)), float(np.max(np.abs(raw_q-issued_q)))

def fresh_record(env):
    m,d = env.model,env.data
    mujoco.mj_forward(m,d)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.time)
    obj = m.body('obj0').id
    gripper = m.body('gripper').id
    contacts = baseline.contacts(env)
    released = not any('gripper' in (c[0],c[1]) or 'moving_jaw_so101_v1' in (c[0],c[1]) for c in contacts)
    position = d.xpos[obj].copy()
    extent = np.abs(d.xmat[obj].reshape(3,3))@np.full(3,.015)
    va = int(m.joint('obj0_free').dofadr[0])
    row = {'time':float(d.time),'object_xyz':position.tolist(),
        'object_quaternion_wxyz':d.xquat[obj].tolist(), 'extent':extent.tolist(),
        'linear_speed_m_s':float(np.linalg.norm(d.qvel[va:va+3])), 'released':released}
    env.grading_trace[-1] = row
    tcp = d.xpos[gripper]+d.xmat[gripper].reshape(3,3)@baseline.TCP_LOCAL
    return row,contacts,tcp

def bilateral_finger_contact(env):
    m,d = env.model,env.data
    cube = m.body('obj0').id
    finger = m.geom('fixed_finger_contact').id
    fixed = moving = False
    for contact_index,contact in enumerate(d.contact):
        g1,g2 = int(contact.geom1),int(contact.geom2)
        b1,b2 = int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
        if cube not in [b1,b2]:
            continue
        other_geom = g2 if b1==cube else g1
        other_body = b2 if b1==cube else b1
        force = np.zeros(6)
        mujoco.mj_contactForce(m,d,contact_index,force)
        fixed |= other_geom==finger and force[0]>.001
        moving |= 'moving_jaw' in m.body(other_body).name and force[0]>.001
    return fixed and moving

def run_episode(policy, original_config, mode, seed, chunks, prompt, folder):
    folder.mkdir(parents=True,exist_ok=False)
    policy.config = copy.deepcopy(original_config)
    if mode == 'hosted_median':
        policy.config.state_normalization = copy.deepcopy(MEDIAN['state'])
        policy.config.action_normalization = copy.deepcopy(MEDIAN['action'])
    local_runtime.reset_cached_policy(policy)
    # Match the hosted virtual camera projection before the model's saved
    # resize to 256x256; rendering a square changes the horizontal field.
    env = baseline.make_env(seed,width=320,height=240,offscreen=True)
    m,d = env.model,env.data
    mujoco.mj_forward(m,d)
    mujoco.mj_saveLastXML(str(folder/'cell.xml'),m)
    history = deque(maxlen=8)
    initial = env.state_deg().copy()
    initial_cube = d.body('obj0').xpos.copy() # Diagnostic only.
    initial_rgb = [policy_rgb(env.render(name)) for name in ['scene','wrist']]
    for name,img in zip(['scene','wrist'],initial_rgb):
        Image.fromarray(img).save(folder/f'initial-{name}.png')
    entry = (*initial_rgb,transform(initial,mode),transform(initial,mode))
    history.extend([entry]*8)
    arrays = {k:[] for k in ['qpos_before','qvel_before','qpos_after','qvel_after','time_s',
        'raw_commands_sim_units','applied_commands_sim_units','measured_states_sim_units',
        'tcp_m','cube_truth_m','scene_rgb','wrist_rgb','controls']}
    contacts_log = []
    timings = []
    clipped = 0
    max_clip = 0.
    bilateral_lift = False
    held_ticks = 0
    result = {'mode':mode,'seed':seed,'prompt':prompt,'control':'FLUX 3 Action SO101 checkpoint plus trained LoRA',
        'learned_policy':True,'scripted_pickup':False,'object_attachment':False,
        'object_pose_writes_after_reset':False,'physics_geometry':'Existing documented fixed-finger hull split',
        'source_sha256':sha(__file__),'mujoco_xml_sha256':sha(folder/'cell.xml'),
        'calibration_status':'Trained simulator-native coordinate adapter; identity five joint degrees and gripper percentage',
        'guidance_scale':policy.config.guidance_scale,
        'normalization':{'state':policy.config.state_normalization,'action':policy.config.action_normalization}}
    try:
        for chunk_index in range(chunks):
            window = {key:np.stack([item[j] for item in history]) for j,key in enumerate(['scene','wrist','states','commands'])}
            np.savez_compressed(folder/f'input-{chunk_index:03d}.npz',**window)
            batch = local_runtime.history_batch(policy,**window,prompt=prompt)
            actions,timing = predict_adapted(policy,batch)
            np.save(folder/f'actions-policy-{chunk_index:03d}.npy',actions,allow_pickle=False)
            commands = transform(actions,mode,inverse=True)
            timing['chunk'] = chunk_index
            timings.append(timing)
            for raw in commands[:policy.config.n_action_steps]:
                issued,was_clipped,clip_magnitude = safe_command(env,raw)
                clipped += int(was_clipped)
                max_clip = max(max_clip,clip_magnitude)
                arrays['qpos_before'].append(d.qpos.copy())
                arrays['qvel_before'].append(d.qvel.copy())
                env.step(issued)
                row,contacts,tcp = fresh_record(env)
                bilateral_lift |= bool(bilateral_finger_contact(env) and row['object_xyz'][2]>.09)
                rgb = [policy_rgb(env.render(name)) for name in ['scene','wrist']]
                measured = env.state_deg().copy()
                for key,value in [('qpos_after',d.qpos.copy()),('qvel_after',d.qvel.copy()),
                    ('time_s',d.time),('raw_commands_sim_units',raw),('applied_commands_sim_units',issued),
                    ('measured_states_sim_units',measured),('tcp_m',tcp),('cube_truth_m',row['object_xyz']),
                    ('scene_rgb',rgb[0]),('wrist_rgb',rgb[1]),('controls',d.ctrl.copy())]:
                    arrays[key].append(value)
                contacts_log.append(contacts)
                # This state precedes the NEXT control command; history records
                # the previous actually applied command, not the future queue.
                history.append((*rgb,transform(measured,mode),transform(issued,mode)))
            progress = {'mode':mode,'seed':seed,'completed_chunks':chunk_index+1,'planned_chunks':chunks,
                'simulated_seconds':len(arrays['time_s'])/30, 'last_inference_seconds':timing['inference_seconds'],
                'grader_so_far':baseline.task_success(env),'clipped_commands':clipped}
            (folder/'progress.json').write_text(json.dumps(progress,indent=2))
            print(json.dumps(progress),flush=True)
        # A bounded fixed-command hold measures physical settling. It uses the
        # last FLUX-issued command, with no scripted opening or object changes.
        for _ in range(45):
            arrays['qpos_before'].append(d.qpos.copy());arrays['qvel_before'].append(d.qvel.copy())
            env.step(issued);row,contacts,tcp = fresh_record(env)
            held_ticks += 1
            rgb = [policy_rgb(env.render(name)) for name in ['scene','wrist']]
            for key,value in [('qpos_after',d.qpos.copy()),('qvel_after',d.qvel.copy()),('time_s',d.time),
                ('raw_commands_sim_units',raw),('applied_commands_sim_units',issued),
                ('measured_states_sim_units',env.state_deg().copy()),('tcp_m',tcp),
                ('cube_truth_m',row['object_xyz']),('scene_rgb',rgb[0]),('wrist_rgb',rgb[1]),('controls',d.ctrl.copy())]:
                arrays[key].append(value)
            contacts_log.append(contacts)
        original_grade = baseline.task_success(env)
        result.update(complete=True,success=bool(original_grade['success'] and bilateral_lift),
            grader=original_grade,bilateral_contact_lift_over_90mm_world_z=bilateral_lift)
    except Exception as exc:
        result.update(complete=False,success=False,error=str(exc),error_type=type(exc).__name__)
        (folder/'error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        if arrays['time_s']:
            np.savez_compressed(folder/'trajectory.npz',**{k:np.asarray(v) for k,v in arrays.items()})
            result['trajectory_sha256'] = sha(folder/'trajectory.npz')
            p = np.asarray(arrays['cube_truth_m']);tcp = np.asarray(arrays['tcp_m'])
            result.update(max_cube_displacement_m=float(np.linalg.norm(p-initial_cube,axis=1).max()),
                closest_tcp_to_cube_m=float(np.linalg.norm(tcp-p,axis=1).min()))
        result.update(physics_ticks=len(arrays['time_s']),simulated_seconds=len(arrays['time_s'])/30,
            clipped_commands=clipped,max_clipping_joint_radians=max_clip,inference=timings,
            planned_settling_hold_seconds=1.5,completed_settling_hold_ticks=held_ticks,
            completed_settling_hold_seconds=held_ticks/30,initial_cube_truth_m=initial_cube.tolist())
        (folder/'contacts.json').write_text(json.dumps(contacts_log))
        (folder/'report.json').write_text(json.dumps(result,indent=2))
        env.close()
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--guidances',default='3,1')
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--restoration',choices=['unmerged','merged'],default='unmerged')
    p.add_argument('--role',choices=['development','held-out'],default='development')
    p.add_argument('--seeds',default='920,921,922')
    p.add_argument('--chunks',type=int,default=24)
    p.add_argument('--resident-gib',type=float,default=6.)
    p.add_argument('--prompt',default='Grasp the red block and place it in the gray bin.')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    guidances=[float(v) for v in a.guidances.split(',')]
    assert guidances and all(v in [1.,3.] for v in guidances) and len(set(guidances))==len(guidances)
    modes=[f'native_cfg{g:g}' for g in guidances]
    seeds = [int(s) for s in a.seeds.split(',')]
    assert 1<=a.chunks<=48
    a.output.mkdir(parents=True,exist_ok=False)
    spec = {'frozen_before_outcomes':True,'role':a.role,
        'modes':modes,'seeds':seeds,'chunks':a.chunks,'execute_per_chunk':32,'fps':30,
        'prompt':a.prompt,'settling_hold_seconds':1.5,'resident_gib':a.resident_gib,
        'camera_render_pixels':[320,240],'policy_camera_resize_pixels':[256,256],
        'recorded_policy_rgb_pixels':[256,256],
        'camera_preprocessing':'Native320x240 RGB -> uint8 PIL bilinear256x256, identical to expert replay preparation -> official side_by_side canvas',
        'success_requires':'Existing strict lift/release/containment/rest grade PLUS bilateral jaw contact during >90mm world-Z lift',
        'runner_filename':Path(__file__).name,'guidance_scales':guidances,
        'guidance_reason':'All adapted caches contain task text; evaluate original CFG3 versus conditional-only CFG1 prospectively, keeping all outcomes',
        'runner_sha256':sha(__file__),'runtime_helper_sha256':sha(local_runtime.__file__),
        'baseline_sha256':sha(baseline.__file__),'source_revision':local_runtime.SOURCE_REVISION,
        'model_revision':local_runtime.REVISION,'no_training':False,
        'adapter_sha256':sha(a.adapter/'adapter.safetensors'),
        'adapter_metadata_sha256':sha(a.adapter/'adapter-metadata.json'),
        'adapter_config_sha256':sha(a.adapter/'policy-config.json'),
        'adapter_loader_sha256':sha(ROOT/'scripts/flux_adapted_inference.py'),
        'adapter_restoration':a.restoration,
        'adapter_units':'Five simulator arm joint degrees and gripper percent; identity transform','object_truth_inputs':'Reset/strict grading/diagnostics only; never passed to FLUX'}
    (a.output/'frozen-specification.json').write_text(json.dumps(spec,indent=2))
    def progress(event):
        (a.output/'runtime-progress.json').write_text(json.dumps(event,indent=2))
        print(json.dumps(event),flush=True)
    policy,metadata = load_adapted_policy(a.adapter,[a.prompt],resident_gib=a.resident_gib,progress=progress,restoration=a.restoration)
    (a.output/'runtime.json').write_text(json.dumps(metadata,indent=2))
    original_config = copy.deepcopy(policy.config)
    results = []
    for mode,guidance in zip(modes,guidances,strict=True):
        variant_config=copy.deepcopy(original_config)
        variant_config.guidance_scale=guidance
        for seed in seeds:
            result = run_episode(policy,variant_config,mode,seed,a.chunks,a.prompt,a.output/mode/f'seed-{seed}')
            results.append(result)
            (a.output/'summary.json').write_text(json.dumps({'complete':len(results)==len(modes)*len(seeds),
                'successful':sum(bool(r['success']) for r in results),'total_completed':len(results),'episodes':results},indent=2))
            if not result['complete']:
                raise RuntimeError('Runtime failure retained; stop before additional trials')

if __name__=='__main__':
    main()
