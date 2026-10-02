"""Independent CPU-only saved-state audit of genuine FLUX SO101 rollouts.

Never imports the policy runtime, torch or a renderer. A failed task is a valid
audit outcome if its retained measurements and report agree.
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'runtime/flux-action-so101-sim'))
import sim

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runner_for_spec(spec):
    adapted='adapter_loader_sha256' in spec
    filename=spec.get('runner_filename','run_flux_so101_adapted.py' if adapted else 'run_flux_so101_retry.py')
    allowed={'run_flux_so101_adapted.py','run_flux_so101_adapted_cfg.py'} if adapted else {'run_flux_so101_retry.py'}
    if filename not in allowed:
        raise ValueError(f'Unsupported runner filename: {filename!r}')
    return ROOT/'scripts'/filename

def transformed(values,mode,spec,inverse=False):
    x=np.asarray(values,dtype=float)
    if mode!='legacy_candidate':return x.copy()
    contract=spec['legacy_candidate_affine']
    scale=np.asarray(contract['scale']);offset=np.asarray(contract['offset'])
    return (x-offset)/scale if inverse else x*scale+offset

def q_from_command(command):
    q=np.deg2rad(np.asarray(command,dtype=float)).copy()
    q[...,5]=sim.GRIPPER_CLOSED+np.asarray(command)[...,5]/100*(sim.GRIPPER_OPEN-sim.GRIPPER_CLOSED)
    return q

def state_from_q(q):
    x=np.rad2deg(np.asarray(q,dtype=float)).copy()
    x[...,5]=(np.asarray(q)[...,5]-sim.GRIPPER_CLOSED)/(sim.GRIPPER_OPEN-sim.GRIPPER_CLOSED)*100
    return x

def source_inspection(runner):
    tree=ast.parse(runner.read_text())
    writes=[];prohibited=[]
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                label=ast.unparse(target)
                if 'qpos' in label or 'qvel' in label:writes.append(label)
        if isinstance(node,ast.Call):
            label=ast.unparse(node.func)
            if label.endswith(('.Controller','.solve','.pose','.mj_resetData')):prohibited.append(label)
    # arrays dictionary assignments with recorded state are permitted; live writes aren't.
    return {'live_state_write_targets':writes,'hidden_ik_or_reset_calls':prohibited,'passes':not writes and not prohibited}

def verify_episode(folder,spec):
    report=json.loads((folder/'report.json').read_text())
    # Materialize once: repeatedly indexing compressed camera arrays would
    # otherwise decompress the complete episode for every history check.
    with np.load(folder/'trajectory.npz',allow_pickle=False) as archive:
        z={key:archive[key] for key in archive.files}
    m=mujoco.MjModel.from_xml_path(str(folder/'cell.xml'))
    d=mujoco.MjData(m)
    n=len(z['time_s']);mode=report['mode']
    checks={};metrics={}
    def check(name,ok,**detail):checks[name]={'passed':bool(ok),**detail}
    def near(name,a,b,atol=1e-9):
        a=np.asarray(a);b=np.asarray(b)
        err=float(np.max(np.abs(a-b))) if a.size else 0.
        check(name,a.shape==b.shape and err<=atol,max_abs_error=err,tolerance=atol)
    check('trajectory_hash',sha(folder/'trajectory.npz')==report['trajectory_sha256'])
    check('episode_runner_hash_matches_freeze',report['source_sha256']==spec['runner_sha256'])
    check('episode_mode_matches_freeze',mode in spec['modes'])
    if 'guidance_scales' in spec:
        expected_modes=[f'native_cfg{g:g}' for g in spec['guidance_scales']]
        check('guidance_modes_match_freeze',spec['modes']==expected_modes)
        expected_guidance=dict(zip(expected_modes,spec['guidance_scales'])).get(mode)
        check('episode_guidance_matches_freeze',expected_guidance is not None and report.get('guidance_scale')==expected_guidance,
              reported=report.get('guidance_scale'),expected=expected_guidance)
    check('xml_hash',sha(folder/'cell.xml')==report.get('mujoco_xml_sha256',report.get('model_sha256')))
    check('no_attachments',m.neq==0 and m.nmocap==0,neq=int(m.neq),nmocap=int(m.nmocap))
    check('all_arrays_same_length',all(len(z[k])==n for k in z))
    check('finite_numeric_arrays',all(np.isfinite(z[k]).all() for k in z))
    width,height=spec.get('recorded_policy_rgb_pixels',spec['camera_render_pixels'])
    for camera in ['scene','wrist']:
        check(f'{camera}_recorded_policy_rgb_contract',z[f'{camera}_rgb'].shape==(n,height,width,3) and z[f'{camera}_rgb'].dtype==np.uint8,
              shape=list(z[f'{camera}_rgb'].shape),expected_pixels=[width,height],dtype=str(z[f'{camera}_rgb'].dtype))
    check('free_body_joint',m.joint('obj0_free').type==mujoco.mjtJoint.mjJNT_FREE)
    near('qpos_continuity',z['qpos_before'][1:],z['qpos_after'][:-1])
    near('qvel_continuity',z['qvel_before'][1:],z['qvel_after'][:-1])
    near('control_time_interval',np.diff(z['time_s']),np.full(n-1,1/30),1e-10)
    qadr=np.array([m.joint(j).qposadr[0] for j in sim.JOINTS])
    near('measured_state_units',state_from_q(z['qpos_after'][:,qadr]),z['measured_states_sim_units'],1e-8)
    raw_q=q_from_command(z['raw_commands_sim_units'])
    clipped_q=np.clip(raw_q,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
    near('explicit_command_clipping',clipped_q,z['controls'])
    near('issued_command_units',state_from_q(clipped_q),z['applied_commands_sim_units'],1e-8)
    near('issued_to_ctrl_roundtrip',q_from_command(z['applied_commands_sim_units']),z['controls'],1e-9)
    expected_clip=np.any(np.abs(raw_q-clipped_q)>1e-9,axis=1)
    policy_ticks=n-report.get('completed_settling_hold_ticks',0)
    check('reported_tick_count',n==report['physics_ticks'])
    if report.get('complete'):
        check('complete_episode_tick_contract',policy_ticks==spec['chunks']*32 and n-policy_ticks==45)
    check('clipping_count',int(expected_clip[:policy_ticks].sum())==report['clipped_commands'],independent=int(expected_clip[:policy_ticks].sum()),reported=report['clipped_commands'])
    near('mapping_roundtrip',transformed(transformed(z['applied_commands_sim_units'],mode,spec),mode,spec,True),z['applied_commands_sim_units'],1e-8)
    inputs=sorted(folder.glob('input-*.npz'));actions=sorted(folder.glob('actions-policy-*.npy'))
    check('paired_input_and_action_chunks',len(inputs)==len(actions)==len(report['inference']))
    chunk_metrics=[]
    for k,(inp,act) in enumerate(zip(inputs,actions)):
        window=np.load(inp,allow_pickle=False);predicted=np.load(act,allow_pickle=False)
        start=k*32;stop=min(start+32,policy_ticks)
        near(f'chunk{k}_raw_actions_one_inverse_transform',transformed(predicted[:stop-start],mode,spec,True),z['raw_commands_sim_units'][start:stop],1e-8)
        if k==0:
            init=transformed(state_from_q(z['qpos_before'][0,qadr]),mode,spec)
            near('initial_state_history',window['states'],np.repeat(init[None],8,axis=0),1e-8)
            near('initial_command_history',window['commands'],np.repeat(init[None],8,axis=0),1e-8)
            for cam in ['scene','wrist']:
                rgb=np.array(Image.open(folder/f'initial-{cam}.png'))
                check(f'initial_{cam}_image_history',np.array_equal(window[cam],np.repeat(rgb[None],8,axis=0)))
        else:
            indices=np.arange(start-8,start)
            near(f'chunk{k}_state_history_alignment',window['states'],transformed(z['measured_states_sim_units'][indices],mode,spec),1e-8)
            near(f'chunk{k}_applied_command_history_alignment',window['commands'],transformed(z['applied_commands_sim_units'][indices],mode,spec),1e-8)
            for cam in ['scene','wrist']:check(f'chunk{k}_{cam}_rgb_alignment',np.array_equal(window[cam],z[f'{cam}_rgb'][indices]))
        distances=np.linalg.norm(z['tcp_m'][start:stop]-z['cube_truth_m'][start:stop],axis=1)
        chunk_metrics.append({'chunk':k,'start_tick':start,'end_tick_exclusive':stop,'closest_tcp_cube_m':float(distances.min()),'max_cube_world_z_m':float(z['cube_truth_m'][start:stop,2].max())})
    if report.get('completed_settling_hold_ticks',0):
        near('settling_hold_last_issued_command',z['controls'][policy_ticks:],np.repeat(z['controls'][policy_ticks-1][None],n-policy_ticks,axis=0))
    obj=m.body('obj0').id;finger=m.geom('fixed_finger_contact').id
    tcp=[];xyz=[];quats=[];extent=[];speed=[];released=[];bilateral=[];force_rows=[]
    contactlog=json.loads((folder/'contacts.json').read_text())
    logmatch=True;contact_mismatches=[];contact_distance_max_error=0.
    # mj_saveLastXML reloads serialized mesh transforms/vertices; its roundtrip
    # is not bit-exact. 0.1 micrometre is far below contact/task tolerances.
    contact_distance_tolerance=1e-7
    for t in range(n):
        d.qpos[:]=z['qpos_after'][t];d.qvel[:]=z['qvel_after'][t];d.ctrl[:]=z['controls'][t];d.time=float(z['time_s'][t])
        mujoco.mj_forward(m,d)
        pos=d.xpos[obj].copy();R=d.xmat[obj].reshape(3,3)
        xyz.append(pos);quats.append(d.xquat[obj].copy());extent.append(np.abs(R)@np.full(3,.015))
        va=int(m.joint('obj0_free').dofadr[0]);speed.append(float(np.linalg.norm(d.qvel[va:va+3])))
        gb=m.body('gripper').id;tcp.append(d.xpos[gb]+d.xmat[gb].reshape(3,3)@np.array([.0071,-.000218121,-.088]))
        fixed_force=moving_force=0.;rel=True;current=[]
        for ci,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if obj not in [b1,b2]:continue
            othergeom=int(c.geom2 if b1==obj else c.geom1);otherbody=b2 if b1==obj else b1
            name=m.body(otherbody).name
            force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
            if othergeom==finger:fixed_force=max(fixed_force,float(force[0]))
            if name=='moving_jaw_so101_v1':moving_force=max(moving_force,float(force[0]))
            if name in ['gripper','moving_jaw_so101_v1']:rel=False
            current.append([m.body(b1).name,m.body(b2).name,float(c.dist)])
        released.append(rel);bilateral.append(fixed_force>.001 and moving_force>.001 and pos[2]>.09)
        force_rows.append({'tick':t,'cube_z_m':float(pos[2]),'fixed_finger_peak_contact_force_n':fixed_force,'moving_jaw_peak_contact_force_n':moving_force,'bilateral_lift':bilateral[-1]})
        if len(current)==len(contactlog[t]):
            contact_distance_max_error=max(contact_distance_max_error,max([abs(a[2]-b[2]) for a,b in zip(current,contactlog[t])]+[0.]))
        if len(current)!=len(contactlog[t]) or any(a[:2]!=b[:2] or abs(a[2]-b[2])>contact_distance_tolerance for a,b in zip(current,contactlog[t])):
            logmatch=False
            contact_mismatches.append({'tick':t,'recomputed':current,'recorded':contactlog[t]})
    xyz=np.asarray(xyz);quats=np.asarray(quats);extent=np.asarray(extent)
    near('cube_truth_fresh_fk',xyz,z['cube_truth_m'],1e-10)
    near('tcp_fresh_fk',tcp,z['tcp_m'],1e-10)
    check('contact_geometry_log_matches',logmatch,mismatched_ticks=len(contact_mismatches),max_distance_error_m=contact_distance_max_error,tolerance_m=contact_distance_tolerance,first_mismatches=contact_mismatches[:10])
    last=slice(max(0,n-45),n);p=xyz[last];ex=extent[last]
    xy=np.abs(p[:,:2]-np.array([.24,-.16]))+ex[:,:2]
    inside=bool(n>=45 and np.all(xy[:,0]<.070) and np.all(xy[:,1]<.055) and np.all((p[:,2]-ex[:,2]>.003)&(p[:,2]-ex[:,2]<.005)))
    spread=float(np.linalg.norm(np.ptp(p,axis=0)))
    angle=float(np.rad2deg(np.max(2*np.arccos(np.clip(np.abs(quats[last]@quats[last][0]),-1,1)))))
    maxspeed=float(max(speed[-45:]));stable=spread<.001 and angle<1 and maxspeed<.002
    grade={'lifted':bool(xyz[:,2].max()>.09),'inside_tray_for_1_5_s':inside,'released_for_1_5_s':bool(n>=45 and all(released[-45:])),'stable':bool(stable),'max_object_height_m':float(xyz[:,2].max()),'settle_position_spread_m':spread,'settle_orientation_spread_deg':angle,'settle_max_linear_speed_m_s':maxspeed}
    grade['success']=all(grade[k] for k in ['lifted','inside_tray_for_1_5_s','released_for_1_5_s','stable']) and any(bilateral)
    check('task_success_report_matches',grade['success']==report.get('success',False))
    if report.get('complete'):
        for key in ['lifted','inside_tray_for_1_5_s','released_for_1_5_s','stable']:check(f'grader_{key}_matches',grade[key]==report['grader'][key])
        check('bilateral_lift_report_matches',any(bilateral)==report['bilateral_contact_lift_over_90mm_world_z'])
    # Per-frame independent dynamics from saved measured state: cold solver warmstart
    # may differ slightly. Record errors; large jumps fail, task failure doesn't.
    prediction=[];velocity=[]
    replay=mujoco.MjData(m)
    for t in range(n):
        mujoco.mj_resetData(m,replay)
        replay.qpos[:]=z['qpos_before'][t];replay.qvel[:]=z['qvel_before'][t];replay.ctrl[:]=z['controls'][t];replay.time=float(z['time_s'][t]-1/30)
        mujoco.mj_forward(m,replay)
        for _ in range(20):mujoco.mj_step(m,replay)
        prediction.append(float(np.max(np.abs(replay.qpos-z['qpos_after'][t]))))
        velocity.append(float(np.max(np.abs(replay.qvel-z['qvel_after'][t]))))
    check('physics_step_replay_no_teleports',max(prediction)<1e-3,qpos_max_abs_error=max(prediction),tolerance=1e-3,caveat='Cold solver warmstart; position array mixes metres and radians. Source AST also audited.')
    runs=[];start=None
    for i,active in enumerate([*bilateral,False]):
        if active and start is None:start=i
        elif not active and start is not None:
            count=i-start
            runs.append({'start_tick':start,'end_tick_inclusive':i-1,'frames':count,'observed_endpoint_span_s':(count-1)/30})
            start=None
    pair_anyheight=[r['fixed_finger_peak_contact_force_n']>.001 and r['moving_jaw_peak_contact_force_n']>.001 for r in force_rows]
    for row in chunk_metrics:
        a,b=row['start_tick'],row['end_tick_exclusive']
        row['bilateral_lift_frames']=int(sum(bilateral[a:b]))
        row['bilateral_jaw_contact_any_height_frames']=int(sum(pair_anyheight[a:b]))
    metrics.update(grade=grade,bilateral_lift_ticks=int(sum(bilateral)),bilateral_lift_runs=runs,
        longest_observed_bilateral_lift_span_s=max([r['observed_endpoint_span_s'] for r in runs]+[0.]),
        bilateral_jaw_contact_any_height_frames=int(sum(pair_anyheight)),
        contact_duration_note='Contacts reconstructed at 30Hz saved states; endpoint spans are sampled evidence, not a claim of uninterrupted substep contact.',
        fixed_finger_peak_contact_force_n=max(r['fixed_finger_peak_contact_force_n'] for r in force_rows),
        moving_jaw_peak_contact_force_n=max(r['moving_jaw_peak_contact_force_n'] for r in force_rows),
        closest_tcp_cube_m=float(np.linalg.norm(np.asarray(tcp)-xyz,axis=1).min()),per_chunk=chunk_metrics,physics_qpos_error_max=max(prediction),physics_qvel_error_max=max(velocity),cold_replay_worst_tick=int(np.argmax(prediction)))
    out={'episode':str(folder),'audit_gpu_used':False,'mujoco_version':mujoco.__version__,'checker_sha256':sha(__file__),'checks':checks,'passed':all(c['passed'] for c in checks.values()),'metrics':metrics}
    (folder/'independent-cpu-audit.json').write_text(json.dumps(out,indent=2))
    (folder/'independent-contact-forces.json').write_text(json.dumps(force_rows,indent=2))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path);args=p.parse_args()
    spec=json.loads((args.suite/'frozen-specification.json').read_text())
    adapted='adapter_loader_sha256' in spec
    runner=runner_for_spec(spec)
    source=source_inspection(runner)
    source['runner_hash_matches_freeze']=sha(runner)==spec['runner_sha256']
    source['baseline_hash_matches_freeze']=sha(ROOT/'scripts/so101_pick_place_baseline.py')==spec['baseline_sha256']
    source['runtime_helper_hash_matches_freeze']=sha(ROOT/'runtime/flux-so101-work/local_runtime_fast.py')==spec['runtime_helper_sha256']
    if adapted:
        source['adapter_loader_hash_matches_freeze']=sha(ROOT/'scripts/flux_adapted_inference.py')==spec['adapter_loader_sha256']
        runtime=json.loads((args.suite/'runtime.json').read_text())
        recorded=runtime['trained_adapter']
        source['adapter_runtime_hash_matches_freeze']=recorded['sha256']==spec['adapter_sha256']
        source['adapter_metadata_runtime_hash_matches_freeze']=recorded['metadata_sha256']==spec['adapter_metadata_sha256']
        source['adapter_config_runtime_hash_matches_freeze']=recorded['config_sha256']==spec['adapter_config_sha256']
        source['identity_simulator_calibration']=bool(spec['modes']) and all(mode in {'native_adapted','native_cfg3','native_cfg1'} for mode in spec['modes'])
        if runner.name=='run_flux_so101_adapted_cfg.py':
            source['prospective_guidance_contract']=bool(spec.get('guidance_scales')) and all(g in [1.,3.] for g in spec['guidance_scales']) and len(set(spec['guidance_scales']))==len(spec['guidance_scales']) and spec['modes']==[f'native_cfg{g:g}' for g in spec['guidance_scales']]
        source['restoration_mode_matches_freeze']=recorded['restoration']['mode']==spec['adapter_restoration']
        source['original_base_checkpoint_preserved']=recorded['metadata']['original_model_sha256']==runtime['sha256']['model.safetensors']
    episodes=[]
    for folder in sorted(args.suite.glob('*/seed-*')):
        if (folder/'report.json').exists() and (folder/'trajectory.npz').exists():episodes.append(verify_episode(folder,spec))
    out={'source_inspection':source,'episodes':episodes,'passed':bool(episodes) and all(source[key] for key in source if isinstance(source[key],bool)) and all(e['passed'] for e in episodes)}
    (args.suite/'independent-cpu-audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'passed':out['passed'],'source':source,'episodes':[{'episode':e['episode'],'passed':e['passed'],'failed_checks':[k for k,v in e['checks'].items() if not v['passed']],'metrics':e['metrics']} for e in episodes]},indent=2))
    sys.exit(0 if out['passed'] else 1)

if __name__=='__main__':main()
