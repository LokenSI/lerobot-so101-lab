"""Replay absolute demonstration actions and verify observations and task grades."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
from so101_pick_place_baseline import ROOT,SOURCE,make_env,task_success,sim
base=ROOT/'runs/so101-teaser-dataset'
records=[]
for path in sorted(base.glob('seed-*/episode.npz')):
 report=json.loads(path.with_name('report.json').read_text())
 data=np.load(path)
 env=make_env(report['seed'],width=96,height=96,offscreen=False)
 maxerr=0.
 for state,action in zip(data['states'],data['actions']):
  maxerr=max(maxerr,float(np.max(np.abs(env.data.qpos[env.qadr]-state))))
  env.step(sim.q_to_deg(action))
 grade=task_success(env);env.close()
 assert maxerr<1e-8,(path,maxerr)
 assert data['images_front'].shape==(525,96,96,3)
 assert data['images_wrist'].shape==(525,96,96,3)
 assert data['images_front'].dtype==np.uint8
 assert data['images_wrist'].dtype==np.uint8
 assert data['states'].shape==data['actions'].shape==(525,6)
 assert np.isfinite(data['states']).all() and np.isfinite(data['actions']).all()
 report.update(grade)
 report['independent_action_replay_max_state_error_rad']=maxerr
 report['final_grader_replay_verified']=True
 path.with_name('report.json').write_text(json.dumps(report,indent=2))
 row={'seed':report['seed'],'success':grade['success'],'replay_max_error_rad':maxerr,'sha256_episode':hashlib.sha256(path.read_bytes()).hexdigest(),'path':str(path.relative_to(ROOT)),'frame_count':525}
 records.append(row)
 print(row,flush=True)
manifest={
 'purpose':'Scripted simulator demonstrations for a learned SO-101 image-and-joint-state policy',
 'task':'Lift the 30 mm red cube and release it fully inside the gray tray',
 'control_source':'Privileged-pose scripted Cartesian IK, no learned policy in demonstration controller',
 'learned_policy':False,'object_attachments':False,'object_pose_written_during_rollout':False,
 'engine':'MuJoCo','version':mujoco.__version__,
 'source':{'space':'https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim','revision':'d7da38e032da0e136d6de21c218dc616982a8cc9','pinned_files_modified':False,'source_model_sha256':hashlib.sha256((SOURCE/'so101.xml').read_bytes()).hexdigest()},
 'collision_adjustment':'Fixed-jaw concave visual mesh originally used as one convex collision hull. Replace only that collision geom with two overlapping convex hulls of original body-local vertices z<-0.055 m and z>-0.060 m. Original visual mesh, inertial properties, all joints and position actuators retained.',
 'script_sha256':hashlib.sha256((ROOT/'scripts/so101_pick_place_baseline.py').read_bytes()).hexdigest(),
 'observation_order':'images and joint state immediately BEFORE corresponding absolute action target is applied',
 'state_action_units':['radian']*6,'joint_names':sim.JOINTS,'control_hz':30,'episode_frames':525,
 'images':{'images_front':{'camera':'scene','shape':[96,96,3],'dtype':'uint8'},'images_wrist':{'camera':'wrist','shape':[96,96,3],'dtype':'uint8'},'rendered_natively_square':True},
 'randomization':'Source reset samples cube x and y independently within ±15 mm; cube yaw0.3 rad, lighting and cameras fixed. Seed is recorded.',
 'success_grader':{'lift_center_height_m':'>0.09','tray':'Entire oriented cube inside inner walls, base resting within1mm of tray floor','release':'No finger or gripper-body contacts for final45frames','rest_position_spread_m':'<0.001','rest_orientation_spread_deg':'<1','rest_max_linear_speed_m_s':'<0.002','consecutive_settle_frames':45},
 'total':len(records),'successful':sum(r['success'] for r in records),'runs':records,
}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2))
(base/'summary.json').write_text(json.dumps({'successful':manifest['successful'],'total':len(records),'runs':records,'verified_manifest':'manifest.json'},indent=2))
print('VERIFIED',manifest['successful'],'/',manifest['total'],flush=True)
