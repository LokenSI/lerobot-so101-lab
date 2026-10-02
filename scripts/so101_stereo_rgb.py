"""Virtual calibrated RGB stereo and marked-object contact control; no policy inference.

Uses the existing unchanged SO101 baseline and bootstrap-fetched simulator.
Camera images, not simulator object poses, enter the localization function.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H=640,480
BASELINE=.06
CAMERA_Z=.50
FOV=42.
MARKER_OFFSET=.01515
LEFT=np.array([.19,-.02,CAMERA_Z]);RIGHT=LEFT+np.array([BASELINE,0,0])
FY=H/(2*np.tan(np.deg2rad(FOV/2)))
CALIBRATION={'image_size':[W,H],'baseline_m':BASELINE,'K':[[FY,0,W/2],[0,FY,H/2],[0,0,1]],'left_center_world_m':LEFT.tolist(),'right_center_world_m':RIGHT.tolist(),'world_to_camera_rotation':[[1,0,0],[0,-1,0],[0,0,-1]],'distortion':[0,0,0,0,0],'rectification':'Parallel identical ideal pinhole cameras: native render is rectified','marker_to_cube_center_world_m':[0,0,-MARKER_OFFSET],'scope':'Generic virtual pair, not calibrated physical hardware'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_baseline(base_repo):
 spec=importlib.util.spec_from_file_location('stereo_existing_baseline',base_repo/'scripts/so101_pick_place_baseline.py')
 baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
 return baseline

def find_marker(rgb):
 """RGB-only color prior; no segmentation/depth/object-ID simulator buffers."""
 a=np.asarray(rgb).astype(float)
 mask=(a[:,:,0]>80)&(a[:,:,2]>70)&(a[:,:,0]>1.65*a[:,:,1])&(a[:,:,2]>1.55*a[:,:,1])&(a[:,:,2]>.65*a[:,:,0])
 remaining=set(zip(*np.where(mask)));blobs=[]
 while remaining:
  start=remaining.pop();stack=[start];points=[start]
  while stack:
   y,x=stack.pop()
   for neighbour in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
    if neighbour in remaining:remaining.remove(neighbour);stack.append(neighbour);points.append(neighbour)
  if len(points)>=12:
   y,x=np.asarray(points).T;blobs.append({'u':float(x.mean()+.5),'v':float(y.mean()+.5),'area':len(x)})
 if len(blobs)!=1:return {'valid':False,'reason':'missing or ambiguous colored marker','components':len(blobs)}
 return {'valid':True,**blobs[0]}

def localize(left_rgb,right_rgb,calibration=CALIBRATION):
 """Only RGB arrays and fixed public calibration enter this estimator."""
 left,right=find_marker(left_rgb),find_marker(right_rgb)
 result={'left':left,'right':right,'valid':False}
 if not left['valid'] or not right['valid']:result['reason']='marker missing/ambiguous in one or both cameras';return result
 disparity=left['u']-right['u'];vertical=abs(left['v']-right['v'])
 result.update(disparity_px=disparity,epipolar_error_px=vertical)
 if disparity<=1 or vertical>1.5:result['reason']='invalid disparity or epipolar mismatch';return result
 K=np.asarray(calibration['K']);depth=K[0,0]*calibration['baseline_m']/disparity
 if not .15<depth<.8:result['reason']='outside calibrated work-volume';return result
 camera=np.array([(left['u']-K[0,2])*depth/K[0,0],((left['v']+right['v'])/2-K[1,2])*depth/K[1,1],depth])
 world=np.asarray(calibration['left_center_world_m'])+np.asarray(calibration['world_to_camera_rotation']).T@camera
 cube=world+np.asarray(calibration['marker_to_cube_center_world_m'])
 result.update(valid=True,reason='RGB marker stereo triangulation',depth_m=float(depth),marker_world_m=world.tolist(),cube_center_world_m=cube.tolist())
 return result

def build_env(baseline,seed,condition,folder):
 base=baseline.make_env(seed,width=W,height=H,offscreen=False)
 folder.mkdir(parents=True,exist_ok=True);xmlpath=folder/'cell.xml'
 mujoco.mj_saveLastXML(str(xmlpath),base.model);tree=ET.parse(xmlpath);world=tree.getroot().find('worldbody')
 for name,pos in [('stereo_left',LEFT),('stereo_right',RIGHT)]:ET.SubElement(world,'camera',name=name,pos=' '.join(map(str,pos)),xyaxes='1 0 0 0 1 0',fovy=str(FOV))
 obj=world.find(".//body[@name='obj0']")
 ET.SubElement(obj,'geom',name='rgb_marker',type='cylinder',size='.0045 .0001',pos=f'0 0 {MARKER_OFFSET-.0001}',rgba='1 0.05 0.9 1',contype='0',conaffinity='0',mass='0')
 if condition=='left_occluded':ET.SubElement(world,'geom',name='camera_occluder',type='box',size='.012 .012 .001',pos='.19 -.02 .46',rgba='.12 .12 .12 1',contype='0',conaffinity='0')
 tree.write(xmlpath,encoding='unicode');model=mujoco.MjModel.from_xml_path(str(xmlpath))
 env=object.__new__(baseline.GradedSim);env.scene='Single cube';env.width=W;env.height=H;env.seed=seed;env.offscreen=True
 env.model=model;env.data=mujoco.MjData(model);env.qadr=np.array([model.joint(j).qposadr[0] for j in baseline.sim.JOINTS]);env.renderer=mujoco.Renderer(model,H,W);env.reset()
 if condition=='shifted_start':
  adr=model.joint('obj0_free').qposadr[0];env.data.qpos[adr:adr+2]+=np.array([.020,.025]);env.data.qvel[:]=0;mujoco.mj_forward(model,env.data)
  for _ in range(120):mujoco.mj_step(model,env.data)
 if condition=='unmarked':model.geom_rgba[model.geom('rgb_marker').id,3]=0
 mujoco.mj_forward(model,env.data);env.grading_trace=[]
 base.close();return env,xmlpath

def capture(env):
 mujoco.mj_forward(env.model,env.data)
 return env.render('stereo_left'),env.render('stereo_right')

def calibration_checks(baseline,output):
 folder=output/'calibration-checks';env,xmlpath=build_env(baseline,123,'nominal',folder);adr=env.model.joint('obj0_free').qposadr[0];checks=[]
 for i,xyz in enumerate([[.18,-.05,.015],[.20,-.01,.015],[.23,.03,.015],[.25,-.08,.030],[.22,.05,.060]]):
  env.data.qpos[adr:adr+3]=xyz;env.data.qvel[:]=0;mujoco.mj_forward(env.model,env.data)
  l,r=capture(env);estimate=localize(l,r);Image.fromarray(l).save(folder/f'probe-{i}-left.png');Image.fromarray(r).save(folder/f'probe-{i}-right.png')
  error=float(np.linalg.norm(np.array(estimate.get('cube_center_world_m',[9,9,9]))-xyz));checks.append({'truth_cube_center_m':xyz,'measurement':estimate,'error_m':error})
  if not estimate['valid'] or error>.002:raise AssertionError(f'RGB stereo calibration check failed: {checks[-1]}')
 negatives={'swapped_pair':localize(r,l),'vertical_correspondence_shift':localize(l,np.roll(r,16,axis=0)),'blank_pair':localize(np.zeros_like(l),np.zeros_like(r))}
 assert not any(v['valid'] for v in negatives.values())
 env.close();result={'calibration':CALIBRATION,'positive_probes':checks,'negative_checks':negatives,'max_error_m':max(c['error_m'] for c in checks),'max_epipolar_error_px':max(c['measurement']['epipolar_error_px'] for c in checks),'scope':'RGB-derived known colored marker; truth is used only in this independent diagnostic check'}
 (folder/'report.json').write_text(json.dumps(result,indent=2));return result

def run_episode(baseline,seed,condition,folder):
 env,xmlpath=build_env(baseline,seed,condition,folder);m,d=env.model,env.data
 left,right=capture(env);estimate=localize(left,right);Image.fromarray(left).save(folder/'initial-left.png');Image.fromarray(right).save(folder/'initial-right.png')
 actual=d.body('obj0').xpos.copy() # Diagnostic only; never enters controller goals.
 result={'condition':condition,'seed':seed,'control':'Scripted IK from one RGB-only stereo estimate at reset','learned_policy':False,'calibration':CALIBRATION,'initial_measurement':estimate,'diagnostic_initial_truth_m':actual.tolist(),'diagnostic_initial_error_m':float(np.linalg.norm(np.asarray(estimate['cube_center_world_m'])-actual)) if estimate['valid'] else None,'perception_inputs':'Two RGB arrays plus fixed calibration and known marker offset; no simulator pose/depth/object IDs','object_attachment':False,'object_pose_writes_after_reset':False,'model_sha256':sha(xmlpath),'renderer_runner_sha256':sha(__file__)}
 if not estimate['valid']:
  result.update(success=False,abstained=True,reason='No grasp motion: invalid stereo observation',grader={'success':False,'reason':'Perception abstention before motion'},frames=0)
  (folder/'report.json').write_text(json.dumps(result,indent=2));env.close();return result
 measured=np.asarray(estimate['cube_center_world_m']);xy=measured[:2];ctrl=baseline.Controller(env);yaw=.3;R=baseline.rotz(yaw);tray=np.array([.24,-.16]);open_q=baseline.sim.deg_to_q([0,0,0,0,0,45])[-1];close_q=baseline.sim.deg_to_q([0,0,0,0,0,0])[-1]
 requests=[('lift clear',[.22,0,.10],baseline.tilted([.22,0]),open_q,2.),('orient above measured cube',np.r_[xy,.06],R,open_q,2.),('approach measured cube',np.r_[xy,.017],R,open_q,2.),('close grip',np.r_[xy,.017],R,close_q,1.5),('lift cube',np.r_[xy,.12],baseline.tilted(xy),close_q,2.),('transfer to known tray',np.r_[tray,.12],baseline.tilted(tray),close_q,2.5),('lower cube',np.r_[tray,.06],baseline.tilted(tray),close_q,1.5),('release',np.r_[tray,.06],baseline.tilted(tray),open_q,1.),('retreat',np.r_[tray,.15],baseline.tilted(tray),open_q,1.5),('settle',np.r_[tray,.15],baseline.tilted(tray),open_q,1.5)]
 arrays={k:[] for k in ['qpos_before','qpos_after','qvel_before','qvel_after','time_s','controls','tcp_m','phase','scripted_target_tcp_m','cube_truth_m','capture_ticks','left_rgb','right_rgb']};contacts=[];measurements=[];waypoints=[];index=0
 for phase,target,Rtarget,grip,seconds in requests:
  target=np.asarray(target);q,err,rerr=ctrl.solve(target,Rtarget,grip);waypoints.append({'phase':phase,'target_tcp_m':target.tolist(),'position_error_m':err,'rotation_error':rerr,'seconds':seconds})
  start=d.ctrl.copy();pstart,Rstart=ctrl.pose(start)
  for tick in range(round(seconds*30)):
   u=(tick+1)/round(seconds*30);blend=u*u*(3-2*u);action=start+(q-start)*blend;active=pstart+(target-pstart)*blend
   if phase in ['approach measured cube','lift cube','transfer to known tray','lower cube','retreat']:
    Rt=Rstart+(Rtarget-Rstart)*blend;U,S,Vt=np.linalg.svd(Rt);Rt=U@Vt;action,_,_=ctrl.solve(active,Rt,start[-1]+(grip-start[-1])*blend)
   arrays['qpos_before'].append(d.qpos.copy());arrays['qvel_before'].append(d.qvel.copy());env.step(baseline.sim.q_to_deg(action));mujoco.mj_forward(m,d)
   # Replace cached grader row with fresh FK at the same integrated state.
   pos=d.body('obj0').xpos.copy();rot=d.body('obj0').xmat.reshape(3,3);va=m.joint('obj0_free').dofadr[0];cs=baseline.contacts(env)
   env.grading_trace[-1].update(object_xyz=pos.tolist(),object_quaternion_wxyz=d.body('obj0').xquat.tolist(),extent=(np.abs(rot)@np.full(3,.015)).tolist(),linear_speed_m_s=float(np.linalg.norm(d.qvel[va:va+3])))
   for key,value in [('qpos_after',d.qpos.copy()),('qvel_after',d.qvel.copy()),('time_s',d.time),('controls',d.ctrl.copy()),('tcp_m',ctrl.pose(d.qpos[env.qadr])[0]),('phase',phase),('scripted_target_tcp_m',active),('cube_truth_m',pos)]:arrays[key].append(value)
   contacts.append(cs)
   if index%3==0:
    l,r=capture(env);arrays['capture_ticks'].append(index);arrays['left_rgb'].append(l);arrays['right_rgb'].append(r);measurements.append(localize(l,r))
   index+=1
 grader=baseline.task_success(env);result.update(success=grader['success'],abstained=False,grader=grader,frames=index,simulation_seconds=index/30,waypoints=waypoints,contact_frames=sum(bool(c) for c in contacts),capture_hz=10,controller_uses_estimated_xy=True,grasp_z_m=.017,known_fixed_task_tray_xy_m=tray.tolist())
 np.savez_compressed(folder/'trajectory.npz',**{k:np.asarray(v) for k,v in arrays.items()});(folder/'measurements.json').write_text(json.dumps(measurements));(folder/'contacts.json').write_text(json.dumps(contacts));(folder/'report.json').write_text(json.dumps(result,indent=2));env.close();print(condition,seed,result['success'],grader,flush=True);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--base-repo',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ROOT/'runs/so101-stereo');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 spec={'frozen_before_outcomes':True,'seeds':[810,811],'conditions':['nominal','shifted_start'],'negative_conditions':['unmarked','left_occluded'],'positive_trial_count':4,'calibration':CALIBRATION,'scope':'Marked-cube calibrated virtual stereo plus scripted contact control; no new ACT model','source_sha256':sha(__file__)}
 specpath=args.output/'frozen-specification.json'
 if specpath.exists():raise RuntimeError('Refusing to overwrite frozen experiment output')
 specpath.write_text(json.dumps(spec,indent=2));baseline=load_baseline(args.base_repo.resolve());checks=calibration_checks(baseline,args.output);reports=[]
 for condition in spec['conditions']:
  for seed in spec['seeds']:reports.append(run_episode(baseline,seed,condition,args.output/condition/f'seed-{seed}'))
 for condition in spec['negative_conditions']:reports.append(run_episode(baseline,810,condition,args.output/condition/'seed-810'))
 summary={'complete':True,'positive_successes':sum(r['success'] for r in reports[:4]),'positive_trials':4,'negative_abstentions':sum(r.get('abstained',False) for r in reports[4:]),'negative_trials':2,'episodes':reports,'calibration_checks':checks,'baseline_source_sha256':sha(args.base_repo/'scripts/so101_pick_place_baseline.py'),'new_source_sha256':sha(__file__)}
 (args.output/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='episodes' and k!='calibration_checks'}),flush=True)

if __name__=='__main__':main()
