"""SO-101 pick/place control in MuJoCo, with genuine contacts and no attachment.

The baseline uses privileged object pose for IK waypoint construction. It is
scripted control, not FLUX or a learned policy. Only position actuators are driven
after reset; the object's free joint is never written during a rollout.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runtime/flux-action-so101-sim'
sys.path.insert(0,str(SOURCE))
import sim

TCP_LOCAL=np.array([0.0071,-0.000218121,-0.088])

class GradedSim(sim.Sim):
 def reset(self):
  super().reset()
  self.grading_trace=[]
  self.initial_object_xyz=self.object_positions()['obj0'].copy()
 def step(self,command_deg):
  super().step(command_deg)
  pos=self.object_positions()['obj0'].copy()
  rot=self.data.body('obj0').xmat.reshape(3,3)
  extent=np.abs(rot)@np.full(3,.015)
  cs=contacts(self)
  released=not any('gripper' in (c[0],c[1]) or 'moving_jaw_so101_v1' in (c[0],c[1]) for c in cs)
  va=int(self.model.joint('obj0_free').dofadr[0])
  speed=float(np.linalg.norm(self.data.qvel[va:va+3]))
  self.grading_trace.append({'time':float(self.data.time),'object_xyz':pos.tolist(),'object_quaternion_wxyz':self.data.body('obj0').xquat.copy().tolist(),'linear_speed_m_s':speed,'extent':extent.tolist(),'released':released})

def task_success(env):
 """Grade lift and 1.5 s of released, stable rest wholly inside the tray."""
 trace=env.grading_trace
 if len(trace)<45:return {'success':False,'reason':'Fewer than 45 grading frames'}
 last=trace[-45:];p=np.array([r['object_xyz'] for r in last]);ext=np.array([r['extent'] for r in last])
 xy=np.abs(p[:,:2]-np.array([.24,-.16]))+ext[:,:2]
 inside=bool(np.all(xy[:,0]<.070) and np.all(xy[:,1]<.055) and np.all((p[:,2]-ext[:,2]>.003)&(p[:,2]-ext[:,2]<.005)))
 released=all(r['released'] for r in last)
 spread=float(np.linalg.norm(np.ptp(p,axis=0)))
 quat=np.array([r['object_quaternion_wxyz'] for r in last]);dots=np.abs(quat@quat[0]);angle=float(np.max(2*np.arccos(np.clip(dots,-1,1))))
 speed=max(r['linear_speed_m_s'] for r in last)
 stable=bool(spread<.001 and angle<np.deg2rad(1.) and speed<.002)
 lifted=max(r['object_xyz'][2] for r in trace)>.09
 return {'success':inside and released and stable and lifted,'lifted':lifted,'inside_tray_for_1_5_s':inside,'released_for_1_5_s':released,'stable':stable,'settle_position_spread_m':spread,'settle_orientation_spread_deg':float(np.rad2deg(angle)),'settle_max_linear_speed_m_s':speed,'max_object_height_m':max(r['object_xyz'][2] for r in trace),'final_object_xyz_m':p[-1].tolist(),'grading_frames':45}

def make_env(seed=0,width=640,height=480,offscreen=True,original_geometry=False):
 if original_geometry:
  return GradedSim('Single cube',width=width,height=height,seed=seed,offscreen=offscreen)
 # The source collision hull spans the concave fixed-finger/motor bracket.
 # Decompose it along Z into two overlapping hulls of its original vertices.
 # The visible geometry, inertial properties and every motor stay unchanged.
 original=sim.build_model('Single cube')
 gid=31;mid=original.geom_dataid[gid]
 va=original.mesh_vertadr[mid];vn=original.mesh_vertnum[mid]
 Rg=np.empty(9);mujoco.mju_quat2Mat(Rg,original.geom_quat[gid])
 verts=original.mesh_vert[va:va+vn]@Rg.reshape(3,3).T+original.geom_pos[gid]
 robot=ET.fromstring((SOURCE/'so101.xml').read_text())
 robot.find('compiler').set('meshdir',str(SOURCE/'assets'))
 fixed=robot.find(".//body[@name='gripper']")
 for g in fixed.findall('geom'):
  if g.get('class')=='collision' and g.get('mesh')=='wrist_roll_follower_so101_v1':
   g.set('contype','0');g.set('conaffinity','0')
 for name,points in [('fixed_finger',verts[verts[:,2]<-.055]),('fixed_housing',verts[verts[:,2]>-.060])]:
  ET.SubElement(robot.find('asset'),'mesh',name=name,vertex=' '.join(f'{v:.9g}' for v in points.ravel()))
  ET.SubElement(fixed,'geom',name=name+'_contact',type='mesh',mesh=name,**{'class':'collision','friction':'1.5 0.02 0.001','condim':'4'})
 scene=ET.fromstring(sim.scene_xml('Single cube'))
 include=scene.find('include');scene.remove(include)
 for child in robot:scene.append(child)
 model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
 env=object.__new__(GradedSim);env.scene='Single cube';env.width=width;env.height=height;env.seed=seed;env.offscreen=offscreen
 env.model=model;env.data=mujoco.MjData(model);env.qadr=np.array([model.joint(j).qposadr[0] for j in sim.JOINTS]);env.renderer=mujoco.Renderer(model,height,width) if offscreen else None
 env.reset()
 return env

def rotz(theta):
 c,s=np.cos(theta),np.sin(theta)
 return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def tilted(xy,yaw=.3,tilt=.5):
 theta=np.arctan2(xy[1],xy[0]-.0692345)
 c,s=np.cos(-tilt),np.sin(-tilt)
 return rotz(theta)@np.array([[c,0,s],[0,1,0],[-s,0,c]])@rotz(yaw-theta)

class Controller:
 def __init__(self, env):
  self.env=env;self.m=env.model;self.kin=mujoco.MjData(self.m)
  self.gid=self.m.body('gripper').id
  self.kin.qpos[:]=env.data.qpos
  self.lastq=env.data.qpos[env.qadr].copy()
 def pose(self,q):
  self.kin.qpos[self.env.qadr]=q;mujoco.mj_forward(self.m,self.kin)
  R=self.kin.xmat[self.gid].reshape(3,3)
  return self.kin.xpos[self.gid]+R@TCP_LOCAL,R
 def solve(self,target,Rtarget,grip,initial=None):
  q=self.lastq.copy() if initial is None else initial.copy()
  q[-1]=grip
  lo,hi=self.m.actuator_ctrlrange[:5].T
  def residual(q):
   p,R=self.pose(q)
   # All nine rotation components avoid orientation-axis ambiguity.
   return np.r_[p-target,.065*(R-Rtarget).ravel()]
  for _ in range(100):
   e=residual(q);J=np.empty((len(e),5))
   for j in range(5):
    qp=q.copy();qp[j]+=1e-5
    J[:,j]=(residual(qp)-e)/1e-5
   dq=np.linalg.solve(J.T@J+np.eye(5)*1e-6,-J.T@e)
   dq*=min(1,.15/max(np.max(np.abs(dq)),1e-12))
   q[:5]=np.clip(q[:5]+dq,lo+1e-4,hi-1e-4)
   if np.max(np.abs(dq))<1e-8:break
  p,R=self.pose(q)
  err=float(np.linalg.norm(p-target));rerr=float(np.linalg.norm(R-Rtarget))
  self.lastq=q.copy()
  return q,err,rerr

def contacts(env):
 names=[];objbid=env.model.body('obj0').id
 for i in range(env.data.ncon):
  c=env.data.contact[i];b1=env.model.geom_bodyid[c.geom1];b2=env.model.geom_bodyid[c.geom2]
  if objbid in [b1,b2]:
   names.append([env.model.body(b1).name,env.model.body(b2).name,float(c.dist)])
 return names

def run(seed,out,render=False,save_data=False,grasp_z=.017,grip_percent=0,original_geometry=False,data_size=96):
 out.mkdir(parents=True,exist_ok=True)
 env=make_env(seed,width=640 if render else data_size,height=480 if render else data_size,offscreen=render or save_data,original_geometry=original_geometry)
 ctrl=Controller(env);m,d=env.model,env.data
 obj0=env.object_positions()['obj0'].copy();yaw=.3
 R=rotz(yaw)
 q0=d.qpos[env.qadr].copy();startp,startR=ctrl.pose(q0)
 open_q=sim.deg_to_q([0,0,0,0,0,45])[-1]
 close_q=sim.deg_to_q([0,0,0,0,0,grip_percent])[-1]
 records=[];frames=[];dataset={'images_front':[],'images_wrist':[],'states':[],'actions':[]}
 waypoints=[]
 tray=np.array([.24,-.16])
 requests=[
  ('lift clear', np.array([.22,0,.10]),tilted([.22,0]),open_q,2.),
  ('orient above cube',np.r_[obj0[:2],.06],R,open_q,2.),
  ('approach',np.r_[obj0[:2],grasp_z],R,open_q,2.),
  ('close grip',np.r_[obj0[:2],grasp_z],R,close_q,1.5),
  ('lift cube',np.r_[obj0[:2],.12],tilted(obj0[:2]),close_q,2.),
  ('transfer to tray',np.r_[tray,.12],tilted(tray),close_q,2.5),
  ('lower cube',np.r_[tray,.06],tilted(tray),close_q,1.5),
  ('release',np.r_[tray,.06],tilted(tray),open_q,1.),
  ('retreat',np.r_[tray,.15],tilted(tray),open_q,1.5),
  ('settle',np.r_[tray,.15],tilted(tray),open_q,1.5),
 ]
 for phase,target,Rtarget,grip,seconds in requests:
  q,err,rerr=ctrl.solve(target,Rtarget,grip)
  waypoints.append({'phase':phase,'target_tcp_m':target.tolist(),'q':q.tolist(),'position_error_m':err,'rotation_error':rerr,'seconds':seconds})
  if err>.012 or rerr>.12:
   print('IK limitation',waypoints[-1],flush=True)
  startctrl=env.data.ctrl.copy()
  pstart,Rstart=ctrl.pose(startctrl)
  for t in range(round(seconds*sim.FPS)):
   u=(t+1)/round(seconds*sim.FPS);blend=u*u*(3-2*u)
   action=startctrl+(q-startctrl)*blend
   if phase in ['approach','lift cube','transfer to tray','lower cube','retreat']:
    Rt=Rstart+(Rtarget-Rstart)*blend
    U,S,Vt=np.linalg.svd(Rt);Rt=U@Vt
    action,_,_=ctrl.solve(pstart+(target-pstart)*blend,Rt,startctrl[-1]+(grip-startctrl[-1])*blend)
   before=env.data.qpos[env.qadr].copy()
   if save_data:
    dataset['images_front'].append(np.asarray(Image.fromarray(env.render('scene')).resize((data_size,data_size))))
    dataset['images_wrist'].append(np.asarray(Image.fromarray(env.render('wrist')).resize((data_size,data_size))))
    dataset['states'].append(before);dataset['actions'].append(action)
   env.step(sim.q_to_deg(action))
   p=env.object_positions()['obj0'].copy()
   records.append({'phase':phase,'time':float(d.time),'object_xyz':p.tolist(),'command_q':action.tolist(),'q':d.qpos[env.qadr].copy().tolist(),'contacts':contacts(env),'qpos':d.qpos.copy().tolist()})
   if render:
    frames.append(env.render('overview'))
  print(seed,phase,'obj',env.object_positions()['obj0'],'grip',env.state_deg()[-1],flush=True)
 final=env.object_positions()['obj0'].copy()
 settle=[r for r in records if r['phase']=='settle']
 settled_xyz=np.array([r['object_xyz'] for r in settle]);delta=np.abs(settled_xyz[:,:2]-tray)
 released=all(not any('gripper' in (c[0],c[1]) or 'moving_jaw_so101_v1' in (c[0],c[1]) for c in r['contacts']) for r in settle)
 inside=bool(np.all(delta[:,0]<.054) and np.all(delta[:,1]<.039) and np.all((settled_xyz[:,2]>.017)&(settled_xyz[:,2]<.025)))
 lifted=bool(max(r['object_xyz'][2] for r in records if r['phase']=='lift cube')>.09)
 stable=bool(np.max(np.linalg.norm(np.diff(settled_xyz,axis=0),axis=1))<.0005)
 report={'seed':seed,'success':inside and released and lifted and stable,'lifted':lifted,'inside_tray_for_1_5_s':inside,'released_for_1_5_s':released,'stable':stable,'initial_object_xyz_m':obj0.tolist(),'final_object_xyz_m':final.tolist(),'max_object_height_m':max(r['object_xyz'][2] for r in records),'control':'scripted Cartesian waypoint IK using simulator object pose','learned_policy':False,'object_attachment':False,'object_pose_write_after_reset':False,'simulation':{'engine':'MuJoCo','version':mujoco.__version__,'fps':sim.FPS,'physics_dt':sim.PHYSICS_DT},'source_model_files_modified':False,'collision_hull_split':not original_geometry,'tcp_local_m':TCP_LOCAL.tolist(),'waypoints':waypoints,'seconds':len(records)/sim.FPS}
 report.update(task_success(env))
 (out/'report.json').write_text(json.dumps(report,indent=2));(out/'telemetry.json').write_text(json.dumps(records))
 if save_data:np.savez_compressed(out/'episode.npz',**{k:np.array(v) for k,v in dataset.items()})
 if render:
  import imageio.v2 as imageio
  font='/mnt/c/Windows/Fonts/segoeui.ttf'
  f=ImageFont.truetype(font,17);fb=ImageFont.truetype('/mnt/c/Windows/Fonts/segoeuib.ttf',21)
  with imageio.get_writer(str(out/'rollout.mp4'),fps=sim.FPS,codec='libx264',quality=8,macro_block_size=16) as writer:
   for i,frame in enumerate(frames):
    im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,640,63),fill=(7,17,29))
    draw.text((15,7),'SO-101 | Pick and place in simulation',font=fb,fill='white')
    draw.text((15,35),'MuJoCo contacts | Scripted IK baseline',font=f,fill=(253,192,97))
    draw.rectangle((0,440,640,480),fill=(7,17,29))
    draw.text((15,449),f"{records[i]['phase'].upper()}  |  {i/sim.FPS:04.1f} s",font=f,fill='white')
    writer.append_data(np.asarray(im))
   im.save(out/'final.png')
 env.close();print(json.dumps(report),flush=True)
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='0');ap.add_argument('--output',type=Path,default=ROOT/'runs/so101-teaser-baseline');ap.add_argument('--render',action='store_true');ap.add_argument('--save-data',action='store_true');ap.add_argument('--grasp-z',type=float,default=.017);ap.add_argument('--grip-percent',type=float,default=0);ap.add_argument('--original-geometry',action='store_true');ap.add_argument('--data-size',type=int,default=96)
 args=ap.parse_args();reports=[]
 for seed in [int(x) for x in args.seeds.split(',')]:reports.append(run(seed,args.output/f'seed-{seed:03d}',args.render,args.save_data,args.grasp_z,args.grip_percent,args.original_geometry,args.data_size))
 (args.output/'summary.json').write_text(json.dumps({'successful':sum(r['success'] for r in reports),'total':len(reports),'runs':reports},indent=2))

if __name__=='__main__':main()
