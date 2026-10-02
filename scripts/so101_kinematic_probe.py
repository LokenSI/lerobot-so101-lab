from pathlib import Path
import sys, json
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/flux-action-so101-sim'))
import sim
env=sim.Sim('Single cube',width=640,height=480,offscreen=True)
m,d=env.model,env.data
print('rest', env.state_deg())
for n in ['gripper','moving_jaw_so101_v1','wrist']:
 print(n, 'pos',d.body(n).xpos, 'rot',d.body(n).xmat.reshape(3,3))
print('site',d.site('gripperframe').xpos,d.site('gripperframe').xmat.reshape(3,3))
print('joints',m.jnt_range)
print('object',env.object_positions())
gbody=m.body('gripper').id
for i in range(m.ngeom):
 if m.geom_bodyid[i] in [gbody,m.body('moving_jaw_so101_v1').id]:
  print('geom',i,m.geom_type[i],m.geom_group[i],m.geom_dataid[i],m.geom_pos[i],d.geom(i).xpos)
  if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
   mid=m.geom_dataid[i]; a=m.mesh_vertadr[mid]; n=m.mesh_vertnum[mid]
   verts=m.mesh_vert[a:a+n] @ d.geom(i).xmat.reshape(3,3).T + d.geom(i).xpos
   local=(verts-d.body(gbody).xpos) @ d.body(gbody).xmat.reshape(3,3)
   print('local bounds',local.min(0),local.max(0))
   if i==31:
    for a,b in [(-.11,-.07),(-.08,-.06),(-.07,-.04),(-.06,0)]:
     v=local[(local[:,2]>=a)&(local[:,2]<=b)]
     print('band',a,b,'bounds',v.min(0),v.max(0))
for perc in [0,20,40,60]:
 d.qpos[env.qadr[-1]]=sim.deg_to_q([0,0,0,0,0,perc])[-1];mujoco.mj_forward(m,d)
 for gid in [31,37]:
  mid=m.geom_dataid[gid];a=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid]
  verts=m.mesh_vert[a:a+n] @ d.geom(gid).xmat.reshape(3,3).T + d.geom(gid).xpos
  local=(verts-d.body(gbody).xpos) @ d.body(gbody).xmat.reshape(3,3)
  tips=local[local[:,2]<-.083]
  if len(tips):print('tip bounds',perc,gid,tips.min(0),tips.max(0))
out=ROOT/'runs/so101-teaser-baseline';out.mkdir(exist_ok=True)
Image.fromarray(env.render('overview')).save(out/'probe.png')
env.close()
