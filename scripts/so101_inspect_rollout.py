import sys,json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/flux-action-so101-sim'))
import sim
from so101_pick_place_baseline import make_env
rows=json.loads((ROOT/'runs/so101-teaser-baseline/seed-000/telemetry.json').read_text())
env=make_env(0,width=640,height=480)
pics=[]
for phase in dict.fromkeys(r['phase'] for r in rows):
 row=[r for r in rows if r['phase']==phase][-1]
 env.data.qpos[:]=row['qpos'];mujoco.mj_forward(env.model,env.data)
 R=env.data.body('gripper').xmat.reshape(3,3)
 print(phase,'tcp',env.data.body('gripper').xpos+R@np.array([.0071,-.000218,-.088]))
 for i in range(env.data.ncon):
  c=env.data.contact[i]
  names=[env.model.body(env.model.geom_bodyid[g]).name for g in [c.geom1,c.geom2]]
  if names!=['world','obj0']:print('contact',c.geom1,c.geom2,names,c.dist,c.pos)
 im=Image.fromarray(env.render('overview'));ImageDraw.Draw(im).text((10,10),phase,fill='black');pics.append(im)
canvas=Image.new('RGB',(640*2,480*5))
for i,im in enumerate(pics):canvas.paste(im,((i%2)*640,(i//2)*480))
canvas.save(ROOT/'runs/so101-teaser-baseline/inspection.png');env.close()
