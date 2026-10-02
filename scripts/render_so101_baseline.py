"""Presentation replay of a verified scripted SO-101 physics demonstration."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from so101_pick_place_baseline import ROOT,make_env
base=ROOT/'runs/so101-teaser-baseline'
trial=ROOT/'runs/so101-teaser-dataset/seed-000'
report=json.loads((trial/'report.json').read_text())
assert report['success'] and report['final_grader_replay_verified']
rows=json.loads((trial/'telemetry.json').read_text())
env=make_env(seed=0,width=96,height=96)
env.model.vis.global_.offwidth=960;env.model.vis.global_.offheight=720
renderer=mujoco.Renderer(env.model,720,960)
fontdir=Path('/mnt/c/Windows/Fonts')
title=ImageFont.truetype(str(fontdir/'segoeuib.ttf'),28)
body=ImageFont.truetype(str(fontdir/'segoeui.ttf'),21)
lift=max(range(len(rows)),key=lambda i:rows[i]['object_xyz'][2])
indices=[0,224,lift,len(rows)-1]
saved={}
try:
 with imageio.get_writer(str(base/'scripted-pick-place.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
  for i,row in enumerate(rows):
   # These writes replay already-recorded world states for rendering only.
   env.data.qpos[:]=row['qpos'];mujoco.mj_forward(env.model,env.data)
   renderer.update_scene(env.data,camera='overview')
   im=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(im)
   d.rectangle((0,0,960,88),fill=(8,20,32))
   d.text((22,8),'SO-101 | Contact-based pick and place',font=title,fill='white')
   d.text((22,49),'SCRIPTED BASELINE  |  MuJoCo physics + virtual cameras',font=body,fill=(136,224,214))
   d.rectangle((0,662,960,720),fill=(8,20,32))
   caption=f"{row['phase'].upper()}  |  seed0  |  {(i+1)/30:.1f}s"
   if i>=len(rows)-45:caption='PASS: lifted, released and resting inside the tray'
   d.text((22,679),caption,font=body,fill=(255,207,126))
   writer.append_data(np.asarray(im))
   if i in indices:
    im.save(base/f'frame-{i:04d}.png');saved[i]=im.copy()
finally:
 renderer.close();env.close()
sheet=Image.new('RGB',(1920,1440))
for n,i in enumerate(indices):sheet.paste(saved[i],((n%2)*960,(n//2)*720))
sheet.save(base/'contact-sheet.png')
metadata={'scope':'Presentation replay of recorded scripted-control physics trajectory','source_trial':str(trial.relative_to(ROOT)),'telemetry_sha256':hashlib.sha256((trial/'telemetry.json').read_bytes()).hexdigest(),'video_sha256':hashlib.sha256((base/'scripted-pick-place.mp4').read_bytes()).hexdigest(),'frames':len(rows),'fps':30,'simulation_success':True,'learned_policy':False,'object_pose_written_in_actual_test':False,'pose_writes_in_this_renderer':'Replay only; all qpos taken directly from saved physics frames','video':'scripted-pick-place.mp4','lift_frame':lift}
(base/'video-provenance.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2))
