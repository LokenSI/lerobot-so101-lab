"""HD actual saved-state replay with actual RGB pair and sparse stereo measurement."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from so101_stereo_rgb import localize,CALIBRATION
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def font(n,bold=False):
 for p in [f'/mnt/c/Windows/Fonts/{"segoeuib" if bold else "segoeui"}.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  if Path(p).is_file():return ImageFont.truetype(p,n)
 return ImageFont.load_default()
def project(scene,points):
 c=mujoco.mjv_averageCamera(scene.camera[0],scene.camera[1]);f=np.asarray(c.forward);f/=np.linalg.norm(f);r=np.cross(f,c.up);r/=np.linalg.norm(r);u=np.cross(r,f);rel=np.asarray(points).reshape(-1,3)-c.pos;depth=rel@f;s=c.frustum_near/np.maximum(depth,1e-12);x,y=(rel@r)*s,(rel@u)*s;half=float(c.frustum_width) or 1440/1080*(c.frustum_top-c.frustum_bottom)/2
 pixels=np.c_[1440*(x-c.frustum_center+half)/(2*half),1080*(1-(y-c.frustum_bottom)/(c.frustum_top-c.frustum_bottom))];valid=(depth>c.frustum_near)&np.all((pixels>0)&(pixels<[1440,1080]),axis=1);return pixels,valid
def render(episode,out,base_repo):
 report=json.loads((episode/'report.json').read_text())
 with np.load(episode/'trajectory.npz') as saved:data={key:saved[key] for key in saved.files}
 tree=ET.parse(episode/'cell.xml');tree.getroot().find('compiler').set('meshdir',str(base_repo/'runtime/flux-action-so101-sim/assets'));model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));state=mujoco.MjData(model);model.vis.global_.offwidth=1440;model.vis.global_.offheight=1080;r=mujoco.Renderer(model,1080,1440);sr=mujoco.Renderer(model,480,640);out.mkdir(parents=True,exist_ok=True)
 state.qpos[:]=data['qpos_after'][0];mujoco.mj_forward(model,state);r.update_scene(state,camera='overview');c=mujoco.mjv_averageCamera(r.scene.camera[0],r.scene.camera[1]);point=c.pos+np.asarray(c.forward)*.5;pixel,valid=project(r.scene,[point]);r.scene.ngeom=1;r.scene.nlight=0;mujoco.mjv_initGeom(r.scene.geoms[0],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.0025),point,np.eye(3).ravel(),np.array([1.,0.,1.,1.]));r.scene.geoms[0].emission=1;rgb=r.render();ys,xs=np.where((rgb[:,:,0]>180)&(rgb[:,:,1]<70)&(rgb[:,:,2]>180));projection_error=float(np.linalg.norm([xs.mean()+.5-pixel[0,0],ys.mean()+.5-pixel[0,1]]));assert projection_error<1.5
 capture_ticks=data['capture_ticks'];ticks=list(capture_ticks);last=len(data['qpos_after'])-1
 if ticks[-1]!=last:ticks.append(last)
 records=[];images=[];video=out/'stereo-replay.mp4';big=font(28,True);medium=font(20);small=font(17);cyan=(73,225,230);amber=(255,190,78);pink=(255,135,240);ink=(8,18,29)
 with imageio.get_writer(str(video),fps=10,codec='libx264',quality=9,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
  for frame,tick in enumerate(ticks):
   state.qpos[:]=data['qpos_after'][tick];state.qvel[:]=data['qvel_after'][tick];state.ctrl[:]=data['controls'][tick];state.time=data['time_s'][tick];mujoco.mj_forward(model,state);r.update_scene(state,camera='overview');r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;im=Image.fromarray(r.render().copy());draw=ImageDraw.Draw(im)
   trail=data['tcp_m'][max(0,tick-120):tick+1];px,vis=project(r.scene,trail)
   for a,b,va,vb in zip(px[:-1],px[1:],vis[:-1],vis[1:]):
    if va and vb:draw.line([tuple(a),tuple(b)],fill=cyan,width=3)
   target=data['scripted_target_tcp_m'][tick];tp,tv=project(r.scene,[target])
   if tv[0]:x,y=tp[0];draw.ellipse((x-9,y-9,x+9,y+9),outline=amber,width=3)
   if tick in capture_ticks:
    ci=int(np.searchsorted(capture_ticks,tick));left,right=data['left_rgb'][ci],data['right_rgb'][ci];kind='Saved actual RGB capture'
   else:
    sr.update_scene(state,camera='stereo_left');left=sr.render().copy();sr.update_scene(state,camera='stereo_right');right=sr.render().copy();kind='Terminal RGB saved-state render'
   measurement=localize(left,right)
   if measurement['valid']:
    mp,mv=project(r.scene,[measurement['cube_center_world_m']])
    if mv[0]:x,y=mp[0];draw.rectangle((x-7,y-7,x+7,y+7),outline=pink,width=3)
   draw.rectangle((0,0,1440,110),fill=ink);draw.text((25,12),'SO-101 / calibrated virtual RGB stereo',font=big,fill='white');draw.text((25,56),f"{report['condition']} / seed {report['seed']}  |  SCRIPTED IK from RGB estimate  |  {state.time:.2f} simulated s",font=medium,fill=amber);draw.text((25,86),'Marked cube / sparse centroid disparity / generic virtual cameras / no learned policy or hardware test',font=small,fill=(185,200,215))
   draw.rounded_rectangle((1080,128,1430,858),radius=9,fill=ink)
   for name,img,y0,feature in [('LEFT',left,178,measurement['left']),('RIGHT',right,477,measurement['right'])]:
    draw.text((1096,y0-33),name+' RGB / 640 x 480',font=small,fill='white');im.paste(Image.fromarray(img).resize((320,240)),(1095,y0))
    if feature['valid']:
     x=1095+feature['u']/2;y=y0+feature['v']/2;draw.line((1095,y,1415,y),fill=pink,width=1);draw.ellipse((x-5,y-5,x+5,y+5),outline=pink,width=2)
   draw.text((1095,738),'Native parallel rectified pair',font=small,fill='white');draw.text((1095,768),'Baseline 60 mm / f = 625.22 px',font=small,fill='white');draw.text((1095,799),'Pink: image marker correspondence',font=small,fill=pink)
   draw.rectangle((0,875,1440,1080),fill=ink);draw.text((25,889),f"PHASE: {str(data['phase'][tick])}  |  {kind}",font=medium,fill='white')
   if measurement['valid']:
    xyz=measurement['cube_center_world_m'];desc=f"RGB stereo cube: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m  |  disparity {measurement['disparity_px']:.2f} px  |  epipolar error {measurement['epipolar_error_px']:.2f} px"
   else:desc='RGB localization INVALID: '+measurement['reason']+' / controller retains initial valid estimate'
   draw.text((25,930),desc,font=medium,fill=pink);draw.text((25,969),'Cyan observed TCP / amber scripted target / pink RGB triangulation. Marker offset and fixed tray task prior are known.',font=small,fill=cyan)
   terminal=bool(tick==last);status=f"FINAL STRICT CONTACT GRADER: {'PASS' if report['success'] else 'FAIL'}" if terminal else 'Final strict contact grader pending / no future success shown';draw.text((25,1005),status,font=medium,fill=(141,230,163) if terminal and report['success'] else amber);draw.text((25,1041),'Physics: actual free cube, contacts, no attachment; object pose written at reset only. Final 1.5 s presentation hold adds no sim time.',font=small,fill=(185,200,215))
   for _ in range(15 if terminal else 1):writer.append_data(np.asarray(im))
   if frame in [0,len(ticks)//2,len(ticks)-1]:im.save(out/f'frame-{tick:04d}.png');images.append(im.copy())
   if frame==len(ticks)//2:im.save(out/'poster.png')
   records.append({'video_frame':frame,'source_tick':int(tick),'simulation_time_s':float(state.time),'rgb_source':kind,'measurement':measurement,'scripted_target_world_m':target.tolist(),'terminal':terminal})
 r.close();sr.close();sheet=Image.new('RGB',(2160,540),ink)
 for i,img in enumerate(images):sheet.paste(img.resize((720,540)),(i*720,0))
 sheet.save(out/'contact-sheet.png');(out/'overlay-record.json').write_text(json.dumps(records));provenance={'source_episode':str(episode.relative_to(ROOT)),'trajectory_sha256':sha(episode/'trajectory.npz'),'source_report_sha256':sha(episode/'report.json'),'saved_model_sha256':sha(episode/'cell.xml'),'renderer_sha256':sha(__file__),'video_sha256':sha(video),'width':1440,'height':1080,'fps':10,'video_frames':len(ticks)+14,'control_hz':30,'camera_projection_check_error_px':projection_error,'rgb_capture_hz':10,'rendering':'Saved qpos/qvel replay; no dynamics integration. Mesh path resolved through explicit base-repo for portability','perception':'RGB colored marker only; no ground-truth pose, segmentation buffer or depth buffer input','scope':'New scripted stereo experiment; separate from previous ACT 2/5 and visual ACT 10/12','success':report['success'],'grader':report['grader']};(out/'provenance.json').write_text(json.dumps(provenance,indent=2));return provenance
def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-repo',type=Path,default=ROOT);a=p.parse_args();print(json.dumps(render(a.episode.resolve(),a.output,a.base_repo.resolve()),indent=2))
if __name__=='__main__':main()
