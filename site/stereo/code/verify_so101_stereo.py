"""Independent saved-state/image audit and portable scene export; no rollout tuning."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from so101_stereo_rgb import localize
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--base-repo',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];run=a.run.resolve();results=[]
 summary=json.loads((run/'report.json').read_text())
 for e in summary['episodes']:
  folder=run/e['condition']/f"seed-{e['seed']}";tree=ET.parse(folder/'cell.xml');compiler=tree.getroot().find('compiler');compiler.set('meshdir',str(a.base_repo.resolve()/'runtime/flux-action-so101-sim/assets'));m=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));d=mujoco.MjData(m)
  # The replay export changes only mesh search path, preserving graded XML.
  depth=len(folder.relative_to(root).parts);compiler.set('meshdir','../'*depth+'runtime/flux-action-so101-sim/assets');tree.write(folder/'cell-portable.xml',encoding='unicode')
  portable={'source_model_sha256':sha(folder/'cell.xml'),'portable_model_sha256':sha(folder/'cell-portable.xml'),'only_change':'compiler meshdir points to repo bootstrap runtime','relative_meshdir':compiler.get('meshdir')};(folder/'portable-scene.json').write_text(json.dumps(portable,indent=2))
  if e['abstained']:
   from PIL import Image
   measurement=localize(np.asarray(Image.open(folder/'initial-left.png')),np.asarray(Image.open(folder/'initial-right.png')));assert not measurement['valid'];results.append({'condition':e['condition'],'seed':e['seed'],'abstention_reproduced_from_rgb':True});continue
  arr=np.load(folder/'trajectory.npz');np.testing.assert_allclose(arr['qpos_before'][1:],arr['qpos_after'][:-1],atol=1e-12,rtol=0);np.testing.assert_allclose(arr['qvel_before'][1:],arr['qvel_after'][:-1],atol=1e-12,rtol=0)
  original=json.loads((folder/'measurements.json').read_text());recomputed=[localize(l,r) for l,r in zip(arr['left_rgb'],arr['right_rgb'])];assert original==recomputed
  obj=m.body('obj0').id;trace=[];bilateral_lift=False;max_truth_error=0.
  for i,(q,v) in enumerate(zip(arr['qpos_after'],arr['qvel_after'])):
   d.qpos[:]=q;d.qvel[:]=v;d.ctrl[:]=arr['controls'][i];mujoco.mj_forward(m,d);pos=d.xpos[obj].copy();extent=np.abs(d.xmat[obj].reshape(3,3))@np.full(3,.015);va=m.joint('obj0_free').dofadr[0];pairs=[]
   for c in d.contact:
    b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
    if obj in (b1,b2):pairs.append(m.body(b2 if b1==obj else b1).name)
   fixed=any(name=='gripper' for name in pairs);moving=any('moving_jaw' in name for name in pairs);bilateral_lift|=bool(fixed and moving and pos[2]>.09)
   max_truth_error=max(max_truth_error,float(np.linalg.norm(pos-arr['cube_truth_m'][i])));trace.append((pos,extent,not(fixed or moving),float(np.linalg.norm(v[va:va+3])),d.xquat[obj].copy()))
  last=trace[-45:];xyz=np.array([r[0] for r in last]);extent=np.array([r[1] for r in last]);xy=np.abs(xyz[:,:2]-[.24,-.16])+extent[:,:2];inside=bool(np.all(xy[:,0]<.070) and np.all(xy[:,1]<.055) and np.all((xyz[:,2]-extent[:,2]>.003)&(xyz[:,2]-extent[:,2]<.005)));released=all(t[2] for t in last);quats=np.array([t[4] for t in last]);angle=float(np.max(2*np.arccos(np.clip(np.abs(quats@quats[0]),-1,1))));stable=bool(np.linalg.norm(np.ptp(xyz,axis=0))<.001 and max(t[3] for t in last)<.002 and angle<np.deg2rad(1))
  independent=inside and released and stable and bilateral_lift;assert independent==e['success']
  assert max_truth_error<1e-12 and m.neq==0 and m.joint('obj0_free').type[0]==mujoco.mjtJoint.mjJNT_FREE
  mujoco.mj_forward(m,d);positions=[d.cam_xpos[m.camera(name).id].copy() for name in ['stereo_left','stereo_right']];parallel=np.max(np.abs(d.cam_xmat[m.camera('stereo_left').id]-d.cam_xmat[m.camera('stereo_right').id]));assert parallel<1e-12;assert abs(np.linalg.norm(positions[1]-positions[0])-.06)<1e-12
  results.append({'condition':e['condition'],'seed':e['seed'],'independent_success':independent,'bilateral_contact_lift_over_90mm_world_z':bilateral_lift,'fresh_contact_release_final_45_ticks':released,'full_rotated_cube_containment':inside,'stable_rest':stable,'rest_orientation_spread_deg':float(np.rad2deg(angle)),'saved_truth_fresh_fk_error_m':max_truth_error,'all_saved_rgb_measurements_reproduced_exactly':True,'native_camera_rotation_difference':float(parallel),'actual_camera_baseline_m':float(np.linalg.norm(positions[1]-positions[0])),'no_equality_attachments':m.neq==0})
 (run/'independent-verification.json').write_text(json.dumps({'complete':True,'episodes':results,'scope':'Independent RGB recomputation and fresh saved-state free-body/contact audit, no controller tuning'},indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
