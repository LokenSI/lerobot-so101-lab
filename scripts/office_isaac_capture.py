"""Bounded native RTX capture of baked MuJoCo replay; requires coordinated GPU launch."""
import argparse,hashlib,json,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=60);p.add_argument('--pose-stride',type=int,default=8);p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=360);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':a.width,'height':a.height})
    result={'scope':'native RTX visual replay of measured MuJoCo poses; no Isaac physics or new policy result','stage':str(a.stage),
            'stage_sha256':hashlib.sha256(a.stage.read_bytes()).hexdigest(),'rendered_frames':0,'complete':False,'pose_stride':a.pose_stride,'original_fps':30,'pose_frame_indices':[]}
    try:
        import carb,numpy as np,omni.usd,omni.timeline,omni.replicator.core as rep
        from pxr import UsdGeom,UsdLux,Gf
        carb.settings.get_settings().set_bool('/app/player/playSimulations',False)
        ctx=omni.usd.get_context();ctx.open_stage(str(a.stage.resolve()))
        for _ in range(20):app.update()
        stage=ctx.get_stage()
        result['source_stage_end_frame']=stage.GetEndTimeCode()
        from pxr import UsdPhysics
        if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):raise RuntimeError('Visual replay stage unexpectedly contains physics schemas')
        if not any(prim.IsA(UsdLux.DomeLight) for prim in stage.Traverse()):UsdLux.DomeLight.Define(stage,'/ReplayLight').GetIntensityAttr().Set(600)
        bounds=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        center=bounds.GetMidpoint();size=bounds.GetSize();span=max(float(size[0]),float(size[1]),1.0)
        camera=UsdGeom.Camera.Define(stage,'/ReplayCamera');center=Gf.Vec3d(center[0],center[1],.22)
        eye=center+Gf.Vec3d(span*.55,-span*.85,span*.75)
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye,center,Gf.Vec3d(0,0,1)).GetInverse())
        camera.GetFocalLengthAttr().Set(24);camera.GetHorizontalApertureAttr().Set(36)
        product=rep.create.render_product(str(camera.GetPath()),(a.width,a.height))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        rep.orchestrator.set_capture_on_play(False);timeline=omni.timeline.get_timeline_interface();timeline.pause()
        for i in range(a.frames):
            pose_frame=min(a.start_frame+i*a.pose_stride,int(stage.GetEndTimeCode()))
            timeline.set_current_time(pose_frame/30);timeline.commit()
            rep.orchestrator.step(rt_subframes=2,delta_time=0.0,pause_timeline=True,wait_for_render=True)
            image=np.asarray(rgb.get_data())
            if image.shape[:2]!=(a.height,a.width) or image.ndim!=3 or image.shape[2]<3:raise RuntimeError(f'Invalid native RGB shape {image.shape}')
            pixels=np.ascontiguousarray(image[:,:,:3],dtype=np.uint8)
            (a.output/f'frame-{i:04d}.ppm').write_bytes(f'P6\n{a.width} {a.height}\n255\n'.encode()+pixels.tobytes())
            result['rendered_frames']+=1
            result['pose_frame_indices'].append(pose_frame)
        result['complete']=True;result['schema_assertion']='no rigid-body or physics scene schemas'
        result['covers_final_source_pose']=result['pose_frame_indices'][-1]==int(stage.GetEndTimeCode())
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:
        (a.output/'native-render-report.json').write_text(json.dumps(result,indent=2));app.close()
if __name__=='__main__':main()
