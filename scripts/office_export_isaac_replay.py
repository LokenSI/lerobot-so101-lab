"""Bake measured MuJoCo poses into USD for Isaac inspection; this is not Isaac physics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import office_environment as office

def triple(v):return '('+', '.join(f'{float(x):.8g}' for x in v)+')'
def matrix(rotation,position):
    m=np.eye(4);m[:3,:3]=rotation.T;m[3,:3]=position
    return '('+', '.join(triple(row) for row in m)+')'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--presentation-legibility',action='store_true');a=p.parse_args();meta=json.loads((a.episode/'episode.json').read_text())
    env=office.make_env(meta['seed'],offscreen=False,scene=meta['scene']);poses=np.load(a.episode/('trajectory.npz' if (a.episode/'trajectory.npz').exists() else 'episode.npz'))
    qpos=poses['qpos_after'];qvel=poses['qvel_after'];indices=list(range(0,len(qpos),a.stride))
    if indices[-1]!=len(qpos)-1:indices.append(len(qpos)-1)
    transforms={};geometries={}
    try:
        for gid in range(env.model.ngeom):
            if env.model.geom_group[gid]==3:continue
            # Visual-only duplicate room walls; no physics is present in this replay.
            if a.presentation_legibility and gid in [2,3]:continue
            kind=int(env.model.geom_type[gid]);mid=int(env.model.geom_dataid[gid])
            if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                va=int(env.model.mesh_vertadr[mid]);vn=int(env.model.mesh_vertnum[mid]);fa=int(env.model.mesh_faceadr[mid]);fn=int(env.model.mesh_facenum[mid])
                points=env.model.mesh_vert[va:va+vn];faces=env.model.mesh_face[fa:fa+fn].ravel();counts=[3]*fn
            elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):
                points=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*env.model.geom_size[gid]
                faces=[0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7];counts=[4]*6
            else:continue
            geometries[gid]=(points,faces,counts);transforms[gid]=[]
        for i in indices:
            env.data.qpos[:]=qpos[i];env.data.qvel[:]=qvel[i];mujoco.mj_forward(env.model,env.data)
            for gid in geometries:transforms[gid].append((i,matrix(env.data.geom_xmat[gid].reshape(3,3),env.data.geom_xpos[gid])))
        lines=['#usda 1.0','(', '    defaultPrim = "OfficeReplay"','    upAxis = "Z"','    metersPerUnit = 1',f'    timeCodesPerSecond = 30',f'    endTimeCode = {len(qpos)-1}',')',
               'def Xform "OfficeReplay" (','    customData = {','        string evidenceType = "MuJoCo pose replay; Isaac physics disabled; no domain transfer evaluated"',
               '        string instruction = '+json.dumps(meta['task']['instruction']),'    }',')','{']
        for gid,(points,faces,counts) in geometries.items():
            color=env.model.geom_rgba[gid,:3]
            if a.presentation_legibility:
                material=int(env.model.geom_matid[gid])
                if material>=0:color=env.model.mat_rgba[material,:3]
                if int(env.model.geom_group[gid])==2:
                    color=np.asarray([.90,.91,.94]) if gid%3 else np.asarray([.97,.53,.12])
            lines.extend([f'    def Mesh "geom_{gid}"','    {','        uniform token subdivisionScheme = "none"',
                '        int[] faceVertexCounts = ['+', '.join(map(str,counts))+']','        int[] faceVertexIndices = ['+', '.join(map(str,faces))+']',
                '        point3f[] points = ['+', '.join(triple(v) for v in points)+']',
                '        color3f[] primvars:displayColor = ['+triple(color)+']',
                '        uniform bool doubleSided = '+('true' if a.presentation_legibility else 'false'),
                '        matrix4d xformOp:transform.timeSamples = {'])
            lines.extend(f'            {i}: {m},' for i,m in transforms[gid])
            lines.extend(['        }','        uniform token[] xformOpOrder = ["xformOp:transform"]','    }'])
        lines.append('}');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines))
        provenance={'scope':'baked measured MuJoCo trajectory displayed in Isaac; no Isaac physics evaluation','episode':str(a.episode),
            'instruction':meta['task']['instruction'],'original_success':meta['success'],'controller':meta.get('controller','checkpoint policy'),
            'trajectory_sha256':hashlib.sha256((a.episode/('trajectory.npz' if (a.episode/'trajectory.npz').exists() else 'episode.npz')).read_bytes()).hexdigest(),
            'usd_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'geometries':len(geometries),'frames':len(qpos),'sample_stride':a.stride,
            'renderer_difference':'USD simple display colors; original MuJoCo materials/textures/lighting are not reproduced','learned_policy':meta.get('learned_policy',meta.get('policy',{}).get('learned_policy',False)),
            'presentation_legibility':a.presentation_legibility,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'visual_only_modifications':['hide duplicated room-wall geoms2,3','resolve source material color','light white/orange visual robot mesh palette','double-sided meshes'] if a.presentation_legibility else [],
            'poses_modified':False,'physics_executed':False}
        a.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance))
    finally:poses.close();env.close()
if __name__=='__main__':main()
