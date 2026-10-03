"""Multitask SO-101 MuJoCo environment; privileged truth is reserved for expert/grader."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runtime/action-training-phase/office'
BASELINE = ROOT / 'scripts/so101_pick_place_baseline.py'
spec = importlib.util.spec_from_file_location('office_legacy_baseline', BASELINE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
sim = baseline.sim
COLORS = ('red', 'blue', 'yellow')
TRAYS = {'left': (.235, .15), 'right': (.235, -.16)}

def scene_spec(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    slots = np.array([[.20, -.02], [.25, .045], [.145, .045]])
    slots = slots[rng.permutation(3)] + rng.uniform(-.008, .008, (3, 2))
    return {'seed': seed, 'objects': [{'color': c, 'xy': slots[i].tolist(), 'yaw': float(rng.uniform(.15,.45))} for i,c in enumerate(COLORS)],
            'trays': {k:list(v) for k,v in TRAYS.items()}, 'side_reference':'robot base coordinates: left=positive Y, right=negative Y'}

def instructions(seed: int, split: str) -> list[dict]:
    rows=[]
    for color in COLORS:
        for tray in TRAYS:
            text = f'Pick up the {color} cube and put it in the {tray} tray.' if split == 'train' else f'Move the {color} block into the {tray} tray.'
            rows.append({'id':f'{color}-{tray}', 'instruction':text, 'goals':[[color,tray]], 'kind':'object-destination'})
    scene=scene_spec(seed)
    for relation,choose in [('leftmost',max),('rightmost',min)]:
        obj=choose(scene['objects'],key=lambda o:o['xy'][1])
        rows.append({'id':f'relation-{relation}', 'instruction':f'Put the {relation} cube in the right tray.',
                     'goals':[[obj['color'],'right']],'kind':'spatial-relation'})
    rows.extend([
        {'id':'exclude-blue','instruction':'Leave the blue and yellow cubes alone. Put only the red cube in the left tray.', 'goals':[['red','left']], 'kind':'exclusion'},
        {'id':'exclude-red','instruction':'Do not move the red or yellow cubes. Transfer the blue cube to the right tray.', 'goals':[['blue','right']], 'kind':'exclusion'},
        {'id':'sequence-red-blue','instruction':'First put the red cube in the right tray, then put the blue cube in the left tray.', 'goals':[['red','right'],['blue','left']], 'kind':'sequence'},
        {'id':'sequence-blue-red','instruction':'First put the blue cube in the left tray, then put the red cube in the right tray.', 'goals':[['blue','left'],['red','right']], 'kind':'sequence'},
    ])
    return rows

class OfficeSim(sim.Sim):
    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        q=sim.deg_to_q(sim.REST_DEG)
        self.data.qpos[self.qadr]=q; self.data.ctrl[:]=q
        mujoco.mj_forward(self.model,self.data)
        for _ in range(120): mujoco.mj_step(self.model,self.data)
        self.initial_positions=self.object_positions()
        self.grading_trace=[]
    def step(self, command_deg):
        super().step(command_deg)
        row={'time':float(self.data.time),'objects':{}}
        for i,color in enumerate(COLORS):
            body=self.data.body(f'obj{i}')
            adr=int(self.model.joint(f'obj{i}_free').dofadr[0])
            touching=False
            for k in range(self.data.ncon):
                c=self.data.contact[k]
                bodies=[int(self.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
                if body.id in bodies:
                    other=bodies[1] if bodies[0]==body.id else bodies[0]
                    touching |= self.model.body(other).name in ('gripper','moving_jaw_so101_v1')
            row['objects'][color]={'xyz':body.xpos.copy().tolist(),'quat':body.xquat.copy().tolist(),
                'extent':(np.abs(body.xmat.reshape(3,3))@np.full(3,.015)).tolist(),
                'speed':float(np.linalg.norm(self.data.qvel[adr:adr+3])), 'released':not touching}
        self.grading_trace.append(row)

def make_env(seed: int, width=256, height=256, offscreen=True, scene=None):
    scene=scene or scene_spec(seed)
    original=baseline.make_env(seed=0,width=width,height=height,offscreen=False)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.xml',dir=OUTPUT,delete=False) as f: path=Path(f.name)
    try:
        mujoco.mj_saveLastXML(str(path),original.model)
        xml=ET.parse(path).getroot()
    finally:
        path.unlink(missing_ok=True); original.close()
    world=xml.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name') in ('obj0','tray'):world.remove(body)
    for i,obj in enumerate(scene['objects']):
        world.append(ET.fromstring(sim._object_xml(i,obj['color'],'cube',obj['xy'],obj['yaw'])))
    for side,xy in scene['trays'].items():
        tray=ET.fromstring(sim._tray_xml(xy));tray.set('name',f'tray_{side}')
        for g in tray.findall('geom'):g.set('rgba','0.18 0.6 0.35 1' if side=='left' else '0.6 0.25 0.65 1')
        world.append(tray)
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    env=object.__new__(OfficeSim);env.scene='Office multitask';env.scene_spec=scene
    env.width=width;env.height=height;env.seed=seed;env.offscreen=offscreen
    env.model=model;env.data=mujoco.MjData(model)
    env.qadr=np.array([model.joint(j).qposadr[0] for j in sim.JOINTS])
    env.renderer=mujoco.Renderer(model,height,width) if offscreen else None
    env.reset();return env

def grade(env, goals, settling_frames=45):
    trace=env.grading_trace;results=[];completion_ticks=[]
    for color,side in goals:
        history=[r['objects'][color] for r in trace]
        if len(history)<settling_frames:return {'success':False,'reason':'insufficient settling trace'}
        last=history[-settling_frames:];p=np.array([r['xyz'] for r in last]);ext=np.array([r['extent'] for r in last])
        xy=np.abs(p[:,:2]-env.scene_spec['trays'][side])+ext[:,:2]
        inside=bool(np.all(xy[:,0]<.070) and np.all(xy[:,1]<.055) and np.all((p[:,2]-ext[:,2]>.003)&(p[:,2]-ext[:,2]<.006)))
        released=all(r['released'] for r in last)
        quat=np.array([r['quat'] for r in last]);angle=float(np.max(2*np.arccos(np.clip(np.abs(quat@quat[0]),-1,1))))
        stable=bool(np.linalg.norm(np.ptp(p,axis=0))<.001 and angle<np.deg2rad(1) and max(r['speed'] for r in last)<.002)
        lifted=max(r['xyz'][2] for r in history)>.09
        completion=None
        for end in range(settling_frames,len(history)+1):
            segment=history[end-settling_frames:end]
            sp=np.array([r['xyz'] for r in segment]);se=np.array([r['extent'] for r in segment])
            sx=np.abs(sp[:,:2]-env.scene_spec['trays'][side])+se[:,:2]
            sq=np.array([r['quat'] for r in segment])
            sa=float(np.max(2*np.arccos(np.clip(np.abs(sq@sq[0]),-1,1))))
            if (np.all(sx[:,0]<.070) and np.all(sx[:,1]<.055) and
                np.all((sp[:,2]-se[:,2]>.003)&(sp[:,2]-se[:,2]<.006)) and
                np.linalg.norm(np.ptp(sp,axis=0))<.001 and sa<np.deg2rad(1) and
                all(r['released'] for r in segment) and max(r['speed'] for r in segment)<.002 and
                max(r['xyz'][2] for r in history[:end])>.09):
                completion=end-1;break
        completion_ticks.append(completion)
        results.append({'object':color,'destination':side,'inside':inside,'released':released,'stable':stable,'lifted':lifted,'success':inside and released and stable and lifted})
    target_colors={g[0] for g in goals}
    distractors={color:max(float(np.linalg.norm(np.array(r['objects'][color]['xyz'])-env.initial_positions[f'obj{i}'])) for r in trace) for i,color in enumerate(COLORS) if color not in target_colors}
    untouched=all(v<.02 for v in distractors.values())
    ordered=all(t is not None for t in completion_ticks) and all(a<b for a,b in zip(completion_ticks,completion_ticks[1:]))
    return {'success':all(r['success'] for r in results) and untouched and ordered,'goals':results,'completion_ticks':completion_ticks,'sequence_order_correct':ordered,'distractor_max_displacement_m':distractors,'distractors_untouched':untouched,'grading_frames':settling_frames}

def expert_actions(env, goals):
    ctrl=baseline.Controller(env)
    opening=sim.deg_to_q([0,0,0,0,0,45])[-1];closing=sim.GRIPPER_CLOSED
    for color,side in goals:
        i=COLORS.index(color);xy=env.object_positions()[f'obj{i}'][:2].copy()
        yaw=env.scene_spec['objects'][i]['yaw'];R=baseline.rotz(yaw);tray=np.array(env.scene_spec['trays'][side])
        requests=[('lift clear',[.22,0,.10],baseline.tilted([.22,0]),opening,2.),
            ('orient',np.r_[xy,.06],R,opening,2.),('approach',np.r_[xy,.017],R,opening,2.),
            ('close',np.r_[xy,.017],R,closing,1.5),('lift',np.r_[xy,.12],baseline.tilted(xy),closing,2.),
            ('transfer',np.r_[tray,.12],baseline.tilted(tray),closing,2.5),
            ('lower',np.r_[tray,.06],baseline.tilted(tray),closing,1.5),('release',np.r_[tray,.06],baseline.tilted(tray),opening,1.),
            ('retreat',np.r_[tray,.15],baseline.tilted(tray),opening,1.5),('settle',np.r_[tray,.15],baseline.tilted(tray),opening,1.5)]
        for phase,target,Rtarget,grip,seconds in requests:
            target=np.array(target);q,err,rerr=ctrl.solve(target,Rtarget,grip)
            start=env.data.ctrl.copy();pstart,Rstart=ctrl.pose(start)
            for tick in range(round(seconds*sim.FPS)):
                u=(tick+1)/round(seconds*sim.FPS);blend=u*u*(3-2*u)
                action=start+(q-start)*blend
                if phase in ('approach','lift','transfer','lower','retreat'):
                    Rt=Rstart+(Rtarget-Rstart)*blend;U,_,Vt=np.linalg.svd(Rt);Rt=U@Vt
                    action,err,rerr=ctrl.solve(pstart+(target-pstart)*blend,Rt,start[-1]+(grip-start[-1])*blend)
                yield action.copy(),phase

def source_hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASELINE,Path(sim.__file__),baseline.SOURCE/'so101.xml')}
