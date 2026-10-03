"""Overlay recorded policy clips with measured TCP and kinematic issued-chunk waypoints."""
import argparse,hashlib,json,re
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
import office_environment as office
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cumulative-step',type=int,required=True);p.add_argument('--chunk-execution',type=int,default=8);a=p.parse_args()
    meta=json.loads((a.episode/'episode.json').read_text());source=a.episode/'trajectory.npz'
    with np.load(source) as src:qpos=src['qpos_after'];states=src['states'];actions=src['actions'];latencies=src['inference_seconds']
    full_predictions=None;prediction_ticks=None
    if (a.episode/'predictions.npz').exists():
        with np.load(a.episode/'predictions.npz') as src:full_predictions=src['chunks'];prediction_ticks=src['ticks']
    env=office.make_env(meta['seed'],offscreen=False,scene=meta['scene']);kin=mujoco.MjData(env.model);camera=env.model.camera('overview').id
    body=env.model.body('gripper').id;lo,hi=env.model.actuator_ctrlrange.T;events=((actions<lo-1e-6)|(actions>hi+1e-6)).any(1).cumsum()
    trail=[];trace=[];reader=imageio.get_reader(str(a.episode/'rollout.mp4'));a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        def tcp(data):return data.xpos[body]+data.xmat[body].reshape(3,3)@office.baseline.TCP_LOCAL
        def pixel(point):
            local=(point-env.data.cam_xpos[camera])@env.data.cam_xmat[camera].reshape(3,3);depth=-local[2]
            if depth<=0:return None
            f=128/np.tan(np.deg2rad(env.model.cam_fovy[camera])/2)
            return (int((128+f*local[0]/depth)*2),int((128-f*local[1]/depth-44)*2+96))
        with imageio.get_writer(str(a.output),fps=30,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i in range(len(qpos)):
                env.data.qpos[:]=qpos[i];mujoco.mj_forward(env.model,env.data);measured=tcp(env.data).copy();trail.append(measured)
                chunk_start=(i//a.chunk_execution)*a.chunk_execution;chunk=actions[chunk_start:chunk_start+a.chunk_execution]
                if full_predictions is not None:
                    prediction_index=int(np.searchsorted(prediction_ticks,i,side='right')-1)
                    chunk_start=int(prediction_ticks[prediction_index]);chunk=full_predictions[prediction_index]
                waypoints=[];waypoint_indices=np.linspace(0,len(chunk)-1,min(8,len(chunk))).astype(int)
                for command in chunk[waypoint_indices]:
                    kin.qpos[:]=qpos[i];kin.qpos[env.qadr]=command;mujoco.mj_forward(env.model,kin);waypoints.append(tcp(kin).copy())
                frame=Image.new('RGB',(512,584));frame.paste(Image.fromarray(reader.get_data(i)).crop((0,44,256,256)).resize((512,424)),(0,96));draw=ImageDraw.Draw(frame)
                try:draw.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
                except OSError:pass
                points=[p for point in trail[-180:] if (p:=pixel(point)) is not None]
                future=[p for point in waypoints if (p:=pixel(point)) is not None]
                if len(points)>1:draw.line(points,fill='cyan',width=3)
                if len(future)>1:draw.line(future,fill='yellow',width=2)
                for x,y in future:draw.ellipse((x-3,y-3,x+3,y+3),outline='yellow',width=2)
                physical=meta.get('task_success',meta['grader']['success']);status='TASK PASS' if physical else 'TASK FAIL'
                draw.text((6,5),f"{meta['policy'].get('model','Policy')} checkpoint {a.cumulative_step} | {status}",fill='lime' if physical else 'orange')
                draw.text((6,22),meta['task']['instruction'],fill='white')
                latency=float(latencies[chunk_start]);draw.text((6,39),f'Fresh chunk#{chunk_start//a.chunk_execution}: {latency*1000:.0f}ms | execute {a.chunk_execution}ticks',fill='white')
                draw.text((6,56),f'Decoded command-limit events: {events[i]} / {i+1}',fill='orange')
                condition='DEVELOPMENT OVERFIT CONTROL' if meta.get('development_overfit_control') else 'held-out development scene'
                draw.text((6,73),f"Seed{meta['seed']} | {condition} | wording {meta.get('wording','validation')}",fill='yellow')
                draw.text((6,527),f'Cyan: measured TCP. Yellow: 8 FK samples/{len(chunk)} predicted targets.',fill='white')
                draw.text((6,546),'FK visualization; no oracle or command correction.',fill='white')
                draw.text((6,565),'MuJoCo evaluation; clock pauses for inference.',fill='yellow');writer.append_data(np.asarray(frame))
                trace.append({'frame':i,'measured_tcp_m':measured.tolist(),'chunk_start':chunk_start,'issued_waypoint_tcp_m':[x.tolist() for x in waypoints],
                              'selected_command_indices_within_chunk':waypoint_indices.tolist(),
                              'fresh_inference_seconds':latency,'cumulative_action_limit_events':int(events[i])})
    finally:reader.close();env.close()
    a.output.with_suffix('.overlay-data.json').write_text(json.dumps(trace))
    a.output.with_suffix('.provenance.json').write_text(json.dumps({'scope':'CPU overlay of original learned MuJoCo rollout; no policy rerun or altered physical commands',
        'cumulative_step':a.cumulative_step,'success':meta['success'],'task_success':meta.get('task_success'),'strict_zero_raw_limit_events_success':meta.get('strict_zero_raw_limit_events_success'),
        'episode':str(a.episode),'checkpoint':meta['policy'],'chunk_execution':a.chunk_execution,
        'waypoints':'eight evenly spaced FK samples from saved full fresh native prediction chunks' if full_predictions is not None else 'eight evenly spaced FK samples from actually issued policy-decoded chunk; full future predictions not retained',
        'native_decoder_caveat':'GR00T MIN_MAX native normalized clipping differs from unbounded ACT/Smol MEAN_STD; events describe decoded commands before common actuator supervisor',
        'source_full_predictions_sha256':hashlib.sha256((a.episode/'predictions.npz').read_bytes()).hexdigest() if full_predictions is not None else None,
        'video_crop':'top44 pixels containing original video header removed; original video unchanged',
        'source_trajectory_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_video_sha256':hashlib.sha256((a.episode/'rollout.mp4').read_bytes()).hexdigest(),
        'output_video_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
