"""Export accepted RGB expert episodes as separate LeRobot v3 or GR00T v2.1 splits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']

def stats(values):
    return {k:np.asarray(v).tolist() for k,v in {'mean':values.mean(0),'std':values.std(0),'min':values.min(0),'max':values.max(0),'q01':np.quantile(values,.01,axis=0),'q99':np.quantile(values,.99,axis=0),'count':[len(values)]}.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--format',choices=['v3','v2.1'],default='v3');p.add_argument('--split',choices=['train','validation'],required=True)
    p.add_argument('--tasks',nargs='+',help='Explicit task IDs to export as a separate scoped dataset')
    a=p.parse_args();manifest=json.loads((a.dataset/'manifest.json').read_text())
    rows=[r for r in manifest['episodes'] if r['success'] and r['rendered'] and r['split']==a.split]
    if a.tasks:rows=[r for r in rows if r['task']['id'] in a.tasks]
    if not rows:raise ValueError('No successful rendered episodes available for selected split')
    if a.output.exists():raise FileExistsError('Export refuses existing output; use a fresh immutable export directory')
    first=np.load(a.dataset/rows[0]['path']/'episode.npz');size=first['images_scene'].shape[1];first.close()
    features={k:{'dtype':'float32','shape':(6,),'names':JOINTS} for k in ['observation.state','action']}
    features.update({f'observation.images.{c}':{'dtype':'video','shape':(size,size,3),'names':['height','width','channels']} for c in ['scene','wrist']})
    if a.format=='v3':
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.configs.video import RGBEncoderConfig
        ds=LeRobotDataset.create(repo_id=f'local/office-so101-{a.split}',root=a.output,fps=30,robot_type='so101_mujoco_radians',features=features,use_videos=True,
                                streaming_encoding=True,rgb_encoder=RGBEncoderConfig(vcodec='h264',pix_fmt='yuv420p',crf=18,preset='fast'),encoder_threads=2)
        for row in rows:
            with np.load(a.dataset/row['path']/'episode.npz') as ep:
                states=ep['states'].astype(np.float32);actions=ep['actions'].astype(np.float32)
                images={c:ep[f'images_{c}'] for c in ['scene','wrist']}
                for t in range(len(states)):
                    ds.add_frame({'observation.state':states[t],'action':actions[t],
                                  **{f'observation.images.{c}':images[c][t] for c in ['scene','wrist']},'task':row['task']['instruction']})
                ds.save_episode()
                print(json.dumps({'exported_episode':row['path'],'frames':len(states)}),flush=True)
        ds.finalize()
    else:
        import imageio.v2 as imageio
        import pyarrow as pa
        import pyarrow.parquet as pq
        (a.output/'meta').mkdir(parents=True);(a.output/'data/chunk-000').mkdir(parents=True)
        tasks=list(dict.fromkeys(r['task']['instruction'] for r in rows));episodes=[];episode_stats=[];all_states=[];all_actions=[];index=0
        sampled_pixels={c:[] for c in ['scene','wrist']}
        for i,row in enumerate(rows):
            with np.load(a.dataset/row['path']/'episode.npz') as ep:
                n=len(ep['states']);ti=tasks.index(row['task']['instruction'])
                table=pa.table({'observation.state':pa.array(ep['states'].astype(np.float32).tolist(),type=pa.list_(pa.float32(),6)),
                    'action':pa.array(ep['actions'].astype(np.float32).tolist(),type=pa.list_(pa.float32(),6)),
                    'timestamp':pa.array((np.arange(n)/30).astype(np.float32)), 'frame_index':np.arange(n,dtype=np.int64),
                    'episode_index':np.full(n,i,dtype=np.int64),'index':np.arange(index,index+n,dtype=np.int64),'task_index':np.full(n,ti,dtype=np.int64)})
                pq.write_table(table,a.output/f'data/chunk-000/episode_{i:06d}.parquet')
                for c in ['scene','wrist']:
                    vp=a.output/f'videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4';vp.parent.mkdir(parents=True,exist_ok=True)
                    images=ep[f'images_{c}']
                    sampled_pixels[c].append(images[np.linspace(0,n-1,min(n,32)).astype(int),::8,::8].reshape(-1,3).astype(np.float32)/255)
                    with imageio.get_writer(str(vp),fps=30,codec='libx264',quality=9,macro_block_size=1) as writer:
                        for frame in images:writer.append_data(frame)
                all_states.append(ep['states'].copy());all_actions.append(ep['actions'].copy())
                episode_stats.append({'episode_index':i,'stats':{'observation.state':stats(ep['states']),'action':stats(ep['actions'])}})
                episodes.append({'episode_index':i,'tasks':[tasks[ti]],'length':n});index+=n
        meta=a.output/'meta'
        for name,content in [('tasks.jsonl',[{'task_index':i,'task':t} for i,t in enumerate(tasks)]),('episodes.jsonl',episodes),('episodes_stats.jsonl',episode_stats)]:
            (meta/name).write_text(''.join(json.dumps(r)+'\n' for r in content))
        computed={'observation.state':stats(np.concatenate(all_states)),'action':stats(np.concatenate(all_actions))}
        for c in ['scene','wrist']:
            key=f'observation.images.{c}';features[key]['info']={'video.fps':30,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}
            pixels=np.concatenate(sampled_pixels[c]);image_stats=stats(pixels)
            computed[key]={k:np.asarray(v).reshape(3,1,1).tolist() if k!='count' else v for k,v in image_stats.items()}
        features.update({k:{'dtype':dtype,'shape':[1],'names':None} for k,dtype in [('timestamp','float32'),('frame_index','int64'),('episode_index','int64'),('index','int64'),('task_index','int64')]})
        info={'codebase_version':'v2.1','robot_type':'so101_mujoco_radians','total_episodes':len(rows),'total_frames':index,'total_tasks':len(tasks),
              'total_videos':len(rows)*2,'total_chunks':1,'chunks_size':1000,'fps':30,'splits':{'train':f'0:{len(rows)}'},
              'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
              'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
        (meta/'info.json').write_text(json.dumps(info,indent=2));(meta/'stats.json').write_text(json.dumps(computed,indent=2))
        modality={'state':{'single_arm':{'start':0,'end':5},'gripper':{'start':5,'end':6}},'action':{'single_arm':{'start':0,'end':5},'gripper':{'start':5,'end':6}},
                  'video':{c:{'original_key':f'observation.images.{c}'} for c in ['scene','wrist']},'annotation':{'human.task_description':{'original_key':'task_index'}}}
        (meta/'modality.json').write_text(json.dumps(modality,indent=2))
    contract={'format':a.format,'split':a.split,'fps':30,'joint_order':JOINTS,'state_action_units':'radians_all_six',
              'raw_actions':'absolute actuator targets; relative conversion only in policy processor','camera_order':['scene','wrist'],
              'expert_uses_privileged_pose':True,'source_manifest_sha256':hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),
              'source_episodes':rows,'normalization_scope':a.split,'not_a_model_success':True}
    contract['explicit_task_filter']=a.tasks
    if a.format=='v2.1':contract['image_statistics']='measured:32 evenly spaced frames per episode, spatial stride8; count is sampled pixels'
    (a.output/'office-contract.json').write_text(json.dumps(contract,indent=2))
    (a.output/'meta/office-contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps({'format':a.format,'split':a.split,'episodes':len(rows),'output':str(a.output)}))
if __name__=='__main__':main()
