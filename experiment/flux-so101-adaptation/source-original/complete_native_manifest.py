"""Merge all20 separately verified native replay reports without losing19-run summary."""
import json,hashlib
from pathlib import Path
root=Path(__file__).parent/'native320x240-v1'
rows=[]
for seed in range(20):
    folder=root/f'seed-{seed:03d}'
    row=json.loads((folder/'report.json').read_text())
    assert row['seed']==seed and row['success'] and row['replay_max_state_error_rad']==0.
    assert hashlib.sha256((folder/'episode.npz').read_bytes()).hexdigest()==row['native_episode_sha256']
    row['split']='val' if seed in (0,7,14) else 'train'
    rows.append(row)
manifest={'scope':'20 genuine-contact expert replays for FLUX adaptation, not FLUX policy successes',
    'successful':20,'total':20,'train_episodes':17,'validation_episodes':3,
    'camera_projection':'Native320x240 scene/wrist, PIL bilinear resize256x256perview',
    'command_alignment':'Measuredstate and images before corresponding unchanged original expert action',
    'runs':rows}
(root/'combined-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Verified all20 native expert episodes:17train/3validation,strict20/20,state replayerror0')
