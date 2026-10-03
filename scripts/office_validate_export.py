"""Validate exported office data through the real local LeRobot reader (CPU only)."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info=json.loads((a.dataset/'meta/info.json').read_text());contract=json.loads((a.dataset/'meta/office-contract.json').read_text())
    assert contract['fps']==30 and contract['state_action_units']=='radians_all_six'
    ds=LeRobotDataset('local/office-reader-check',root=a.dataset,video_backend='pyav',download_videos=False)
    samples=[]
    for i in sorted(set([0,len(ds)//2,len(ds)-1])):
        row=ds[i];s={'index':i,'task':row['task'],'keys':sorted(row)}
        for key in ['observation.state','action']:
            assert tuple(row[key].shape)==(6,) and np.isfinite(row[key].numpy()).all();s[key]=list(row[key].shape)
        for key in ['observation.images.scene','observation.images.wrist']:
            expected=(3,*info['features'][key]['shape'][:2]);assert tuple(row[key].shape)==expected
            assert 0<=row[key].min() and row[key].max()<=1;s[key]=list(row[key].shape)
        samples.append(s)
    result={'reader':'official LeRobotDataset, PyAV CPU video decoder','frames':len(ds),'episodes':info['total_episodes'],
            'tasks':info['total_tasks'],'samples':samples,'passed':True}
    (a.dataset/'reader-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
