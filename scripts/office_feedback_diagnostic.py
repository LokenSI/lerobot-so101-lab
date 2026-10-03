"""Compare retained on-expert predictions with prior closed-loop commands, without causal claims."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing diagnostic')
    teacher=json.loads((a.teacher/'teacher-forced-report.json').read_text());meta=json.loads((a.rollout/'episode.json').read_text())
    if meta['seed']!=1000 or meta['task']['id']!='red-left' or meta['wording']!='train':raise ValueError('Require exact familiar train-scene/instruction control')
    with np.load(a.original) as source,np.load(a.rollout/'trajectory.npz') as trajectory:
        rows=[]
        for tick in [0,8,16,24,32,48]:
            anchor=next(row for row in teacher['rows'] if row['frame']==tick)
            rows.append({'tick':tick,'closed_loop_state_minus_expert_rad':(trajectory['states'][tick]-source['states'][tick]).tolist(),
               'closed_loop_issued_command_rad':trajectory['actions'][tick].tolist(),
               'teacher_forced_first_command_rad':anchor['first_command_rad'],'expert_command_rad':source['actions'][tick].tolist(),
               'teacher_forced_first8_mae_rad':anchor['first8_mae_rad']})
    result={'scope':'retrospective familiar training-scene feedback diagnostic; no new rollout or causal proof',
       'teacher_report_sha256':hashlib.sha256((a.teacher/'teacher-forced-report.json').read_bytes()).hexdigest(),
       'rollout_trajectory_sha256':hashlib.sha256((a.rollout/'trajectory.npz').read_bytes()).hexdigest(),'rows':rows,
       'limits':'Prior rollout and fresh teacher probes differ in process/runtime and stochastic draw sequence after omitted anchor frames. Values support a feedback-sensitivity hypothesis but cannot isolate camera, state, noise or phase causes.',
       'recommended_next_step':'Use frozen same-A100 rollouts and retained actual fresh inputs; if failures persist, collect corrective off-expert demonstrations or test a declared phase/temporal-continuity training change rather than blind repetition.'}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
