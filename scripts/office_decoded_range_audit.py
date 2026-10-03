"""Count decoded command range violations by joint without changing any outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import office_environment as office

def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);a=p.parse_args();env=office.make_env(10000,offscreen=False)
    limits=env.model.actuator_ctrlrange.copy();env.close();rows=[]
    for name in json.loads((a.comparison/'frozen-contract.json').read_text())['order']:
        for episode in sorted((a.comparison/f'rollout-{name}').glob('seed-*')):
            report=json.loads((episode/'episode.json').read_text())
            with np.load(episode/'trajectory.npz') as data:
                commands=data['actions'];violations=(commands<limits[:,0]-1e-6)|(commands>limits[:,1]+1e-6)
                row={'model':name,'task':report['task']['id'],'steps':len(commands),'task_success':report['task_success'],
                  'decoded_min_radians':commands.min(0).tolist(),'decoded_max_radians':commands.max(0).tolist(),
                  'per_joint_limit_events':violations.sum(0).tolist(),'any_joint_limit_events':int(violations.any(1).sum()),
                  'max_lower_excess_radians':np.maximum(limits[:,0]-commands,0).max(0).tolist(),
                  'max_upper_excess_radians':np.maximum(commands-limits[:,1],0).max(0).tolist(),
                  'trajectory_sha256':hashlib.sha256((episode/'trajectory.npz').read_bytes()).hexdigest()}
                assert row['any_joint_limit_events']==report['command_limit_events'];rows.append(row)
    output={'scope':'policy-decoded commands before existing common actuator supervisor; native decoder bounds differ, not neural-head safety',
      'joint_order':office.sim.JOINTS,'authored_actuator_limits_radians':limits.tolist(),'rows':rows,'outcomes_modified':False}
    target=a.comparison/'decoded-range-audit.json'
    if target.exists():raise FileExistsError('Preserve range audits')
    target.write_text(json.dumps(output,indent=2));print(json.dumps({'rows':[{k:r[k] for k in ['model','task','per_joint_limit_events','any_joint_limit_events']} for r in rows]}))

if __name__=='__main__':main()
