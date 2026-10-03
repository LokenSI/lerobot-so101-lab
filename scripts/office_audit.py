"""Independently replay issued expert commands with CPU physics; no saved pose writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import office_environment as office

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.dataset/'manifest.json').read_text());rows=[]
    for item in manifest['episodes']:
        folder=a.dataset/item['path'];meta=json.loads((folder/'episode.json').read_text())
        env=office.make_env(meta['seed'],offscreen=False,scene=meta['scene']);qerr=verr=0.
        try:
            with np.load(folder/'episode.npz') as src:
                for command,qpos,qvel in zip(src['actions'],src['qpos_after'],src['qvel_after'],strict=True):
                    env.step(office.sim.q_to_deg(command));qerr=max(qerr,float(np.max(np.abs(env.data.qpos-qpos))))
                    verr=max(verr,float(np.max(np.abs(env.data.qvel-qvel))))
            graded=office.grade(env,meta['task']['goals'])
            rows.append({'path':item['path'],'dynamics_match':qerr<1e-7 and verr<1e-7,'max_qpos_error':qerr,'max_qvel_error':verr,
                         'original_success':meta['success'],'replayed_success':graded['success'],'grade_agrees':graded['success']==meta['success']})
        finally:env.close()
    report={'scope':'CPU command replay, no post-reset object or robot pose writes','episodes':rows,'dynamics_pass':all(r['dynamics_match'] for r in rows),
            'grade_agreement':all(r['grade_agrees'] for r in rows),'expert_successes':sum(r['replayed_success'] for r in rows),
            'expert_failures':sum(not r['replayed_success'] for r in rows),'learned_policy_evaluations':0}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))
    if not report['dynamics_pass'] or not report['grade_agreement']:raise SystemExit(1)
if __name__=='__main__':main()
