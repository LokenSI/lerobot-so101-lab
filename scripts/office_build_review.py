"""Build a replay matrix from retained expert/checkpoint episodes, including failures."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expert-episodes',type=Path,nargs='*',default=[])
    p.add_argument('--evaluation-dirs',type=Path,nargs='*',default=[]);p.add_argument('--tasks',nargs='*');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=4);p.add_argument('--columns',type=int,default=3);p.add_argument('--presentation-legibility',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    episodes=list(a.expert_episodes);incomplete=[]
    for directory in a.evaluation_dirs:
        if not (directory/'report.json').exists():
            status=json.loads((directory/'load-status.json').read_text())
            if (directory/'guard-report.json').exists():status.update(json.loads((directory/'guard-report.json').read_text()))
            incomplete.append({'directory':str(directory),'error':status.get('error'),'complete':False,'no_trajectory':True});continue
        report=json.loads((directory/'report.json').read_text())
        for row in report['episodes']:
            if a.tasks and row['task']['id'] not in a.tasks:continue
            episodes.append(directory/f'seed-{row["seed"]}-{row["task"]["id"]}')
        if not report.get('complete'):incomplete.append({'directory':str(directory),'error':report.get('error'),'complete':False})
    if not episodes:raise ValueError('No retained trajectories available; do not fabricate missing rollouts')
    replays=[];cells=[]
    for i,episode in enumerate(episodes):
        meta=json.loads((episode/'episode.json').read_text());replay=a.output/f'cell-{i:03d}.usda'
        if meta.get('steps',meta.get('frames',0))==0:
            incomplete.append({'episode':str(episode),'error':meta.get('error'),'complete':False,'no_trajectory':True});continue
        subprocess.run([sys.executable,str(Path(__file__).with_name('office_export_isaac_replay.py')),'--episode',str(episode),'--output',str(replay),'--stride',str(a.stride),*(['--presentation-legibility'] if a.presentation_legibility else [])],check=True)
        replays.append(replay);cells.append({'cell':i,'episode':str(episode),'instruction':meta['task']['instruction'],'seed':meta['seed'],
              'success':meta['success'],'expert':meta.get('learned_policy') is False,'policy':meta.get('policy'),
              'failure_events':meta.get('command_limit_events'),'grader':meta['grader']})
        cells[-1].update({'wording':meta.get('wording'),'chunk_execution':meta.get('chunk_execution'),'development_overfit_control':meta.get('development_overfit_control',False)})
    matrix=a.output/'matrix.usda'
    subprocess.run([sys.executable,str(Path(__file__).with_name('office_isaac_matrix.py')),'--replays',*[str(r) for r in replays],'--output',str(matrix),'--columns',str(a.columns)],check=True)
    (a.output/'review-contract.json').write_text(json.dumps({'scope':'native Isaac presentation of retained MuJoCo evidence; no new policy or Isaac physics evaluation',
       'cells':cells,'failures_included':sum(not c['success'] for c in cells),'excluded_outcome_filter':None,'presentation_legibility':a.presentation_legibility,
       'explicit_task_filter':a.tasks,'incomplete_evaluations':incomplete,'all_failures_from_selected_completed_cases_retained':True},indent=2))
    print(json.dumps({'matrix':str(matrix),'cells':len(cells),'failures':sum(not c['success'] for c in cells)}))
if __name__=='__main__':main()
