"""Audit matched teacher-forced inputs and summarize measured commands, not success."""
import argparse, hashlib, json, statistics
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comparison',type=Path,required=True);parser.add_argument('--models',nargs='+',default=['smolvla','groot','act']);a=parser.parse_args()
    reports={name:json.loads((a.comparison/f'teacher-{name}'/'teacher-forced-report.json').read_text()) for name in a.models}
    anchors=reports['smolvla']['anchor_order'];matched=[];models={}
    for i,frame in enumerate(anchors):
        values=[r['rows'][i]['input_sha256'] for r in reports.values()]
        if not all(value==values[0] for value in values):raise ValueError(f'Input mismatch at frame{frame}')
        matched.append({'frame':frame,'input_sha256':values[0]})
    for name,report in reports.items():
        if not report['complete'] or report['anchor_order']!=anchors:raise ValueError('Incomplete/unmatched anchors')
        guard=json.loads((a.comparison/f'teacher-{name}'/'guard-report.json').read_text())
        session_string=report['policy']['session']
        # Reports created in WSL use POSIX workspace paths; relocate explicitly.
        if session_string.startswith('/mnt/d/'):
            session_string='D:/'+session_string[7:]
        session=Path(session_string)
        cloud=json.loads((session/'cloud-resource-report.json').read_text())
        rows=report['rows'];models[name]={'mean_first8_mae_rad':statistics.mean(r['first8_mae_rad'] for r in rows),
          'per_joint_mean_first8_mae_rad':[statistics.mean(r['per_joint_first8_mae_rad'][j] for r in rows) for j in range(6)],
          'server_seconds_median':statistics.median(r['server_synchronized_seconds'] for r in rows),
          'end_to_end_seconds_median':statistics.median(r['end_to_end_seconds'] for r in rows),
          'phase_rows':[{k:r[k] for k in ['frame','first8_mae_rad','per_joint_first8_mae_rad','first_command_rad','expert_first_command_rad']} for r in rows],
          'guard':guard,'cloud_resources':cloud,
          'teacher_report_sha256':hashlib.sha256((a.comparison/f'teacher-{name}'/'teacher-forced-report.json').read_bytes()).hexdigest()}
    summary={'scope':'identical original training-scene teacher-forced diagnostic; no robot task success or fair training-budget ranking',
       'all_17_actual_input_hash_sets_identical':True,'matched_inputs':matched,'models':models,
       'interpretation':'Small expert-state command errors alone do not establish closed-loop competence; compare actual off-expert observations before attributing failures.'}
    target=a.comparison/'teacher-forced-summary.json'
    if target.exists():raise FileExistsError('Retain existing summaries')
    target.write_text(json.dumps(summary,indent=2));print(json.dumps({n:{k:v for k,v in row.items() if k not in ['phase_rows','guard','cloud_resources']} for n,row in models.items()},indent=2))

if __name__=='__main__':main()
