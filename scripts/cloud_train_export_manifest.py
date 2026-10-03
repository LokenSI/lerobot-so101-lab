"""Hash checkpoint exports, including saved processors and resume files."""
import argparse,hashlib,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['smolvla','groot'],required=True);p.add_argument('--reload-report',type=Path);a=p.parse_args()
    rows=[]
    for path in sorted(a.checkpoint.rglob('*')):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            rows.append({'path':str(path.relative_to(a.checkpoint)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    reload=json.loads(a.reload_report.read_text()) if a.reload_report else None
    result={'timestamp_unix':time.time(),'model':a.model,'checkpoint':str(a.checkpoint),'files':rows,
        'reload_probe':reload,'deployable_reload_verified':bool(reload and reload.get('passed')),
        'resume':'Keep the complete checkpoint including optimizer/scheduler/RNG; a model-only export cannot resume exactly.',
        'units':'Six absolute joint targets, all radians including gripper actuator angle',
        'input':'Current measured6joint radians + current scene/wrist RGB + text instruction; no object poses',
        'closed_loop_success_claim':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'files':len(rows),'reload_verified':result['deployable_reload_verified']}))
if __name__=='__main__':main()
