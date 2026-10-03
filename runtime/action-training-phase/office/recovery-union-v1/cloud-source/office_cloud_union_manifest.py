"""Hash complete native final checkpoints directly, without duplicate archives."""
import argparse,hashlib,json
from pathlib import Path
from office_cloud_union_training import ROOT

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['smolvla','groot'],required=True);p.add_argument('--attempt',choices=['v1','v2'],default='v1');a=p.parse_args()
    report_root=ROOT if a.attempt=='v1' else ROOT/'groot-worker1-v2'
    run=Path('/data/action-training/runs')/f'union-eighteen-{a.attempt}-{a.model}-main'
    checkpoint=run/'checkpoint-2000' if a.model=='groot' else run/'checkpoints/002000'
    gate=json.loads((report_root/f'{a.model}-main-reload-guard.json').read_text())
    execution=json.loads((report_root/f'{a.model}-main-execution-report.json').read_text())
    if not gate['passed'] or not execution['complete'] or execution['exit_code']!=0:raise ValueError('Native final save/reload not complete')
    rows=[{'path':path.relative_to(checkpoint).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)} for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    metadata={'scope':'complete native final checkpoint including resume state; no robot-success claim','model':a.model,'native_checkpoint_root':str(checkpoint),
      'files':rows,'total_bytes':sum(row['bytes'] for row in rows),'file_count':len(rows),'native_final_save_complete':True,'fresh_native_reload_passed':True,
      'recipe_sha256':execution['recipe_sha256'],'source_pin':json.loads((ROOT/'recipe.json').read_text())['models'][a.model]['source_pin'],
      'archive_created':False,'resume_state_retained':any('optimizer' in row['path'] or 'optim' in row['path'] for row in rows)}
    destination=report_root/f'{a.model}-main-file-manifest.json'
    if destination.exists():raise FileExistsError('Keep original full manifest')
    destination.write_text(json.dumps(metadata,indent=2));print(json.dumps({'manifest':str(destination),'sha256':digest(destination),'total_bytes':metadata['total_bytes'],'files':len(rows)}))

if __name__=='__main__':main()
