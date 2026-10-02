"""Read-only hash/schema/source verification; no neural model or GPU imports."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=json.loads((ROOT/'experiment/flux-so101-retry/package-manifest.json').read_text())
    failures=[]
    for name,entry in manifest['files'].items():
        path=ROOT/name
        if not path.is_file():failures.append({'path':name,'reason':'missing'});continue
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            failures.append({'path':name,'reason':'hash/size mismatch or LFS pointer'})
    base=ROOT/'runs/flux-so101-retry/dev-v1'
    spec=json.loads((base/'frozen-specification.json').read_text())
    archived=ROOT/'experiment/flux-so101-retry/source-measured'
    for name,key in [('run_flux_so101_retry.py','runner_sha256'),('local_runtime_fast.py','runtime_helper_sha256'),('so101_pick_place_baseline.py','baseline_sha256')]:
        if sha(archived/name)!=spec[key]:failures.append({'path':str(archived/name),'reason':'measured source hash differs'})
    for mode in spec['modes']:
        episode=base/mode/'seed-910'
        report=json.loads((episode/'report.json').read_text())
        if sha(episode/'trajectory.npz')!=report['trajectory_sha256']:
            failures.append({'path':str(episode),'reason':'trajectory hash differs'})
        if sha(episode/'cell.xml')!=report['model_sha256']:
            failures.append({'path':str(episode),'reason':'legacy XML hash differs'})
        for chunk in range(spec['chunks']):
            for name in [f'input-{chunk:03d}.npz',f'actions-policy-{chunk:03d}.npy']:
                if not (episode/name).is_file():failures.append({'path':str(episode/name),'reason':'missing chunk'})
        if report['success'] or not report['complete']:
            failures.append({'path':str(episode),'reason':'retained development outcome changed'})
    runtime=json.loads((base/'runtime.json').read_text())
    lock=json.loads((ROOT/'experiment/flux-so101-retry/source-lock.json').read_text())
    if runtime['sha256']!=lock['so101_files_sha256']:
        failures.append({'path':'source-lock.json','reason':'model package hash inventory differs'})
    for p in (ROOT/'scripts').glob('*flux*.py'):ast.parse(p.read_text())
    for p in (ROOT/'vendor/flux_retry_runtime').glob('*.py'):ast.parse(p.read_text())
    result={'passed':not failures,'files_hashed':len(manifest['files']),
            'neural_weights_bundled':False,'neural_weights_reread':False,
            'neural_weights_sha256_from_measured_runtime':runtime['sha256']['model.safetensors'],
            'episode_model_sha256_is_xml':True,'measured_task_successes':0,'measured_trials':2,
            'canonical_repository_written':False,'failures':failures}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
