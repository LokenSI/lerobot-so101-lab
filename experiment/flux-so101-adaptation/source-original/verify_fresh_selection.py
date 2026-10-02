"""Independent stdlib-only frozen selection/hash audit; no model import."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SELECTION=ROOT/'runs/flux-so101-retry/selected-adapter-0256-cfg3.json'
EXPECTED_SELECTION_SHA='9540e16b15f37cbb06409a3867d026783ff867b9f0d3077b24ce0e327d403eda'
EXPECTED_CHECKER_SHA='68e8a1e09949f834e2e9ad7cffce41bbfee5ddafcbdd6bc6acf4c092aeca2dea'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
selection=json.loads(SELECTION.read_text())
suite=ROOT/selection['fresh_output']
spec=json.loads((suite/'frozen-specification.json').read_text())
checks={}
checks['selection_bytes_match_pre_fresh_recorded_hash']=sha(SELECTION)==EXPECTED_SELECTION_SHA
checked_files={p:{'expected':expected,'actual':sha(ROOT/p)} for p,expected in selection['sha256'].items()}
checks['all_selected_files_match_pre_fresh_hashes']=all(v['expected']==v['actual'] for v in checked_files.values())
checks['checker_frozen_exact']=sha(ROOT/'runtime/flux-sim-audit/verify_flux_retry.py')==EXPECTED_CHECKER_SHA
checks['selected_adapter_config_loader_and_sources_match_fresh_spec']=all([
    spec['adapter_sha256']==selection['adapter_sha256'],
    spec['adapter_metadata_sha256']==selection['sha256'][selection['adapter']+'/adapter-metadata.json'],
    spec['adapter_config_sha256']==selection['sha256'][selection['adapter']+'/policy-config.json'],
    spec['runner_sha256']==selection['sha256']['scripts/'+spec['runner_filename']],
    spec['adapter_loader_sha256']==selection['sha256']['scripts/flux_adapted_inference.py'],
    spec['baseline_sha256']==selection['sha256']['scripts/so101_pick_place_baseline.py'],
    spec['runtime_helper_sha256']==selection['sha256']['runtime/flux-so101-work/local_runtime_fast.py']])
checks['fresh_protocol_matches_selection']=all([
    selection['optimizer_updates']==256,spec['guidance_scales']==[selection['guidance_scale']],
    spec['modes']==['native_cfg3'],spec['seeds']==selection['fresh_seeds'],
    spec['chunks']==selection['chunks'],spec['execute_per_chunk']==selection['execute_per_chunk'],
    spec['fps']==30,spec['settling_hold_seconds']*spec['fps']==selection['settling_hold_ticks'],
    spec['adapter_restoration']==selection['restoration'],spec['prompt']==selection['prompt'],
    spec['role']==selection['role'],spec['resident_gib']==selection['resident_gib']])
reports=[]
for folder in sorted(suite.glob('*/seed-*')):
    if not (folder/'report.json').exists():continue
    report=json.loads((folder/'report.json').read_text())
    checks[f'seed{report["seed"]}_belongs_to_frozen_protocol']=report['seed'] in selection['fresh_seeds'] and report['mode']=='native_cfg3' and report['guidance_scale']==selection['guidance_scale'] and report['prompt']==selection['prompt']
    audit=json.loads((folder/'independent-cpu-audit.json').read_text())
    checks[f'seed{report["seed"]}_independent_audit_complete_and_frozen']=audit['passed'] and audit['checker_sha256']==EXPECTED_CHECKER_SHA and audit['model_source']=='reconstructed_frozen_baseline'
    reports.append({'seed':report['seed'],'success':audit['metrics']['grade']['success'],'report_sha256':sha(folder/'report.json'),'audit_sha256':sha(folder/'independent-cpu-audit.json')})
out={'passed':all(checks.values()),'selection_sha256':sha(SELECTION),'checker_sha256':EXPECTED_CHECKER_SHA,
     'checks':checks,'selected_files':checked_files,'completed_episodes':reports,
     'assessment_complete':len(reports)==len(selection['fresh_seeds']),
     'scope':'Checks immutable selection/source/adapter files and protocol agreement. Task grading is performed by the separately frozen CPU verifier; this script does not infer unseen outcomes.'}
output=HERE/f'fresh-selection-independent-{len(reports):02d}-episodes.json'
output.write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':out['passed'],'checks':checks,'completed_episodes':reports,'assessment_complete':out['assessment_complete'],'output':str(output)},indent=2))
raise SystemExit(0 if out['passed'] else 1)
