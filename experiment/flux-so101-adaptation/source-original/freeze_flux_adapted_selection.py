"""Freeze an independently audited development selection before fresh starts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    suite = ROOT / 'runs/flux-so101-retry/adapted-dev-0256-cfg'
    adapter = ROOT / 'runtime/flux-so101-adaptation/train-broad-v1/checkpoint-0256'
    destination = ROOT / 'runs/flux-so101-retry/selected-adapter-0256-cfg3.json'
    assert not destination.exists(), 'Never overwrite a frozen selection'
    audit = json.loads((suite / 'independent-cpu-audit.json').read_text())
    summary = json.loads((suite / 'summary.json').read_text())
    spec = json.loads((suite / 'frozen-specification.json').read_text())
    assert audit['passed'] and len(audit['episodes']) == 2
    assert summary['complete'] and summary['total_completed'] == 2
    assert spec['guidance_scales'] == [3.0, 1.0] and spec['seeds'] == [910]
    selected = json.loads((suite / 'native_cfg3/seed-910/report.json').read_text())
    assert selected['complete'] and selected['success']
    assert selected['bilateral_contact_lift_over_90mm_world_z']
    assert digest(adapter / 'adapter.safetensors') == spec['adapter_sha256']
    fresh_output = ROOT / 'runs/flux-so101-retry/adapted-fresh-0256-cfg3'
    assert not fresh_output.exists(), 'Fresh outcomes must not precede selection'
    tracked = [
        adapter / name for name in ['adapter.safetensors', 'adapter-metadata.json',
                                    'policy-config.json', 'normalization.json']
    ] + [
        ROOT / name for name in [
            'scripts/run_flux_so101_adapted_cfg.py', 'scripts/flux_adapted_inference.py',
            'scripts/so101_pick_place_baseline.py',
            'runtime/flux-so101-work/local_runtime_fast.py',
            'runtime/flux-sim-audit/verify_flux_retry.py',
            'runs/flux-so101-retry/adaptation-evaluation-plan.json',
            'runs/flux-so101-retry/guidance-ablation-plan.json']
    ] + [suite / name for name in ['summary.json', 'frozen-specification.json',
                                   'independent-cpu-audit.json',
                                   'native_cfg3/seed-910/report.json',
                                   'native_cfg1/seed-910/report.json']]
    frozen = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_before_outcomes': True,
        'selection_reason': 'Earliest independently verified successful development checkpoint; original guidance3 selected under the prospective plan. Both planned guidance outcomes retained.',
        'optimizer_updates': 256, 'guidance_scale': 3.0,
        'adapter_sha256': spec['adapter_sha256'],
        'selected_adapter_sha256': spec['adapter_sha256'],
        'adapter': str(adapter.relative_to(ROOT)).replace('\\', '/'),
        'restoration': 'unmerged', 'prompt': selected['prompt'],
        'fresh_seeds': [920, 921, 922], 'seeds': [920, 921, 922], 'role': 'held-out',
        'chunks': 24, 'execute_per_chunk': 32, 'settling_hold_ticks': 45,
        'resident_gib': 12, 'fresh_output': str(fresh_output.relative_to(ROOT)).replace('\\', '/'),
        'no_retuning_against_fresh_outcomes': True,
        'assessment_scope': 'Three fresh simulated initial positions; exploratory evidence for one fixed instruction, not hardware or production reliability.',
        'sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in tracked},
    }
    destination.write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps({'selection': str(destination), 'adapter_sha256': spec['adapter_sha256']}))


if __name__ == '__main__':
    main()
