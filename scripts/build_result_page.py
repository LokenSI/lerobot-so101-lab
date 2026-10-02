"""Build an honest static FLUX page from measured reports, without running inference.

This generates local staging only. It copies small audited JSON records and
already-rendered presentation files. It does not publish, copy model weights,
copy simulator code or include social drafts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, unit='', digits=2):
    return 'unavailable' if value is None else f'{float(value):.{digits}f}{unit}'


def build(run, presentation, output):
    specification = run / 'frozen-specification.json'
    spec = json.loads(specification.read_text())
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / 'evidence'
    evidence.mkdir()
    shutil.copy2(specification, evidence / specification.name)
    runtime = None
    if (run / 'runtime.json').is_file():
        runtime = json.loads((run / 'runtime.json').read_text())
        shutil.copy2(run / 'runtime.json', evidence / 'runtime.json')
    rows, cards, records = [], [], []
    for mode in spec['modes']:
        for seed in spec['seeds']:
            episode = run / mode / f'seed-{seed}'
            report_path = episode / 'report.json'
            report = json.loads(report_path.read_text()) if report_path.is_file() else None
            label = 'PENDING' if report is None else ('INCOMPLETE' if not report.get('complete') else ('PASS' if report['success'] else 'FAIL'))
            basename = f'{mode}-seed-{seed}'
            report_href = ''
            if report is not None:
                destination = evidence / f'{basename}-report.json'
                shutil.copy2(report_path, destination)
                report_href = f'evidence/{destination.name}'
            times = [r['inference_seconds'] for r in (report or {}).get('inference', [])]
            mean = sum(times) / len(times) if times else None
            timing = f"mean {number(mean,' s')} / {len(times)} predictions"
            rows.append(f'<tr><td>{esc(mode)}</td><td>{seed}</td><td class="{label.lower()}">{label}</td><td>{esc(timing)}</td><td>{esc((report or {}).get("clipped_commands", "pending"))}</td></tr>')
            source_media = presentation / basename
            media = ''
            provenance = source_media / 'provenance.json'
            if report is not None and provenance.is_file():
                proof = json.loads(provenance.read_text())
                if proof['source_report_sha256'] != sha(report_path):
                    raise ValueError('Presentation report provenance differs from measured report')
                if proof['success'] != report['success']:
                    raise ValueError('Presentation outcome differs from measured report')
                target = output / 'media' / basename
                target.mkdir(parents=True)
                for filename in ['flux-replay.mp4', 'poster.png', 'contact-sheet.png', 'provenance.json', 'frame-provenance.json', 'raw-predicted-tcp.npz']:
                    path = source_media / filename
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if filename == 'flux-replay.mp4' and sha(path) != proof['video_sha256']:
                        raise ValueError('Rendered video hash differs from provenance')
                    shutil.copy2(path, target / filename)
                media = f'<video controls preload="metadata" poster="media/{basename}/poster.png"><source src="media/{basename}/flux-replay.mp4" type="video/mp4"></video><p><a href="media/{basename}/contact-sheet.png">First / middle / final frames</a> · <a href="media/{basename}/provenance.json">Video provenance</a> · <a href="media/{basename}/flux-replay.mp4">MP4</a></p>'
            else:
                media = '<p class="quiet">Completed replay will appear after rendering and independent verification.</p>'
            grader = (report or {}).get('grader', {})
            body = '<p>Measured report pending.</p>' if report is None else f'<p>Strict grade: {label}. Cube maximum world height: {number(grader.get("max_object_height_m"), " m", 3)}. Bilateral contact above 90 mm world height: {esc(report.get("bilateral_contact_lift_over_90mm_world_z", "see report"))}.</p>'
            if report and report.get('error'):
                body += f'<p>Execution stopped: {esc(report.get("error_type", "runtime error"))}. The incomplete result is retained.</p>'
            if report_href:
                body += f'<p><a href="{report_href}">Measured episode report</a></p>'
            cards.append(f'<article><h3>{esc(mode)} / seed {seed}</h3>{body}{media}</article>')
            records.append({'mode': mode, 'seed': seed, 'status': label, 'report': report_href or None, 'mean_inference_seconds': mean,
                            'first_inference_seconds': times[0] if times else None,
                            'median_inference_seconds': statistics.median(times) if times else None,
                            'prediction_count': len(times)})
    terminal = all(r['status'] in ['PASS', 'FAIL', 'INCOMPLETE'] for r in records)
    status = 'Completed exploratory adapter comparison' if terminal else 'Exploratory adapter comparison in progress'
    successful = sum(r['status'] == 'PASS' for r in records)
    completed = sum(r['status'] in ['PASS', 'FAIL'] for r in records)
    weight_hash = runtime.get('sha256', {}).get('model.safetensors') if runtime else None
    clarification = {'episode_model_sha256_means': 'SHA256 of measured MuJoCo cell.xml; not neural weights',
                     'neural_weights_sha256': weight_hash, 'neural_weights_hash_source': 'runtime.json.sha256.model.safetensors',
                     'original_measured_reports_modified': False}
    (evidence / 'schema-clarification.json').write_text(json.dumps(clarification, indent=2))
    stylesheet = '''body{margin:0;background:#09141f;color:#e6edf5;font:17px/1.55 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:34px 24px}h1{font-size:42px;line-height:1.15}h2{margin-top:38px}a{color:#79dfe4}article,.panel{background:#122432;border:1px solid #244052;border-radius:14px;padding:22px;margin:24px 0}video{width:100%;max-height:800px;background:#000;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #315064;padding:12px;text-align:left}.quiet{color:#b5c5d5}.fail,.incomplete{color:#ffbc43}.pass{color:#92ea9b}.tag{color:#ffbc43}code{overflow-wrap:anywhere}li{margin:9px 0}@media(max-width:700px){h1{font-size:32px}main{padding:18px}table{font-size:12px}th,td{padding:5px}}'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FLUX SO-101 — measured local simulation</title><style>{stylesheet}</style></head><body><main>
<p><a href="../index.html">SO-101 experiments</a></p><p class="tag">{esc(status)}</p><h1>Can an instruction become a completed robot task?</h1>
<p>FLUX 3 Action SO-101 takes two camera views, measured joints, recent commands and a written instruction. Here it controls a simulated arm and a free cube through real contacts.</p>
<div class="panel"><p><strong>{successful} successful / {completed} completed episodes.</strong> This small development comparison explores adapter choices. It is not a production reliability estimate, and missing or incomplete episodes are excluded from the completed denominator and shown below.</p><p>Instruction: <q>{esc(spec['prompt'])}</q></p><p>Pretrained original BF16 weights with CPU offload. No additional training, scripted pickup or object attachment. Physics runs at simulated 30 Hz while model inference pauses it.</p></div>
<h2>Measured outcomes</h2><table><thead><tr><th>Adapter</th><th>Seed</th><th>Outcome</th><th>Inference timing</th><th>Clipped commands</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}
<h2>How to read the overlay</h2><p>Amber dots show forward kinematics of raw FLUX joint targets. Cyan traces the tool's observed motion. Predictions refresh once per 32 executed ticks; 42 are predicted and the final ten are not executed. Saved scene and wrist RGB views feed the policy. This replay is neither real-time inference nor generated future camera video.</p>
<h2>What changes between adapters?</h2><p><strong>Hosted median:</strong> native new-calibration coordinates and the public demo's declared median quantiles. <strong>Legacy candidate:</strong> original saved statistics with a geometry-supported coordinate hypothesis. Exact source-dataset motor calibration is unavailable, so this second mapping is not verified.</p>
<h2>What counts as completion?</h2><p>The cube must exceed 90 mm in world Z with positive contact force on both jaws, then finish fully inside the tray, released and stably resting for the final 1.5 seconds. A final picture alone does not establish success. All failures and runtime interruptions remain visible.</p>
<h2>Earlier evidence remains separate</h2><p>The original hosted FLUX SO-101 attempt failed to place the cube. DROID's accelerated test predicted commands from recorded observations without executing them on this arm. ACT's earlier 2/5 and later exploratory 10/12 results use a separately trained model. The stereo experiment uses measured perception and scripted control.</p>
<h2>Reproducibility and attribution</h2><p><a href="evidence/frozen-specification.json">Frozen protocol and source hashes</a> · <a href="METHOD.md">Method</a> · <a href="result-summary.json">Structured results</a> · <a href="evidence/schema-clarification.json">Scene / neural model hash clarification</a></p>
<p>Neural weights SHA256: <code>{esc(weight_hash or 'runtime record pending')}</code>. The original episode field <code>model_sha256</code> identifies the physical scene XML, not these neural weights.</p>
<p>Model: <a href="https://huggingface.co/black-forest-labs/flux-3-action-so101/tree/{esc(spec['model_revision'])}">Black Forest Labs FLUX 3 Action SO-101</a>. Model usage follows its original <a href="https://huggingface.co/black-forest-labs/flux-3-action-so101/blob/{esc(spec['model_revision'])}/LICENSE.md">FLUX Kommunity License</a>; repository code licensing does not replace those terms. <a href="https://github.com/black-forest-labs/flux-action/tree/{esc(spec['source_revision'])}">Official inference source</a>. Model weights remain local.</p>
<p class="quiet">Independent experiment; no endorsement by model or robot creators. Simulation evidence does not establish physical arm or industrial performance.</p></main></body></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('METHOD.md'), output / 'METHOD.md')
    result = {'status': status, 'successful': successful, 'completed': completed, 'planned': len(records),
              'episodes': records, 'source_specification_sha256': sha(specification), 'builder_sha256': sha(__file__),
              'model_weights_included': False, 'published': False}
    (output / 'result-summary.json').write_text(json.dumps(result, indent=2))
    manifest = {str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in output.rglob('*') if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--presentation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root.resolve(), args.presentation_root.resolve(), args.output.resolve()), indent=2))
