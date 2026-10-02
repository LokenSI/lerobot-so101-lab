"""Build a trained-FLUX viewer from independently audited actual reports only."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def esc(value):return html.escape(str(value))

def read_suite(path,expected_role):
    spec=json.loads((path/'frozen-specification.json').read_text())
    runtime=json.loads((path/'runtime.json').read_text())
    if spec['role']!=expected_role or not set(spec['modes'])<=set(['native_adapted','native_cfg3','native_cfg1']) or spec['no_training']:
        raise ValueError('Wrong experiment role or model family')
    if spec['adapter_restoration']!='unmerged':raise ValueError('This recipe requires recorded unmerged restoration')
    if spec['adapter_sha256']!=runtime['trained_adapter']['sha256']:raise ValueError('Adapter hash mismatch')
    audit=json.loads((path/'independent-cpu-audit.json').read_text())
    if not audit.get('passed'):raise ValueError('Suite needs completed independent CPU audit')
    rows=[]
    for mode,seed in [(mode,seed) for mode in spec['modes'] for seed in spec['seeds']]:
        episode=path/mode/f'seed-{seed}'
        report=json.loads((episode/'report.json').read_text())
        verified=json.loads((episode/'independent-cpu-audit.json').read_text())
        if not verified['passed']:raise ValueError('Episode independent audit failed')
        xml_hash=report.get('mujoco_xml_sha256',report.get('model_sha256'))
        if report['trajectory_sha256']!=sha(episode/'trajectory.npz') or xml_hash!=sha(episode/'cell.xml'):
            raise ValueError('Measured evidence hashes changed')
        if report['success'] and not report['complete']:raise ValueError('Incomplete episode cannot pass')
        times=[r['inference_seconds'] for r in report['inference']]
        rows.append({'suite':path.name,'role':expected_role,'seed':seed,'mode':report['mode'],
            'guidance_scale':report.get('guidance_scale',runtime['config'].get('guidance_scale')),
            'outcome':'PASS' if report['success'] else ('FAIL' if report['complete'] else 'INCOMPLETE'),
            'success':bool(report['success']),'complete':bool(report['complete']),
            'updates':runtime['trained_adapter']['metadata']['optimizer_updates'],
            'adapter_sha256':spec['adapter_sha256'],'report_sha256':sha(episode/'report.json'),
            'trajectory_sha256':report['trajectory_sha256'],'episode_xml_sha256':xml_hash,
            'first_inference_seconds':times[0] if times else None,'median_inference_seconds':statistics.median(times) if times else None,
            'predictions':len(times),'clipped_commands':report['clipped_commands'],
            'grader':report.get('grader'),'bilateral_lift':report.get('bilateral_lift'),
            'episode_path':episode,'audit':verified})
    return {'path':path,'spec':spec,'runtime':runtime,'audit':audit,'rows':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--development-suite',type=Path,action='append',required=True)
    p.add_argument('--assessment-suite',type=Path)
    p.add_argument('--selection-plan',type=Path,help='For later stages: explicit pre-frozen selected adapter SHA256 and fresh seeds')
    p.add_argument('--progress',action='store_true',help='Show measured progress without a final adaptation verdict')
    p.add_argument('--additional-stage-json',type=Path,action='append',default=[],
                   help='Explicit later pre-frozen plan plus development suite list; never pool with original stage')
    p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    dev=[read_suite(x,'development') for x in a.development_suite]
    later=[]
    for description in a.additional_stage_json:
        stage=json.loads(description.read_text())
        stage_plan=Path(stage['frozen_plan'])
        if not stage_plan.is_file():raise ValueError('Later stage needs its own pre-frozen plan')
        suites=[read_suite(Path(x),'development') for x in stage['development_suites']]
        later.append({'name':stage['name'],'plan':stage_plan,'suites':suites})
    assessment=read_suite(a.assessment_suite,'held-out') if a.assessment_suite else None
    if any(x['spec']['seeds']!=[plan['development']['seed']] for x in dev):raise ValueError('Development seeds changed')
    candidates=plan['training']['checkpoint_candidates']
    attempted=[x['runtime']['trained_adapter']['metadata']['optimizer_updates'] for x in dev]
    if attempted!=candidates[:len(attempted)]:raise ValueError('Candidate sequence differs from frozen selection plan')
    successful=[x for x in dev if any(r['success'] for r in x['rows'])]
    if assessment:
        if len(assessment['spec']['modes'])!=1:raise ValueError('Fresh assessment must use one frozen selected condition')
        if assessment['spec']['seeds']!=plan['fresh_start_assessment']['seeds']:raise ValueError('Fresh starts changed')
        if later:
            if not a.selection_plan:raise ValueError('Later-stage fresh assessment needs separately frozen selected-adapter plan')
            selected_plan=json.loads(a.selection_plan.read_text())
            if not selected_plan.get('frozen_before_outcomes') or selected_plan['seeds']!=assessment['spec']['seeds']:
                raise ValueError('Later selected-adapter plan must be frozen before the fresh outcomes')
            selected=selected_plan['selected_adapter_sha256']
            successful_all=[s for stage in later for s in stage['suites'] if any(r['success'] for r in s['rows'])]+successful
            if selected not in [s['spec']['adapter_sha256'] for s in successful_all]:raise ValueError('Selected adapter lacks independently verified development success')
            if 'guidance_scale' in selected_plan:
                if any(r['guidance_scale']!=selected_plan['guidance_scale'] for r in assessment['rows']):raise ValueError('Fresh guidance differs from selected frozen condition')
                if not any(r['success'] and r['adapter_sha256']==selected and r['guidance_scale']==selected_plan['guidance_scale'] for s in successful_all for r in s['rows']):
                    raise ValueError('Selected guidance has no verified development success')
        else:
            if not successful:raise ValueError('Fresh assessment requires selected successful development adapter')
            selected=successful[0]['spec']['adapter_sha256']
        if selected!=assessment['spec']['adapter_sha256']:raise ValueError('Selected checkpoint differs')
        if not all(r['complete'] for r in assessment['rows']):raise ValueError('Retain all fresh starts before final assessment page')
    elif successful and not a.progress:
        raise ValueError('A successful development candidate requires the separately frozen fresh assessment before final page')
    elif attempted!=candidates and not a.progress:
        raise ValueError('Final unsuccessful-adaptation page requires all frozen development candidates; intermediate failures remain development evidence')
    output=a.output;output.mkdir(parents=True,exist_ok=False)
    (output/'evidence').mkdir();(output/'media').mkdir()
    shutil.copy2(a.plan,output/'evidence/adaptation-evaluation-plan.json')
    if a.selection_plan:shutil.copy2(a.selection_plan,output/'evidence/selected-adapter-plan.json')
    for index,stage in enumerate(later):shutil.copy2(stage['plan'],output/f"evidence/later-stage-{index+1}-frozen-plan.json")
    shutil.copy2(Path(__file__).with_name('TRAINING-RECIPE.md'),output/'TRAINING-RECIPE.md')
    root=Path(__file__).resolve().parents[2]
    license_path=root/'models/flux-action/so101/LICENSE.md'
    if sha(license_path)!=dev[0]['runtime']['sha256']['LICENSE.md']:raise ValueError('License bytes differ from measured package')
    shutil.copy2(license_path,output/'evidence/FLUX-SO101-KOMMUNITY-LICENSE.md')
    rows=[];cards=[]
    for suite in dev+[suite for stage in later for suite in stage['suites']]+([assessment] if assessment else []):
        dest=output/'evidence'/suite['path'].name;dest.mkdir()
        for name in ['frozen-specification.json','runtime.json','summary.json','independent-cpu-audit.json']:
            shutil.copy2(suite['path']/name,dest/name)
        for row in suite['rows']:
            episode=row.pop('episode_path');audit=row.pop('audit')
            suffix='' if row['mode']=='native_adapted' else '-'+row['mode']
            key=f"{row['suite']}{suffix}-seed-{row['seed']}";media=a.media_root/key
            if not media.exists():raise ValueError(f'Missing verified replay media: {media}')
            proof=json.loads((media/'provenance.json').read_text())
            if proof['source_report_sha256']!=row['report_sha256'] or proof['source_trajectory_sha256']!=row['trajectory_sha256']:
                raise ValueError('Replay provenance differs from measured episode')
            if proof['video_sha256']!=sha(media/'flux-replay.mp4'):raise ValueError('Replay video hash differs')
            target=output/'media'/key;shutil.copytree(media,target)
            evidence_key=f"{row['mode']}-seed-{row['seed']}"
            report_dest=dest/f"{evidence_key}-report.json";shutil.copy2(episode/'report.json',report_dest)
            shutil.copy2(episode/'independent-cpu-audit.json',dest/f"{evidence_key}-independent-cpu-audit.json")
            times=f"first {row['first_inference_seconds']:.2f} s / median {row['median_inference_seconds']:.2f} s"
            cards.append(f'''<section><h2>{esc(row['role'])}: checkpoint {row['updates']}, guidance {esc(row['guidance_scale'])}, seed {row['seed']} — {row['outcome']}</h2>
<video controls playsinline preload="metadata" poster="media/{key}/poster.png" src="media/{key}/flux-replay.mp4"></video>
<p>{esc(times)} per prediction · {row['predictions']} predictions · {row['clipped_commands']} clipped commands.</p>
<p><a href="evidence/{row['suite']}/{report_dest.name}">Original outcome</a> · <a href="evidence/{row['suite']}/{evidence_key}-independent-cpu-audit.json">Independent CPU audit</a> · <a href="media/{key}/provenance.json">Replay provenance</a></p></section>''')
            rows.append(row)
    fresh=assessment['rows'] if assessment else []
    passes=sum(r['success'] for r in fresh)
    title=(f'Trained FLUX: {passes}/{len(fresh)} fresh simulated starts' if fresh else
           ('FLUX adaptation: measured development progress' if a.progress or later else 'Trained FLUX: unsuccessful frozen-stage candidates'))
    updates=assessment['runtime']['trained_adapter']['metadata']['optimizer_updates'] if assessment else attempted[-1]
    ongoing=bool(a.progress or assessment is None)
    summary={'title':title,'research_ongoing':ongoing,'final_adaptation_verdict_claimed':bool(assessment and not a.progress),
             'later_stages':[{'name':s['name'],'frozen_plan_sha256':sha(s['plan']),'suites':[x['path'].name for x in s['suites']]} for s in later],
             'pretrained_zero_shot':{'successes':0,'trials':2,'role':'Earlier development; separate denominator'},
             'development_attempted_checkpoints':attempted,'fresh_assessment':{'successes':passes,'trials':len(fresh),'seeds':[r['seed'] for r in fresh]},
             'selected_or_latest_updates':updates,'maximum_frozen_updates':400,'train_episodes':17,'validation_episodes':3,
             'train_windows':255,'validation_windows':45,'training_required':True,'hardware_result_claimed':False,'rows':rows}
    (output/'result-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>
body{{margin:0;background:#09141f;color:#eef5fb;font:17px/1.65 system-ui}}main{{max-width:1100px;margin:auto;padding:36px 24px}}h1{{font-size:clamp(34px,5vw,56px);line-height:1.15}}section{{margin:28px 0;padding:24px;background:#142231;border:1px solid #2d465b;border-radius:12px}}video{{width:100%;border-radius:9px}}a{{color:#62e0d7}}.scope{{color:#ffbc55}}code{{overflow-wrap:anywhere}}p{{max-width:940px}}</style></head><body><main>
<p><a href="../index.html">SO101 lab</a> · <a href="../flux-retry/index.html">Retained pretrained failures: 0/2</a></p><h1>{esc(title)}</h1>
<p>The original pretrained model did not complete our two initial simulated placements. We then trained the genuine FLUX 3 Action SO101 model for our scene using 17 demonstrations, reserving three other demonstrations for validation. The selected or latest tested model used {updates} optimizer updates. Checkpoint selection and guidance comparisons have separate recorded outcomes.</p>
<p>FLUX learns to turn two camera views, measured joints, previous issued commands and a short instruction into the next joint targets. Training uses the original native flow-matching loss, rank-8 LoRA and trainable action heads. In closed-loop tests it receives no object coordinates or scripted pickup corrections.</p>
<p class="scope">Development conditions and fresh starts have separate denominators. {"Research is ongoing; this page reports measured progress without a final adaptation verdict." if ongoing else f"The frozen selected model completed {passes}/{len(fresh)} fresh same-task placements; every planned start is retained."} These are small simulation tests; physical SO101, Orin deployment and real-time control remain untested. A 30 fps replay pauses physics while model inference runs.</p>
<section><h2>What the overlay shows</h2><p>Amber is forward kinematics of the raw 42-command prediction, refreshed after 32 executed commands; the last ten points were not executed. Cyan is the actual saved tool path. Insets retain the processed 256×256 RGB pixels sent to the model, derived from the physical 320×240 virtual-camera projection. Inference restores original BF16 bases with FP32 LoRA/full heads under BF16 autocast. Each stage's frozen specification records its effective sampler, history and normalization settings.</p><p>Success requires positive bilateral jaw contact above 90 mm cube world Z, then full containment, release and stable rest for 1.5 seconds. Independent CPU replay verifies saved histories, actions, clipping, contacts and physical state transitions.</p></section>
{''.join(cards)}
<section><h2>Inspect the full evidence</h2><p><a href="result-summary.json">Measured result summary</a> · <a href="evidence/adaptation-evaluation-plan.json">Pre-frozen selection and seed plan</a> · <a href="TRAINING-RECIPE.md">Training recipe and reproduction</a></p><p>Original 14 GB pretrained weights and encoders are fetched from pinned sources. Their original <a href="evidence/FLUX-SO101-KOMMUNITY-LICENSE.md">FLUX Kommunity terms</a> remain applicable; the repository does not relabel them as Apache-licensed weights.</p></section></main></body></html>'''
    (output/'index.html').write_text(text,encoding='utf-8')
    manifest={p.relative_to(output).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'fresh_successes':passes,'fresh_trials':len(fresh),'development_candidates':attempted},indent=2))

if __name__=='__main__':main()
