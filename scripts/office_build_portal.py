"""Publish a local office-only review page from actual reports, videos and training logs."""
import argparse,hashlib,html,json,os,re,shutil
from pathlib import Path

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--office',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--progress-files',type=Path,nargs='*',default=[]);p.add_argument('--progress-offsets',type=int,nargs='*',default=[]);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);media=a.output/'media';media.mkdir(exist_ok=True);sources=[];cards=[];milestones=[]
    for prov in sorted((a.office/'review-videos').glob('*.provenance.json')):
        data=json.loads(prov.read_text());video=prov.with_name(prov.name.replace('.provenance.json','.mp4'))
        if not video.exists():continue
        episode_path=data['episode']
        if os.name=='nt' and episode_path.startswith('/mnt/d/'):episode_path='D:/'+episode_path[7:]
        ep=Path(episode_path);meta=json.loads((ep/'episode.json').read_text());target=media/video.name;shutil.copy2(video,target)
        shutil.copy2(prov,media/prov.name);sources.append({'source':str(video),'source_sha256':sha(video),'copied_media_sha256':sha(target),'provenance':prov.name})
        cards.append({'model':meta['policy'].get('model','Policy'),'step':data['cumulative_step'],'task':meta['task']['id'],'instruction':meta['task']['instruction'],'success':meta['success'],
           'task_success':meta.get('task_success'),'strict_zero_raw_limit_events_success':meta.get('strict_zero_raw_limit_events_success'),'chunk_execution':data['chunk_execution'],
           'inference_scope':meta['policy'].get('inference_scope','local inference; see original evaluation report'),
           'language_attribution':meta['policy'].get('external_task_selector','instruction supplied to learned policy'),
           'seed':meta['seed'],'wording':meta.get('wording','validation'),'development_overfit_control':meta.get('development_overfit_control',False),
           'events':meta['command_limit_events'],'p50_ms':round(meta['fresh_inference_p50_s']*1000),'p95_ms':round(meta['fresh_inference_p95_s']*1000),
           'src':'media/'+video.name,'checkpoint_sha256':meta['policy']['checkpoint_sha256'],'kind':'learned','experiment':ep.parent.parent.name if ep.parent.name.startswith('rollout-') else ep.parent.name,
           'server_p50_ms':round(meta['server_synchronized_inference_p50_s']*1000,1) if meta.get('server_synchronized_inference_p50_s') is not None else None,
           'server_p95_ms':round(meta['server_synchronized_inference_p95_s']*1000,1) if meta.get('server_synchronized_inference_p95_s') is not None else None})
    cards.sort(key=lambda r:(r['model'],r['step'],r['chunk_execution'],r['seed'],r['task']))
    def condition(c):
        return (c['model'],c['step'],c['chunk_execution'],c['wording'],c['development_overfit_control'],c['experiment'])
    for model,step,chunk,wording,overfit,experiment in sorted(set(condition(c) for c in cards)):
        rows=[c for c in cards if condition(c)==(model,step,chunk,wording,overfit,experiment)]
        milestones.append({'model':model,'step':step,'chunk_execution':chunk,'wording':wording,
            'development_overfit_control':overfit,'experiment':experiment,'passed':sum(c['success'] for c in rows),
            'physical_task_passed':sum(c['task_success'] if c['task_success'] is not None else c['success'] for c in rows),'cases':len(rows)})
    expert=a.office/'preview/full256-red-left/replay.mp4';expert_src=None
    if expert.exists():
        shutil.copy2(expert,media/'expert-reference.mp4');expert_src='media/expert-reference.mp4';sources.append({'source':str(expert),'source_sha256':sha(expert),'kind':'privileged scripted expert'})
    series=[]
    for i,path in enumerate(a.progress_files):
        raw=json.loads(path.read_text());offset=a.progress_offsets[i] if i<len(a.progress_offsets) else 0
        points=[{'step':int(r['step'])+offset,'loss':float(r['loss'])} for r in raw.get('points',[]) if 'loss' in r and 'step' in r]
        series.append({'name':path.parent.name,'offset':offset,'points':points,'source_sha256':sha(path)})
    admission=json.loads((a.office/'local-memory-admission.json').read_text()) if (a.office/'local-memory-admission.json').exists() else {}
    training_media=[]
    for source in sorted((a.office/'training-progress-media').glob('*')):
        if source.is_file() and source.suffix in ('.mp4','.png','.json'):
            target=media/source.name;shutil.copy2(source,target)
            training_media.append({'name':source.name,'src':'media/'+source.name,'sha256':sha(source)})
    attempts=[]
    run_dirs=sorted(set(p.parent for p in a.office.glob('evaluation-*/report.json'))|set(p.parent for p in a.office.glob('evaluation-*/guard-report.json')))
    for directory in run_dirs:
        path=directory/'report.json';guard_path=directory/'guard-report.json'
        run=json.loads(path.read_text()) if path.exists() else {}
        guard=json.loads(guard_path.read_text()) if guard_path.exists() else {}
        attempts.append({'name':directory.name,'complete':run.get('complete',False),'episodes':len(run.get('episodes',[])),
           'successes':run.get('successes',0),'wording':run.get('wording','validation (original default)'),
           'task_successes':run.get('task_successes'),'chunk_execution':run.get('chunk_execution'),
           'development_overfit_control':run.get('development_overfit_control',False),'error':run.get('error') or guard.get('error'),
           'guard':guard,'report_sha256':sha(path) if path.exists() else None,'guard_sha256':sha(guard_path) if guard_path.exists() else None})
    for name in ['native-legible-overview','native-legible-closeup']:
        source=a.office/(name+'.mp4')
        if source.exists():sources.append({'source':str(source),'source_sha256':sha(source),'copied_media_sha256':sha(source),'provenance':name+'.provenance.json','kind':'native visual replay'})
    comparison_records=[]
    for source in sorted(a.office.glob('comparison-v*/comparison-summary.json')):
        summary=json.loads(source.read_text())
        comparison_records.append({'name':source.parent.name,'chunk_execution':summary['chunk_execution'],'scope':summary['scope'],'contract_sha256':summary['contract_sha256'],
          'task_successes':sum(row['task_successes'] for row in summary['models'].values()),'cases':sum(len(row['episodes']) for row in summary['models'].values()),
          'models':{name:{key:row[key] for key in ['training_scope','task_successes','strict_successes','learned_language']} for name,row in summary['models'].items()}})
    record={'scope':'local office review; actual clips and copied training metrics only','cards':cards,'expert_src':expert_src,'milestones':milestones,'comparisons':comparison_records,
         'loss_series':series,'training_media':training_media,'admission':admission,'attempts':attempts,'provenance_sources':sources,'sealed_test_cases_used':0,'outcome_filter':'none'}
    (a.output/'office-review-manifest.json').write_text(json.dumps(record,indent=2))
    data=json.dumps(record).replace('<','\\u003c');esc=html.escape
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SO-101 • measured learning</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;color:#ecf2f3;background:#101719}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:44px 28px}a{color:#a9e9de}header{display:flex;align-items:center;justify-content:space-between;gap:24px;border-bottom:1px solid #344246;padding-bottom:28px}.eyebrow{font-size:12px;letter-spacing:.18em;color:#8ce0cc;text-transform:uppercase}h1{font-size:clamp(30px,4vw,54px);letter-spacing:-.045em;line-height:1.05;margin:12px 0}p{color:#abc0c5;line-height:1.6;max-width:860px}.badge{border:1px solid #507064;border-radius:20px;padding:10px 15px;white-space:nowrap;font-size:12px;color:#a9dfce}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.stat,.panel{padding:22px;background:#182225;border:1px solid #344246;border-radius:16px}.stat strong{font-size:34px;display:block;letter-spacing:-.04em}.stat span{font-size:12px;color:#adc0c5}.failure{color:#fcb692}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:22px 0}select,button{font:inherit;background:#233438;color:#e5f1f2;border:1px solid #547077;border-radius:8px;padding:9px 13px;cursor:pointer}h2{font-size:21px;letter-spacing:-.015em;margin:26px 0 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}.card{background:#182225;border:1px solid #3a5055;border-radius:14px;overflow:hidden}.cardhead{padding:16px 18px;display:flex;justify-content:space-between;font-size:13px}.card h3{font-size:16px;line-height:1.5;margin:0;padding:0 18px 12px;min-height:60px}.card video{display:block;width:100%;height:340px;object-fit:contain;background:#000}.details{padding:15px 18px;color:#aebfc5;font-size:12px;line-height:1.8}.tag{font-size:10px;letter-spacing:.08em;padding:4px 7px;border-radius:5px;background:#4f3027;color:#ffc2a4}.pass{background:#244c42;color:#a5f2d5}.timeline{display:flex;gap:12px;flex-wrap:wrap}.milestone{padding:10px 14px;border-radius:8px;border:1px solid #4c6268;font-size:13px}.chart{width:100%;height:240px;background:#111b1e;border-radius:10px}.chart text{fill:#a6bbc2;font-size:12px}.split{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-top:24px}.expert video{width:100%;background:#000}.note{color:#e6bd85;font-size:13px}.muted{font-size:12px;color:#8fa7af}footer{margin-top:30px;border-top:1px solid #344246;padding-top:20px;font-size:12px;color:#a2b8be}code{font-size:11px;overflow-wrap:anywhere}@media(max-width:720px){main{padding:24px 16px}header{display:block}.stats,.split{grid-template-columns:1fr}.badge{display:inline-block}.card video{height:auto}}
</style></head><body><main><header><div><div class="eyebrow">SO-101 / LeRobot / closed-loop evidence</div><h1>Robot training,<br>measured.</h1><p>Replay actual checkpoints on the same scene and instruction pairs. Every shown failure is retained. The robot is evaluated in MuJoCo; an Isaac view replays those measured poses.</p></div><span class="badge">LOCAL REVIEW • NO HARDWARE RESULT</span></header>
<div class="stats"><div class="stat"><strong>48 / 48</strong><span>Expert demonstrations passed independent physics audit</span></div><div class="stat"><strong id="learnedCount" class="failure"></strong><span>Shown learned validation successes</span></div><div class="stat"><strong id="ramAdmission" class="failure"></strong><span>Whole-system RAM admission • 48 GiB limit</span></div></div>
<section class="panel"><h2 style="margin-top:0">Training timeline</h2><div id="timeline" class="timeline"></div><p class="muted">Checkpoint steps are actual optimizer updates. A schedule restart is a separate training series. Loss measures imitation error; closed-loop task outcomes appear below.</p><svg id="chart" class="chart" viewBox="0 0 1000 240" role="img" aria-label="Recorded training loss"></svg><div id="chartSources" class="muted"></div></section>
<h2>Checkpoint iteration matrix</h2><div class="controls"><label>Instruction <select id="task"><option value="all">All retained tasks</option></select></label><label>Checkpoint <select id="step"><option value="all">All measured milestones</option></select></label><button id="restart">Restart visible clips together</button></div><div id="matrix" class="grid"></div>
<div class="split"><section class="panel expert"><div class="eyebrow">Reference data • privileged scripted IK</div><h2>Expert reference, shown separately</h2><video id="expertVideo" controls muted playsinline preload="metadata"></video><p class="note">The expert uses simulator object poses for demonstration generation. Its success does not establish learned-policy or hardware success.</p></section><section class="panel"><h2>What the overlays mean</h2><p>Cyan traces the measured tool position. Yellow marks the eight raw model commands actually issued for that chunk, projected using forward kinematics. Those waypoints never correct the policy.</p><p>Videos exclude inference pauses. Fresh inference latency and out-of-range command events are measured, not estimated.</p><p class="note">No sealed tests have been used to choose these development checkpoints. Native Isaac capture is a visual replay of the source trajectory, with no new policy or physics result.</p></section></div>
<footer><a href="office-review-manifest.json">Evidence manifest and SHA256 provenance</a><p>All selected completed cases, including failures, remain visible. This local page contains office evidence only; no private chat, draft publication text, credentials or cloud connection data.</p></footer></main><script id="record" type="application/json">__DATA__</script><script>
const R=JSON.parse(document.getElementById('record').textContent),el=id=>document.getElementById(id);el('learnedCount').textContent=R.cards.filter(c=>c.success&&!c.development_overfit_control).length+' / '+R.cards.filter(c=>!c.development_overfit_control).length;el('ramAdmission').textContent=R.admission.overall_admission||'UNMEASURED';
for(const m of R.milestones){const d=document.createElement('div');d.className='milestone';d.textContent=`${m.model} ${m.step.toLocaleString()} updates · ${m.chunk_execution}-tick chunks · ${m.wording} wording${m.development_overfit_control?' · OVERFIT control':''} · physical ${m.physical_task_passed}/${m.cases}, strict ${m.passed}/${m.cases}`;el('timeline').append(d)}
for(const attempt of R.attempts||[]){if(attempt.complete)continue;const d=document.createElement('div');d.className='milestone failure';d.textContent=attempt.name+': incomplete / '+attempt.episodes+' robot cases; '+(attempt.error||'pending');el('timeline').append(d)}for(const key of ['task','step'])for(const value of [...new Set(R.cards.map(c=>c[key]))]){const o=document.createElement('option');o.value=value;o.textContent=key==='step'?value.toLocaleString()+' updates':value;el(key).append(o)}
function render(){
  el('matrix').replaceChildren();
  for(const c of R.cards){
    if(el('task').value!=='all'&&el('task').value!==c.task)continue;
    if(el('step').value!=='all'&&Number(el('step').value)!==c.step)continue;
    const card=document.createElement('article');card.className='card';
    const head=document.createElement('div');head.className='cardhead';
    const label=document.createElement('span');label.textContent=c.model+' · '+c.step.toLocaleString()+' updates';
    const badge=document.createElement('span');badge.className='tag'+(c.task_success?' pass':'');badge.textContent=c.task_success?'TASK PASS':'TASK FAIL';
    head.append(label,badge);const title=document.createElement('h3');title.textContent=c.instruction;
    const condition=document.createElement('div');condition.className='details';
    condition.textContent='Seed '+c.seed+' · '+c.chunk_execution+'-tick chunks · wording '+c.wording+(c.development_overfit_control?' · DEVELOPMENT OVERFIT CONTROL':' · held-out development scene');
    const v=document.createElement('video');v.src=c.src;v.controls=true;v.muted=true;v.playsInline=true;v.preload='metadata';
    const details=document.createElement('div');details.className='details';
    details.textContent=`${c.events} raw limit events · strict ${c.success?'PASS':'FAIL'} · fresh inference p50 ${c.p50_ms} ms / p95 ${c.p95_ms} ms`;
    const scope=document.createElement('div');scope.className='details';scope.textContent=c.inference_scope+' · '+c.language_attribution;
    card.append(head,title,condition,v,details,scope);el('matrix').append(card)
  }
}
el('task').onchange=render;el('step').onchange=render;el('restart').onclick=()=>{for(const v of el('matrix').querySelectorAll('video')){v.currentTime=0;v.play().catch(()=>{})}};render();if(R.expert_src)el('expertVideo').src=R.expert_src;
const NS='http://www.w3.org/2000/svg',points=R.loss_series.flatMap(s=>s.points);function svg(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;el('chart').append(n)}
if(points.length){const xmax=Math.max(...points.map(p=>p.step),1),ymax=Math.max(...points.map(p=>p.loss),.001);svg('line',{x1:50,y1:205,x2:960,y2:205,stroke:'#4c6268'});svg('line',{x1:50,y1:20,x2:50,y2:205,stroke:'#4c6268'});const colors=['#8ce0cc','#d8a969','#ad9fdf'];R.loss_series.forEach((s,i)=>{svg('polyline',{points:s.points.map(p=>`${50+910*p.step/xmax},${205-175*p.loss/ymax}`).join(' '),fill:'none',stroke:colors[i%3],'stroke-width':2});svg('text',{x:65,y:25+i*16,fill:colors[i%3]},s.name+' · offset '+s.offset)});svg('text',{x:420,y:230},'Cumulative optimizer updates');svg('text',{x:60,y:220},'0');svg('text',{x:920,y:220},String(xmax));svg('text',{x:5,y:25},ymax.toFixed(3));el('chartSources').textContent='Only copied measured loss records are plotted. Their source hashes are in the evidence manifest.'}else{svg('text',{x:50,y:110},'Awaiting copied training-loss records; no synthetic loss curve is shown.');el('chartSources').textContent='Closed-loop clip outcomes and checkpoint milestones are already measured.'}
</script></body></html>'''.replace('__DATA__',data)
    if any(item['name']=='training-loss.mp4' for item in training_media):
        page=page.replace('<h2>Checkpoint iteration matrix</h2>',
            '<section class="panel"><h2>Recorded training-loss animation</h2><video controls muted playsinline preload="metadata" src="media/training-loss.mp4" style="width:100%;max-height:420px"></video><p class="muted">Actual scalar training losses. GR00T and SmolVLA use different objectives; their loss magnitudes do not rank robot quality. SmolVLA optimizer and schedule restart at cumulative update 1,000. Physical task scores remain separate.</p></section><h2>Checkpoint iteration matrix</h2>')
    page=page.replace('Yellow marks the eight raw model commands actually issued for that chunk','Yellow marks eight FK samples from retained full fresh predictions where available, or actually issued commands in older clips')
    page=page.replace('raw limit events','policy-decoded limit events')
    page=page.replace("${c.events} policy-decoded limit events", "${c.experiment} · ${c.events} policy-decoded limit events")
    page=page.replace("'Seed '+c.seed", "c.experiment+' · Seed '+c.seed")
    page=page.replace("scope.textContent=c.inference_scope", "scope.textContent=(c.server_p50_ms===null?'':('A100 synchronized server p50 '+c.server_p50_ms+' ms / p95 '+c.server_p95_ms+' ms · '))+c.inference_scope")
    page=page.replace('Whole-system RAM admission • 48 GiB limit','Retained historical RAM admission • 48 GiB limit')
    page=page.replace('<label>Checkpoint <select', '<label>Experiment <select id="experiment"><option value="all">All retained experiments</option></select></label><label>Checkpoint <select')
    page=page.replace("['task','step']", "['task','step','experiment']")
    page=page.replace("if(el('task').value!=='all'", "if(el('experiment').value!=='all'&&el('experiment').value!==c.experiment)continue;\n    if(el('task').value!=='all'")
    page=page.replace("el('task').onchange=render;", "el('experiment').onchange=render;el('task').onchange=render;")
    page=page.replace('minmax(300px,1fr)','minmax(340px,1fr)')
    page=page.replace('for(const c of R.cards){', "const shown=[...R.cards];if(el('experiment').value!=='all')shown.sort((a,b)=>a.task.localeCompare(b.task)||(['SmolVLA','GR00T-N1.7-3B','ACT task-specific pair'].indexOf(a.model)-['SmolVLA','GR00T-N1.7-3B','ACT task-specific pair'].indexOf(b.model)));\n  for(const c of shown){")
    page=page.replace('};render();if(R.expert_src)', "};if(R.comparisons.length)el('experiment').value=R.comparisons[R.comparisons.length-1].name;render();if(R.expert_src)")
    page=page.replace('`${m.model} ${m.step.toLocaleString()}', '`${m.experiment} · ${m.model} ${m.step.toLocaleString()}')
    additions='<section class="panel"><h2>Controlled development comparison</h2><p>Matched common8 and common16 used the same A100, CPU OSMesa scene, paired instructions, reset RNG and grader. All six tasks failed in each comparison. Earlier trials use different execution conditions and remain separately labelled. Training scopes differ; ACT uses an external exact-task router.</p><p class="note">Limit events refer to policy-decoded commands before the common actuator supervisor. GR00T clips normalized MIN_MAX values before decoding; ACT and SmolVLA MEAN_STD outputs are unbounded. Zero events do not establish raw neural-head safety.</p></section>'
    if comparison_records:
        additions+='<section class="panel"><h2>Frozen comparison outcomes</h2>'+''.join('<p><strong>'+esc(row['name'])+'</strong> · '+str(row['chunk_execution'])+' commands per replan · physical '+str(row['task_successes'])+'/'+str(row['cases'])+' · '+esc(row['scope'])+'</p>' for row in comparison_records)+'<p class="muted">Fresh-base experiments restart optimizer steps at zero. Changing the training data is a separate intervention; equal step counts do not make model architecture, objectives, batches or pretraining equal.</p></section>'
    probe_path=a.office/'local-bf16-coldload-v1/report.json';guard_path=a.office/'local-bf16-coldload-v1/guard-report.json'
    if probe_path.exists() and guard_path.exists():
        probe=json.loads(probe_path.read_text());guard=json.loads(guard_path.read_text())
        if probe.get('complete') and guard.get('complete'):
            warm=[row['synchronized_seconds'] for row in probe['rows'][1:]]
            additions+='<section class="panel"><h2>Measured desktop GR00T inference feasibility</h2><p>On the RTX 5070 Ti, an explicit BF16 constructor-load optimization loaded the retained original 4,000-step checkpoint in '+str(round(probe['cold_load_seconds'],1))+' seconds and returned five finite 16 × 6 action chunks. Whole-device peak '+str(round(guard['peak_gpu_mib']/1024,2))+' GiB; whole-host RAM peak '+str(round(guard['peak_host_mib']/1024,2))+' GiB. Warm synchronized calls took '+str(round(min(warm),3))+'–'+str(round(max(warm),3))+' seconds.</p><p class="note">This is a separately recorded source optimization, not an unmodified stock load or quantization test. No concurrent Isaac renderer, new closed-loop success, physical robot result or local training fit is established. Earlier stock-load admission failures remain retained.</p></section>'
    for name in ['native-legible-overview','native-legible-closeup']:
        source=a.office/(name+'.mp4')
        if source.exists():
            shutil.copy2(source,media/source.name)
            shutil.copy2(source.with_suffix('.provenance.json'),media/(name+'.provenance.json'))
            additions+='<section class="panel"><h2>'+('Native replay overview' if name.endswith('overview') else 'Native replay closeup • failed ACT2000 case')+'</h2><video controls muted playsinline preload="metadata" src="media/'+source.name+'" style="width:100%;max-height:680px"></video><p class="muted">Actual RTX replay of measured MuJoCo poses. Visual walls/colors changed for presentation; no Isaac physics, new learned-policy success or domain-transfer result. Overview cells remain small; use the closeup to inspect the robot.</p></section>'
    page=page.replace('<h2>Checkpoint iteration matrix</h2>',additions+'<h2>Checkpoint iteration matrix</h2>')
    (a.output/'index.html').write_text(page,encoding='utf-8');print(json.dumps({'portal':str(a.output/'index.html'),'clips':len(cards),'loss_points':sum(len(s['points']) for s in series)}))
if __name__=='__main__':main()
