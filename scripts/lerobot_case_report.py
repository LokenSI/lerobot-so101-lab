"""Build a local, self-contained viewer for measured LeRobot simulator cases.

Only reads case reports and existing videos; performs no inference or simulation.
Run again after the overlay renderer finishes to refresh available annotations.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/lerobot-cases'
CASE_LABELS = {'nominal':'Original scene','blue_target':'Blue target','dim_light':'Lower light','visual_distractor':'Visual distractor'}
DESCRIPTIONS = {
    'nominal':'Three new cube starts in the original scene.',
    'blue_target':'Target recoloured blue; geometry and layout unchanged.',
    'dim_light':'Simulator diffuse/ambient light scaled to 0.75; not a lux measurement.',
    'visual_distractor':'An extra blue cube appears in the image; its geometry has no physical collisions.',
}

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 · Same policy, different scenes</title>
<style>
:root{color-scheme:dark;--bg:#0a111b;--panel:#111e2d;--line:#293c50;--ink:#edf4fb;--muted:#b1bfd0;--cyan:#5fdfd1;--amber:#ffbf60;--red:#ff9a94}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(ellipse at 85% 0,#1b3444 0,transparent 45%),var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1320px;margin:auto;padding:38px 28px 55px}a{color:var(--cyan);text-underline-offset:4px}button,select{font:inherit;color:var(--ink);background:#192c3e;border:1px solid #456078;border-radius:8px;padding:9px 13px}button{cursor:pointer}button:hover{background:#244357}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid var(--amber);outline-offset:3px}button:disabled{cursor:default;opacity:.55}.kicker{color:var(--cyan);font-size:13px;letter-spacing:.17em;text-transform:uppercase}h1{max-width:950px;font-size:clamp(35px,5vw,64px);line-height:1.07;letter-spacing:-.035em;margin:15px 0 21px}h2{font-size:29px;line-height:1.25;margin:0 0 16px;letter-spacing:-.02em}h3{margin:0 0 10px;font-size:21px}.lead{max-width:880px;color:var(--muted);font-size:19px;line-height:1.7}.note{background:#192536;border-left:3px solid var(--amber);padding:18px 22px;border-radius:4px;margin:25px 0;color:#d3deeb}.note strong{color:var(--ink)}.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.tile{text-align:left;background:var(--panel);padding:18px 21px;border:1px solid var(--line);min-height:150px}.tile .number{font-size:36px;font-weight:700;line-height:1.2;margin:8px 0;color:var(--cyan)}.tile .label{font-weight:650}.tile .detail{font-size:14px;color:var(--muted)}.tile.active{border-color:var(--cyan)}section{margin:36px 0 0}.player-layout{display:grid;grid-template-columns:minmax(0,1fr) 315px;gap:22px}.screen{background:#000;border:1px solid var(--line);border-radius:12px;overflow:hidden}.screen video{display:block;width:100%;aspect-ratio:4/3;background:#000}.toolbar{display:flex;gap:13px;align-items:center;flex-wrap:wrap;margin:16px 0}.toolbar label{display:flex;align-items:center;gap:9px}.toolbar input{accent-color:var(--cyan);width:18px;height:18px}.sidebar{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:23px;height:fit-content}.badge{display:inline-block;border-radius:20px;padding:4px 12px;font-size:13px;font-weight:700;letter-spacing:.03em;background:#183e39;color:var(--cyan)}.badge.fail{background:#482c30;color:var(--red)}.metric{display:flex;justify-content:space-between;gap:14px;padding:10px 0;border-bottom:1px solid var(--line);font-size:14px}.metric span{color:var(--muted)}.small{font-size:14px;color:var(--muted)}.links{display:flex;gap:15px;flex-wrap:wrap;font-size:14px;margin-top:16px}.legend{display:flex;gap:22px;flex-wrap:wrap;margin:12px 0;color:var(--muted);font-size:14px}.legend i{display:inline-block;width:24px;height:4px;vertical-align:middle;margin-right:6px;border-radius:4px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:18px}.compare video{width:100%;display:block;background:#000;aspect-ratio:4/3;border:1px solid var(--line);border-radius:8px}.compare .caption{margin:10px 0;font-size:14px;color:var(--muted)}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;min-width:610px;font-size:15px}th,td{text-align:left;padding:15px 18px;border-bottom:1px solid var(--line)}th{background:var(--panel)}td:last-child{white-space:nowrap}tbody tr:hover{background:#152437}tbody tr:last-child td{border-bottom:0}.pass-text{color:var(--cyan)}.fail-text{color:var(--red)}.failure-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.failure{background:var(--panel);padding:23px;border:1px solid var(--line);border-radius:10px}.technical{margin-top:28px;border-top:1px solid var(--line);padding-top:22px}.technical code{font-size:12px;overflow-wrap:anywhere}footer{color:var(--muted);font-size:14px;padding-top:24px}summary{cursor:pointer;font-weight:650}details>div{padding-top:15px}.reference{display:none}.reference.ready{display:block}@media(max-width:900px){.player-layout{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}.sidebar{display:grid;grid-template-columns:1fr 1fr;gap:14px}.sidebar .intro{grid-row:span 2}.sidebar .links{grid-column:1/-1}}@media(max-width:600px){main{padding:26px 16px}.compare,.failure-grid{grid-template-columns:1fr}.sidebar{display:block}.tile{padding:15px}.tile .number{font-size:30px}.lead{font-size:17px}.toolbar{align-items:flex-start}.toolbar label{font-size:14px}h2{font-size:25px}}
</style></head>
<body><main>
<div class="kicker">SO-101 · LeRobot ACT · simulator exploration</div>
<h1>Same policy.<br>Different scenes.</h1>
<p class="lead">A frozen learned policy, three new cube starts and four visual conditions. Watch the predicted joint-target path beside the motion that actually happened.</p>
<div class="note"><strong>10/12 successful rollouts in this exploration.</strong> These are <strong>three paired starts × four conditions</strong>, not 12 independent starts. The earlier untouched assessment remains <strong>2/5</strong>; this exploration does not replace it. All results are simulation.</div>
<div class="tiles" id="tiles"></div>
<section aria-labelledby="viewer-title"><h2 id="viewer-title">Watch a rollout</h2>
<div class="toolbar"><label>Condition <select id="condition"></select></label><label>Start <select id="seed"></select></label><label><input type="checkbox" id="overlay" checked>Annotated replay</label><span class="small" id="availability"></span></div>
<div class="player-layout"><div><div class="screen"><video id="main-video" controls playsinline preload="metadata"></video></div><div class="legend"><span><i style="background:var(--amber)"></i>FK projection of 16 predicted joint targets</span><span><i style="background:var(--cyan)"></i>Actual TCP trail</span></div><p class="small">The amber path refreshes every eight control ticks (0.267 simulated seconds). It is a kinematic projection of the ACT chunk, not a validated future trajectory or object forecast. Overlays replay logged simulator states and real policy RGB inputs; this is not a live hardware feed.</p></div>
<aside class="sidebar"><div class="intro"><span id="result-badge" class="badge"></span><h3 id="episode-title" style="margin-top:16px"></h3><p id="condition-detail" class="small"></p><p id="failure-detail"></p></div><div id="metrics"></div><div class="links" id="episode-links"></div></aside></div></section>
<section aria-labelledby="comparison-title"><h2 id="comparison-title">One start. A different ending.</h2><p class="lead" style="font-size:17px">Seed 401 succeeds in the original scene. With simulator lighting scaled to 0.75, it lifts the cube but fails to release and settle it. Same initial object position, same frozen policy.</p>
<div class="toolbar"><button id="play-both">Play both</button><button id="pause-both">Pause both</button><button id="restart-both">Restart both</button><span class="small">Playback and seeking are synchronized.</span></div>
<div class="compare"><div><video id="compare-normal" controls playsinline preload="metadata"></video><p class="caption"><span class="pass-text">PASS</span> · Original scene · Seed 401</p></div><div><video id="compare-dim" controls playsinline preload="metadata"></video><p class="caption"><span class="fail-text">FAIL</span> · Lower light · Seed 401</p></div></div></section>
<section aria-labelledby="all-title"><h2 id="all-title">All 12 rollouts</h2><p class="small">Each row uses the same seed across conditions. Open any episode in the viewer, including both failures. Pass means lift above 9 cm, full tray containment, release and stable rest over the final 45 control frames.</p><div class="table-wrap"><table><thead><tr><th>Condition</th><th>Seed</th><th>Result</th><th>Evidence</th></tr></thead><tbody id="episode-table"></tbody></table></div></section>
<section aria-labelledby="failures-title"><h2 id="failures-title">What the failures tell us</h2><div class="failure-grid"><article class="failure"><h3>Lower light · Seed 401</h3><p>The cube is lifted but remains held and moving at the end. It fails containment, release and stability across the final grading window.</p><button data-open="dim_light:401">Watch this failure</button></article><article class="failure"><h3>Visual distractor · Seed 402</h3><p>The cube ends in the tray. That final frame is insufficient: it fails the required 1.5-second containment, release and stability window. This is a completion/timing failure, not evidence of a wrong final placement.</p><button data-open="visual_distractor:402">Watch this failure</button></article></div></section>
<section><h2>What to test next</h2><p class="lead" style="font-size:17px">This gives us concrete camera and task-completion questions for the physical SO-101: lighting variation, release confirmation and a measurable definition of “done.” After cameras arrive, those checks need real observations and a separate physical validation. Edge-compute integration remains a future experiment.</p></section>
<section class="technical"><details><summary>Evidence, scope and reproducibility</summary><div><p class="small">Official LeRobot ACT; one frozen 8,000-step checkpoint; RGB from two 96 × 96 cameras plus six joint states. No object coordinates or phase counters enter the policy. Established simulator actuator clipping remains active. No new training or tuning occurred for this suite.</p><p class="small">Blue target changes colour only. Lower light scales simulator diffuse/ambient values by 0.75; it is not a calibrated 25% lux reduction. The extra blue distractor is render-only and does not collide with the robot or target. Cameras, task geometry and strict grader are otherwise shared.</p><p class="small">Model SHA-256: <code id="model-hash"></code><br>Strict baseline/grader SHA-256: <code id="grader-hash"></code></p><div class="links"><a href="report.json">Suite report</a><a href="telemetry-validation.json">Telemetry validation</a><a href="overlays/report.json">Overlay manifest</a><a href="artifact-verification.json">Artifact verification</a><a href="overlays/tcp-consistency-validation.json">TCP consistency</a><a href="findings.md">Findings</a><a href="../lerobot-teaser/experiment-roles.json">Earlier experiment roles</a><a href="../robotics-teasers/verification.md">Independent review</a></div><p id="reference" class="small reference"></p></div></details></section>
<footer>MuJoCo replay · desktop RTX 5070 Ti · no physical robot, Orin, Thor, ROS 2 or Omniverse execution. Successful simulator clips demonstrate this fixed task; broad visual understanding and hardware transfer remain untested.</footer>
</main><script id="study-data" type="application/json">__DATA__</script>
<script>
const study=JSON.parse(document.getElementById('study-data').textContent);
const byId=id=>document.getElementById(id), esc=v=>String(v).replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const condition=byId('condition'), seed=byId('seed'), toggle=byId('overlay'), video=byId('main-video');
study.cases.forEach(c=>{condition.add(new Option(c.label,c.id)); const t=study.case_totals[c.id];const tile=document.createElement('button');tile.className='tile';tile.dataset.case=c.id;tile.innerHTML=`<div class="label">${esc(c.label)}</div><div class="number">${t.successful}/${t.total}</div><div class="detail">${esc(c.short)}</div>`;tile.addEventListener('click',()=>openEpisode(c.id,401));byId('tiles').append(tile)});
function fillSeeds(preferred=401){seed.replaceChildren();study.episodes.filter(e=>e.case===condition.value).forEach(e=>seed.add(new Option(`${e.seed} · ${e.success?'pass':'fail'}`,e.seed)));seed.value=String(preferred);if(!seed.value)seed.selectedIndex=0}
function selected(){return study.episodes.find(e=>e.case===condition.value&&e.seed===Number(seed.value))}
function render(){const e=selected(), c=study.cases.find(c=>c.id===e.case), o=e.overlay;toggle.disabled=!o;toggle.checked=!!o&&toggle.checked;byId('availability').textContent=o?'':'Raw replay available; this episode has no annotated version.';video.pause();video.src=toggle.checked?o.video:e.raw_video;video.poster=toggle.checked?o.poster:'';video.load();byId('result-badge').className=`badge ${e.success?'':'fail'}`;byId('result-badge').textContent=e.success?'PASS':'FAIL';byId('episode-title').textContent=`${c.label} · Seed ${e.seed}`;byId('condition-detail').textContent=c.detail;byId('failure-detail').textContent=e.failure_context||'Lift, transfer, release and the full rest window passed the same strict grader.';const g=e.grader;byId('metrics').innerHTML=[['Cube lifted',g.lifted],['Full containment window',g.inside_tray_for_1_5_s],['Release window',g.released_for_1_5_s],['Stable rest',g.stable]].map(([k,v])=>`<div class="metric"><span>${k}</span><b class="${v?'pass-text':'fail-text'}">${v?'Pass':'Fail'}</b></div>`).join('');byId('episode-links').innerHTML=`<a href="${esc(e.report)}">Report</a><a href="${esc(e.trajectory)}">Telemetry</a><a href="${esc(e.raw_video)}">Raw video</a>${o?`<a href="${esc(o.provenance)}">Overlay provenance</a>`:''}`;document.querySelectorAll('.tile').forEach(t=>t.classList.toggle('active',t.dataset.case===e.case))}
function openEpisode(c,s){condition.value=c;fillSeeds(s);toggle.checked=!!selected().overlay;render();byId('viewer-title').scrollIntoView({behavior:'smooth',block:'start'})}
condition.addEventListener('change',()=>{fillSeeds(Number(seed.value)||401);toggle.checked=!!selected().overlay;render()});seed.addEventListener('change',()=>{toggle.checked=!!selected().overlay;render()});toggle.addEventListener('change',render);
study.episodes.forEach(e=>{const row=document.createElement('tr');row.innerHTML=`<td>${esc(study.cases.find(c=>c.id===e.case).label)}</td><td>${e.seed}</td><td class="${e.success?'pass-text':'fail-text'}">${e.success?'Pass':'Fail'}</td><td><button data-open="${e.case}:${e.seed}">Watch</button> &nbsp; <a href="${esc(e.report)}">Report</a></td>`;byId('episode-table').append(row)});
document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',()=>{const[c,s]=b.dataset.open.split(':');openEpisode(c,Number(s))}));
condition.value='nominal';fillSeeds(401);toggle.checked=!!selected().overlay;render();
const normal=study.episodes.find(e=>e.case==='nominal'&&e.seed===401), dim=study.episodes.find(e=>e.case==='dim_light'&&e.seed===401);const useOverlays=!!(normal.overlay&&dim.overlay);const players=[byId('compare-normal'),byId('compare-dim')];[normal,dim].forEach((e,i)=>{players[i].src=useOverlays?e.overlay.video:e.raw_video;players[i].poster=useOverlays?e.overlay.poster:''});
players.forEach((p,i)=>{const other=players[1-i];p.addEventListener('play',()=>{if(other.paused)other.play().catch(()=>{})});p.addEventListener('pause',()=>{if(!other.paused)other.pause()});p.addEventListener('seeked',()=>{if(Math.abs(other.currentTime-p.currentTime)>.05)other.currentTime=p.currentTime});p.addEventListener('ratechange',()=>{if(other.playbackRate!==p.playbackRate)other.playbackRate=p.playbackRate})});
byId('play-both').addEventListener('click',()=>players.forEach(p=>p.play().catch(()=>{})));byId('pause-both').addEventListener('click',()=>players.forEach(p=>p.pause()));byId('restart-both').addEventListener('click',()=>players.forEach(p=>{p.pause();p.currentTime=0}));
players[0].addEventListener('timeupdate',()=>{if(!players[0].paused&&!players[1].paused&&Math.abs(players[0].currentTime-players[1].currentTime)>.18)players[1].currentTime=players[0].currentTime});
byId('model-hash').textContent=study.checkpoint_sha256;byId('grader-hash').textContent=study.baseline_and_grader_sha256;
if(study.reference){const r=byId('reference');r.classList.add('ready');r.innerHTML=`Previously successful seed 301 is available as a <a href="${esc(study.reference.video)}">reference replay</a>, with <a href="${esc(study.reference.provenance)}">provenance</a>. It is excluded from the 12 exploratory rollouts.`}
</script></body></html>'''

def existing(relative: str) -> str:
    path=(OUT/relative).resolve()
    assert path.is_relative_to(OUT.resolve()), relative
    assert path.is_file(), f'Missing linked asset: {relative}'
    return relative.replace('\\','/')

def main():
    source=json.loads((OUT/'report.json').read_text())
    assert source['complete'] and len(source['episodes'])==12
    assert source['success_count']==sum(e['success'] for e in source['episodes'])
    manifest=OUT/'overlays/report.json'
    clips=json.loads(manifest.read_text()).get('clips',[]) if manifest.exists() else []
    overlays={}
    reference=None
    for clip in clips:
        if not clip.get('complete',False):continue
        overlay={key:existing('overlays/'+clip[key]) for key in ('video','poster','provenance')}
        if clip.get('reference_excluded_from_fresh_count'):reference=overlay
        else:overlays[(clip['case'],clip['seed'])]=overlay
    episodes=[]
    for e in source['episodes']:
        path=e['episodepath']
        failure=''
        if e['case']=='dim_light' and e['seed']==401:failure='The cube was lifted but remains held and moving. The full containment, release and stability windows fail.'
        if e['case']=='visual_distractor' and e['seed']==402:failure='The cube ends in the tray, but fails the required final 1.5-second containment, release and stability window.'
        episodes.append({'case':e['case'],'seed':e['seed'],'success':e['success'],'grader':e['grader'],'failure_context':failure,'raw_video':existing(path+'/rollout.mp4'),'report':existing(path+'/report.json'),'trajectory':existing(path+'/trajectory.npz'),'overlay':overlays.get((e['case'],e['seed']))})
    labels=[{'id':k,'label':CASE_LABELS[k],'detail':DESCRIPTIONS[k],'short':{'nominal':'New starts','blue_target':'Colour only','dim_light':'Light scale 0.75','visual_distractor':'Render-only extra cube'}[k]} for k in CASE_LABELS]
    data={'cases':labels,'case_totals':source['case_totals'],'episodes':episodes,'reference':reference,'checkpoint_sha256':source['checkpoint_sha256'],'baseline_and_grader_sha256':source['baseline_and_grader_sha256']}
    packed=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    (OUT/'index.html').write_text(HTML.replace('__DATA__',packed),encoding='utf-8')
    findings='''# Same learned policy, four visual conditions

The frozen local LeRobot ACT checkpoint completed **10 of 12 exploratory MuJoCo rollouts**. This is **three paired cube starts (seeds 400–402) across four conditions**, not 12 independent starts. No model training or tuning occurred during this suite. The prior untouched benchmark remains **2/5** and is not superseded by these exploratory cases.

| Condition | Passed | Change |
|---|---:|---|
| Original scene | 3/3 | New initial cube positions |
| Blue target | 3/3 | Target colour only; same layout |
| Lower light | 2/3 | Diffuse/ambient simulator light scaled to 0.75; not calibrated lux |
| Visual distractor | 2/3 | Extra blue cube rendered without physical collisions |

All cases use the same strict grader: lift the cube above 9 cm, contain it wholly inside the tray, release it and remain stable through the last 45 control frames. The scene and wrist RGB inputs are 96 × 96, with six measured joints; object coordinates and task phases do not enter the learned policy. Existing simulator actuator limits remain active.

**Lower light, seed 401:** the cube lifts but remains held and moving; the complete containment/release/rest window fails. **Visual distractor, seed 402:** the cube ends in the tray, but fails the required final 1.5-second window. A successful-looking final frame is not enough to pass.

The annotated replay overlays amber forward-kinematics projections of the 16 predicted ACT joint targets and the cyan actual TCP trail. The prediction refreshes every eight executed ticks (0.267 simulated seconds). It is a kinematic projection, not a validated dynamic forecast or predicted object trajectory. Insets show the actual logged policy RGB inputs; annotated videos replay simulator telemetry rather than a live hardware stream. Some episodes have raw video only. Seed 301 is a previous successful reference, excluded from the 12-rollout count.

This fixed-task exploration identifies useful physical tests after the cameras arrive: lighting changes, colour changes, scene distraction and reliable release confirmation. It does not establish general visual understanding, physical transfer or production reliability. Physical SO-101 transfer and Orin inference remain untested.

Evidence: [interactive viewer](index.html), [suite report](report.json), [telemetry validation](telemetry-validation.json), [overlay manifest](overlays/report.json), [artifact verification](artifact-verification.json), [TCP consistency validation](overlays/tcp-consistency-validation.json), [earlier experiment roles](../lerobot-teaser/experiment-roles.json).
'''
    (OUT/'findings.md').write_text(findings,encoding='utf-8')
    print(json.dumps({'episodes':len(episodes),'successes':source['success_count'],'annotated_cases':len(overlays),'reference_available':bool(reference),'index':str(OUT/'index.html')},indent=2))

if __name__=='__main__':main()
