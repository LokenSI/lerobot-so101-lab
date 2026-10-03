
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
