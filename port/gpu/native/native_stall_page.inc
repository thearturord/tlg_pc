R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stall report</title>
<style>
:root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa3ad;--card:#1d2127;--line:#2c323a;--blue:#4da3ff;--red:#ff5c5c;--green:#3ecf8e;--orange:#ffb347;--purple:#c792ea;--teal:#4fd1c5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px;font-weight:600}
.sub{color:var(--mut);margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:22px}
.card span{color:var(--mut);font-size:12px}
.verdict{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--orange);border-radius:6px;padding:10px 14px;margin:14px 0;font-size:15px}
.bars{display:grid;gap:6px;margin-top:8px}
.bar{display:grid;grid-template-columns:170px 1fr 60px;gap:8px;align-items:center}
.bar i{display:block;height:14px;border-radius:3px;background:var(--blue)}
.track{background:#0f1215;border-radius:3px}
canvas{width:100%;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0;color:var(--mut);font-size:13px}
.legend label{cursor:pointer;user-select:none}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:5px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500;font-size:12px}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.45 ui-monospace,Consolas,monospace}
#tip{position:fixed;pointer-events:none;background:#000d;color:#fff;border:1px solid #444;border-radius:5px;padding:4px 8px;font-size:12px;display:none;z-index:5}
.up{color:var(--red)}
.note{color:var(--mut);font-size:12px;margin-top:6px}
</style></head><body><main>
<h1>Stall report</h1>
<div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>
<div class="verdict" id="verdict"></div>
<h2>Where a normal frame goes</h2>
<div class="bars" id="shares"></div>
<div class="note">Each bar is the share of a normal frame that thread (or the GPU) is busy. The frame thread's bar is its waiting.</div>
<h2>Frame time, every frame (click a frame)</h2>
<canvas id="c1" data-h="260"></canvas>
<h2>What each part does, per frame</h2>
<div class="legend" id="legend"></div>
<canvas id="c2" data-h="260"></canvas>
<h2>Selected frame</h2>
<div class="card"><pre id="detail">Click a bar in the first chart.</pre></div>
<h2>What the frame thread waits on (ms per frame)</h2>
<table id="t_waits"></table>
<h2>Why it waits: the longest wait of every frame, followed to what released it</h2>
<table id="t_chains"></table>
<h2>Our threads and the GPU (ms per frame)</h2>
<table id="t_metrics"></table>
<h2>The worst frames</h2>
<table id="t_worst"></table>
<h2>Every game thread wait (idle workers included)</h2>
<table id="t_all"></table>
<div class="note">The capture itself slows the game a little (GPU timestamps), so compare shapes, not absolute fps. Call sites are Ghidra addresses of the game code that made the blocking call; "host" means inside the runtime.</div>
<div id="tip"></div>
</main>
<script>
const D = /*DATA*/null;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f1 = x => (Math.round(x*10)/10).toFixed(1), f2 = x => (Math.round(x*100)/100).toFixed(2);
$('sub').textContent = `${D.seconds.toFixed(1)} s, ${D.frame_count} frames from ${D.frame_source}. Frame thread: ${D.frame_thread}.`;
$('cards').innerHTML = [
  [f1(1000/D.median_ms)+' fps', 'median frame '+f2(D.median_ms)+' ms'],
  [f1(D.p99_ms)+' ms', '99th percentile (worst '+f1(D.max_ms)+' ms)'],
  [D.slow_frames+' ('+f1(100*D.slow_frames/D.frame_count)+' %)', 'slow frames (> '+f1(D.slow_threshold_ms)+' ms)'],
  [f1(D.normal_mean)+' → '+f1(D.slow_mean)+' ms', 'mean normal → slow frame'],
].map(c => `<div class="card"><b>${c[0]}</b><span>${c[1]}</span></div>`).join('');
$('verdict').textContent = D.limiter;
const maxShare = Math.max(100, ...D.shares.map(s => s.pct));
$('shares').innerHTML = D.shares.map(s => `<div class="bar"><span>${esc(s.name)}</span><div class="track"><i style="width:${Math.min(100,s.pct/maxShare*100)}%;background:${s.pct>70?'var(--orange)':'var(--blue)'}"></i></div><span>${s.pct.toFixed(0)} %</span></div>`).join('');

// frames: [t, ms, slow, gpu, raster, front, encoder, frameThreadWait]
const F = D.frames, N = F.length;
let sel = -1;
function setup(c) {
  const dpr = window.devicePixelRatio || 1, h = +c.dataset.h;
  c.style.height = h + 'px'; const w = c.clientWidth;
  c.width = w*dpr; c.height = h*dpr;
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  return {g, w, h};
}
const PAD = {l:44, r:10, t:10, b:22};
function axes(g, w, h, maxY, unit) {
  g.strokeStyle = '#2c323a'; g.fillStyle = '#9aa3ad'; g.font = '11px system-ui'; g.lineWidth = 1;
  const steps = 5;
  for (let i = 0; i <= steps; i++) {
    const y = PAD.t + (h-PAD.t-PAD.b)*(1-i/steps);
    g.beginPath(); g.moveTo(PAD.l, y); g.lineTo(w-PAD.r, y); g.stroke();
    g.fillText((maxY*i/steps).toFixed(maxY<10?1:0)+unit, 4, y+3);
  }
  const t1 = F[N-1][0];
  for (let i = 0; i <= 5; i++) g.fillText((t1*i/5).toFixed(1)+' s', PAD.l + (w-PAD.l-PAD.r)*i/5 - 12, h-6);
}
const xOf = (i, w) => PAD.l + (w-PAD.l-PAD.r) * (i+0.5)/N;
function draw1() {
  const c = $('c1'), {g, w, h} = setup(c);
  g.clearRect(0,0,w,h);
  const maxY = Math.max(D.max_ms*1.08, D.slow_threshold_ms*1.2);
  axes(g, w, h, maxY, '');
  const bw = Math.max(1, (w-PAD.l-PAD.r)/N - 0.5), yOf = v => PAD.t + (h-PAD.t-PAD.b)*(1-v/maxY);
  for (let i = 0; i < N; i++) {
    g.fillStyle = i === sel ? '#ffffff' : (F[i][2] ? '#ff5c5c' : '#4da3ff');
    const x = xOf(i, w) - bw/2, y = yOf(F[i][1]);
    g.fillRect(x, y, bw, h-PAD.b-y);
  }
  g.setLineDash([5,4]); g.lineWidth = 1;
  g.strokeStyle = '#ffb347'; g.beginPath(); g.moveTo(PAD.l, yOf(D.slow_threshold_ms)); g.lineTo(w-PAD.r, yOf(D.slow_threshold_ms)); g.stroke();
  g.strokeStyle = '#3ecf8e'; g.beginPath(); g.moveTo(PAD.l, yOf(D.median_ms)); g.lineTo(w-PAD.r, yOf(D.median_ms)); g.stroke();
  g.setLineDash([]); g.fillStyle = '#ffb347'; g.fillText('slow limit', w-PAD.r-60, yOf(D.slow_threshold_ms)-3);
  g.fillStyle = '#3ecf8e'; g.fillText('median', w-PAD.r-44, yOf(D.median_ms)-3);
  g.fillStyle = '#9aa3ad'; g.fillText('ms', 4, 10);
}
const series = [
  {name:'frame thread waiting', idx:7, color:'#c792ea', on:true},
  {name:'raster thread working', idx:4, color:'#ff5c5c', on:true},
  {name:'front thread working', idx:5, color:'#ffb347', on:true},
  {name:'GPU busy', idx:3, color:'#3ecf8e', on:true},
  {name:'encoder working', idx:6, color:'#4fd1c5', on:false},
  {name:'frame time', idx:1, color:'#e6e8eb', on:true},
];
$('legend').innerHTML = series.map((s,i) => `<label><input type="checkbox" data-i="${i}" ${s.on?'checked':''}> <span class="sw" style="background:${s.color}"></span>${s.name}</label>`).join('');
$('legend').addEventListener('change', e => { series[e.target.dataset.i].on = e.target.checked; draw2(); });
function draw2() {
  const c = $('c2'), {g, w, h} = setup(c);
  g.clearRect(0,0,w,h);
  let maxY = 1;
  for (const s of series) if (s.on) for (let i = 0; i < N; i++) maxY = Math.max(maxY, F[i][s.idx]);
  maxY *= 1.08;
  axes(g, w, h, maxY, '');
  const yOf = v => PAD.t + (h-PAD.t-PAD.b)*(1-v/maxY);
  g.lineWidth = 1.4;
  for (const s of series) {
    if (!s.on) continue;
    g.strokeStyle = s.color; g.beginPath();
    for (let i = 0; i < N; i++) { const x = xOf(i, w), y = yOf(F[i][s.idx]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
  }
  if (sel >= 0) { g.strokeStyle = '#fff8'; g.beginPath(); g.moveTo(xOf(sel, w), PAD.t); g.lineTo(xOf(sel, w), h-PAD.b); g.stroke(); }
}
function detail(i) {
  const r = F[i];
  $('detail').textContent =
`frame ${i} at +${r[0].toFixed(2)} s: ${r[1].toFixed(2)} ms ${r[2] ? '(SLOW)' : ''}
  frame thread waiting   ${r[7].toFixed(2)} ms
  raster thread working  ${r[4].toFixed(2)} ms
  front thread working   ${r[5].toFixed(2)} ms
  GPU busy               ${r[3].toFixed(2)} ms
  encoder working        ${r[6].toFixed(2)} ms

longest wait of the frame thread, and what released it:
${D.frame_chains[i] || '(the frame thread did not wait in this frame)'}`;
}
function pick(ev, c) {
  const rect = c.getBoundingClientRect(), x = ev.clientX - rect.left;
  const i = Math.floor((x-PAD.l) / (rect.width-PAD.l-PAD.r) * N);
  return i >= 0 && i < N ? i : -1;
}
const tip = $('tip');
for (const id of ['c1','c2']) {
  const c = $(id);
  c.addEventListener('mousemove', ev => {
    const i = pick(ev, c);
    if (i < 0) { tip.style.display = 'none'; return; }
    const r = F[i];
    tip.style.display = 'block'; tip.style.left = (ev.clientX+14)+'px'; tip.style.top = (ev.clientY+10)+'px';
    tip.textContent = `+${r[0].toFixed(2)} s  ${r[1].toFixed(1)} ms  raster ${r[4].toFixed(1)}  front ${r[5].toFixed(1)}  GPU ${r[3].toFixed(1)}  waits ${r[7].toFixed(1)}`;
  });
  c.addEventListener('mouseleave', () => tip.style.display = 'none');
  c.addEventListener('click', ev => { const i = pick(ev, c); if (i >= 0) { sel = i; detail(i); draw1(); draw2(); } });
}
function table(id, head, rows) {
  $(id).innerHTML = '<tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr>' + rows.map(r => '<tr>' + r.map((c,i) => `<td class="${typeof c === 'number' ? 'n' : ''}">${typeof c === 'number' ? c.toFixed(2) : esc(c)}</td>`).join('') + '</tr>').join('');
}
table('t_waits', ['what', 'normal', 'slow'], D.frame_thread_waits.map(w => [w.what, w.normal, w.slow]));
$('t_chains').innerHTML = '<tr><th>normal frames</th><th>slow frames</th><th>chain</th></tr>' + D.why_it_waits.map(c => `<tr><td class="n">${c.normal}</td><td class="n">${c.slow}</td><td><pre>${esc(c.chain)}</pre></td></tr>`).join('');
table('t_metrics', ['what', 'normal', 'slow'], D.metrics.map(m => [m.what, m.normal, m.slow]));
$('t_worst').innerHTML = '<tr><th>frame</th><th>ms</th><th>chain</th></tr>' + D.worst.map(i => `<tr><td class="n"><a href="#" onclick="sel=${i};detail(${i});draw1();draw2();window.scrollTo(0,document.getElementById('detail').offsetTop-200);return false" style="color:var(--blue)">+${F[i][0].toFixed(2)} s</a></td><td class="n">${F[i][1].toFixed(1)}</td><td><pre>${esc(D.frame_chains[i] || '(no wait)')}</pre></td></tr>`).join('');
table('t_all', ['thread / wait', 'normal', 'slow'], D.all_waits.map(w => [w.what, w.normal, w.slow]));
function drawAll() { draw1(); draw2(); }
window.addEventListener('resize', drawAll);
drawAll();
if (D.worst.length) { sel = D.worst[0]; detail(sel); drawAll(); }
</script></body></html>)HTML"
