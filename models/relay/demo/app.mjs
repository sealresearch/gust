// Relay sandbox page: environments, tool editor, Relay in a worker, result views and the act / confirm / hold flow.
import {buildRecord, nowISO} from './schema.mjs';

const ENV_KEYS = ['phone', 'home', 'vacuum', 'watch', 'glasses', 'car', 'laptop', 'search', 'extract', 'bench'];
const $ = id => document.getElementById(id);
const COLORS = ['#ff6b6b', '#4dabf7', '#69db7c', '#ffd43b', '#da77f2', '#ff922b', '#38d9a9', '#f783ac'];
const NAMED = {red: '#ff4d4f', blue: '#4dabf7', green: '#3ecf8e', purple: '#b197fc', orange: '#ff922b', pink: '#f783ac',
  yellow: '#ffd43b', teal: '#38d9a9', white: '#f2f2f4', cyan: '#3bc9db', gold: '#fcc419'};
const ACT_MARGIN = 2;  // router margin at which a call runs without asking (Relay's own second-call gate)
const OPT_SURE = 0.6;   // a required option picked with less probability than this makes the page ask (audit: catches
                        // "left side" -> all windows; also asks on genuinely ambiguous "open the windows")
const TRIGGER_FLOOR = -1;  // a matched trigger does not force a call Relay scores below this
const EXTRACT_FLOOR = 0;   // extract: no schema fits below this router margin (audit AUC 0.995)

const envs = {};
let env = null, state = null, worker = null, ready = false, busy = false, runId = 0, lastToolsText = '';
let pending = null, lastResult = null, tab = 'calls', benchRows = null;

// ------------------------------------------------------------------ engine
function startWorker(threads) {
  ready = false;
  setBusy(false);
  if (worker) worker.terminate();
  worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
  status(`loading Relay (${threads} thread${threads > 1 ? 's' : ''})…`);
  worker.onmessage = ({data}) => {
    if (data.type === 'ready') {
      ready = true;
      const i = data.info;
      status(`ready · ${(i.loadMs / 1000).toFixed(1)} s · ${i.threads} thread${i.threads > 1 ? 's' : ''}`, 'ok');
      $('loadInfo').textContent = `model ${(i.modelBytes / 1e6).toFixed(2)} MB + runtime ${(i.wasmBytes / 1e3).toFixed(0)} KB${i.isolated ? '' : ' · not cross-origin isolated: single thread'}`;
      setBusy(false);
    } else if (data.type === 'result' || data.type === 'error') {
      if (data.type === 'error' && data.id == null) {
        status(`engine error: ${data.error.split('\n')[0]}`, 'err');
        return;
      }
      const cb = calls.get(data.id);
      calls.delete(data.id);
      if (cb) data.type === 'error' ? cb.reject(new Error(data.error)) : cb.resolve(data);
    }
  };
  worker.onerror = e => status(`engine error: ${e.message}`, 'err');
  worker.postMessage({type: 'init', threads});
}
const calls = new Map();
function relay(record) {
  return new Promise((resolve, reject) => {
    const id = ++runId;
    calls.set(id, {resolve, reject});
    worker.postMessage({type: 'run', id, record});
  });
}

// ------------------------------------------------------------------ ui helpers
function status(t, cls = '') { const b = $('status'); b.textContent = t; b.className = `badge ${cls}`; }
function setBusy(b) { busy = b; $('run').disabled = b || !ready; document.querySelectorAll('#chips button').forEach(x => { x.disabled = b || !ready; }); }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
function jsonHTML(v, ind = '') {
  if (Array.isArray(v)) return v.length ? `[\n${v.map(x => ind + '  ' + jsonHTML(x, ind + '  ')).join(',\n')}\n${ind}]` : '[]';
  if (v && typeof v === 'object') {
    const e = Object.entries(v);
    return e.length ? `{\n${e.map(([k, x]) => `${ind}  <span class="k">"${esc(k)}"</span>: ${jsonHTML(x, ind + '  ')}`).join(',\n')}\n${ind}}` : '{}';
  }
  if (typeof v === 'string') return `<span class="s">${esc(JSON.stringify(v))}</span>`;
  return `<span class="n">${esc(JSON.stringify(v))}</span>`;
}
const callText = c => `${c.tool}(${Object.entries(c.args || {}).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')})`;
const sleep = ms => new Promise(r => setTimeout(r, ms));
function setAccent(c) {
  const v = NAMED[String(c).toLowerCase().trim()] || (/^#[0-9a-f]{3,8}$/i.test(c) ? c : null);
  if (v) document.documentElement.style.setProperty('--acc', v);
}
const ctx = {
  render: () => drawStage(), sleep, setAccent,
  speak: t => { try { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(String(t))); } catch {} },
  toast: t => { $('envNote').textContent = t; },
  openUrl: u => { const url = /^https?:/.test(u) ? u : `https://${u}`; const w = window.open(url, '_blank', 'noopener'); return w ? 'opened' : 'popup blocked'; },
  mailto: href => { const a = document.createElement('a'); a.href = href; a.click(); },
  clipboard: t => navigator.clipboard.writeText(String(t)).then(() => 'copied', () => 'clipboard blocked'),
};

// ------------------------------------------------------------------ environments
async function loadEnvs() {
  for (const k of ENV_KEYS) {
    try { envs[k] = (await import(`./envs/${k}.mjs`)).default; } catch (e) { console.warn(`env ${k} not available`, e); }
  }
  const list = $('envList');
  list.innerHTML = '';
  for (const k of ENV_KEYS) if (envs[k]) {
    const b = document.createElement('button');
    b.innerHTML = `${esc(envs[k].label)}<small>${esc(envs[k].subtitle || '')}</small>`;
    b.onclick = () => { $('menu').classList.remove('open'); selectEnv(k); };
    b.dataset.k = k;
    list.appendChild(b);
  }
}

function selectEnv(k) {
  env = envs[k];
  location.hash = `env=${k}`;
  state = env.init ? env.init() : {};
  document.documentElement.style.setProperty('--acc', '#ff4d4f');
  $('envBtn').textContent = `${env.label} ▾`;
  document.querySelectorAll('#envList button').forEach(b => b.classList.toggle('cur', b.dataset.k === k));
  $('tagline').textContent = env.tagline || env.subtitle || '';
  $('q').placeholder = env.placeholder || 'ask something...';
  $('q').value = '';
  $('tools').value = JSON.stringify(env.tools || [], null, 2);
  $('tools').readOnly = !!env.custom;
  lastToolsText = $('tools').value;
  $('toolCount').textContent = `${(env.tools || []).length} tools`;
  $('envNote').textContent = '';
  $('confirm').style.display = 'none';
  out(`<span class="dim">${esc(env.help || 'run a request to see Relay\'s output here')}</span>`);
  $('stats').textContent = '';
  drawStage();
  const chips = $('chips');
  chips.innerHTML = '';
  for (const ex of env.examples || []) {
    const b = document.createElement('button');
    b.innerHTML = `<b>${esc(ex.title)}</b><span>${esc(ex.sub || '')}</span>`;
    b.title = ex.q;
    b.onclick = () => { $('q').value = ex.q; run(); };
    chips.appendChild(b);
  }
  setBusy(busy);
}

function drawStage() {
  if (!env) return;
  if (env.custom === 'bench') return drawBench();
  $('stage').innerHTML = env.render ? env.render(state) : '';
}

function currentTools() {
  let tools;
  try { tools = JSON.parse($('tools').value); } catch (e) { throw new Error(`tools JSON does not parse: ${e.message}`); }
  if (!Array.isArray(tools)) throw new Error('tools must be a JSON array');
  const n = Number($('distract').value);
  if (n) {
    const names = new Set(tools.map(t => t.name));
    const pool = Object.entries(envs).filter(([k, e]) => e !== env && !e.custom).flatMap(([, e]) => e.tools || [])
      .filter(t => !names.has(t.name) && names.add(t.name));
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const extra = pool.map(t => [rnd(), t]).sort((a, b) => a[0] - b[0]).slice(0, n).map(x => x[1]);
    // interleave (deterministically) so the real tools are not always first
    tools = [...tools, ...extra].map((t, i) => [((i * 2654435761) >>> 0) % 1000, t]).sort((a, b) => a[0] - b[0]).map(x => x[1]);
  }
  return tools;
}

// ------------------------------------------------------------------ result views
function out(html) { $('out').innerHTML = html; }
function topMargin(r) { return Math.max(-99, ...r.chunks.flatMap(c => (c.margins || []).map(m => m.margin))); }
const conf = m => 1 / (1 + Math.exp(-m / 2));

function showResult() {
  const R = lastResult;
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === tab));
  if (!R) return;
  const r = R.result;
  if (tab === 'raw') return out(jsonHTML(r));
  if (tab === 'calls') {
    const shown = {function_calls: r.calls.map(c => ({name: c.tool, arguments: c.args}))};
    if (r.dropped.length) shown.held_calls = r.dropped.map(c => ({name: c.tool, arguments: c.args}));
    shown.confidence = Number(conf(R.margin).toFixed(3));
    shown.router_margin = Number(R.margin.toFixed(2));
    if (R.routedBy.length) shown.routed_by_trigger = R.routedBy;
    shown.latency_ms = Math.round(R.ms);
    shown.encoder_passes = r.chunks.length;
    return out(jsonHTML(shown));
  }
  if (tab === 'why') return out(whyHTML(R.query, [...r.calls, ...r.dropped]));
  if (tab === 'rank') {
    const ms = r.chunks.flatMap(c => c.margins || []).sort((a, b) => b.margin - a.margin);
    const called = new Set([...r.calls, ...r.dropped].map(c => c.tool));
    const lo = Math.min(-12, ...ms.map(m => m.margin)), hi = Math.max(12, ...ms.map(m => m.margin));
    return out(`<div class="dim" style="margin-bottom:8px">Router score per tool: tool logit minus the learned "none" logit. Above 0 the best tool is called; extra tools need ${ACT_MARGIN}.</div>` +
      ms.slice(0, 40).map(m => `<div class="bar ${called.has(m.tool) ? 'called' : ''}"><span class="t">${esc(m.tool)}</span><span class="b"><i style="width:${(100 * (m.margin - lo) / (hi - lo)).toFixed(1)}%"></i></span><span>${m.margin.toFixed(1)}</span></div>`).join('') +
      (ms.length > 40 ? `<div class="dim">… ${ms.length - 40} more</div>` : ''));
  }
}

function spansOf(prog) {
  if (!prog) return [];
  const s = [];
  for (const k of ['span', 'date', 'time']) if (Array.isArray(prog[k])) s.push(prog[k]);
  if (Array.isArray(prog.spans)) s.push(...prog.spans);
  if (Array.isArray(prog.items)) prog.items.forEach(p => s.push(...spansOf(p)));
  return s.filter(x => Array.isArray(x) && x.length === 2 && x[1] > x[0]);
}
function whyHTML(text, cs) {
  const marks = [];
  const rows = [];
  let ci = 0;
  cs.forEach((c, i) => Object.entries(c.why || {}).forEach(([name, w]) => {
    const color = COLORS[ci++ % COLORS.length];
    const sp = spansOf(w.prog);
    sp.forEach(([a, b]) => marks.push({a, b, color, label: name}));
    const pct = w.optProb != null ? `option ${(100 * w.optProb).toFixed(0)}%` : `${(100 * w.conf).toFixed(0)}%`;
    rows.push(`<div><span style="color:${color}">●</span> ${esc(c.tool)}.<b>${esc(name)}</b> = ${esc(JSON.stringify(c.args[name]))} <span class="dim">· ${esc(w.op)}${w.op === 'ref' ? ' (the output of an earlier call)' : sp.length ? '' : ' (picked from the options, not a span)'} · ${pct}</span></div>`);
  }));
  marks.sort((x, y) => x.a - y.a || y.b - x.b);
  let html = '', at = 0;
  for (const m of marks) {
    if (m.a < at) continue;
    html += esc(text.slice(at, m.a)) + `<mark style="--c:${m.color}">${esc(text.slice(m.a, m.b))}<sub>${esc(m.label)}</sub></mark>`;
    at = m.b;
  }
  html += esc(text.slice(at));
  return `<div class="why-text">${html}</div><div class="why-list">${rows.join('') || '<div class="dim">no arguments</div>'}</div>
<div class="dim" style="margin-top:10px">Relay never writes values: each one is a program over your words (copy a span, parse a number or date, pick an enum option…).</div>`;
}

// ------------------------------------------------------------------ run
const HELP = /^\s*(hi|hello|hey|help|what can you do)\b[\s?!.]*$/i;

async function run() {
  if (busy || !ready || !env) return;
  if (env.custom === 'bench') return runBench(20);
  const q = $('q').value.trim();
  if (!q) return;
  $('confirm').style.display = 'none';
  if (HELP.test(q)) { out(`<span class="dim">${esc(env.help || '')}</span>`); return; }
  let tools;
  try { tools = currentTools(); } catch (e) { out(`<span class="fail">${esc(e.message)}</span>`); return; }
  setBusy(true);
  status('thinking…');
  const {record, routedBy} = buildRecord(tools, q, {now: nowISO(), mode: env.mode || 'call'});
  try {
    const {result, ms} = await relay(record);
    const margin = topMargin(result);
    if (!$('requireArgs').checked && result.dropped.length) { result.calls = result.dropped; result.dropped = []; }
    lastResult = {result, ms, margin, routedBy, query: q};
    showResult();
    let forced = routedBy.length > 0 || env.mode === 'extract';
    let verdict;
    if (forced && margin < (env.mode === 'extract' ? EXTRACT_FLOOR : TRIGGER_FLOOR)) {
      result.calls = [];
      forced = false;
    }
    const unsure = result.calls.flatMap(c => {
      const t = tools.find(x => x.name === c.tool);
      const req = (t && t.parameters && t.parameters.required) || [];
      return Object.entries(c.why || {}).filter(([k, w]) => req.includes(k) && w.optProb != null && w.optProb < OPT_SURE)
        .map(([k, w]) => `${c.tool}.${k} = "${c.args[k]}" (${Math.round(100 * w.optProb)}%)`);
    });
    if (unsure.length && env.mode !== 'extract') {
      verdict = 'not sure which option was meant, so it asks';
      ask(`Relay would call <b>${esc(result.calls.map(callText).join(', then '))}</b>, but it is unsure about ${esc(unsure.join(', '))}. Do it anyway?`, result.calls);
    } else if (result.calls.length && (forced || margin >= ACT_MARGIN)) {
      verdict = forced && routedBy.length ? `routed by the ${routedBy.join(', ')} trigger` : 'acting';
      $('stats').textContent = stats(ms, result, margin, verdict);
      await execute(result.calls);
    } else if (result.calls.length) {
      verdict = `below the act threshold, so it asks first`;
      ask(`Relay is ${(100 * conf(margin)).toFixed(0)}% sure it should <b>${esc(result.calls.map(callText).join(', then '))}</b>. Do it?`, result.calls);
    } else if (result.dropped.length) {
      const miss = result.dropped.flatMap(c => {
        const t = tools.find(x => x.name === c.tool);
        return ((t && t.parameters && t.parameters.required) || []).filter(p => !(p in c.args)).map(p => `${c.tool}.${p}`);
      });
      verdict = 'held: a required value is missing, so it would ask';
      ask(`Relay would call <b>${esc(result.dropped.map(callText).join(', '))}</b> but the request never says <b>${esc(miss.join(', '))}</b>. Ask the user, or do it anyway?`, result.dropped);
    } else verdict = 'no call: nothing here fits the request';
    $('stats').textContent = stats(ms, result, margin, verdict);
    status(`done · ${Math.round(ms)} ms`, 'ok');
  } catch (e) {
    out(`<span class="fail">${esc(e.message)}</span>`);
    status('error', 'err');
  } finally { setBusy(false); }
}
const stats = (ms, r, m, v) => `${Math.round(ms)} ms · ${r.chunks.length} encoder pass${r.chunks.length > 1 ? 'es' : ''} · confidence ${conf(m).toFixed(2)} (margin ${m.toFixed(1)}) · ${v}`;

function ask(html, cs) {
  pending = cs;
  $('confirmText').innerHTML = html;
  $('confirm').style.display = 'block';
}

// "#k" values are Relay's ref program: "the output of call k" (e.g. move photos into the album call 0 created).
// The sandbox resolves them to that call's first argument value.
function resolveRefs(c, done) {
  const args = {};
  for (const [k, v] of Object.entries(c.args || {})) {
    const m = typeof v === 'string' && v.match(/^#(\d+)$/);
    const src = m && done[Number(m[1])];
    args[k] = src ? Object.values(src.args || {})[0] ?? v : v;
  }
  return {...c, args};
}

async function execute(cs) {
  const done = [];
  for (const c0 of cs) {
    const c = resolveRefs(c0, done);
    done.push(c);
    try {
      const note = await env.apply(c, state, ctx);
      if (note) ctx.toast(note);
    } catch (e) { ctx.toast(`${c.tool}: ${e.message}`); }
    drawStage();
    await sleep(120);
  }
}

// ------------------------------------------------------------------ benchmark samples (Mobile Actions, CC BY 4.0)
const norm = v => typeof v === 'string' ? v.trim().toLowerCase().replace(/\s+/g, ' ') : Array.isArray(v) ? v.map(norm) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b)).map(([k, x]) => [k, norm(x)])) : v;
const same = (p, g) => JSON.stringify(p.map(c => [c.tool, norm(c.args)])) === JSON.stringify(g.map(c => [c.tool, norm(c.args)]));
let bench = {n: 0, ok: 0, rows: []};
async function runBench(k) {
  if (!benchRows) benchRows = (await (await fetch('/demo/data/mobile_actions_sample.jsonl')).text()).split('\n').filter(Boolean).map(JSON.parse);
  setBusy(true);
  for (let i = 0; i < k && bench.n < benchRows.length; i++) {
    const r = benchRows[bench.n];
    status(`row ${bench.n + 1} / ${benchRows.length}`);
    const {result, ms} = await relay({id: r.id, text: r.text, tools: r.tools, kind: 'call', context: r.context});
    const ok = same(result.calls, r.calls);
    bench.n++; bench.ok += ok;
    bench.rows.unshift({text: r.text, gold: r.calls, pred: result.calls, ok, ms});
    drawBench();
  }
  status(`benchmark · ${bench.ok}/${bench.n}`, 'ok');
  setBusy(false);
}
function drawBench() {
  const acc = bench.n ? (100 * bench.ok / bench.n).toFixed(1) : '–';
  $('stage').innerHTML = `<div style="padding:20px 24px;height:100%;overflow:auto">
  <div style="font-size:44px;font-weight:700">${acc}${bench.n ? '%' : ''}</div>
  <div class="dim">${bench.ok} of ${bench.n} Mobile Actions rows exactly right (simplified scorer; official full-set score 92.2)</div>
  ${bench.rows.slice(0, 30).map(x => `<div class="bench-row"><span class="${x.ok ? 'pass' : 'fail'}">${x.ok ? '✓' : '✗'}</span> ${esc(x.text)}
    <div style="font:11px var(--mono);margin-top:3px"><span class="dim">gold</span> ${esc(x.gold.map(callText).join(' · '))}<br><span class="dim">relay</span> ${esc(x.pred.map(callText).join(' · ') || '(no call)')} <span class="dim">· ${Math.round(x.ms)} ms</span></div></div>`).join('')}
  </div>`;
}

// ------------------------------------------------------------------ mic (browser speech recognition)
function setupMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $('mic').style.display = 'none'; return; }
  let rec = null;
  $('mic').onclick = () => {
    if (rec) { rec.stop(); return; }
    rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = true;
    $('mic').classList.add('on');
    status('listening… (browser speech recognition)');
    rec.onresult = e => { $('q').value = [...e.results].map(x => x[0].transcript).join(''); };
    rec.onend = () => { $('mic').classList.remove('on'); rec = null; if ($('q').value.trim()) run(); else status('no speech heard'); };
    rec.onerror = e => status(`mic: ${e.error}`, 'err');
    rec.start();
  };
}

// ------------------------------------------------------------------ wiring
$('run').onclick = run;
$('q').onkeydown = e => { if (e.key === 'Enter') run(); };
$('reset').onclick = () => { if (env.custom === 'bench') bench = {n: 0, ok: 0, rows: []}; selectEnv(ENV_KEYS.find(k => envs[k] === env)); };
$('envBtn').onclick = e => { e.stopPropagation(); $('menu').classList.toggle('open'); };
document.addEventListener('click', () => $('menu').classList.remove('open'));
$('tabs').onclick = e => { const t = e.target.dataset.t; if (t) { tab = t; showResult(); } };
$('doit').onclick = async () => { $('confirm').style.display = 'none'; const p = pending; pending = null; if (p) await execute(p); };
$('skip').onclick = () => { $('confirm').style.display = 'none'; pending = null; ctx.toast('skipped'); };
$('distract').oninput = () => { $('distractN').textContent = $('distract').value; };
$('tools').oninput = () => { try { $('toolCount').textContent = `${JSON.parse($('tools').value).length} tools (edited)`; } catch { $('toolCount').textContent = 'invalid JSON'; } };
const hc = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8));
for (const t of [1, 2, 4, 8].filter(t => t <= Math.max(hc, 1))) $('threads').add(new Option(`${t} thread${t > 1 ? 's' : ''}`, t));
$('threads').value = String(window.crossOriginIsolated ? [8, 4, 2, 1].find(t => t <= hc) : 1);
$('threads').onchange = () => startWorker(Number($('threads').value));
setupMic();
await loadEnvs();
const want = (location.hash.match(/env=(\w+)/) || [])[1];
selectEnv(envs[want] ? want : 'phone');
startWorker(Number($('threads').value));
window.addEventListener('hashchange', () => { const k = (location.hash.match(/env=(\w+)/) || [])[1]; if (envs[k] && envs[k] !== env) selectEnv(k); });
