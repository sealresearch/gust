// Relay-2 outputs -> value programs -> calls (port of relay2/decode.py: predict, predict_records, make_program,
// best_span, allowed_ops). Head values are float32; the arithmetic PyTorch does on them in float32 (sums, softmax,
// masking) is reproduced with Math.fround so that argmax ties and saturated confidences break the same way.
import {K, OP, OPS, evalExamples, options, trim} from './features.mjs';
import {DATE_FORMATS, NUM_SCALES, TABLE_NAMES, affixes, run} from './programs.mjs';
import {PY_WS} from './py.mjs';
import {FIX} from './hostfix.mjs';

const f32 = Math.fround;
export const MAX_SPAN = 96;
const ALLOWED = {
  boolean: ['absent', 'bool_true', 'bool_false', 'default'],
  integer: ['absent', 'number', 'copy', 'timepart', 'default', 'enum', 'digits'],
  number: ['absent', 'number', 'copy', 'default', 'enum'],
  array: ['absent', 'list', 'default'],
  object: ['absent', 'default'],
};
const has = (o, k) => o !== null && typeof o === 'object' && Object.prototype.hasOwnProperty.call(o, k);
const ptype = p => (has(p, 'type') ? p.type : 'string');  // param.get("type", "string")

/** allowed_ops: additive mask over OPS (0 or -1e4). */
export function allowedOps(param, nOpts) {
  const t = ptype(param);
  const ok = new Set(typeof t === 'string' && has(ALLOWED, t) ? ALLOWED[t] : OPS.filter(o => o !== 'list'));
  if (!nOpts) ok.delete('enum');
  if (FIX.enumOnly && nOpts && has(param, 'enum') && param.enum && param.enum.length)  // closed enum: options only
    for (const o of ['copy', 'select', 'lookup', 'affix', 'digits', 'ref']) ok.delete(o);  // no copied words; bool/number stay
  if (!has(param, 'default')) ok.delete('default');
  return OPS.map(o => (ok.has(o) ? 0 : -1e4));
}

/** First index of the maximum of a[0..n) (torch.argmax). */
export function argmax(a, n = a.length, from = 0) {
  let best = from;
  for (let i = from + 1; i < from + n; i++) if (a[i] > a[best]) best = i;
  return best - from;
}

/** torch.softmax on a float32 vector: exp(x - max) summed, then scaled by the float32 reciprocal of the sum. */
export function softmaxMax(x, n = x.length) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += f32(Math.exp(f32(x[i] - m)));
  return f32(1 / f32(sum));  // the maximum's probability: exp(0) * (1 / sum)
}

/** best_span: joint argmax of start + end over j >= i, j - i < MAX_SPAN; null if the null slot (index n) wins. */
export function bestSpan(s, e, n) {
  let best = -Infinity, a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const lim = Math.min(n, i + MAX_SPAN);
    const si = s[i];
    for (let j = i; j < lim; j++) {
      const v = f32(si + e[j]);
      if (v > best) { best = v; a = i; b = j; }
    }
  }
  if (f32(s[n] + e[n]) > best) return null;
  return [a, b];
}

const SELECT_PUNCT = new RegExp(`[${PY_WS}]+([,.;:!?])`, 'gu');

/** make_program: head outputs of one argument row -> value program (code point spans) or null. */
export function makeProgram(name, row, ex, param) {
  const n = ex.nq, offs = ex.offs;
  const char = t => trim(ex.cps, offs[t[0]][0], offs[t[1]][1]);
  const aux = row.aux;
  if (['copy', 'number', 'lookup', 'timepart', 'digits', 'affix'].includes(name)) {
    const sp = row.spans();
    const t = bestSpan(sp[0], sp[1], n);
    if (t === null) return null;
    const span = char(t);
    if (name === 'copy' || name === 'digits') return {op: name, span};
    if (name === 'number')
      return {op: 'number', span, scale: NUM_SCALES[argmax(aux, NUM_SCALES.length)], int: param.type === 'integer'};
    if (name === 'timepart') return {op: 'timepart', span, part: ['hour', 'minute'][argmax(aux, 2)]};
    if (name === 'affix') {
      const m = Math.min(affixes().length, aux.length);
      return m ? {op: 'affix', index: argmax(aux, m), span} : null;
    }
    return {op: 'lookup', table: TABLE_NAMES[argmax(aux, TABLE_NAMES.length)], span};
  }
  if (name === 'datetime') {
    const sp = row.spans();
    const d = bestSpan(sp[0], sp[1], n);
    const t = bestSpan(sp[2], sp[3], n);
    if (d === null && t === null) return null;
    return {op: 'datetime', date: d ? char(d) : null, time: t ? char(t) : null,
      format: DATE_FORMATS[argmax(aux, DATE_FORMATS.length)]};
  }
  if (name === 'enum') {
    const [opts, kind] = options(param);
    if (!opts.length) return null;
    const en = row.enum();
    const i = Math.min(argmax(en), opts.length - 1);
    return kind === 'enum' ? {op: 'enum', index: i} : {op: 'option', index: i};
  }
  if (name === 'bool_true' || name === 'bool_false') return {op: 'bool', value: name === 'bool_true'};
  if (name === 'default') return {op: 'default'};
  if (name === 'ref') return {op: 'ref', call: argmax(aux, K)};
  if (name === 'list' || name === 'select') {
    const tg = row.tags();  // [q][3]
    const spans = [];
    let cur = null;
    for (let i = 0; i < n; i++) {
      const t = argmax(tg[i]);
      if (t === 1) {
        if (cur) spans.push(cur);
        cur = [i, i];
      } else if (t === 2 && cur) cur[1] = i;
      else {
        if (cur) spans.push(cur);
        cur = null;
      }
    }
    if (cur) spans.push(cur);
    if (!spans.length) return null;
    if (name === 'select') {
      const value = spans.map(s => { const [a, b] = char(s); return ex.cps.slice(a, b).join(''); }).join(' ')
        .replace(SELECT_PUNCT, '$1');
      return {op: 'const', value};
    }
    return {op: 'list', items: spans.map(s => ({op: 'copy', span: char(s)}))};
  }
  return null;
}

/** Code point spans of a program -> UTF-16 spans (programs.run slices JavaScript strings). */
function toUtf16(prog, map) {
  if (!map) return prog;
  const conv = sp => (sp ? [map[sp[0]], map[sp[1]]] : sp);
  const out = {...prog};
  for (const k of ['span', 'date', 'time']) if (out[k]) out[k] = conv(out[k]);
  if (out.items) out.items = out.items.map(p => toUtf16(p, map));
  return out;
}

function setKey(o, k, v) {
  if (k === '__proto__') Object.defineProperty(o, k, {value: v, enumerable: true, writable: true, configurable: true});
  else o[k] = v;
}

/**
 * predict for one example (one chunk): route, place instances, fill every parameter of each instance.
 * `net` computes the heads: route(ex) -> {tool: Float32Array, count: Float32Array[], none}; anchors(ex, insts) ->
 * Float32Array[] (one row of nq logits per instance); args(ex, rows) -> per row {op, aux, spans(), tags(), enum()}.
 */
export function predictExample(net, ex, {gate = 0, gate2 = null, exclusive = true, explain = false} = {}) {
  const nt = ex.tools.length;
  if (!nt) return {id: ex.id, calls: [], margin: -1e9};
  const r = net.route(ex);
  const margin = new Float32Array(nt);
  for (let i = 0; i < nt; i++) margin[i] = f32(r.tool[i] - r.none);
  const g2 = gate2 === null || gate2 === undefined ? gate : Math.max(gate, gate2);
  const called = Array.from(margin, m => m > g2);
  const j = argmax(margin);
  called[j] = called[j] || margin[j] > gate || ((ex.rec && ex.rec.kind) === 'extract');
  const counts = called.map((c, i) => (c ? argmax(r.count[i]) + 1 : 0));
  const out = {id: ex.id, calls: [], margin: margin[j]};
  if (explain) out.margins = ex.tools.map((t, i) => ({tool: t.name, margin: margin[i], count: counts[i]}));
  const insts = [];
  for (let ti = 0; ti < nt; ti++) for (let k = 0; k < counts[ti]; k++) insts.push([ti, k]);
  if (!insts.length) return out;

  const al = net.anchors(ex, insts);
  const anchor = new Map(), used = new Map();
  insts.forEach(([ti, k], i) => {
    const lg = Float32Array.from(al[i]);
    for (const w of used.get(ti) || []) lg[w] = -1e9;
    const w = argmax(lg);
    if (!used.has(ti)) used.set(ti, []);
    used.get(ti).push(w);
    anchor.set(`${ti},${k}`, w);
  });
  const rows = [];
  for (const [ti, k] of insts)
    for (let pj = 0; pj < ex.tools[ti].params.length; pj++) rows.push({ti, k, pj, anchor: anchor.get(`${ti},${k}`)});
  const a = rows.length ? net.args(ex, rows) : [];

  ex.cps = ex.cps || Array.from(ex.text);
  let map = null;
  if (ex.cps.length !== ex.text.length) {
    map = new Int32Array(ex.cps.length + 1);
    let u = 0;
    ex.cps.forEach((c, i) => { map[i] = u; u += c.length; });
    map[ex.cps.length] = u;
  }
  const calls = new Map();
  for (const [ti, k] of insts)
    calls.set(`${ti},${k}`, {ti, k, tool: ex.tools[ti].name, args: {}, pos: anchor.get(`${ti},${k}`), cands: []});
  const ctx = ex.rec ? ex.rec.context : null;
  rows.forEach((rw, jr) => {
    const tool = ex.tools[rw.ti].tool;
    const param = tool.params[rw.pj];
    const c = calls.get(`${rw.ti},${rw.k}`);
    const nOpts = ex.tools[rw.ti].params[rw.pj].opts.length;
    const row = a[jr];
    const mask = allowedOps(param, nOpts);
    const logits = new Float32Array(OPS.length);
    for (let o = 0; o < OPS.length; o++) logits[o] = f32(row.op[o] + mask[o]);
    const op = argmax(logits);
    if (op === OP.absent) return;
    let conf = softmaxMax(logits);
    const prog = makeProgram(OPS[op], row, ex, param);
    if (prog === null) return;
    let v;
    try {
      v = run(toUtf16(prog, map), ex.text, param, ctx);
    } catch {
      return;
    }
    if (v === null || v === undefined) return;
    if (FIX.numReject && ['integer', 'number'].includes(ptype(param)) &&
        !(typeof v === 'number' || (v && typeof v === 'object' && typeof v.v === 'number'))) return;
    let optProb = null;
    if (prog.op === 'enum' || prog.op === 'option') optProb = softmaxMax(row.enum());
    if ('span' in prog) {
      const sp = row.spans();
      conf *= f32(softmaxMax(sp[0], ex.nq + 1) * softmaxMax(sp[1], ex.nq + 1));
    }
    // only text-copying ops compete for a stretch of text; '10:15' may give both an hour and a minute
    const span = (['copy', 'lookup', 'digits', 'affix'].includes(prog.op) && ptype(param) === 'string') ||
      (FIX.numSpan && prog.op === 'number') ? prog.span : null;
    c.cands.push([conf, param.name, v, span, prog.op === 'ref', explain ? toUtf16(prog, map) : null, optProb,
      prog.span || prog.date || prog.time || null]);
  });
  for (const c of calls.values()) {
    const taken = [];
    const cands = c.cands.map((x, i) => [x, i]).sort((x, y) => y[0][0] - x[0][0] || x[1] - y[1]).map(x => x[0]);
    c.why = {};
    c.first = Infinity;
    for (const [conf, name, v, span, isRef, prog, optProb, where] of cands) {
      if (exclusive && span !== null) {
        if (taken.some(([x, y]) => span[0] < y && x < span[1])) continue;
        taken.push(span);
      }
      setKey(c.args, name, v);
      if (where) c.first = Math.min(c.first, where[0]);
      if (explain) c.why[name] = {op: prog.op, conf, span: prog.span || null, prog, optProb};
      c.ref = (c.ref || false) || isRef;
    }
  }
  const order = [...calls.values()].sort((x, y) =>
    (x.ref ? 1 : 0) - (y.ref ? 1 : 0) || x.pos - y.pos || (FIX.orderTie ? x.first - y.first : 0) || x.ti - y.ti || x.k - y.k);
  out.calls = order.map(c => (explain ? {tool: c.tool, args: c.args, why: c.why, anchor: c.pos, margin: margin[c.ti]} :
    {tool: c.tool, args: c.args}));
  if (FIX.dedupe) {
    const seen = new Set();
    out.calls = out.calls.filter(c => { const k = c.tool + JSON.stringify(c.args); if (seen.has(k)) return false; seen.add(k); return true; });
  }
  return out;
}

/**
 * predict_records for one record: long tool lists are split into several sequences; the chunk with the most
 * confident tool keeps its calls, other chunks add calls only above the stricter gate2. Returns {calls, chunks}.
 */
export function predictRecord(net, rec, tk, {gate = 0, gate2 = null, ...kw} = {}) {
  let exs;
  try {
    exs = evalExamples(rec, tk);
  } catch {
    exs = [];  // an unbuildable record simply gets no calls
  }
  const preds = exs.map(ex => predictExample(net, ex, {gate, gate2, ...kw}));
  const g2 = gate2 === null || gate2 === undefined ? gate : Math.max(gate, gate2);
  const ps = preds.map((p, i) => [p, i]).sort((x, y) => y[0].margin - x[0].margin || x[1] - y[1]).map(x => x[0]);
  const calls = ps.length ? [...ps[0].calls] : [];
  // FIX.chunks: a later chunk's best tool must also come within 2 of the best chunk's (it used to need only gate2)
  const bar = FIX.chunks && ps.length ? Math.max(g2, ps[0].margin - 2) : g2;
  if (rec.kind !== 'extract') for (const p of ps.slice(1)) if (p.margin > bar) calls.push(...p.calls);
  return {calls, chunks: exs.map((ex, i) => ({ex, ...preds[i]}))};
}
