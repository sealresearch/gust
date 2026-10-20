// Relay-2 inputs (port of relay2/features.py, inference path): one joint sequence per example,
//     [CLS] request [SEP] [T] tool text [P] param text [O] option ... [P] ... [T] ... [SEP]
// with long tool lists split into several sequences (eval_examples). Request offsets are code point indices.
import {descOptions} from './programs.mjs';
import {pyIter, pyStr, strip, truthy} from './py.mjs';

export const K = 6;
export const OPS = ['absent', 'copy', 'number', 'datetime', 'timepart', 'enum', 'bool_true', 'bool_false', 'lookup',
  'default', 'ref', 'list', 'digits', 'affix', 'select'];
export const OP = Object.fromEntries(OPS.map((o, i) => [o, i]));
export const LIMITS = {request: 192, tool: 48, param: 40, option: 8, options: 16, params: 32, max_len: 1024};
export const T_TOK = '[unused0]', P_TOK = '[unused1]', O_TOK = '[unused2]';

const has = (o, k) => o !== null && typeof o === 'object' && Object.prototype.hasOwnProperty.call(o, k);

/** Tok: the tokenizer plus the marker ids features.py reads from it. */
export class Tok {
  constructor(tokenizer) {
    this.tk = tokenizer;
    this.cls = tokenizer.id('[CLS]');
    this.sep = tokenizer.id('[SEP]');
    this.pad = tokenizer.id('[PAD]');
    [this.T, this.P, this.O] = [T_TOK, P_TOK, O_TOK].map(t => tokenizer.id(t));
  }
  ids(text, n) { return this.tk.ids(text, n); }
  request(text, n) { return this.tk.request(text, n); }
}

// ---------------------------------------------------------------- texts

const CAMEL = /(?<=[a-z0-9])(?=[A-Z])/g;
const SEPS = /[_\-.]+/g;
/** humanize: playlist_owner -> 'playlist owner'; restaurantName -> 'restaurant name'. */
export const humanize = name => strip(pyStr(name).replace(CAMEL, ' ').replace(SEPS, ' ')).toLowerCase();

export function toolText(t) {
  return `${pyStr(t.name)} (${humanize(t.name)}) : ${truthy(t.description) ? pyStr(t.description) : ''}`;
}

export function paramText(p) {
  const typ = truthy(p.raw_type) ? p.raw_type : (has(p, 'type') ? p.type : 'string');
  let s = `${pyStr(p.name)} (${pyStr(typ)}${truthy(p.required) ? ', required' : ''}) : ${truthy(p.description) ? pyStr(p.description) : ''}`;
  if (has(p, 'default')) s += ` default: ${pyStr(p.default)}`;
  if (p.type === 'array' && truthy(p.items)) {
    const it = p.items;
    s += ` items: ${pyStr(truthy(it.raw_type) ? it.raw_type : it.type)}`;
  }
  return s;
}

/** Choices the model can point at: the enum, else values listed in the description. -> [list, kind] */
export function options(p) {
  if (truthy(p.enum)) return [pyIter(p.enum).map(pyStr), 'enum'];
  return [descOptions(p), 'option'];
}

// ---------------------------------------------------------------- spans

export const OPEN = Array.from('"\'“‘([{`'), CLOSE = Array.from('"\'”’)]}`');

/** trim: drop unbalanced quotes/brackets at the edges of a decoded span. `cps` = text as code point strings. */
export function trim(cps, a, b) {
  const within = (ch, x, y) => { for (let i = x; i < y; i++) if (cps[i] === ch) return true; return false; };
  while (a < b && OPEN.includes(cps[a]) && !within(CLOSE[OPEN.indexOf(cps[a])], a + 1, b)) a++;
  while (b > a && CLOSE.includes(cps[b - 1]) && !within(OPEN[CLOSE.indexOf(cps[b - 1])], a, b - 1)) b--;
  return [a, b];
}

// ---------------------------------------------------------------- examples

export function toolSegment(t, tk) {
  const seg = [tk.T, ...tk.ids(toolText(t), LIMITS.tool)];
  const prow = [];
  for (const p of t.params.slice(0, LIMITS.params)) {
    const pp = seg.length;
    seg.push(tk.P, ...tk.ids(paramText(p), LIMITS.param));
    const opos = [];
    for (const o of options(p)[0].slice(0, LIMITS.options)) {
      opos.push(seg.length);
      seg.push(tk.O, ...tk.ids(o, LIMITS.option));
    }
    prow.push([pp, opos]);
  }
  return [seg, prow];
}

/**
 * build_example(train=False, targets=False): record -> {ids, nq, offs, text, tools: [{name, tool, pos, params:
 * [{pos, opts}]}], rec} or null when unusable.
 */
export function buildExample(rec, tk) {
  const tools = [...(rec.tools || [])];
  if (!tools.length) return null;
  const text = rec.text;
  const {ids: qIds, offs} = tk.request(text, LIMITS.request);
  if (!qIds.length) return null;
  const segs = tools.map(t => [t, ...toolSegment(t, tk)]);
  const budget = LIMITS.max_len - qIds.length - 3;
  if (segs.reduce((n, x) => n + x[1].length, 0) > budget) return null;  // eval_examples splits long lists
  const keep = segs.slice();
  // keep.sort(key=order[name]) with order = last index of each name: duplicate names move together
  const order = new Map();
  tools.forEach((t, i) => order.set(t.name, i));
  keep.sort((x, y) => order.get(x[0].name) - order.get(y[0].name));
  const ids = [tk.cls, ...qIds, tk.sep];
  const toolRows = [];
  for (const [t, seg, prow] of keep) {
    const base = ids.length;
    for (const x of seg) ids.push(x);
    toolRows.push({name: t.name, tool: t, pos: base,
      params: prow.map(([pp, opos]) => ({pos: base + pp, opts: opos.map(o => base + o)}))});
  }
  ids.push(tk.sep);
  return {id: rec.id, ids, q0: 1, nq: qIds.length, offs, text, tools: toolRows, rec};
}

/** eval_examples: the record's tools in order, split into as many sequences as needed so every tool fits. */
import {FIX} from './hostfix.mjs';

export function evalExamples(rec, tk) {
  const {ids: qIds} = tk.request(rec.text, LIMITS.request);
  const budget = LIMITS.max_len - qIds.length - 3;
  const tools = [...(rec.tools || [])];
  const chunks = [];
  let cur = [], used = 0;
  for (const t of tools) {
    const n = toolSegment(t, tk)[0].length;
    if (cur.length && used + n > budget) {
      chunks.push(cur);
      cur = [];
      used = 0;
    }
    cur.push(t);
    used += n;
  }
  if (cur.length) chunks.push(cur);
  if (FIX.chunks && chunks.length > 1) {  // same number of chunks, filled evenly (no lone tool in a last chunk)
    const sizes = tools.map(t => toolSegment(t, tk)[0].length);
    const target = sizes.reduce((x, y) => x + y, 0) / chunks.length;
    const even = [];
    let c = [], u = 0;
    tools.forEach((t, i) => {
      if (c.length && (u + sizes[i] > budget || (u >= target && even.length < chunks.length - 1))) { even.push(c); c = []; u = 0; }
      c.push(t);
      u += sizes[i];
    });
    if (c.length) even.push(c);
    chunks.splice(0, chunks.length, ...even);
  }
  const out = [];
  chunks.forEach((ch, i) => {
    const ex = buildExample({...rec, tools: ch, pool: null}, tk);
    if (ex !== null) {
      ex.chunk = i;
      out.push(ex);
    }
  });
  return out;
}

