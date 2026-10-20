// Retrieval prefilter (Relay-P): with long tool lists, keep only the top-k tools by vector similarity before full
// Relay runs. Tool vectors come from the same Gust encoder run on each tool's own segment and are cached per tool,
// so an app pays for them once; a request costs one short encoder pass plus a dot product per tool.
// Head weights: relay-prefilter.bin (research/prefilter_export.py): u32 header length, JSON header, then float16 data.
import {LIMITS, toolSegment} from './features.mjs';

function half(h) {  // float16 -> float32
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

function parseHead(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(4, 4 + n)));
  let off = 4 + n;
  const t = {};
  for (const [name, shape] of meta.tensors) {
    const len = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = half(dv.getUint16(off + 2 * i, true));
    off += 2 * len;
    t[name] = {shape, data: out};
  }
  return {meta, t};
}

function linear(x, W, b) {  // W [out, in] row-major
  const [o, n] = W.shape, y = new Float32Array(o);
  for (let i = 0; i < o; i++) {
    let s = b.data[i];
    const row = i * n;
    for (let j = 0; j < n; j++) s += W.data[row + j] * x[j];
    y[i] = s;
  }
  return y;
}
const gelu = x => x.map(v => 0.5 * v * (1 + erf(v / Math.SQRT2)));
function erf(x) {  // Abramowitz-Stegun 7.1.26
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}
function normalize(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map(x => x / s); }

export class Prefilter {
  /** relay: a Relay instance (its encoder and tokenizer are reused); head: relay-prefilter.bin bytes */
  /** budget: also stop adding tools (best first) once their tool tokens would pass this many; the encoder costs
   *  ~0.1 ms per token at 8 threads, so the budget bounds latency however long the app's tool list is. minKeep: tools
   *  always kept, budget or not. With a budget, any list over it is cut, even below minTools. */
  constructor(relay, head, {k = 10, minTools = 13, budget = 600, minKeep = 3} = {}) {
    this.relay = relay;
    this.h = parseHead(head);
    this.k = k;
    this.minTools = minTools;
    this.budget = budget;
    this.minKeep = minKeep;
    this.cache = new Map();
    this.lens = new Map();
    if (this.h.meta.feat === 'lower' && !relay.net.late) throw new Error('this prefilter needs a late-fusion Relay');
    this.stats = {toolPasses: 0, requestPasses: 0};
  }

  _feat(ids, tool = false) {  // first-token state and mean state of one encoder pass
    const net = this.relay.net, d = net.d;
    // feat "lower" (late-fusion model): the layer-S states the host caches per tool and computes per request anyway
    const p = this.h.meta.feat === 'lower' ? net._piece(ids, tool) : net.hidden({ids});
    const H = p.st || p;  // one-way tool pieces carry their keys / values too
    const first = H.slice(0, d), mean = new Float32Array(d);
    const n = ids.length;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += H[i * d + j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    const x = new Float32Array(2 * d);
    x.set(first, 0);
    x.set(mean, d);
    return x;
  }

  _side(prefix, x) {
    const t = this.h.t;
    return normalize(linear(gelu(linear(x, t[`${prefix}.net.0.weight`], t[`${prefix}.net.0.bias`])),
      t[`${prefix}.net.2.weight`], t[`${prefix}.net.2.bias`]));
  }

  toolVector(tool) {
    const key = JSON.stringify(tool);
    let v = this.cache.get(key);
    if (!v) {
      const seg = toolSegment(tool, this.relay.tok)[0].slice(0, LIMITS.max_len >> 1);
      v = this._side('t', this._feat(seg, true));
      this.stats.toolPasses++;
      this.cache.set(key, v);
    }
    return v;
  }

  /** Precompute (cache) the vectors of an app's tools, e.g. at startup. */
  warm(tools) { for (const t of tools) this.toolVector(t); }

  /** Tools ranked by similarity to the request: [{tool, score}] */
  rank(text, tools) {
    const tk = this.relay.tok;
    const {ids} = tk.request(text, LIMITS.request);
    const r = this._side('r', this._feat([tk.cls, ...ids, tk.sep]));
    this.stats.requestPasses++;
    return tools.map(t => {
      const v = this.toolVector(t);
      let s = 0;
      for (let i = 0; i < v.length; i++) s += v[i] * r[i];
      return {tool: t, score: s};
    }).sort((a, b) => b.score - a.score);
  }

  /** Tokens of a tool's segment in the full Relay sequence (cached). */
  toolTokens(tool) {
    const key = JSON.stringify(tool);
    let n = this.lens.get(key);
    if (n === undefined) this.lens.set(key, n = toolSegment(tool, this.relay.tok)[0].length);
    return n;
  }

  /** The record with its tool list cut to the top k (and, with a budget, to the best tools that fit it); unchanged
   *  below minTools tools when within the budget; keeps the original order. */
  filter(record) {
    const tools = record.tools || [];
    if (record.kind === 'extract') return record;  // extraction reads every schema part
    const over = this.budget > 0 && tools.reduce((n, t) => n + this.toolTokens(t), 0) > this.budget;
    if (tools.length < this.minTools && !over) return record;
    const keep = new Set();
    let used = 0;
    for (const {tool} of this.rank(record.text, tools)) {
      if (keep.size >= this.k) break;
      const n = this.toolTokens(tool);
      if (this.budget > 0 && keep.size >= this.minKeep && used + n > this.budget) break;
      keep.add(tool);
      used += n;
    }
    return {...record, tools: tools.filter(t => keep.has(t))};
  }
}
