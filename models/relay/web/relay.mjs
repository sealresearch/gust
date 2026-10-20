// Relay-2 on the Gust WebAssembly runtime: tokenizer, sequence building, the four float32 graphs (encoder, route,
// anchor, args), the host's key . query dot products, and decoding into calls. Port of the Python host
// (relay2/gust_enc.py, relay2/gust_heads.py, relay2/decode.py). No dependencies; works in Node >= 18 and browsers.
//
//   const relay = await Relay.create({wasm, artifacts: {encoder, route, anchor, args}, tokenizer});
//   relay.predict({id, text, tools: [...], kind?, context?})  ->  [{tool, args}]
import {Tok} from './features.mjs';
import {predictRecord} from './decode.mjs';
import {Tokenizer} from './tokenizer.mjs';
import {FIX, setFixes} from './hostfix.mjs';
export {setFixes};

export {parseJSON} from './py.mjs';

// Row capacities of the head graphs (relay2/gust_build.py); larger sets are split into several runs.
export const TOOLS = 128, REQ = 192, INSTS = 32, ROWS = 32, OPTS = 64;
const f32 = Math.fround;

function dot(a, ia, b, ib, n) {
  let s = 0;  // float64 accumulation of exact float32 products
  for (let i = 0; i < n; i++) s += a[ia + i] * b[ib + i];
  return s;
}

/** The Gust graphs plus the host-side arithmetic of relay2/gust_heads.py, per example (one chunk). */
export class GustNet {
  constructor(models) {
    this.m = models;
    const shape = (g, name) => g.tensor(name).shape;
    this.late = !models.encoder;  // Relay-L: lower (per piece, tool pieces cached) + upper (joint) graphs
    this.L = shape(this.late ? models.lower : models.encoder, 'tokens')[0];
    this.d = shape(this.late ? models.upper : models.encoder, 'hidden')[2];
    this.pieceCache = new Map();
    // one-way late fusion: the lower graph also reads cached tool keys / values ("count", "kc{i}", "vc{i}")
    this.oneway = this.late && models.lower.inputs.some(t => t.name === 'count');
    this.dk = shape(models.anchor, 'anchor_k')[2];
    this.K1 = shape(models.route, 'tool_out')[2];
    this.nOps = shape(models.args, 'op')[2];
    this.nAux = shape(models.args, 'aux')[2];
    this.scale = f32(this.dk ** 0.5);
    this.inputs = {};
    for (const [k, g] of Object.entries(models))
      this.inputs[k] = Object.fromEntries(g.inputs.map(t => [t.name, new Float32Array(t.size / 4)]));
    this.stats = {runs: 0, ms: 0};
  }

  _run(name, feed, valid) {
    const g = this.m[name];
    const inp = this.inputs[name];
    for (const t of g.inputs) {
      const buf = inp[t.name];
      buf.fill(0);
      const v = feed[t.name];
      if (v) buf.set(v.length > buf.length ? v.subarray(0, buf.length) : v);
    }
    const t0 = performance.now();
    const out = g.runFloat(inp, {validFrames: valid});
    this.stats.ms += performance.now() - t0;
    this.stats.runs++;
    return out;
  }

  /** Layer-S states of one piece of token ids (lower graph); tool pieces are cached by their tokens. One-way:
   *  `kv` (the request's tools) = [{rows, k: [per layer], v: [per layer]}] is read by the request's attention, and
   *  tool pieces keep their per-layer keys / values (piece.kv). Returns the states (tool pieces: {st, kv}). */
  _piece(ids, cache, kv = null) {
    const key = ids.join(',');
    if (cache && this.pieceCache.has(key)) return this.pieceCache.get(key);
    if (!cache && !kv && this.lastReq && this.lastReq.key === key) return this.lastReq.st;  // prefilter + decode
    const n = ids.length, d = this.d;
    const feed = {tokens: Float32Array.from(ids)};
    if (this.oneway) {
      let rows = 0;
      if (kv) {
        const total = kv.reduce((s, p) => s + p.rows, 0);
        for (let l = 0; l < kv[0].k.length; l++) {
          const kc = new Float32Array(total * d), vc = new Float32Array(total * d);
          let at = 0;
          for (const p of kv) { kc.set(p.k[l], at * d); vc.set(p.v[l], at * d); at += p.rows; }
          feed[`kc${l}`] = kc;
          feed[`vc${l}`] = vc;
        }
        rows = total;
      }
      feed.count = Float32Array.of(rows);
    }
    const out = this._run('lower', feed, n);
    const names = Object.keys(out);
    const stName = names.find(k => !/\.[kv]$/.test(k));
    const st = out[stName].slice(0, n * d);
    let res = st;
    if (this.oneway && cache) {
      const layers = names.filter(k => k.endsWith('.k')).length;
      res = {st, rows: n, k: [], v: []};
      for (let l = 0; l < layers; l++) {
        res.k.push(out[`l${l}.k`].slice(0, n * d));
        res.v.push(out[`l${l}.v`].slice(0, n * d));
      }
    }
    if (cache) this.pieceCache.set(key, res);
    else if (!kv) this.lastReq = {key, st};
    return res;
  }

  /** Late fusion: [CLS] request [SEP] and every tool segment run alone below the split, the rest jointly. */
  _hiddenLate(ex) {
    const n = ex.ids.length, d = this.d;
    const starts = (ex.tools || []).map(t => t.pos).sort((a, b) => a - b);
    const spans = starts.length ? [[0, starts[0], false]] : [[0, n, false]];
    starts.forEach((s, i) => spans.push([s, i + 1 < starts.length ? starts[i + 1] : n - 1, true]));
    if (starts.length) spans.push([n - 1, n, true]);
    const state = new Float32Array(n * d);
    if (this.oneway) {  // tools first (cached), then the request reading their keys and values
      const tools = spans.filter(([a, b, cache]) => cache && b > a).map(([a, b]) => [a, this._piece(ex.ids.slice(a, b), true)]);
      for (const [a, p] of tools) state.set(p.st, a * d);
      const [a0, b0] = spans[0];
      state.set(this._piece(ex.ids.slice(a0, b0), false, tools.length ? tools.map(([, p]) => p) : null), a0 * d);
    } else {
      for (const [a, b, cache] of spans) if (b > a) state.set(this._piece(ex.ids.slice(a, b), cache), a * d);
    }
    return this._run('upper', {state}, n).hidden.slice(0, n * d);
  }

  /** Encoder hidden states of the example's sequence, [len, d] (cached on the example). */
  hidden(ex) {
    if (ex.H) return ex.H;
    const n = ex.ids.length;
    if (n > this.L) throw new Error(`sequence of ${n} tokens exceeds the encoder's ${this.L}`);
    if (this.late) return (ex.H = this._hiddenLate(ex));
    const tok = new Float32Array(n);
    for (let i = 0; i < n; i++) tok[i] = ex.ids[i];
    const out = this._run('encoder', {tokens: tok}, n);
    ex.H = out.hidden.slice(0, n * this.d);
    return ex.H;
  }

  rows(H, positions) {
    const d = this.d, out = new Float32Array(positions.length * d);
    positions.forEach((p, i) => out.set(H.subarray(p * d, (p + 1) * d), i * d));
    return out;
  }

  req(ex) {
    const H = this.hidden(ex);
    return H.subarray(ex.q0 * this.d, (ex.q0 + ex.nq) * this.d);
  }

  route(ex) {
    const H = this.hidden(ex), d = this.d, K1 = this.K1;
    const nt = ex.tools.length;
    const tool = new Float32Array(nt), count = [];
    let none = 0;
    const cls = H.subarray(0, d);
    for (let s = 0; s < Math.max(1, nt); s += TOOLS) {
      const part = ex.tools.slice(s, s + TOOLS);
      const r = part.length
        ? this._run('route', {tools: this.rows(H, part.map(t => t.pos)), cls}, part.length)
        : this._run('route', {tools: new Float32Array(d), cls}, 1);
      part.forEach((_, i) => {
        tool[s + i] = r.tool_out[i * K1];
        count[s + i] = r.tool_out.slice(i * K1 + 1, (i + 1) * K1);
      });
      none = r.none[0];
    }
    return {tool, count, none};
  }

  anchors(ex, insts) {
    const H = this.hidden(ex), d = this.d, dk = this.dk, nq = ex.nq;
    const req = this.req(ex);
    const out = [];
    for (let s = 0; s < insts.length; s += INSTS) {
      const part = insts.slice(s, s + INSTS);
      const r = this._run('anchor', {req, inst: this.rows(H, part.map(([ti]) => ex.tools[ti].pos)),
        inst_slot: Float32Array.from(part, ([, k]) => k)}, nq);
      const aq = r.anchor_q, ak = r.anchor_k;
      part.forEach((_, i) => {
        const lg = new Float32Array(nq);
        for (let q = 0; q < nq; q++) lg[q] = f32(f32(dot(aq, i * dk, ak, q * dk, dk)) / this.scale);
        out.push(lg);
      });
    }
    return out;
  }

  args(ex, rows) {
    const H = this.hidden(ex), d = this.d, dk = this.dk, nq = ex.nq, scale = this.scale;
    const req = this.req(ex);
    // global parameter / option numbering of the example (collate order)
    const pidx = new Map(), nopt = [], ostart = [], optPos = [];
    for (const [ti, t] of ex.tools.entries())
      t.params.forEach((p, pj) => {
        pidx.set(`${ti},${pj}`, nopt.length);
        ostart.push(optPos.length);
        nopt.push(p.opts.length);
        optPos.push(...p.opts);
      });
    const E = Math.max(1, ...nopt);
    const prow = rows.map(r => pidx.get(`${r.ti},${r.pj}`));
    const params = [...new Set(prow)].sort((a, b) => a - b);
    const opts = [];
    for (const p of params) for (let o = ostart[p]; o < ostart[p] + nopt[p]; o++) opts.push(o);
    const ochunks = [];
    for (let k = 0; k < opts.length; k += OPTS) ochunks.push(opts.slice(k, k + OPTS));
    const okAll = new Map();
    let sk = null, tk = null;
    const res = [];
    for (let s = 0; s < rows.length; s += ROWS) {
      const part = rows.slice(s, s + ROWS);
      const first = s === 0;
      const ha = new Float32Array(part.length * d);
      part.forEach((r, i) => {
        if (r.anchor >= 0) ha.set(H.subarray((ex.q0 + r.anchor) * d, (ex.q0 + r.anchor + 1) * d), i * d);
      });
      const feed = {
        req,
        hp: this.rows(H, part.map(r => ex.tools[r.ti].params[r.pj].pos)),
        ht: this.rows(H, part.map(r => ex.tools[r.ti].pos)),
        slot: Float32Array.from(part, r => r.k),
        ha,
        opt: first && ochunks.length ? this.rows(H, ochunks[0].map(o => optPos[o])) : new Float32Array(d),
      };
      const r = this._run('args', feed, nq);
      if (first) {
        sk = r.span_k.slice(0, nq * 4 * dk);
        tk = r.tag_k.slice(0, nq * 3 * dk);
        if (ochunks.length) ochunks[0].forEach((o, i) => okAll.set(o, r.opt_k.slice(i * dk, (i + 1) * dk)));
      }
      const m = part.length;
      const op = r.op.slice(0, m * this.nOps), aux = r.aux.slice(0, m * this.nAux), nul = r.null.slice(0, m * 4);
      const sq = r.span_q.slice(0, m * 4 * dk), tq = r.tag_q.slice(0, m * 3 * dk), oq = r.opt_q.slice(0, m * dk);
      if (first)
        for (const oc of ochunks.slice(1)) {
          const r2 = this._run('args', {...feed, opt: this.rows(H, oc.map(o => optPos[o]))}, nq);
          oc.forEach((o, i) => okAll.set(o, r2.opt_k.slice(i * dk, (i + 1) * dk)));
        }
      part.forEach((rw, i) => {
        const p = prow[s + i];
        let spans = null, tags = null, en = null;
        res.push({
          op: op.subarray(i * this.nOps, (i + 1) * this.nOps),
          aux: aux.subarray(i * this.nAux, (i + 1) * this.nAux),
          spans() {  // [4][nq + 1], last = null slot
            if (!spans) {
              spans = [];
              for (let k = 0; k < 4; k++) {
                const v = new Float32Array(nq + 1);
                for (let q = 0; q < nq; q++) v[q] = f32(f32(dot(sk, (q * 4 + k) * dk, sq, (i * 4 + k) * dk, dk)) / scale);
                v[nq] = nul[i * 4 + k];
                spans.push(v);
              }
            }
            return spans;
          },
          tags() {  // [nq][3]
            if (!tags) {
              tags = [];
              for (let q = 0; q < nq; q++) {
                const v = new Float32Array(3);
                for (let c = 0; c < 3; c++) v[c] = f32(f32(dot(tk, (q * 3 + c) * dk, tq, (i * 3 + c) * dk, dk)) / scale);
                tags.push(v);
              }
            }
            return tags;
          },
          enum() {  // [E], -1e4 beyond the parameter's options
            if (!en) {
              en = new Float32Array(E).fill(-1e4);
              for (let e = 0; e < nopt[p]; e++) en[e] = f32(f32(dot(okAll.get(ostart[p] + e), 0, oq, i * dk, dk)) / scale);
            }
            return en;
          },
        });
      });
    }
    return res;
  }
}

/**
 * A call that leaves a required parameter without a value is a guess, not an answer: the user has not said enough yet
 * (an assistant would ask). Such a record gets no call at all. Battle test (held-out Glaive no-call rows) 57.9 -> 72.4
 * correct, BFCL 55.9 -> 57.6, no Gauge suite lower.
 */
export function requireArgs(calls, tools) {
  const by = new Map((tools || []).map(t => [t.name, t]));
  const incomplete = c => { const t = by.get(c.tool); return t && t.params.some(p => p.required && !(p.name in c.args)); };
  if (FIX.perCallHold) return calls.filter(c => !incomplete(c));  // hold only the incomplete calls
  return calls.some(incomplete) ? [] : calls;
}

/** ALL-CAPS requests -> lower case (the cased tokenizer splits capitals into fragments). */
function capsFix(record) {
  if (!FIX.caps) return record;
  const letters = (record.text || '').match(/\p{L}/gu) || [];
  const upper = letters.filter(ch => ch !== ch.toLowerCase()).length;
  return letters.length >= 6 && upper / letters.length > 0.7 ? {...record, text: record.text.toLowerCase()} : record;
}

const bytes = x => (x instanceof Uint8Array ? x : x instanceof ArrayBuffer ? new Uint8Array(x) :
  ArrayBuffer.isView(x) ? new Uint8Array(x.buffer, x.byteOffset, x.byteLength) : x);

export class Relay {
  constructor(net, tok, {gate = 0, gate2 = 2, require = true} = {}) {
    this.net = net;
    this.tok = tok;
    this.gate = gate;
    this.gate2 = gate2;
    this.require = require;
  }

  /**
   * @param wasm       Gust runtime (bytes, or a URL/string to fetch), e.g. gust-relay-simd.wasm
   * @param artifacts  {encoder, route, anchor, args}: the .gust files as bytes
   * @param tokenizer  relay-tokenizer.json, parsed or as text/bytes
   * @param require    drop the calls of a record when one leaves a required parameter empty (requireArgs)
   * @param threads    > 1: run on that many threads; `wasm` must then be the threaded runtime (gust-relay-mt.wasm,
   *                   bytes) and SharedArrayBuffer available (gust-threads.mjs); call close() when done
   * @param GustEngine optional Gust binding class; defaults to the vendored ./gust.mjs
   * @param fixes      host decoding fixes from the audit (hostfix.mjs): true (default), false (Python parity) or {key: bool}
   */
  static async create({wasm, artifacts, tokenizer, gate = 0, gate2 = 2, require = true, threads = 1,
    GustEngine = null, fixes = true}) {
    setFixes(fixes);  // host decoding fixes (hostfix.mjs), on by default since 2026-10-17; false = Python-parity decoder
    const GE = GustEngine || (await import('./gust.mjs')).GustEngine;
    const threaded = threads > 1 || (typeof wasm !== 'string' && !(wasm instanceof URL) &&
      WebAssembly.Module.imports(new WebAssembly.Module(bytes(wasm))).some(i => i.kind === 'memory'));
    const engine = threaded
      ? await (await import('./gust-threads.mjs')).createThreadedEngine(bytes(wasm), {threads})
      : await GE.create(typeof wasm === 'string' || wasm instanceof URL ? wasm : bytes(wasm));
    const models = {};
    for (const k of artifacts.lower ? ['lower', 'upper', 'route', 'anchor', 'args'] : ['encoder', 'route', 'anchor', 'args']) {
      if (!artifacts[k]) throw new Error(`missing artifact ${k}`);
      models[k] = engine.load(bytes(artifacts[k]));
    }
    if (tokenizer instanceof Uint8Array || tokenizer instanceof ArrayBuffer)
      tokenizer = new TextDecoder().decode(bytes(tokenizer));
    const relay = new Relay(new GustNet(models), new Tok(new Tokenizer(tokenizer)), {gate, gate2, require});
    relay.engine = engine;
    return relay;
  }

  _record(record, explain = false) {
    record = capsFix(record);
    const r = predictRecord(this.net, record, this.tok, {gate: this.gate, gate2: this.gate2, explain});
    if (this.require) r.calls = requireArgs(r.calls, record.tools);
    return r;
  }

  /** Stops the worker threads of a threaded instance (no-op otherwise). */
  close() {
    if (this.engine && this.engine.close) this.engine.close();
  }

  /** Calls for one record {id?, text, tools, kind?, context?}: [{tool, args}]. */
  predict(record) {
    return this._record(record).calls;
  }

  /** Per-sequence results: [{ids, offs, margin, calls}] (long tool lists are split into several sequences). */
  predictChunks(record) {
    return this._record(record).chunks.map(c => ({ids: c.ex.ids, offs: c.ex.offs, margin: c.margin, calls: c.calls}));
  }

  /**
   * For demos and debugging: {calls, dropped, chunks}. Each call carries `why` (per argument: value program, its
   * confidence and the UTF-16 span of the request it read) and `anchor`; each chunk carries per-tool route margins
   * (tool score minus the 'none' score). `dropped` holds calls that requireArgs withheld.
   */
  explain(record) {
    record = capsFix(record);
    const r = predictRecord(this.net, record, this.tok, {gate: this.gate, gate2: this.gate2, explain: true});
    const kept = this.require ? requireArgs(r.calls, record.tools) : r.calls;
    return {calls: kept, dropped: r.calls.filter(c => !kept.includes(c)),
      chunks: r.chunks.map(c => ({ids: c.ex.ids, offs: c.ex.offs, margin: c.margin, margins: c.margins}))};
  }

  /** Both at once: {calls, chunks}. */
  predictFull(record) {
    const r = this._record(record);
    return {calls: r.calls,
      chunks: r.chunks.map(c => ({ids: c.ex.ids, offs: c.ex.offs, margin: c.margin, calls: c.calls}))};
  }
}
