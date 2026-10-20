// Relay on the native C host compiled to WebAssembly (host/relay_wasm.c): the same code as the native builds, so
// browsers, Node and devices give byte-identical calls. This file only moves bytes in and out of the module.
//
//   const relay = await RelayNative.create({wasm, files: {encoder, route, anchor, args, tokenizer, prefilter?}});
//   relay.predict({text, tools: [...], kind?, context?})  ->  [{tool, args}]
const ORDER = ['encoder', 'route', 'anchor', 'args', 'tokenizer', 'prefilter'];

const bytes = x => (x instanceof Uint8Array ? x : x instanceof ArrayBuffer ? new Uint8Array(x) :
  ArrayBuffer.isView(x) ? new Uint8Array(x.buffer, x.byteOffset, x.byteLength) :
  typeof x === 'string' ? new TextEncoder().encode(x) : null);

// The module is a WASI reactor that never touches files; these stubs satisfy its imports.
function wasiStubs(getMemory) {
  return {
    fd_write(fd, iovs, n, written) {  // stderr / stdout of an abort message
      const m = new DataView(getMemory().buffer);
      let total = 0;
      for (let i = 0; i < n; i++) total += m.getUint32(iovs + 8 * i + 4, true);
      m.setUint32(written, total, true);
      return 0;
    },
    fd_close: () => 0,
    fd_seek: () => 70,  // ESPIPE
    proc_exit(code) { throw new Error(`relay wasm exited (${code})`); },
  };
}

async function spawn(module, memory, sp, id) {
  const url = new URL('./relay-native-worker.mjs', import.meta.url);
  if (typeof process !== 'undefined' && process.versions?.node) {
    const {Worker} = await import('node:worker_threads');
    const w = new Worker(url);
    await new Promise((resolve, reject) => { w.once('message', resolve); w.once('error', reject); w.postMessage({module, memory, sp, id}); });
    w.unref(); // parked workers must not keep the process alive
    return w;
  }
  const w = new Worker(url, {type: 'module'});
  await new Promise((resolve, reject) => { w.onmessage = resolve; w.onerror = reject; w.postMessage({module, memory, sp, id}); });
  return w;
}

export class RelayNative {
  constructor(instance, handle) {
    this.x = instance.exports;
    this.h = handle;
    this.workers = [];
    this.enc = new TextEncoder();
    this.dec = new TextDecoder();
  }

  /**
   * @param wasm   relay-simd.wasm (bytes, or a URL / path string to fetch)
   * @param files  {encoder, route, anchor, args, tokenizer, prefilter?} as bytes (tokenizer may be text)
   * @param fixes  host decoding fixes (default on); prefilter: rank long tool lists when files.prefilter is given
   * @param threads > 1: relay-mt.wasm on that many threads (Web Workers / worker_threads; browsers need a cross-origin
   *                isolated page for SharedArrayBuffer); results are identical for any thread count. close() when done.
   */
  static async create({wasm, files, fixes = true, prefilter = true, k = 10, budget = 600, gate = 0, gate2 = 2,
    require = true, threads = 1}) {
    const src = typeof wasm === 'string' || wasm instanceof URL ? await (await fetch(wasm)).arrayBuffer() : bytes(wasm);
    const module = await WebAssembly.compile(src);
    const shared = WebAssembly.Module.imports(module).some(i => i.kind === 'memory');
    let memory = shared ? new WebAssembly.Memory({initial: 1024, maximum: 16384, shared: true}) : null;
    const imports = {wasi_snapshot_preview1: wasiStubs(() => memory)};
    if (shared) imports.env = {memory};
    const instance = await WebAssembly.instantiate(module, imports);
    memory = instance.exports.memory;
    instance.exports._initialize();
    const x = instance.exports;
    const table = x.relay_wasm_alloc(64);
    const owned = [];
    ORDER.forEach((name, i) => {
      const b = files[name] ? bytes(files[name]) : null;
      let p = 0;
      if (b) {
        p = x.relay_wasm_alloc(b.length);
        new Uint8Array(memory.buffer, p, b.length).set(b);
        owned.push(p);
      }
      const v = new DataView(memory.buffer);
      v.setUint32(table + 8 * i, p, true);
      v.setUint32(table + 8 * i + 4, b ? b.length : 0, true);
    });
    const h = x.relay_wasm_open(table, (fixes ? 1 : 0) | (prefilter ? 2 : 0), k, budget, gate, gate2, require ? 1 : 0);
    for (const p of owned) x.relay_wasm_free(p);  // the host keeps its own copies
    x.relay_wasm_free(table);
    if (!h) {
      const e = x.relay_wasm_error();
      const m = new Uint8Array(memory.buffer, e);
      throw new Error(`relay: ${new TextDecoder().decode(m.slice(0, m.indexOf(0)))}`);
    }
    const relay = new RelayNative(instance, h);
    if (shared && threads > 1) {
      for (let id = 1; id < threads; id++) {
        const stack = 1 << 20, base = x.relay_wasm_alloc(stack);
        relay.workers.push(await spawn(module, memory, base + stack, id));
      }
      relay.threads = x.gust_set_threads(threads);
    }
    return relay;
  }

  _call(fn, record) {
    const x = this.x, b = this.enc.encode(typeof record === 'string' ? record : JSON.stringify(record));
    const p = x.relay_wasm_alloc(b.length);
    new Uint8Array(x.memory.buffer, p, b.length).set(b);
    const out = fn(this.h, p, b.length);
    x.relay_wasm_free(p);
    if (!out) throw new Error('relay: the record is not valid JSON');
    const m = new Uint8Array(x.memory.buffer, out);
    const s = this.dec.decode(m.slice(0, m.indexOf(0)));  // a copy: browsers do not decode shared memory
    x.relay_wasm_free(out);
    return s;
  }

  /** Calls of one record as the JSON text the native CLI writes (Python float literals kept). */
  predictJSON(record) { return this._call(this.x.relay_wasm_call, record); }

  /** Calls for one record {text, tools, kind?, context?}: [{tool, args}]. */
  predict(record) { return JSON.parse(this.predictJSON(record)); }

  /**
   * For demos and debugging, as relay.explain() of the JavaScript host: {calls, dropped, chunks}. Each call carries
   * `why` (per argument: value program with UTF-16 spans of the request, conf, optProb), `anchor` and `margin`; each
   * chunk carries per-tool route margins. `dropped` holds the calls requireArgs withheld.
   */
  explain(record) { return JSON.parse(this._call(this.x.relay_wasm_explain, record)); }

  /** Ordered retrieval, token/sequence and graph events plus the existing explain result. Diagnostic overhead only. */
  trace(record, {operators = false} = {}) {
    return JSON.parse(this._call((h, p, n) => this.x.relay_wasm_trace(h, p, n, operators ? 1 : 0), record));
  }

  /** Bytes held by the loaded graphs. */
  modelBytes() { return this.x.relay_wasm_model_bytes(this.h) >>> 0; }

  close() {
    if (this.workers.length) { this.x.gust_set_threads(1); for (const w of this.workers) w.terminate(); this.workers = []; }
    if (this.h) { this.x.relay_wasm_close(this.h); this.h = 0; }
  }
}
