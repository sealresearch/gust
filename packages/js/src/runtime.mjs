// The Gust model API (sdk/include/gust_model.h) compiled to WebAssembly, and the few lines that talk to it. Nothing here knows
// about any one model: it moves typed buffers in and out of linear memory and runs operator threads on workers.
//
// Two builds of the same C code: gust-models.wasm (one thread) and gust-models-mt.wasm (shared memory, workers). Their answers
// are bit for bit the same, and the same as the native libraries.
import { clockTimeGet } from './clock.mjs';

export const isNode = typeof process !== 'undefined' && !!process.versions?.node;

export const STATUS = ['ok', 'argument', 'unknown_model', 'files', 'open', 'input', 'run', 'unsupported', 'memory', 'state'];
export const BUF = { BYTES: 0, TEXT: 1, JSON: 2, PCM_F32: 3, PCM_I16: 4, IMAGE_GRAY8: 5, IMAGE_RGBA8: 6 };
export const ABI_VERSION = 1;

export class GustError extends Error {
  /** `status` names the failure: "argument", "unknown_model", "files", "open", "input", "run", "unsupported", "memory", "state". */
  constructor(code, message) {
    super(message);
    this.name = 'GustError';
    this.code = code;
    this.status = STATUS[code] ?? String(code);
  }
}

// Where the modules are. A page that serves them elsewhere (or a bundler that renames them) says so with configure().
const sources = { st: new URL('../wasm/gust-models.wasm', import.meta.url), mt: new URL('../wasm/gust-models-mt.wasm', import.meta.url) };

/** Points the package at its WebAssembly modules: `{ st, mt }`, each a URL, a path or the module's bytes. */
export function configure({ wasm } = {}) {
  if (wasm?.st) sources.st = wasm.st;
  if (wasm?.mt) sources.mt = wasm.mt;
  for (const k of Object.keys(compiled)) delete compiled[k];
}

const compiled = {}; // kind -> Promise<WebAssembly.Module>

async function wasmBytes(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  if (isNode && (source instanceof URL ? source.protocol === 'file:' : !/^https?:/.test(source))) {
    const fs = await import('node:fs/promises');
    return new Uint8Array(await fs.readFile(source));
  }
  const res = await fetch(source);
  if (!res.ok) throw new Error(`${source}: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}

function compile(kind) {
  compiled[kind] ??= wasmBytes(sources[kind]).then((b) => WebAssembly.compile(b)).catch((e) => { delete compiled[kind]; throw e; });
  return compiled[kind];
}

// The module is a WASI reactor that never touches files; these satisfy its imports.
function wasi(getMemory) {
  return {
    fd_write(fd, iovs, n, written) { // an abort message on stderr or stdout
      const m = new DataView(getMemory().buffer);
      let total = 0;
      for (let i = 0; i < n; i++) total += m.getUint32(iovs + 8 * i + 4, true);
      m.setUint32(written, total, true);
      return 0;
    },
    fd_close: () => 0,
    fd_seek: () => 70, // ESPIPE
    clock_time_get(id, _precision, out) { return clockTimeGet(getMemory(), id, out); }, // deadlines of models that have them
    proc_exit(code) { throw new Error(`the Gust module exited (${code})`); },
  };
}

async function spawn(module, memory, sp, id) {
  const url = new URL('./worker.mjs', import.meta.url);
  const hello = { module, memory, sp, id };
  if (isNode) {
    const { Worker } = await import('node:worker_threads');
    const w = new Worker(url);
    await new Promise((resolve, reject) => { w.once('message', resolve); w.once('error', reject); w.postMessage(hello); });
    w.unref(); // parked workers must not keep the process alive
    return w;
  }
  const w = new Worker(url, { type: 'module' });
  await new Promise((resolve, reject) => { w.onmessage = resolve; w.onerror = reject; w.postMessage(hello); });
  return w;
}

/** The most threads worth asking for on this machine. */
export async function cores() {
  if (isNode) return Math.min(8, (await import('node:os')).availableParallelism());
  return Math.min(8, globalThis.navigator?.hardwareConcurrency || 1);
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** One instance of the module. Models opened on the same Runtime share its memory and its threads. */
export class Runtime {
  static async create(kind) {
    const module = await compile(kind);
    const rt = new Runtime(kind, module);
    let memory = null;
    const imports = { wasi_snapshot_preview1: wasi(() => memory) };
    if (kind === 'mt') {
      memory = new WebAssembly.Memory({ initial: 1024, maximum: 16384, shared: true });
      imports.env = { memory };
    }
    const instance = await WebAssembly.instantiate(module, imports);
    memory = instance.exports.memory ?? memory;
    instance.exports._initialize();
    rt.x = instance.exports;
    rt.memory = memory;
    if (rt.x.gust_model_abi_version() !== ABI_VERSION) throw new Error(`the module speaks model API version ${rt.x.gust_model_abi_version()}, this package needs ${ABI_VERSION}`);
    rt.scratch = rt.x.malloc(16);
    return rt;
  }

  constructor(kind, module) {
    this.kind = kind;
    this.module = module;
    this.refs = 0;
    this.workers = [];
    this.stacks = [];
    this.threads = 1;
  }

  // ---- memory ----
  get u8() { return new Uint8Array(this.memory.buffer); }
  get view() { return new DataView(this.memory.buffer); }

  alloc(n) {
    const p = this.x.malloc(Math.max(n, 1)) >>> 0;
    if (!p) throw new GustError(8, 'the module is out of memory');
    return p;
  }

  free(p) { if (p) this.x.free(p); }

  /** Copies bytes into the module; the caller frees the address. */
  put(bytes) {
    const p = this.alloc(bytes.length);
    this.u8.set(bytes, p);
    return p;
  }

  /** A NUL-terminated UTF-8 string in the module; null for no string. */
  cstr(text) {
    if (text == null) return 0;
    const b = encoder.encode(text);
    const p = this.alloc(b.length + 1);
    const u8 = this.u8;
    u8.set(b, p);
    u8[p + b.length] = 0;
    return p;
  }

  /** Reads a NUL-terminated string (a copy first: browsers do not decode shared memory). */
  str(p) {
    if (!p) return null;
    const u8 = this.u8;
    let end = p;
    while (u8[end]) end++;
    return decoder.decode(u8.slice(p, end));
  }

  /** The address of the language-neutral scratch cell (for out parameters), zeroed. */
  cell() { this.view.setUint32(this.scratch, 0, true); this.view.setUint32(this.scratch + 4, 0, true); return this.scratch; }

  /** Throws a GustError for a non-zero status, with the library's message. */
  check(code) {
    if (code) {
      const message = this.str(this.x.gust_model_last_error());
      throw new GustError(code, message || `failed (${STATUS[code] ?? code})`);
    }
  }

  // ---- threads ----
  /** Sets the operator threads (workers start or stop to match). Without shared memory the count is 1. Returns the count in use. */
  async setThreads(n) {
    if (this.kind !== 'mt') return 1;
    n = Math.max(1, Math.min(16, n | 0));
    while (this.workers.length < n - 1) {
      const id = this.workers.length + 1;
      const stack = 1 << 20;
      const base = this.alloc(stack);
      const worker = await spawn(this.module, this.memory, base + stack, id);
      this.workers.push(worker);
      this.stacks.push(base);
    }
    this.threads = this.x.gust_model_set_threads(n);
    while (this.workers.length > n - 1) {
      this.workers.pop().terminate();
      this.free(this.stacks.pop());
    }
    return this.threads;
  }

  /** Stops the workers; the instance is garbage once nothing refers to it. */
  shutdown() {
    if (this.workers.length) {
      this.x.gust_model_set_threads(1);
      for (const w of this.workers) w.terminate();
      this.workers = [];
      this.stacks = [];
    }
    this.threads = 1;
  }
}

// One live runtime per kind. The first model opened creates it; the last one closed shuts it down.
const live = {}; // kind -> Promise<Runtime>

/** A runtime for `threads` operator threads (more than one needs SharedArrayBuffer: Node, or a cross-origin isolated page). */
export async function acquire(threads) {
  const kind = threads > 1 && typeof SharedArrayBuffer !== 'undefined' ? 'mt' : 'st';
  live[kind] ??= Runtime.create(kind).catch((e) => { delete live[kind]; throw e; });
  const rt = await live[kind];
  rt.refs++;
  return rt;
}

export function release(rt) {
  if (--rt.refs > 0) return;
  rt.shutdown();
  if (live[rt.kind] && rt.refs <= 0) delete live[rt.kind];
}
