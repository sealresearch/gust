// Models, streams and the values that go in and out of them. Nothing here is specific to a model: a model says what it takes
// and gives in its description (model.info), and this module moves typed buffers. It mirrors the Python package.
import { BUF, GustError, acquire, cores, release } from './runtime.mjs';
import { obtain, registry, resolve } from './fetch.mjs';

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const BUFFER_BYTES = 32; // gust_buffer on wasm32: size, type, data, len, rate, channels, width, height
const FILE_BYTES = 12;   // gust_model_file on wasm32: name, data, len

const DTYPES = { float32: Float32Array, int16: Int16Array };
const bytesOf = (view) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

/** Audio: interleaved samples (Float32Array or Int16Array) with their rate and channel count. A plain array becomes float32;
 *  bytes need a `dtype` of "float32" or "int16". */
export class Pcm {
  constructor(samples, rate, channels = 1, dtype) {
    this.rate = rate | 0;
    this.channels = channels | 0;
    if (samples instanceof Float32Array || samples instanceof Int16Array) this.samples = samples;
    else if (samples instanceof ArrayBuffer || samples instanceof Uint8Array) {
      const T = DTYPES[dtype];
      if (!T) throw new TypeError('bytes need dtype "float32" or "int16"');
      const b = samples instanceof Uint8Array ? samples : new Uint8Array(samples);
      this.samples = new T(b.slice().buffer);
    } else if (Array.isArray(samples) || ArrayBuffer.isView(samples)) this.samples = Float32Array.from(samples);
    else throw new TypeError('samples must be a Float32Array, an Int16Array, an array of numbers, or bytes with a dtype');
  }

  get dtype() { return this.samples instanceof Int16Array ? 'int16' : 'float32'; }
  get length() { return this.samples.length; }
  toString() { return `Pcm(${this.length} samples, ${this.dtype}, ${this.rate} Hz, ${this.channels} ch)`; }
}

/** A frame: `pixels` (Uint8Array, Uint8ClampedArray such as ImageData.data, or an ArrayBuffer), `width`, `height` and a
 *  `format` of "gray8" or "rgba8". */
export class Image {
  constructor(pixels, width, height, format = 'gray8') {
    if (format !== 'gray8' && format !== 'rgba8') throw new TypeError('format must be "gray8" or "rgba8"');
    this.pixels = pixels instanceof ArrayBuffer ? new Uint8Array(pixels) : pixels;
    this.width = width | 0;
    this.height = height | 0;
    this.format = format;
  }

  toString() { return `Image(${this.width}x${this.height}, ${this.format})`; }
}

/** Describes a value as [type, bytes, extra fields] for a gust_buffer. */
function describeValue(value) {
  if (value instanceof Pcm) return [value.dtype === 'int16' ? BUF.PCM_I16 : BUF.PCM_F32, bytesOf(value.samples), { rate: value.rate, channels: value.channels }];
  if (value instanceof Image) return [value.format === 'rgba8' ? BUF.IMAGE_RGBA8 : BUF.IMAGE_GRAY8, bytesOf(value.pixels), { width: value.width, height: value.height }];
  if (typeof value === 'string') return [BUF.TEXT, encoder.encode(value), {}];
  if (value instanceof ArrayBuffer) return [BUF.BYTES, new Uint8Array(value), {}];
  if (ArrayBuffer.isView(value)) return [BUF.BYTES, bytesOf(value), {}];
  const json = JSON.stringify(value === undefined ? null : value); // anything else travels as JSON: objects, arrays, numbers, booleans, null
  return [BUF.JSON, encoder.encode(json), {}];
}

/** Writes the inputs into the module as a gust_buffer array. Everything allocated is pushed on `owned` for the caller to free. */
function writeInputs(rt, inputs, owned) {
  const array = rt.alloc(BUFFER_BYTES * Math.max(inputs.length, 1));
  owned.push(array);
  inputs.forEach((value, i) => {
    const [type, bytes, extra] = describeValue(value);
    const data = rt.put(bytes);
    owned.push(data);
    const v = rt.view;
    const at = array + BUFFER_BYTES * i;
    v.setUint32(at, BUFFER_BYTES, true);
    v.setUint32(at + 4, type, true);
    v.setUint32(at + 8, data, true);
    v.setUint32(at + 12, bytes.length, true);
    v.setUint32(at + 16, extra.rate ?? 0, true);
    v.setUint32(at + 20, extra.channels ?? 0, true);
    v.setUint32(at + 24, extra.width ?? 0, true);
    v.setUint32(at + 28, extra.height ?? 0, true);
  });
  return array;
}

/** Copies a result out of the module into JavaScript values, and frees it. `raw` keeps JSON outputs as the text the model wrote. */
function readResult(rt, result, raw) {
  try {
    const out = [];
    const count = rt.x.gust_result_count(result) >>> 0;
    for (let i = 0; i < count; i++) {
      const at = rt.x.gust_result_get(result, i) >>> 0;
      const v = rt.view;
      const type = v.getUint32(at + 4, true);
      const data = v.getUint32(at + 8, true);
      const len = v.getUint32(at + 12, true);
      const rate = v.getUint32(at + 16, true);
      const channels = v.getUint32(at + 20, true);
      const width = v.getUint32(at + 24, true);
      const height = v.getUint32(at + 28, true);
      const copy = len ? rt.u8.slice(data, data + len) : new Uint8Array(0); // a copy: the module reuses its memory, and browsers do not decode shared memory
      if (type === BUF.JSON) { const text = decoder.decode(copy); out.push(raw ? text : JSON.parse(text)); }
      else if (type === BUF.TEXT) out.push(decoder.decode(copy));
      else if (type === BUF.PCM_F32) out.push(new Pcm(new Float32Array(copy.buffer), rate, channels));
      else if (type === BUF.PCM_I16) out.push(new Pcm(new Int16Array(copy.buffer), rate, channels));
      else if (type === BUF.IMAGE_GRAY8 || type === BUF.IMAGE_RGBA8) out.push(new Image(copy, width, height, type === BUF.IMAGE_RGBA8 ? 'rgba8' : 'gray8'));
      else out.push(copy);
    }
    return out;
  } finally {
    rt.x.gust_result_free(result);
  }
}

const optionsJson = (options) => (options && Object.keys(options).length ? JSON.stringify(options) : null);
const one = (list) => (list.length === 1 ? list[0] : list);

async function withRuntime(fn) {
  const rt = await acquire(1);
  try { return fn(rt); } finally { release(rt); }
}

/** The ids of the models in this build of the module. */
export function models() {
  return withRuntime((rt) => {
    const out = [];
    const base = rt.x.gust_model_ids() >>> 0;
    for (let i = 0; ; i++) {
      const p = rt.view.getUint32(base + 4 * i, true);
      if (!p) break;
      out.push(rt.str(p));
    }
    return out;
  });
}

function describeOn(rt, id) {
  const p = rt.cstr(id);
  try {
    const text = rt.str(rt.x.gust_model_describe(p) >>> 0);
    if (text == null) throw new GustError(2, `no model called "${id}" is built in`);
    return JSON.parse(text);
  } finally { rt.free(p); }
}

/** What a model is, takes and gives, as an object, before it is opened. */
export function describe(modelId) {
  const id = registry.aliases?.[modelId] ?? modelId;
  return withRuntime((rt) => describeOn(rt, id));
}

/** The instruction set this build runs ("WebAssembly SIMD128"). */
export function runtime() {
  return withRuntime((rt) => rt.str(rt.x.gust_model_runtime() >>> 0));
}

async function threadCount(threads) {
  if (threads === 'auto') return cores();
  if (!Number.isInteger(threads) || threads < 1 || threads > 16) throw new TypeError('threads must be a whole number from 1 to 16, or "auto"');
  return threads;
}

async function openOn(rt, id, files, options) {
  const list = files instanceof Map ? [...files] : Array.isArray(files) ? files : Object.entries(files ?? {});
  const owned = [];
  try {
    const table = rt.alloc(FILE_BYTES * Math.max(list.length, 1));
    owned.push(table);
    list.forEach(([name, data], i) => {
      const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : ArrayBuffer.isView(data) ? bytesOf(data) : encoder.encode(String(data));
      const n = rt.cstr(name);
      const d = rt.put(bytes);
      owned.push(n, d);
      const v = rt.view;
      v.setUint32(table + FILE_BYTES * i, n, true);
      v.setUint32(table + FILE_BYTES * i + 4, d, true);
      v.setUint32(table + FILE_BYTES * i + 8, bytes.length, true);
    });
    const idp = rt.cstr(id);
    const opts = rt.cstr(optionsJson(options));
    owned.push(idp, opts);
    const cell = rt.cell();
    rt.check(rt.x.gust_model_open(idp, table, list.length, opts, cell));
    return rt.view.getUint32(cell, true);
  } finally {
    for (const p of owned) rt.free(p);
  }
}

/**
 * A loaded model.
 *
 *     const model = await Model.load('relay-1');
 *     model.run({ text: '...', tools: [...] });      // one input in, one value out
 *
 * The calls are synchronous and take as long as the model takes: in a browser, load and run the model in a Worker.
 */
export class Model {
  #rt;
  #h;

  constructor(rt, handle, id) {
    this.#rt = rt;
    this.#h = handle;
    this.id = id;
    this.info = JSON.parse(rt.str(rt.x.gust_model_info(handle) >>> 0));
  }

  /**
   * Finds the model's files (in `directory`, in the cache, or downloaded from Hugging Face and checked against the registry's
   * SHA-256), then opens it.
   *
   * @param {string} id
   * @param {object} [o]
   * @param {object} [o.options]  What the model's description lists under options.open.
   * @param {number|'auto'} [o.threads=1]  Operator threads. More than one needs SharedArrayBuffer (Node, or a cross-origin
   *   isolated page); without it the count is 1. The answers are the same for every count.
   * @param {string} [o.directory]  Read the files from this folder (Node) instead of downloading them.
   * @param {boolean|string} [o.cache=true]  Keep downloads: true for the default place, a folder path in Node, false for none.
   * @param {string} [o.baseUrl='https://huggingface.co']
   * @param {boolean} [o.optional=true]  Also fetch the files the model lists as optional (Relay's prefilter, for example).
   * @param {boolean} [o.verify=true]  Check files read from `directory` against the registry.
   * @param {string} [o.revision]  The Hugging Face revision to download, instead of the registry's.
   * @param {(p: {file: string, loaded: number, total: number, cached?: boolean}) => void} [o.onProgress]
   */
  static async load(id, { options, threads = 1, directory, cache = true, baseUrl, optional = true, verify = true, revision, onProgress } = {}) {
    const [modelId, entry] = resolve(id);
    const n = await threadCount(threads);
    const rt = await acquire(n);
    try {
      const info = describeOn(rt, modelId);
      let names = null;
      if (info.files) names = new Set([...(info.files.required ?? []), ...(optional ? info.files.optional ?? [] : [])]);
      const files = await obtain(modelId, entry, { directory, cache, baseUrl, names, verify, onProgress, revision });
      return await Model.#create(rt, modelId, files, options, n);
    } catch (e) { release(rt); throw e; }
  }

  /** Opens a model from files you already have in memory (an object of name to bytes, a Map, or a list of `[name, bytes]`):
   *  no registry, no download. `load` is this plus finding the files. */
  static async open(id, files = [], { options, threads = 1 } = {}) {
    const n = await threadCount(threads);
    const rt = await acquire(n);
    try { return await Model.#create(rt, id, files, options, n); } catch (e) { release(rt); throw e; }
  }

  static async #create(rt, id, files, options, threads) {
    if (rt.kind === 'mt') await rt.setThreads(threads);
    const handle = await openOn(rt, id, files, options);
    return new Model(rt, handle, id);
  }

  /** The number of threads the operators run on. */
  get threads() { return this.#rt.kind === 'mt' ? this.#rt.threads : 1; }

  /** Changes the thread count for the models that share this runtime. Returns the count in use. */
  async setThreads(n) {
    this.#open();
    return this.#rt.setThreads(await threadCount(n));
  }

  /** Bytes the loaded model holds (0 when the model does not say). */
  get memoryBytes() {
    this.#open();
    return this.#rt.x.gust_model_memory(this.#h) >>> 0;
  }

  /**
   * One call: the input goes in, the output comes back (a list when the model gives several). What a model takes is in
   * `model.info.inputs`. JavaScript values go in as JSON (objects, arrays, numbers), text (strings), bytes (Uint8Array),
   * audio (Pcm) or images (Image).
   *
   * @param {*} input
   * @param {{options?: object, raw?: boolean}} [o]  `options` for this call; `raw` keeps JSON outputs as the text the model wrote.
   */
  run(input, o) { return one(this.#run([input], o)); }

  /** Like `run` for a model that takes several inputs at once: `runAll([a, b])`. Always returns a list. */
  runAll(inputs, o) { return this.#run(inputs, o); }

  #run(inputs, { options, raw = false } = {}) {
    this.#open();
    const rt = this.#rt;
    const owned = [];
    try {
      const array = writeInputs(rt, inputs, owned);
      const opts = rt.cstr(optionsJson(options));
      owned.push(opts);
      const cell = rt.cell();
      rt.check(rt.x.gust_model_run(this.#h, array, inputs.length, opts, cell));
      return readResult(rt, rt.view.getUint32(cell, true), raw);
    } finally {
      for (const p of owned) rt.free(p);
    }
  }

  /** A stream for models that work on live input (speech in, audio out). See {@link Stream}. */
  stream(options) {
    this.#open();
    const rt = this.#rt;
    const opts = rt.cstr(optionsJson(options));
    try {
      const cell = rt.cell();
      rt.check(rt.x.gust_stream_open(this.#h, opts, cell));
      return new Stream(rt, rt.view.getUint32(cell, true), this);
    } finally { rt.free(opts); }
  }

  #open() { if (!this.#h) throw new GustError(9, 'the model is closed'); }

  /** Frees the model, and stops the worker threads when it was the last one using them. Call it when you are done. */
  close() {
    if (!this.#h) return;
    this.#rt.x.gust_model_close(this.#h);
    this.#h = 0;
    release(this.#rt);
  }

  [Symbol.dispose ?? Symbol.for('Symbol.dispose')]() { this.close(); }
  toString() { return `Model(${this.id})`; }
}

/**
 * Push input as it arrives, pull what is ready, finish when the input ends.
 *
 *     const s = model.stream();
 *     for (const chunk of chunks) { s.push(chunk); for (const out of s.pull()) ...; }   // pull may return an empty list
 *     s.finish();
 *     for (const out of s.drain()) ...;                                                 // everything that is left
 *     s.close();
 */
export class Stream {
  #rt;
  #h;

  constructor(rt, handle, model) {
    this.#rt = rt;
    this.#h = handle;
    this.model = model; // keeps the model object alive
    /** True once the stream is finished and drained. */
    this.done = false;
  }

  push(...inputs) {
    this.#open();
    const rt = this.#rt;
    const owned = [];
    try {
      const array = writeInputs(rt, inputs, owned);
      rt.check(rt.x.gust_stream_push(this.#h, array, inputs.length));
    } finally { for (const p of owned) rt.free(p); }
  }

  /** The outputs that are ready now, as a list (possibly empty). */
  pull({ raw = false } = {}) {
    this.#open();
    const rt = this.#rt;
    const cell = rt.cell();
    rt.check(rt.x.gust_stream_pull(this.#h, cell, cell + 4));
    const v = rt.view;
    this.done = v.getInt32(cell + 4, true) !== 0;
    return readResult(rt, v.getUint32(cell, true), raw);
  }

  finish() {
    this.#open();
    this.#rt.check(this.#rt.x.gust_stream_finish(this.#h));
  }

  /** After finish(): yields every remaining output until the stream is done. */
  * drain(o) {
    for (;;) {
      yield* this.pull(o);
      if (this.done) return;
    }
  }

  #open() { if (!this.#h) throw new GustError(9, 'the stream is closed'); }

  close() {
    if (!this.#h) return;
    this.#rt.x.gust_stream_close(this.#h);
    this.#h = 0;
  }

  [Symbol.dispose ?? Symbol.for('Symbol.dispose')]() { this.close(); }
}
