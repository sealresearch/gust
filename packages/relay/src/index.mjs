// @sealresearch/relay: Relay 1 in Node and in the browser, on the Gust WebAssembly runtime.
//
//   import { Relay } from '@sealresearch/relay';
//   const relay = await Relay.load();                      // downloads the 11.5 MB model once, then keeps it
//   relay.predict({ text: 'Show me the Eiffel Tower on the map', tools });   // [{ tool, args }]
//   relay.close();
//
// The model files come from Hugging Face, are checked against the checksums below, and are cached (a folder in Node,
// the Cache API in a browser). The runtime is the same C code as the native builds, so the calls are byte for byte
// the ones the native `sealpup` program returns, for any thread count.
import { RelayNative } from './relay-native.mjs';

const isNode = typeof process !== 'undefined' && !!process.versions?.node;

/** The model this version of the package runs, pinned to one revision of the Hugging Face repository. */
export const MODEL = Object.freeze({
  id: 'relay-1',
  repo: 'sealresearch/relay',
  revision: 'af0a7d8597b3cede00085975a3470a6def38b90f',
  files: Object.freeze({
    encoder: { name: 'relay.gust', bytes: 9856380, sha256: 'e5d48435515c903fdf4ee15498ef5166fbbefa245f69d27ca5ce353547f11c94' },
    route: { name: 'relay-route.gust', bytes: 79844, sha256: 'cef649607d726bf5559840b56c2c13cf8280b77327831c2c3570f2e46fc08fa4' },
    anchor: { name: 'relay-anchor.gust', bytes: 53976, sha256: 'fb69cbaac642d61c28650bc3f3ed1e7172268da6529f36817191d7af2c0ecf53' },
    args: { name: 'relay-args.gust', bytes: 1484472, sha256: '8ff795abc606e4552cb5dd10da71079c02a69cd633a820ab6d242470c1cf776b' },
    tokenizer: { name: 'relay-tokenizer.json', bytes: 460207, sha256: 'a8af081fa797c442495a3ee22862640dfce4e2a94962dd8a37be0f3396abbe3a' },
    prefilter: { name: 'relay-prefilter.bin', bytes: 2100554, sha256: 'c25ca7222c90584e1376985648e916286b973203069fe547404f29216b188c64' },
  }),
});

const hex = (buffer) => [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, '0')).join('');
const sha256 = async (bytes) => hex(await globalThis.crypto.subtle.digest('SHA-256', bytes));

async function check(bytes, file) {
  if (bytes.length !== file.bytes) throw new Error(`${file.name} is ${bytes.length} bytes, expected ${file.bytes}`);
  const sum = await sha256(bytes);
  if (sum !== file.sha256) throw new Error(`${file.name} does not match its checksum (got ${sum.slice(0, 12)}…, expected ${file.sha256.slice(0, 12)}…)`);
}

/** Where Node keeps downloaded models: $GUST_CACHE, else the platform's cache folder. */
async function defaultCacheDir() {
  const os = await import('node:os');
  const path = await import('node:path');
  if (process.env.GUST_CACHE) return process.env.GUST_CACHE;
  if (process.platform === 'win32') return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local'), 'gust', 'cache');
  if (process.platform === 'darwin') return path.join(os.homedir(), 'Library', 'Caches', 'gust');
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'gust');
}

async function nodeRead(file) {
  const fs = await import('node:fs/promises');
  return new Uint8Array(await fs.readFile(file));
}

async function fetchChecked(url, file, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${file.name}: ${res.status} ${res.statusText} from ${url}`);
  let bytes;
  if (res.body && onProgress) {
    const chunks = [];
    let loaded = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress({ file: file.name, loaded, total: file.bytes });
    }
    bytes = new Uint8Array(loaded);
    let at = 0;
    for (const c of chunks) { bytes.set(c, at); at += c.length; }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
  }
  await check(bytes, file);
  return bytes;
}

/** One model file: from the cache if it is there and intact, otherwise downloaded, checked and cached. */
async function obtain(file, { baseUrl, cacheDir, browserCache, onProgress }) {
  const url = `${baseUrl.replace(/\/+$/, '')}/${MODEL.repo}/resolve/${MODEL.revision}/${file.name}`;
  if (isNode && cacheDir) {
    const path = await import('node:path');
    const fs = await import('node:fs/promises');
    const dir = path.join(cacheDir, MODEL.id, MODEL.revision.slice(0, 12));
    const cached = path.join(dir, file.name);
    try {
      const bytes = await nodeRead(cached);
      await check(bytes, file);
      onProgress?.({ file: file.name, loaded: file.bytes, total: file.bytes, cached: true });
      return bytes;
    } catch { /* missing or damaged: download it again */ }
    const bytes = await fetchChecked(url, file, onProgress);
    await fs.mkdir(dir, { recursive: true });
    const part = `${cached}.${process.pid}.part`;
    await fs.writeFile(part, bytes);
    await fs.rename(part, cached);
    return bytes;
  }
  if (!isNode && browserCache && typeof caches !== 'undefined') {
    const store = await caches.open(`gust-${MODEL.id}-${MODEL.revision.slice(0, 12)}`);
    const hit = await store.match(url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      try {
        await check(bytes, file);
        onProgress?.({ file: file.name, loaded: file.bytes, total: file.bytes, cached: true });
        return bytes;
      } catch { await store.delete(url); }
    }
    const bytes = await fetchChecked(url, file, onProgress);
    await store.put(url, new Response(bytes));
    return bytes;
  }
  return fetchChecked(url, file, onProgress);
}

async function wasmBytes(name) {
  const url = new URL(`../wasm/${name}`, import.meta.url);
  if (isNode) return nodeRead(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function autoThreads() {
  if (isNode) return Math.min(8, (await import('node:os')).availableParallelism());
  return Math.min(8, globalThis.navigator?.hardwareConcurrency || 1);
}

const need = (cond, message) => { if (!cond) throw new TypeError(message); };

export class Relay {
  #native;

  constructor(native, info) {
    this.#native = native;
    /** The number of threads the operators run on. */
    this.threads = info.threads;
    /** The model, as in {@link MODEL}. */
    this.model = MODEL;
  }

  /**
   * Loads Relay: the model files (downloaded the first time), then the runtime.
   *
   * @param {object} [options]
   * @param {number|'auto'} [options.threads=1]  More than one needs worker threads (Node) or a cross-origin isolated
   *   page (browsers); without SharedArrayBuffer it falls back to one. The calls are the same for every count.
   * @param {boolean} [options.prefilter=true]  Rank long tool lists first and read only the best (downloads 2.1 MB more).
   * @param {string} [options.dir]  Read the model files from this folder (Node) instead of downloading them.
   * @param {string} [options.baseUrl='https://huggingface.co']  Where to download from.
   * @param {boolean|string} [options.cache=true]  Keep downloads: true for the default place, a folder path in Node, false for none.
   * @param {(p: {file: string, loaded: number, total: number, cached?: boolean}) => void} [options.onProgress]
   */
  static async load(options = {}) {
    const { dir, baseUrl = 'https://huggingface.co', cache = true, onProgress, prefilter = true, fixes = true, k = 10, budget = 600 } = options;
    let threads = options.threads === 'auto' ? await autoThreads() : options.threads ?? 1;
    need(Number.isInteger(threads) && threads >= 1 && threads <= 16, 'threads must be a whole number from 1 to 16, or "auto"');
    if (threads > 1 && typeof SharedArrayBuffer === 'undefined') threads = 1;

    const names = ['encoder', 'route', 'anchor', 'args', 'tokenizer', ...(prefilter ? ['prefilter'] : [])];
    const files = {};
    if (dir) {
      need(isNode, 'the dir option reads from disk, which only Node can do');
      const path = await import('node:path');
      await Promise.all(names.map(async (n) => {
        const bytes = await nodeRead(path.join(dir, MODEL.files[n].name));
        await check(bytes, MODEL.files[n]);
        files[n] = bytes;
      }));
    } else {
      const cacheDir = isNode && cache ? (typeof cache === 'string' ? cache : await defaultCacheDir()) : null;
      await Promise.all(names.map(async (n) => {
        files[n] = await obtain(MODEL.files[n], { baseUrl, cacheDir, browserCache: !!cache, onProgress });
      }));
    }

    const wasm = await wasmBytes(threads > 1 ? 'relay-mt-simd.wasm' : 'relay-simd.wasm');
    const native = await RelayNative.create({ wasm, files, threads, prefilter, fixes, k, budget });
    return new Relay(native, { threads: native.threads ?? 1 });
  }

  /**
   * The calls for one request, or an empty list when no tool fits.
   * @param {{text: string, tools: object[], kind?: string, context?: {now?: string}}} request
   * @returns {{tool: string, args: Record<string, unknown>}[]}
   */
  predict(request) {
    need(request && typeof request.text === 'string', 'request.text must be a string');
    need(Array.isArray(request.tools), 'request.tools must be an array of tool schemas');
    return this.#native.predict(request);
  }

  /** The calls as the JSON text the native program writes. */
  predictJSON(request) {
    need(request && typeof request.text === 'string', 'request.text must be a string');
    need(Array.isArray(request.tools), 'request.tools must be an array of tool schemas');
    return this.#native.predictJSON(request);
  }

  /** The calls plus why: per argument the value program, the words it came from and its confidence. */
  explain(request) { return this.#native.explain(request); }

  /** An ordered trace of retrieval, token counts and each graph, plus the explanation. Slower. */
  trace(request, options) { return this.#native.trace(request, options); }

  /** Bytes held by the loaded graphs. */
  get modelBytes() { return this.#native.modelBytes(); }

  /** Stops the worker threads and frees the model. Call it when you are done. */
  close() { this.#native.close(); }
}
