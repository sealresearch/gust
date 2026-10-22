// @sealresearch/relay: Relay 1 in Node and in the browser, on the Gust model API (@sealresearch/gust).
//
//   import { Relay } from '@sealresearch/relay';
//   const relay = await Relay.load();                      // downloads the 11.5 MB model once, then keeps it
//   relay.predict({ text: 'Show me the Eiffel Tower on the map', tools });   // [{ tool, args }]
//   relay.close();
//
// This is a thin layer over the generic package: the model files come from Hugging Face, are checked against the checksums in
// its registry and cached, and the runtime is the same C code as the native libraries, so the calls are byte for byte the ones
// the Python package and the `sealpup` command line return, for any thread count.
import { Model, registry } from '@sealresearch/gust';

const ID = 'relay-1';
const entry = registry.models[ID];
const byName = Object.fromEntries(entry.files.map((f) => [f.name, f]));
const file = (name) => ({ name, bytes: byName[name].size, sha256: byName[name].sha256 });

/** The model this version of the package runs, pinned to one revision of the Hugging Face repository. */
export const MODEL = Object.freeze({
  id: ID,
  repo: entry.repo,
  revision: 'af0a7d8597b3cede00085975a3470a6def38b90f',
  files: Object.freeze({
    encoder: file('relay.gust'),
    route: file('relay-route.gust'),
    anchor: file('relay-anchor.gust'),
    args: file('relay-args.gust'),
    tokenizer: file('relay-tokenizer.json'),
    prefilter: file('relay-prefilter.bin'),
  }),
});

const need = (cond, message) => { if (!cond) throw new TypeError(message); };

function check(request) {
  need(request && typeof request.text === 'string', 'request.text must be a string');
  need(Array.isArray(request.tools), 'request.tools must be an array of tool schemas');
}

export class Relay {
  #model;

  constructor(model) {
    this.#model = model;
    /** The model, as in {@link MODEL}. */
    this.model = MODEL;
  }

  /** The number of threads the operators run on. */
  get threads() { return this.#model.threads; }

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
    const { dir, baseUrl, cache = true, onProgress, threads = 1, prefilter = true, fixes = true, k = 10, budget = 600 } = options;
    const model = await Model.load(ID, {
      directory: dir, baseUrl, cache, onProgress, threads, revision: MODEL.revision,
      optional: prefilter, options: { prefilter, fixes, k, budget },
    });
    return new Relay(model);
  }

  /**
   * The calls for one request, or an empty list when no tool fits.
   * @param {{text: string, tools: object[], kind?: string, context?: {now?: string}}} request
   * @returns {{tool: string, args: Record<string, unknown>}[]}
   */
  predict(request) {
    check(request);
    return this.#model.run(request);
  }

  /** The calls as the JSON text the native program writes. */
  predictJSON(request) {
    check(request);
    return this.#model.run(request, { raw: true });
  }

  /** The calls plus why: per argument the value program, the words it came from and its confidence. */
  explain(request) { return this.#model.run(request, { options: { mode: 'explain' } }); }

  /** An ordered trace of retrieval, token counts and each graph, plus the explanation. Slower. */
  trace(request, { operators = false } = {}) {
    return this.#model.run(request, { options: { mode: operators ? 'trace_ops' : 'trace' } });
  }

  /** Bytes held by the loaded graphs. */
  get modelBytes() { return this.#model.memoryBytes; }

  /** Stops the worker threads and frees the model. Call it when you are done. */
  close() { this.#model.close(); }
}
