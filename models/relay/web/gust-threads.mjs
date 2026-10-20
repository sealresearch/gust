// Multi-threaded Gust engine: the threaded runtime (gust-relay-mt.wasm: shared memory, atomics, GUST_THREADS)
// instantiated once on this thread and once per worker over the same shared memory. The workers park in
// gust_worker() and run the operator tasks the executor hands out (runtime/src/float32/threads.c); results are
// bit-identical to the single-threaded runtime for any thread count.
// Node: worker_threads. Browsers: module Web Workers; SharedArrayBuffer needs a cross-origin isolated page
// (Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp). The calling thread busy-waits
// for its tasks (never Atomics.wait), so it may be a page's main thread, though a worker keeps the page responsive.
//
//   const engine = await createThreadedEngine(wasmBytes, {threads: 4});   ...   engine.close();
import {GustEngine} from './gust.mjs';

const PAGES = 3072; // 192 MiB, the runtime's fixed --initial-memory = --max-memory
const STACK = 1 << 20;

async function spawn(module, memory, sp, id) {
  const url = new URL('./gust-worker.mjs', import.meta.url);
  if (typeof process !== 'undefined' && process.versions?.node) {
    const {Worker} = await import('node:worker_threads');
    const w = new Worker(url);
    await new Promise((resolve, reject) => {
      w.once('message', resolve);
      w.once('error', reject);
      w.postMessage({module, memory, sp, id});
    });
    w.unref(); // parked workers must not keep the process alive
    return w;
  }
  const w = new Worker(url, {type: 'module'});
  await new Promise((resolve, reject) => {
    w.onmessage = resolve;
    w.onerror = reject;
    w.postMessage({module, memory, sp, id});
  });
  return w;
}

/** Returns a GustEngine whose operators run on `threads` threads (this one plus threads - 1 workers). */
export async function createThreadedEngine(wasm, {threads = 4} = {}) {
  if (typeof SharedArrayBuffer === 'undefined') throw new Error('threads need SharedArrayBuffer (cross-origin isolation)');
  const module = await WebAssembly.compile(wasm);
  const memory = new WebAssembly.Memory({initial: PAGES, maximum: PAGES, shared: true});
  const instance = await WebAssembly.instantiate(module, {env: {memory}});
  const engine = new GustEngine(instance);
  const workers = [];
  for (let id = 1; id < threads; id++) {
    const stack = engine.allocate(STACK);
    workers.push(await spawn(module, memory, stack.offset + stack.size, id));
  }
  engine.threads = instance.exports.gust_set_threads(threads);
  engine.close = () => {
    instance.exports.gust_set_threads(1);
    for (const w of workers) w.terminate();
  };
  return engine;
}
