import {readFileSync} from 'node:fs';
import {Worker} from 'node:worker_threads';

const module = await WebAssembly.compile(readFileSync(process.argv[2] || new URL('../../out/workshare.wasm', import.meta.url)));
const memory = new WebAssembly.Memory({initial: 1024, maximum: 16384, shared: true});
const workers = [];
let timer;
try {
  const outcome = await Promise.race([
    (async () => {
      const ready = [];
      let completed;
      const result = new Promise((resolve, reject) => { completed = {resolve, reject}; });
      result.catch(() => {}); // Keep early worker failures handled while readiness is still pending.
      for (let id = 0; id < 8; id++) {
        const worker = new Worker(new URL('./workshare_wasm_worker.mjs', import.meta.url), {workerData: {module, memory, id}});
        workers.push(worker);
        ready.push(new Promise((resolve, reject) => {
          worker.on('message', value => {
            if (value.ready) resolve();
            else if (!id) completed.resolve(value);
          });
          worker.on('error', error => { reject(error); completed.reject(error); });
          worker.on('exit', code => {
            if (code) { const error = new Error(`worker ${id} exited ${code}`); reject(error); completed.reject(error); }
          });
        }));
      }
      await Promise.all(ready);
      workers[0].postMessage('run');
      return result;
    })(),
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('WASM workshare exceeded 180 seconds')), 180000); }),
  ]);
  if (outcome.status || outcome.poolBytes !== 104 || BigInt(outcome.generation) <= 0xffffffffn) {
    throw new Error(`WASM workshare failed: ${JSON.stringify(outcome)}`);
  }
  console.log(`WASM workshare: 100000 changing rounds, eight workers, generation ${outcome.generation}, pool ${outcome.poolBytes} bytes`);
} finally {
  clearTimeout(timer);
  await Promise.allSettled(workers.map(worker => worker.terminate()));
}
