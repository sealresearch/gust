// One worker thread of the multi-threaded Gust engine (gust-threads.mjs): instantiates the shared module over the
// shared memory, moves its stack pointer to its own stack region, reports ready, and parks in gust_worker(id).
async function start({module, memory, sp, id}, post) {
  const instance = await WebAssembly.instantiate(module, {env: {memory}});
  instance.exports.__stack_pointer.value = sp;
  post('ready');
  instance.exports.gust_worker(id); // never returns
}

if (typeof process !== 'undefined' && process.versions?.node) {
  const {parentPort} = await import('node:worker_threads');
  parentPort.once('message', m => start(m, x => parentPort.postMessage(x)));
} else {
  self.onmessage = e => start(e.data, x => self.postMessage(x));
}
