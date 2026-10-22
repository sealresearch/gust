// One operator thread (gust-models-mt.wasm). It instantiates the module over the shared memory (the main thread already ran
// _initialize), moves its stack pointer to a region of its own, says it is ready and parks in gust_worker(id), running the
// tasks the executor hands out. It never returns; the main thread terminates it.
const stubs = {
  fd_write: () => 0, fd_close: () => 0, fd_seek: () => 70,
  proc_exit(code) { throw new Error(`the Gust module exited in a worker (${code})`); },
};

async function start({ module, memory, sp, id }, post) {
  const instance = await WebAssembly.instantiate(module, { env: { memory }, wasi_snapshot_preview1: stubs });
  instance.exports.__stack_pointer.value = sp;
  post('ready');
  instance.exports.gust_worker(id);
}

if (typeof process !== 'undefined' && process.versions?.node) {
  const { parentPort } = await import('node:worker_threads');
  parentPort.once('message', (m) => start(m, (x) => parentPort.postMessage(x)));
} else {
  self.onmessage = (e) => start(e.data, (x) => self.postMessage(x));
}
