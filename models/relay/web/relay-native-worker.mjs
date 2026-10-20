// One worker of a threaded Relay host (relay-native.mjs, relay-mt.wasm): instantiates the module over the shared
// memory (no _initialize: the main thread did that), moves its stack pointer to its own region, reports ready and
// parks in gust_worker(id), running the operator tasks the executor hands out.
const stubs = {
  fd_write: () => 0, fd_close: () => 0, fd_seek: () => 70,
  proc_exit(code) { throw new Error(`relay wasm worker exited (${code})`); },
};

async function start({module, memory, sp, id}, post) {
  const instance = await WebAssembly.instantiate(module, {env: {memory}, wasi_snapshot_preview1: stubs});
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
