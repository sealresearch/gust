import {parentPort, workerData} from 'node:worker_threads';
import {WASI} from 'node:wasi';

const {module, memory, id} = workerData;
const wasi = new WASI({version: 'preview1', returnOnExit: true, args: [], env: {}, preopens: {}});
const instance = await WebAssembly.instantiate(module, {env: {memory}, wasi_snapshot_preview1: wasi.wasiImport});
instance.exports.__stack_pointer.value = memory.buffer.byteLength - id * 1048576;
if (!id) wasi.initialize(instance);
parentPort.postMessage({ready: true});
if (id) instance.exports.gust_worker(id);
else parentPort.once('message', () => {
  const status = instance.exports.test_run();
  parentPort.postMessage({status, poolBytes: instance.exports.test_pool_bytes(), generation: String(instance.exports.test_generation())});
});
