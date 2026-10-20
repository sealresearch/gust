// The same engine benchmark as native, through Node's WASI host. Only the model's directory is exposed.
// node runtime/tools/bench_wasm.mjs out/gust-bench.wasm MODEL.gust [lengths=35,350,600] [reps=11]
import {readFileSync} from 'node:fs';
import {basename, dirname, resolve} from 'node:path';
import {WASI} from 'node:wasi';

const [wasm, model, lengths = '35,350,600', reps = '11'] = process.argv.slice(2);
if (!wasm || !model) throw new Error('usage: bench_wasm.mjs WASM MODEL.gust [lengths] [reps]');
const wasi = new WASI({version: 'preview1', returnOnExit: true,
  args: ['gust-bench', `/model/${basename(model)}`, lengths, reps],
  env: process.env.GUST_BENCH_RELAXED ? {GUST_BENCH_RELAXED: process.env.GUST_BENCH_RELAXED} : {},
  preopens: {'/model': dirname(resolve(model))}});
const {instance} = await WebAssembly.instantiate(readFileSync(wasm), {wasi_snapshot_preview1: wasi.wasiImport});
process.exitCode = wasi.start(instance);
console.log(`peak RSS ${process.resourceUsage().maxRSS} KiB, guest memory ${instance.exports.memory.buffer.byteLength} bytes`);
