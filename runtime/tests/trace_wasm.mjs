// Run the same C observer contract under WASI with one model directory preopened read-only by convention.
import {WASI} from 'node:wasi';
import {readFileSync} from 'node:fs';
import {resolve, dirname, basename} from 'node:path';
const [wasm, model] = process.argv.slice(2);
const wasi = new WASI({version:'preview1', args:['trace-test', ...(model ? [`/model/${basename(model)}`] : [])],
  env:{}, preopens:model ? {'/model':dirname(resolve(model))} : {}});
const module = await WebAssembly.compile(readFileSync(wasm));
wasi.start(await WebAssembly.instantiate(module, {wasi_snapshot_preview1:wasi.wasiImport}));
