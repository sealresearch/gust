// Runs one sequence through @sealresearch/gust (packages/js) on the WebAssembly model module, one frame per Model.run, and
// writes PREFIX.wasm.jsonl for tools/compare_api.mjs. One process per sequence: the hosts keep state in the module.
// Usage: node models/squint/tools/wasm_run.mjs out/squint-seq/NAME [wasm-directory]   (default out/squint-wasm)
import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {Image, Model, configure} from '../../../packages/js/src/index.mjs';

const [prefix, dir = fileURLToPath(new URL('../../../out/squint-wasm/', import.meta.url))] = process.argv.slice(2);
configure({wasm: {st: `${dir}/gust-models.wasm`, mt: `${dir}/gust-models-mt.wasm`}});
const cmd = readFileSync(`${prefix}.cmd`, 'utf8').trim().split('\n');
const [w, h, n] = cmd[0].split(' ').map(Number);
const gray = readFileSync(`${prefix}.gray`);
const model = await Model.open('squint-1');
const lines = [];
for (let i = 0; i < n; i++) {
  const [reset, time] = cmd[i + 1].split(' ');
  const frame = new Uint8Array(gray.buffer, gray.byteOffset + i * w * h, w * h);
  const r = model.run(new Image(frame, w, h, 'gray8'), {options: {time: Number(time), ...(reset === '1' ? {reset: true} : {})}});
  lines.push(JSON.stringify({i, r}));
}
model.close();
writeFileSync(`${prefix}.wasm.jsonl`, lines.join('\n') + '\n');
