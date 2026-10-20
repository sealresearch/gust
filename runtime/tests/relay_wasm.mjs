// Relay through a WebAssembly build (out/relay.wasm or out/relay-mt.wasm): writes <out dir>/<suite>.jsonl in the
// CLI's format.   node relay_wasm.mjs WASM MODEL_DIR EVAL_DIR OUT_DIR SUITES [THREADS]
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';

const [wasm, model, evalDir, outDir, suites, threads = '1'] = process.argv.slice(2);
const here = new URL('.', import.meta.url);
const {RelayNative} = await import(new URL('../../models/relay/web/relay-native.mjs', here));
const f = name => { try { return readFileSync(join(model, name)); } catch { return null; } };
const r = await RelayNative.create({wasm: readFileSync(wasm), threads: +threads, prefilter: false, files: {
  encoder: f('relay.gust'), route: f('relay-route.gust'),
  anchor: f('relay-anchor.gust'), args: f('relay-args.gust'), tokenizer: f('relay-tokenizer.json')}});
console.log(`graphs ${(r.modelBytes() / 1e6).toFixed(1)} MB, ${threads} thread(s)`);
mkdirSync(outDir, {recursive: true});
for (const s of suites.split(',')) {
  const ms = [];
  const out = readFileSync(join(evalDir, `${s}.jsonl`), 'utf8').split('\n').filter(Boolean)
    .map(l => {
      const start = performance.now();
      const calls = r.predictJSON(l);
      ms.push(performance.now() - start);
      return `{"id": ${JSON.stringify(JSON.parse(l).id ?? null)}, "calls": ${calls}}`;
    });
  writeFileSync(join(outDir, `${s}.jsonl`), out.join('\n') + '\n');
  ms.sort((a, b) => a - b);
  if (ms.length) console.log(`${s}: ${ms.length} rows, median ${ms[ms.length >> 1].toFixed(1)} ms, ` +
    `p90 ${ms[Math.floor(ms.length * 0.9)].toFixed(1)} ms`);
}
console.log(`peak RSS ${process.resourceUsage().maxRSS} KiB, guest memory ${r.x.memory.buffer.byteLength} bytes`);
r.close();
void pathToFileURL;
