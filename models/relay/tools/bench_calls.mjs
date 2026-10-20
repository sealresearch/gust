import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {RelayNative} from '../web/relay-native.mjs';
const [wasm, model, input, repetitionsArg, threadsArg] = process.argv.slice(2);
const repetitions = Number(repetitionsArg), threads = Number(threadsArg);
if (!(repetitions >= 1 && repetitions <= 31 && [1, 8].includes(threads))) throw Error('benchmark arguments');
const root = model;
const names = {encoder:'relay.gust', route:'relay-route.gust', anchor:'relay-anchor.gust',
  args:'relay-args.gust', tokenizer:'relay-tokenizer.json', prefilter:'relay-prefilter.bin'};
const files = Object.fromEntries(Object.entries(names).map(([key, value]) => [key, readFileSync(join(root, value))]));
const relay = await RelayNative.create({wasm:readFileSync(wasm), threads, prefilter:true, files});
try {
  console.log(`graph_bytes=${relay.modelBytes()} prefilter=1 threads=${threads}`);
  const records = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  if (!records.length) throw Error('empty input');
  for (let record = 0; record < records.length; record++) {
    const line = records[record], tools = JSON.parse(line).tools.length;
    let reference;
    for (let repeat = 0; repeat < repetitions; repeat++) {
      const start = performance.now(), calls = relay.predictJSON(line), ms = performance.now() - start;
      if (reference !== undefined && calls !== reference) throw Error('repeat output differs');
      reference = calls;
      console.log(`record=${record} tools=${tools} repeat=${repeat} elapsed_ms=${ms.toFixed(3)} calls=${calls}`);
    }
  }
  console.log(`peak_rss_kib=${process.resourceUsage().maxRSS} guest_bytes=${relay.x.memory.buffer.byteLength}`);
} finally { await relay.close(); }
