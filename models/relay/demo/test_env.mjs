// Runs every example of one or more demo environments through Relay (C host, relay.wasm) in Node and prints the calls.
//   node models/relay/demo/test_env.mjs [--model DIR] phone home ...        (env keys = file names in demo/envs/)
import {readFileSync} from 'node:fs';
import {RelayNative} from '../web/relay-native.mjs';
import {buildRecord} from './schema.mjs';

const args = process.argv.slice(2);
const mi = args.indexOf('--model');
const G = (mi >= 0 ? args.splice(mi, 2)[1] : 'relay-model') + '/';
const rd = f => readFileSync(G + f);
const relay = await RelayNative.create({wasm: readFileSync(new URL('../../../out/relay.wasm', import.meta.url)),
  files: {encoder: rd('relay.gust'), route: rd('relay-route.gust'), anchor: rd('relay-anchor.gust'),
    args: rd('relay-args.gust'), tokenizer: rd('relay-tokenizer.json'), prefilter: rd('relay-prefilter.bin')}});
const NOW = '2026-10-17T09:41:00';
for (const key of args) {
  const env = (await import(`./envs/${key}.mjs`)).default;
  console.log(`=== ${key}: ${env.tools.length} tools`);
  for (const ex of env.examples) {
    const {record, routedBy} = buildRecord(env.tools, ex.q, {now: NOW, mode: env.mode || 'call'});
    const t0 = performance.now();
    const r = relay.explain(record);
    const ms = performance.now() - t0;
    const top = r.chunks.flatMap(c => c.margins || []).sort((a, b) => b.margin - a.margin)[0];
    const calls = r.calls.map(c => `${c.tool}(${Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`);
    const held = r.dropped.map(c => `HELD ${c.tool}(${Object.keys(c.args).join(',')})`);
    console.log(`- [${ex.title}] ${ex.q}\n    -> ${[...calls, ...held].join('  ') || '(no call)'}${routedBy.length ? `  [trigger: ${routedBy}]` : ''}  margin ${top ? top.margin.toFixed(1) : '-'}  ${ms.toFixed(0)} ms${ex.expect ? `\n    expect: ${ex.expect}` : ''}`);
  }
}
relay.close();
