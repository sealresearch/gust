// Real-model trace regression: tracing must preserve normal calls and the existing explanation.
// node models/relay/tests/trace.mjs WASM MODEL_DIR RECORDS.jsonl EXPECTED.json OUTPUT_DIR [THREADS]
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {RelayNative} from '../web/relay-native.mjs';

const [wasm, model, recordsPath, expectedPath, output, threadsArg = '1'] = process.argv.slice(2);
const threads = Number(threadsArg);
const names = {encoder:'relay.gust', route:'relay-route.gust', anchor:'relay-anchor.gust',
  args:'relay-args.gust', tokenizer:'relay-tokenizer.json', prefilter:'relay-prefilter.bin'};
const files = Object.fromEntries(Object.entries(names).map(([k, v]) => [k, readFileSync(join(model, v))]));
const records = readFileSync(recordsPath, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const expected = JSON.parse(readFileSync(expectedPath, 'utf8'));
mkdirSync(output, {recursive:true});
const relay = await RelayNative.create({wasm:readFileSync(wasm), files, threads});
const graphBytes = relay.modelBytes();
try {
  for (const row of records) {
    const trace = relay.trace(row); // cold tool vectors on the first call
    assert.deepEqual(trace.result, relay.explain(row));
    assert.equal(relay.predictJSON(row), expected[row.id]);
    assert.deepEqual(trace.result.calls.map(({tool,args}) => ({tool,args})), JSON.parse(expected[row.id]));
    assert(trace.events.some(e => e.event === 'selected_tools'));
    assert(trace.events.filter(e => e.event === 'graph').every(e => e.status === 0));
    assert.deepEqual(relay.trace(row).result, trace.result); // warm cache, same decision
    writeFileSync(join(output, `${row.id}.json`), JSON.stringify(trace, null, 2));
  }
  assert.throws(() => relay.trace('{'), /valid JSON/);
  const unusual = {text:'hello '.repeat(250), tools:[{name:'quote"\\\n☃', description:'hello', params:[]}]};
  const trace = relay.trace(unusual);
  assert.deepEqual(trace.result, relay.explain(unusual));
  assert(trace.events.some(e => e.event === 'request_tokens' && e.tokens > e.kept));
  assert(trace.events.some(e => e.event === 'selected_tools' && e.tools[0] === unusual.tools[0].name));
  assert.deepEqual(relay.trace({text:'',tools:[]}).result, relay.explain({text:'',tools:[]}));
  assert.deepEqual(relay.trace({tools:[]}).result, relay.explain({tools:[]}));
  assert.equal(relay.predictJSON(records[0]), expected[records[0].id]);
  assert.equal(relay.modelBytes(), graphBytes);
  const detailed = relay.trace(records[0], {operators:true});
  assert.deepEqual(detailed.result, relay.explain(records[0]));
  const begins = detailed.events.filter(e => e.event === 'operator_begin');
  const ends = detailed.events.filter(e => e.event === 'operator_end');
  assert(begins.length > 0);
  assert.deepEqual(begins.map(({event,...rest}) => rest), ends.map(({event,...rest}) => rest));
  assert(ends.every(e => e.rows > 0 && e.rows <= e.shape[0] && e.status === 0));
  assert(!relay.trace(records[0]).events.some(e => e.event.startsWith('operator_')));
} finally { relay.close(); }
// Without retrieval, all tools must reach sequence construction, possibly in several chunks.
const full = await RelayNative.create({wasm:readFileSync(wasm), files, prefilter:false, threads});
try {
  const trace = full.trace(records[0]);
  assert.deepEqual(trace.result, full.explain(records[0]));
  const chunks = trace.events.filter(e => e.event === 'chunk');
  assert(chunks.length > 1);
  assert.equal(chunks.reduce((n, c) => n + c.tools.length, 0), records[0].tools.length);
  assert(chunks.every(c => c.tokens <= 1024));
} finally { full.close(); }
console.log(`${records.length} real requests: exact calls/explanations, cold/warm trace; invalid input, escaping, truncation, empty input and multi-chunk checks passed`);
