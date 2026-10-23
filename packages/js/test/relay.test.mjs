// Relay through the generic API. Run with a checkout of the Hugging Face repository (sealresearch/relay) in RELAY_MODEL_DIR,
// after `npm run sync`:    RELAY_MODEL_DIR=../../../hf/relay npm test
// Downloads are served from a local HTTP server that mimics Hugging Face's /<repo>/resolve/<revision>/<file>.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Model, registry } from '../src/index.mjs';

const DIR = process.env.RELAY_MODEL_DIR;
const skip = DIR ? false : 'set RELAY_MODEL_DIR to a checkout of the Hugging Face repository';
const entry = registry.models['relay-1'];

const request = {
  text: 'Show me the Eiffel Tower on the map',
  tools: [
    { name: 'show_map', description: 'Shows a location on the map.', params: [{ name: 'query', type: 'string', description: 'The place to show.', required: true }] },
    { name: 'turn_on_flashlight', description: 'Turns the flashlight on.', params: [] },
  ],
};
const expected = [{ tool: 'show_map', args: { query: 'Eiffel Tower' } }];

function serve({ corrupt } = {}) {
  const hits = [];
  const server = http.createServer((req, res) => {
    const prefix = `/${entry.repo}/resolve/${entry.revision}/`;
    if (!req.url.startsWith(prefix)) { res.writeHead(404).end(); return; }
    const name = req.url.slice(prefix.length);
    hits.push(name);
    let body;
    try { body = Buffer.from(readFileSync(join(DIR, name))); } catch { res.writeHead(404).end(); return; }
    if (name === corrupt) body[body.length >> 1] ^= 0xff;
    res.writeHead(200, { 'content-length': body.length }).end(body);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({
    url: `http://127.0.0.1:${server.address().port}`,
    hits,
    close: () => new Promise((r) => server.close(r)),
  })));
}

test('downloads, checks and caches the model', { skip }, async () => {
  const server = await serve();
  const cache = mkdtempSync(join(tmpdir(), 'gust-js-'));
  try {
    const progress = [];
    const first = await Model.load('relay-1', { baseUrl: server.url, cache, onProgress: (p) => progress.push(p) });
    assert.deepEqual(first.run(request), expected);
    assert.ok(first.memoryBytes > 0);
    first.close();
    assert.equal(server.hits.length, 6);
    assert.ok(progress.some((p) => p.file === 'relay.gust' && p.loaded === 9856380));

    const second = await Model.load('relay', { baseUrl: server.url, cache });
    assert.deepEqual(second.run(request), expected);
    second.close();
    assert.equal(server.hits.length, 6, 'the second load should come from the cache');
  } finally {
    await server.close();
    rmSync(cache, { recursive: true, force: true });
  }
});

test('refuses a download that does not match its checksum', { skip }, async () => {
  const server = await serve({ corrupt: 'relay-args.gust' });
  const cache = mkdtempSync(join(tmpdir(), 'gust-js-'));
  try {
    await assert.rejects(Model.load('relay-1', { baseUrl: server.url, cache }), /relay-args\.gust does not match its checksum/);
  } finally {
    await server.close();
    rmSync(cache, { recursive: true, force: true });
  }
});

test('loads from a folder, with or without the optional files', { skip }, async () => {
  const model = await Model.load('relay-1', { directory: DIR, optional: false });
  try {
    assert.deepEqual(model.run(request), expected);
    assert.deepEqual(model.run({ text: 'Tell me a joke', tools: request.tools }), []);
  } finally { model.close(); }
});

test('one thread and four give the same text, and so do the modes', { skip }, async () => {
  const one = await Model.load('relay-1', { directory: DIR, threads: 1 });
  const four = await Model.load('relay-1', { directory: DIR, threads: 4 });
  try {
    assert.equal(four.threads, 4);
    for (const text of ['Show me the Eiffel Tower on the map', 'Turn on the flashlight and show me Paris', 'Set an alarm']) {
      const r = { ...request, text };
      assert.equal(four.run(r, { raw: true }), one.run(r, { raw: true }));
      assert.deepEqual(four.run(r, { options: { mode: 'explain' } }), one.run(r, { options: { mode: 'explain' } }));
    }
    assert.ok(Object.keys(one.run(request, { options: { mode: 'trace' } })).length > 0, 'a trace has content');
  } finally { one.close(); four.close(); }
});

test('says what is wrong with a request', { skip }, async () => {
  const model = await Model.load('relay-1', { directory: DIR, optional: false });
  try {
    assert.throws(() => model.run({ tools: [] }), (e) => e.status === 'input' && /text \(string\) and tools \(array\)/.test(e.message));
    assert.throws(() => model.run({ text: 'hi' }), (e) => e.status === 'input');
    assert.throws(() => model.run('not a request'), (e) => e.status === 'input');
  } finally { model.close(); }
});

test('options reach the model', { skip }, async () => {
  const model = await Model.load('relay-1', { directory: DIR, options: { prefilter: false, k: 3 } });
  try { assert.deepEqual(model.run(request), expected); } finally { model.close(); }
});

// The same two tools in every format an app is likely to have: Relay's own, OpenAI's (bare and wrapped), Anthropic's and MCP's.
test('takes tools written as JSON Schema', { skip }, async () => {
  const defs = [
    ['set_alarm', 'Sets an alarm.', { type: 'object', properties: { time: { type: 'string', description: 'ISO time' }, label: { type: 'string', description: 'Label' } }, required: ['time'] }],
    ['get_weather', 'Weather for a city.', { type: 'object', properties: { city: { type: 'string', description: '' }, unit: { type: 'string', description: '', enum: ['celsius', 'fahrenheit'] } }, required: ['city'] }],
  ];
  const own = [
    { name: 'set_alarm', description: 'Sets an alarm.', params: [{ name: 'time', type: 'string', description: 'ISO time', required: true }, { name: 'label', type: 'string', description: 'Label', required: false }] },
    { name: 'get_weather', description: 'Weather for a city.', params: [{ name: 'city', type: 'string', description: '', required: true }, { name: 'unit', type: 'string', description: '', required: false, enum: ['celsius', 'fahrenheit'] }] },
  ];
  const formats = {
    openai: defs.map(([name, description, parameters]) => ({ name, description, parameters })),
    'openai, wrapped': defs.map(([name, description, parameters]) => ({ type: 'function', function: { name, description, parameters } })),
    anthropic: defs.map(([name, description, input_schema]) => ({ name, description, input_schema })),
    mcp: defs.map(([name, description, inputSchema]) => ({ name, description, inputSchema })),
    'mixed with Relay format': [{ name: 'set_alarm', description: 'Sets an alarm.', parameters: defs[0][2] }, own[1]],
  };
  const model = await Model.load('relay-1', { directory: DIR, optional: false });
  try {
    for (const text of ['Set an alarm for 6:45 tomorrow called gym', 'What is the weather in Paris in fahrenheit', 'Tell me a joke']) {
      const context = { now: '2026-10-23T09:00:00' };
      const want = model.run({ text, tools: own, context }, { raw: true });
      for (const [name, tools] of Object.entries(formats)) assert.equal(model.run({ text, tools, context }, { raw: true }), want, `${name}: ${text}`);
    }
    assert.deepEqual(model.run({ text: 'Weather in Paris in fahrenheit', tools: formats.openai }), [{ tool: 'get_weather', args: { city: 'Paris', unit: 'fahrenheit' } }]);
  } finally { model.close(); }
});
