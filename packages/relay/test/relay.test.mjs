// Run with a checkout of the Hugging Face repository (sealresearch/relay) in RELAY_MODEL_DIR, after `npm run sync`:
//   RELAY_MODEL_DIR=../relay-model npm test
// The downloads are served from a local HTTP server that mimics Hugging Face's /<repo>/resolve/<revision>/<file>.
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MODEL, Relay } from '../src/index.mjs';

const DIR = process.env.RELAY_MODEL_DIR;
const skip = DIR ? false : 'set RELAY_MODEL_DIR to a checkout of the Hugging Face repository';

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
    const prefix = `/${MODEL.repo}/resolve/${MODEL.revision}/`;
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
  const cache = mkdtempSync(join(tmpdir(), 'gust-relay-'));
  try {
    const progress = [];
    const first = await Relay.load({ baseUrl: server.url, cache, onProgress: (p) => progress.push(p) });
    assert.deepEqual(first.predict(request), expected);
    assert.ok(first.modelBytes > 0);
    first.close();
    assert.equal(server.hits.length, 6);
    assert.ok(progress.some((p) => p.file === 'relay.gust' && p.loaded === MODEL.files.encoder.bytes));

    const second = await Relay.load({ baseUrl: server.url, cache });
    assert.deepEqual(second.predict(request), expected);
    second.close();
    assert.equal(server.hits.length, 6, 'the second load should come from the cache');
  } finally {
    await server.close();
    rmSync(cache, { recursive: true, force: true });
  }
});

test('refuses a download that does not match its checksum', { skip }, async () => {
  const server = await serve({ corrupt: 'relay-args.gust' });
  const cache = mkdtempSync(join(tmpdir(), 'gust-relay-'));
  try {
    await assert.rejects(Relay.load({ baseUrl: server.url, cache }), /relay-args\.gust does not match its checksum/);
  } finally {
    await server.close();
    rmSync(cache, { recursive: true, force: true });
  }
});

test('loads from a folder', { skip }, async () => {
  const relay = await Relay.load({ dir: DIR });
  try {
    assert.deepEqual(relay.predict(request), expected);
    assert.deepEqual(relay.predict({ text: 'Tell me a joke', tools: request.tools }), []);
  } finally {
    relay.close();
  }
});

test('gives the same calls on one thread and on four', { skip }, async () => {
  const one = await Relay.load({ dir: DIR, threads: 1 });
  const four = await Relay.load({ dir: DIR, threads: 4 });
  try {
    assert.equal(four.threads, 4);
    for (const text of ['Show me the Eiffel Tower on the map', 'Turn on the flashlight and show me Paris', 'Set an alarm']) {
      assert.equal(four.predictJSON({ ...request, text }), one.predictJSON({ ...request, text }));
    }
  } finally {
    one.close();
    four.close();
  }
});

test('says what is wrong with a request', { skip }, async () => {
  const relay = await Relay.load({ dir: DIR, prefilter: false });
  try {
    assert.throws(() => relay.predict({ tools: [] }), /request\.text must be a string/);
    assert.throws(() => relay.predict({ text: 'hi' }), /request\.tools must be an array/);
  } finally {
    relay.close();
  }
});
