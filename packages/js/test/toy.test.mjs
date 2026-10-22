// The API with models that have nothing to do with Relay: text to text and streaming audio. They are built into a separate
// module for the tests:   GUST_TOY=1 WASM_OUT=out/toy bash build.sh wasm-models wasm-models-mt   (from the repository root)
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { GustError, Image, Model, Pcm, configure, describe, models, runtime } from '../src/index.mjs';

const toy = fileURLToPath(new URL('../../../out/toy/', import.meta.url));
const skip = existsSync(`${toy}gust-models.wasm`) ? false : 'build the toy modules first (see the top of this file)';
if (!skip) configure({ wasm: { st: `${toy}gust-models.wasm`, mt: `${toy}gust-models-mt.wasm` } });

test('lists and describes models', { skip }, async () => {
  assert.deepEqual(await models(), ['relay-1', 'toy-upper', 'toy-gain']);
  assert.match(await runtime(), /wasm32/);
  assert.equal((await describe('toy-upper')).kind, 'text');
  assert.equal((await describe('relay')).id, 'relay-1', 'aliases work');
  await assert.rejects(describe('nothing'), /no model called "nothing"/);
});

test('runs a text model', { skip }, async () => {
  const m = await Model.open('toy-upper');
  try {
    assert.equal(m.run('hello wörld'), 'HELLO WöRLD');
    assert.equal(m.run(''), '');
    assert.deepEqual(m.info, { id: 'toy-upper', kind: 'text' });
  } finally { m.close(); }
});

test('says what went wrong', { skip }, async () => {
  const m = await Model.open('toy-upper');
  try {
    assert.throws(() => m.run({ not: 'text' }), (e) => e instanceof GustError && e.status === 'input');
    assert.throws(() => m.stream(), (e) => e instanceof GustError && e.status === 'unsupported');
  } finally { m.close(); }
  assert.throws(() => m.run('x'), (e) => e instanceof GustError && e.status === 'state' && /closed/.test(e.message));
  await assert.rejects(Model.open('nope'), (e) => e.status === 'unknown_model');
});

test('streams audio: push, pull, finish, drain', { skip }, async () => {
  const m = await Model.open('toy-gain');
  const s = m.stream();
  try {
    s.push(new Pcm([0.5, -1, 0.25], 16000));
    s.push(new Pcm(new Float32Array([0.125]), 16000));
    const first = s.pull();
    const samples = first.flatMap((p) => [...p.samples]);
    s.finish();
    for (const p of s.drain()) samples.push(...p.samples);
    assert.deepEqual(samples, [1, -2, 0.5, 0.25]);
    assert.equal(s.done, true);
    assert.equal(first[0].rate, 16000);
    assert.equal(first[0].channels, 1);
    assert.throws(() => s.push(new Pcm([1], 16000)), GustError, 'a push after finish');
  } finally { s.close(); m.close(); }
});

test('typed buffers: audio, images and bytes', { skip }, async () => {
  const pcm = new Pcm(new Int16Array([1, 2, 3]), 8000, 2);
  assert.equal(pcm.dtype, 'int16');
  assert.equal(pcm.toString(), 'Pcm(3 samples, int16, 8000 Hz, 2 ch)');
  assert.deepEqual([...new Pcm(new Uint8Array(new Float32Array([1.5]).buffer), 8000, 1, 'float32').samples], [1.5]);
  assert.throws(() => new Pcm(new Uint8Array(4), 8000), /dtype/);
  assert.throws(() => new Image(new Uint8Array(4), 2, 2, 'rgb'), /format/);
  const m = await Model.open('toy-upper');
  try {
    assert.throws(() => m.run(pcm), (e) => e.status === 'input', 'a text model does not take audio, and says so');
    assert.throws(() => m.run(new Uint8Array([1, 2, 3])), (e) => e.status === 'input');
  } finally { m.close(); }
});

test('threads: more than one on the threaded build, the same answers', { skip }, async () => {
  const one = await Model.open('toy-upper');
  const four = await Model.open('toy-upper', [], { threads: 4 });
  try {
    assert.equal(one.threads, 1);
    assert.equal(four.threads, 4);
    assert.equal(four.run('threads'), one.run('threads'));
    assert.equal(await four.setThreads(2), 2);
    assert.equal(four.threads, 2);
    assert.equal(await four.setThreads(1), 1);
  } finally { one.close(); four.close(); }
  await assert.rejects(Model.open('toy-upper', [], { threads: 0 }), /threads must be/);
  await assert.rejects(Model.open('toy-upper', [], { threads: 99 }), /threads must be/);
});

test('closing is safe to repeat, and models do not share state', { skip }, async () => {
  const a = await Model.open('toy-upper');
  const b = await Model.open('toy-upper');
  a.close();
  a.close();
  assert.equal(b.run('still here'), 'STILL HERE');
  b.close();
});
