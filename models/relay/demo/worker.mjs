// Runs Relay off the main thread on the C host compiled to WebAssembly (relay-native.mjs): the same code and bytes as
// the native builds. Messages: {type:'init', threads} -> {type:'ready', info};
// {type:'run', id, record} -> {type:'result', id, result, ms}. Long tool lists go through the retrieval prefilter
// (top 10 tools within 600 tool tokens) inside the host.
import {RelayNative} from '/web/relay-native.mjs';

let relay = null;
// i8x16.relaxed_swizzle in a one-function module (wasm-feature-detect): Safari has SIMD128 but not relaxed SIMD
const RELAXED = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
  10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]));
const get = async (u, t) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return t ? r.text() : new Uint8Array(await r.arrayBuffer()); };

async function init(threads) {
  const t0 = performance.now();
  const mt = threads > 1 && self.crossOriginIsolated;
  const parts = ['encoder', 'route', 'anchor', 'args'];
  const file = k => `/model/${k === 'encoder' ? 'relay' : `relay-${k}`}.gust`;
  const [wasm, tokenizer, prefilter, ...bins] = await Promise.all([
    get(`/wasm/relay${mt ? '-mt' : ''}${RELAXED ? '' : '-simd'}.wasm`), get('/model/relay-tokenizer.json', true),
    get('/model/relay-prefilter.bin'), ...parts.map(k => get(file(k)))]);
  const files = Object.fromEntries(parts.map((k, i) => [k, bins[i]]));
  if (relay) relay.close();
  relay = await RelayNative.create({wasm, files: {...files, tokenizer, prefilter}, gate: 0, gate2: 2, fixes: true,
    threads: mt ? threads : 1});
  const bytes = bins.reduce((s, b) => s + b.length, prefilter.length);
  return {threads: mt ? threads : 1, isolated: self.crossOriginIsolated, loadMs: performance.now() - t0,
    modelBytes: bytes, wasmBytes: wasm.length, relaxed: RELAXED};
}

self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') self.postMessage({type: 'ready', info: await init(data.threads || 1)});
    else if (data.type === 'run') {
      const t0 = performance.now();
      const result = relay.explain(data.record);
      result.prefilter = {tools: (data.record.tools || []).length};
      self.postMessage({type: 'result', id: data.id, result, ms: performance.now() - t0});
    }
  } catch (e) {
    self.postMessage({type: 'error', id: data.id, error: String(e && e.stack || e)});
  }
};
