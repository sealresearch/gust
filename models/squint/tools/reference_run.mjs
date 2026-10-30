// Runs every sequence in out/squint-seq/ through the reference JavaScript Squint and records what it returns.
//   node reference_run.mjs [--dir DIR] [--only a,b] [--suffix .ref] [--scalar] [--nowarm] [--default-budgets]
// Writes <name><suffix>.jsonl per sequence (one line per frame), summary.json (when all sequences ran) and
// timing.json. The reference is loaded exactly as js/dist/index.js exports it (`Squint.load()`), one fresh instance
// per sequence. Wall-clock budgets are infinite so that timing cannot change results.
//   --scalar          load the scalar (non-SIMD) wasm builds instead of the SIMD ones, to compare the two
//   --nowarm          Squint.load({warm: false}), to see whether the synthetic warm-up leaves hidden state behind
//   --default-budgets use the library's default time budgets (results may then depend on machine speed)
import {closeSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, writeFileSync} from 'node:fs';
import os from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';

try {
  os.setPriority(0, os.constants.priority.PRIORITY_LOW);
} catch {
  /* best effort */
}
const arg = (n, f) => {
  const i = process.argv.indexOf(n);
  return i < 0 ? f : process.argv[i + 1];
};
const has = n => process.argv.includes(n);
const REF = (process.env.SQUINT_REF ?? 'C:/Users/rickh/Documents/sandbox/squint-release/models/squint').replace(/\\/g, '/');
const DIR = (arg('--dir', null) ?? fileURLToPath(new URL('../../../out/squint-seq/', import.meta.url))).replace(/\\/g, '/').replace(/\/?$/, '/');
const only = arg('--only', null)?.split(',');
const suffix = arg('--suffix', has('--scalar') ? '.ref-scalar' : has('--nowarm') ? '.ref-nowarm' : '.ref');
const {Squint, packageLoader} = await import(pathToFileURL(REF + '/js/dist/index.js').href);

const dv = new DataView(new ArrayBuffer(8));
const hex64 = x => {
  dv.setFloat64(0, x, false);
  return dv.getBigUint64(0, false).toString(16).padStart(16, '0');
};
const hex = bytes => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('hex');

const loadOptions = {};
if (has('--scalar')) {
  const base = packageLoader(pathToFileURL(REF + '/package/').href);
  loadOptions.load = name => base(name.replace(/\.wasm$/, '-scalar.wasm'));
}
if (has('--nowarm')) loadOptions.warm = false;
const budgets = has('--default-budgets') ? {} : {budgetMs: Infinity, acquireBudgetMs: Infinity, trackedAcquireBudgetMs: Infinity};

const names = readdirSync(DIR).filter(f => f.endsWith('.json') && !/^(summary|timing|manifest)(-[a-z]+)?\.json$/.test(f)).map(f => f.slice(0, -5)).sort();
const summary = {}, timing = {};
let totalFrames = 0;
const t00 = process.hrtime.bigint();
for (const name of names) {
  if (only && !only.includes(name)) continue;
  const meta = JSON.parse(readFileSync(DIR + name + '.json', 'utf8'));
  const {width, height, frames, times, resets, truth} = meta;
  const size = width * height;
  const resetAt = new Set(resets);
  const fd = openSync(DIR + name + '.gray', 'r');
  const squint = await Squint.load(loadOptions);
  const lines = [];
  const s = {frames, accepted: 0, wrong: 0, phantom: 0, exceptions: 0, paths: {}, reasons: {}, wrongFrames: [], phantomFrames: []};
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < frames; i++) {
    const gray = new Uint8Array(size);
    if (readSync(fd, gray, 0, size, size * i) !== size) throw new Error(`${name}: short read at frame ${i}`);
    if (resetAt.has(i)) squint.reset();
    let r;
    try {
      r = await squint.read(gray, width, height, {time: times[i], ...budgets});
    } catch (e) {
      r = {accepted: false, reason: 'exception: ' + (e && e.message ? e.message : String(e)), threw: true};
      s.exceptions++;
    }
    const line = {i, accepted: !!r.accepted, reason: null, path: null, version: null, payload: null, corners: null};
    if (r.accepted) {
      line.path = r.path ?? null;
      line.version = r.selectedVersion ?? null;
      line.payload = hex(r.payload);
      line.corners = Array.isArray(r.corners) && r.corners.length === 4 ? r.corners.flat().map(hex64) : null;
      s.accepted++;
      s.paths[line.path] = (s.paths[line.path] ?? 0) + 1;
      if (truth[i] === null) { s.phantom++; s.phantomFrames.push(i); }
      else if (line.payload !== truth[i]) { s.wrong++; s.wrongFrames.push(i); }
    } else {
      line.reason = r.reason ?? null;
      s.reasons[line.reason] = (s.reasons[line.reason] ?? 0) + 1;
    }
    lines.push(JSON.stringify(line));
  }
  closeSync(fd);
  squint.close();
  writeFileSync(DIR + name + suffix + '.jsonl', lines.join('\n') + '\n');
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  timing[name] = {ms: Math.round(ms), msPerFrame: Math.round((ms / frames) * 10) / 10};
  summary[name] = s;
  totalFrames += frames;
  console.log(name.padEnd(34), `${width}x${height}`.padEnd(10), `frames ${frames}`.padEnd(11), `accepted ${s.accepted}`.padEnd(13), `wrong ${s.wrong} phantom ${s.phantom} exc ${s.exceptions}`.padEnd(30), JSON.stringify(s.paths), `${(ms / 1000).toFixed(1)}s`);
}
if (!only) {
  const paths = {}, reasons = {};
  let accepted = 0, wrong = 0, phantom = 0, exceptions = 0;
  for (const s of Object.values(summary)) {
    accepted += s.accepted; wrong += s.wrong; phantom += s.phantom; exceptions += s.exceptions;
    for (const [k, v] of Object.entries(s.paths)) paths[k] = (paths[k] ?? 0) + v;
    for (const [k, v] of Object.entries(s.reasons)) reasons[k] = (reasons[k] ?? 0) + v;
  }
  const out = {suffix, totals: {sequences: Object.keys(summary).length, frames: totalFrames, accepted, wrong, phantom, exceptions, paths, reasons}, sequences: summary};
  writeFileSync(DIR + 'summary' + suffix.replace('.ref', '') + '.json', JSON.stringify(out, null, 1));
  writeFileSync(DIR + 'timing' + suffix.replace('.ref', '') + '.json', JSON.stringify({totalSeconds: Math.round(Number(process.hrtime.bigint() - t00) / 1e8) / 10, sequences: timing}, null, 1));
  console.log('TOTAL', JSON.stringify(out.totals));
}
