// Compares the model API's JSON (PREFIX.api.jsonl or PREFIX.stream.jsonl) with the JavaScript reference, bit for bit.
// Usage: node tools/compare_api.mjs out/squint-seq api|stream
import {readdirSync, readFileSync, existsSync} from 'node:fs';
const [dir = 'out/squint-seq', kind = 'api'] = process.argv.slice(2);
const rows = f => readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const bits = v => { const b = new DataView(new ArrayBuffer(8)); b.setFloat64(0, v); return b.getBigUint64(0).toString(16).padStart(16, '0'); };
let frames = 0, bad = 0, seqs = 0;
for (const f of readdirSync(dir).filter(f => f.endsWith('.ref.jsonl'))) {
  const name = f.replace('.ref.jsonl', ''), c = `${dir}/${name}.${kind}.jsonl`;
  if (!existsSync(c)) continue;
  seqs++;
  const a = rows(`${dir}/${f}`), b = rows(c);
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    frames++;
    const x = a[i], y = b[i]?.r;
    const same = y && !!x.accepted === !!y.accepted && (!x.accepted || (x.version === y.version && x.payload === y.payload && x.path === y.path &&
      JSON.stringify(x.corners) === JSON.stringify(y.corners.flat().map(bits)))) && (x.accepted || (x.reason ?? null) === (y.reason ?? null));
    if (!same) { d++; if (d === 1) console.log(`${name} frame ${i}:`, JSON.stringify(x).slice(0, 200), '|', JSON.stringify(b[i]).slice(0, 200)); }
  }
  bad += d;
}
console.log(`${kind}: ${seqs} sequences, ${frames} frames, ${bad} differing`);
process.exitCode = bad ? 1 : 0;
