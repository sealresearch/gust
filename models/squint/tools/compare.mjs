// Compares the C reader's output with the JavaScript reference, frame by frame and bit for bit.
// Usage: node tools/compare.mjs out/squint-seq [name ...]   (needs <name>.ref.jsonl and <name>.c.jsonl)
import {existsSync, readdirSync, readFileSync} from 'node:fs';
const dir = process.argv[2] ?? 'out/squint-seq';
const only = process.argv.slice(3);
const names = readdirSync(dir).filter(f => f.endsWith('.ref.jsonl')).map(f => f.replace(/\.ref\.jsonl$/, '')).filter(n => !only.length || only.includes(n));
const rows = f => readFileSync(f, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
let frames = 0, diffs = 0, missing = 0, accepted = 0, reasonDiffs = 0;
for (const name of names) {
  if (!existsSync(`${dir}/${name}.c.jsonl`)) { missing++; console.log(`${name}: no C output`); continue; }
  const a = rows(`${dir}/${name}.ref.jsonl`), b = rows(`${dir}/${name}.c.jsonl`);
  let d = 0, first = null;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i], y = b[i];
    frames++;
    if (x?.accepted) accepted++;
    const same = x && y && !y.error && !!x.accepted === !!y.accepted && (x.accepted
      ? x.version === y.version && x.payload === y.payload && JSON.stringify(x.corners) === JSON.stringify(y.corners) && x.path === y.path
      : true);
    if (x && y && !x.accepted && !y.accepted && (x.reason ?? null) !== (y.reason ?? null)) reasonDiffs++;
    if (!same) { d++; first ??= {i, ref: x, c: y}; }
  }
  diffs += d;
  console.log(`${name}: ${a.length} frames, ${d ? d + ' DIFFER' : 'identical'}`);
  if (first) console.log('  first difference:', JSON.stringify(first));
}
console.log(`\n${names.length} sequences, ${frames} frames (${accepted} accepted in the reference), ${diffs} differing, ${missing} without C output, ${reasonDiffs} frames with a different reject reason`);
process.exitCode = diffs || missing ? 1 : 0;
