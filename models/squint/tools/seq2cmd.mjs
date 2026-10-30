// Writes <name>.cmd next to <name>.json for the C parity driver: "W H N", then "reset time" for every frame.
// Usage: node tools/seq2cmd.mjs out/squint-seq
import {readdirSync, readFileSync, writeFileSync} from 'node:fs';
const dir = process.argv[2] ?? 'out/squint-seq';
for (const f of readdirSync(dir).filter(f => f.endsWith('.json') && f !== 'summary.json')) {
  const m = JSON.parse(readFileSync(`${dir}/${f}`, 'utf8'));
  if (!m.times || !m.width) continue;
  const resets = new Set(m.resets ?? []);
  const lines = [`${m.width} ${m.height} ${m.frames}`];
  for (let i = 0; i < m.frames; i++) lines.push(`${resets.has(i) ? 1 : 0} ${m.times[i]}`);
  writeFileSync(`${dir}/${f.replace(/\.json$/, '.cmd')}`, lines.join('\n') + '\n');
}
