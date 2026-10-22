// Copies what the package ships from the rest of the repository, so there is one copy of each in git:
//   the model registry     cli/registry.json                                    ->  src/registry.mjs
//   the WebAssembly build  out/gust-models.wasm and out/gust-models-mt.wasm     ->  wasm/
//   the licence            LICENSE                                              ->  LICENSE
// Build the modules first:   ZIG=zig bash build.sh wasm-models wasm-models-mt
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, '..');
const repo = join(pkg, '..', '..');

const copies = [
  ['out/gust-models.wasm', 'wasm/gust-models.wasm'],
  ['out/gust-models-mt.wasm', 'wasm/gust-models-mt.wasm'],
  ['LICENSE', 'LICENSE'],
];

let missing = false;
for (const from of ['cli/registry.json', ...copies.map(([f]) => f)]) {
  if (!existsSync(join(repo, from))) {
    console.error(`missing ${from}${from.startsWith('out/') ? ' (build it: ZIG=zig bash build.sh wasm-models wasm-models-mt)' : ''}`);
    missing = true;
  }
}
if (missing) process.exit(1);

const registry = JSON.parse(readFileSync(join(repo, 'cli', 'registry.json'), 'utf8'));
writeFileSync(join(pkg, 'src', 'registry.mjs'), `// Generated from cli/registry.json by scripts/sync.mjs.\nexport default ${JSON.stringify(registry, null, 2)};\n`);
console.log('cli/registry.json -> packages/js/src/registry.mjs');

mkdirSync(join(pkg, 'wasm'), { recursive: true });
for (const [from, to] of copies) {
  copyFileSync(join(repo, from), join(pkg, to));
  console.log(`${from} -> packages/js/${to}`);
}
