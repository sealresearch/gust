// Copies what the package ships from the rest of the repository, so there is one copy of each in git:
//   the WebAssembly binding   models/relay/web/relay-native.mjs and relay-native-worker.mjs  ->  src/
//   the runtime builds        out/relay-simd.wasm and out/relay-mt-simd.wasm                 ->  wasm/
//   the licence               LICENSE                                                        ->  LICENSE
// Build the runtime first:   ZIG=zig bash build.sh wasm-simd wasm-mt-simd
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, '..');
const repo = join(pkg, '..', '..');

const copies = [
  ['models/relay/web/relay-native.mjs', 'src/relay-native.mjs'],
  ['models/relay/web/relay-native-worker.mjs', 'src/relay-native-worker.mjs'],
  ['out/relay-simd.wasm', 'wasm/relay-simd.wasm'],
  ['out/relay-mt-simd.wasm', 'wasm/relay-mt-simd.wasm'],
  ['LICENSE', 'LICENSE'],
];

let missing = false;
for (const [from] of copies) {
  if (!existsSync(join(repo, from))) {
    console.error(`missing ${from}${from.startsWith('out/') ? ' (build it: ZIG=zig bash build.sh wasm-simd wasm-mt-simd)' : ''}`);
    missing = true;
  }
}
if (missing) process.exit(1);

mkdirSync(join(pkg, 'wasm'), { recursive: true });
for (const [from, to] of copies) {
  copyFileSync(join(repo, from), join(pkg, to));
  console.log(`${from} -> packages/relay/${to}`);
}
