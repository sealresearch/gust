// Relay demo server: static files with cross-origin isolation (SharedArrayBuffer for the threaded runtime).
//   node models/relay/demo/serve.mjs [--port 8787] [--model relay-model] [--wasm out]
// Routes: / and /demo/* -> demo/, /web/* -> models/relay/web/ (the host wrapper), /wasm/* -> the build output
// (relay.wasm, relay-mt.wasm from ./build.sh wasm wasm-mt), /model/* -> the model folder.
import {createServer} from 'node:http';
import {readFile, stat} from 'node:fs/promises';
import {extname, join, normalize, dirname, resolve as abs} from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..', '..');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i >= 0 ? process.argv[i + 1] : d; };
const PORT = Number(arg('port', 8787));
const MODEL = arg('model', 'relay-model');
const WASM = abs(arg('wasm', join(REPO, 'out')));
const TYPES = {'.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript', '.js': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.gust': 'application/octet-stream',
  '.jsonl': 'application/json', '.svg': 'image/svg+xml'};

function resolve(url) {
  const p = decodeURIComponent(new URL(url, 'http://x').pathname);
  if (p === '/' || p === '/demo' || p === '/demo/') return join(HERE, 'index.html');
  const safe = normalize(p).replace(/^[\\/]+/, '').replace(/\\/g, '/');
  if (safe.includes('..')) return null;
  if (safe.startsWith('demo/')) return join(HERE, safe.slice(5));
  if (safe.startsWith('web/')) return join(HERE, '..', 'web', safe.slice(4));
  if (safe.startsWith('wasm/')) return join(WASM, safe.slice(5));
  if (safe.startsWith('model/')) return join(MODEL, safe.slice(6));
  return null;
}

createServer(async (req, res) => {
  const f = resolve(req.url);
  const headers = {'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cache-Control': 'no-cache'};
  try {
    if (!f || !(await stat(f)).isFile()) throw new Error('not found');
    res.writeHead(200, {...headers, 'Content-Type': TYPES[extname(f)] || 'application/octet-stream'});
    res.end(await readFile(f));
  } catch {
    res.writeHead(404, headers);
    res.end('not found');
  }
}).listen(PORT, '127.0.0.1', () => console.log(`Relay demo: http://localhost:${PORT}/  (model ${MODEL}, wasm ${WASM})`));
