// Finding, downloading and checking model files. The registry is the one the sealpup command line and the Python package
// read: the model ids, and for each model its Hugging Face repository and every file with its size and SHA-256.
import registry from './registry.mjs';
import { isNode } from './runtime.mjs';

export { registry };

/** The registry entry of a model, by id or alias, and its canonical id. */
export function resolve(modelId, reg = registry) {
  const id = reg.aliases?.[modelId] ?? modelId;
  const entry = reg.models[id];
  if (!entry) throw new Error(`no model called "${modelId}"; the registry has ${Object.keys(reg.models).sort().join(', ')}`);
  return [id, entry];
}

const hex = (buffer) => [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, '0')).join('');

async function check(bytes, file) {
  if (bytes.length !== file.size) throw new Error(`${file.name} is ${bytes.length} bytes, expected ${file.size}`);
  if (!globalThis.crypto?.subtle) throw new Error('checking a download needs crypto.subtle, which browsers only give secure pages (https or localhost)');
  const sum = hex(await globalThis.crypto.subtle.digest('SHA-256', bytes));
  if (sum !== file.sha256) throw new Error(`${file.name} does not match its checksum (got ${sum.slice(0, 12)}, expected ${file.sha256.slice(0, 12)})`);
}

/** Where Node keeps downloaded models: $GUST_CACHE, else the platform's cache folder. */
export async function cacheDir() {
  const os = await import('node:os');
  const path = await import('node:path');
  if (process.env.GUST_CACHE) return process.env.GUST_CACHE;
  if (process.platform === 'win32') return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local'), 'gust', 'cache');
  if (process.platform === 'darwin') return path.join(os.homedir(), 'Library', 'Caches', 'gust');
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'gust');
}

async function download(url, file, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${file.name}: ${res.status} ${res.statusText} from ${url}`);
  let bytes;
  if (res.body && onProgress) {
    const chunks = [];
    let loaded = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress({ file: file.name, loaded, total: file.size });
    }
    bytes = new Uint8Array(loaded);
    let at = 0;
    for (const c of chunks) { bytes.set(c, at); at += c.length; }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
  }
  await check(bytes, file);
  return bytes;
}

async function fromDirectory(directory, file, verify) {
  const path = await import('node:path');
  const fs = await import('node:fs/promises');
  const bytes = new Uint8Array(await fs.readFile(path.join(directory, file.name)));
  if (verify) await check(bytes, file);
  return bytes;
}

async function fromNodeCache(root, id, file, url, onProgress) {
  const path = await import('node:path');
  const fs = await import('node:fs/promises');
  const dir = path.join(root, id);
  const cached = path.join(dir, file.name);
  try {
    const bytes = new Uint8Array(await fs.readFile(cached));
    await check(bytes, file);
    onProgress?.({ file: file.name, loaded: file.size, total: file.size, cached: true });
    return bytes;
  } catch { /* missing or damaged: download it again */ }
  const bytes = await download(url, file, onProgress);
  await fs.mkdir(dir, { recursive: true });
  const part = `${cached}.${process.pid}.part`;
  await fs.writeFile(part, bytes);
  await fs.rename(part, cached);
  return bytes;
}

async function fromBrowserCache(id, file, url, onProgress) {
  const store = await caches.open(`gust-${id}`);
  const hit = await store.match(url);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    try {
      await check(bytes, file);
      onProgress?.({ file: file.name, loaded: file.size, total: file.size, cached: true });
      return bytes;
    } catch { await store.delete(url); }
  }
  const bytes = await download(url, file, onProgress);
  await store.put(url, new Response(bytes));
  return bytes;
}

/**
 * The files of a model as `[name, bytes]`: read from `directory`, or from the cache, or downloaded, checked and cached.
 * `names` limits it to some of the files.
 */
export async function obtain(id, entry, { directory, cache = true, baseUrl = 'https://huggingface.co', names, verify = true, onProgress, revision } = {}) {
  if (directory != null && !isNode) throw new TypeError('the directory option reads from disk, which only Node can do');
  const root = isNode && cache ? (typeof cache === 'string' ? cache : await cacheDir()) : null;
  const wanted = entry.files.filter((f) => !names || names.has(f.name));
  return Promise.all(wanted.map(async (file) => {
    if (directory != null) return [file.name, await fromDirectory(directory, file, verify)];
    const url = `${baseUrl.replace(/\/+$/, '')}/${entry.repo}/resolve/${revision ?? entry.revision ?? 'main'}/${file.name}`;
    if (root) return [file.name, await fromNodeCache(root, id, file, url, onProgress)];
    if (!isNode && cache && typeof caches !== 'undefined') return [file.name, await fromBrowserCache(id, file, url, onProgress)];
    return [file.name, await download(url, file, onProgress)];
  }));
}
