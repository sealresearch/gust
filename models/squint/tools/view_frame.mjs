// Dev helper: writes frame N of a .gray sequence as a PNG so that it can be looked at.
//   node view_frame.mjs <name> <frame> [out.png]   (reads out/squint-seq/<name>.gray and .json)
import {readFileSync, writeFileSync, openSync, readSync, closeSync} from 'node:fs';
import {deflateSync} from 'node:zlib';
const outDir = new URL('../../../out/squint-seq/', import.meta.url);
const [name, frame, dest] = process.argv.slice(2);
const meta = JSON.parse(readFileSync(new URL(name + '.json', outDir)));
const size = meta.width * meta.height, buf = Buffer.alloc(size);
const fd = openSync(new URL(name + '.gray', outDir), 'r');
readSync(fd, buf, 0, size, size * Number(frame));
closeSync(fd);
const crcTable = Array.from({length: 256}, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = b => { let c = 0xffffffff; for (const x of b) c = crcTable[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const chunk = (type, data) => { const t = Buffer.from(type), len = Buffer.alloc(4), c = Buffer.alloc(4); len.writeUInt32BE(data.length); c.writeUInt32BE(crc(Buffer.concat([t, data]))); return Buffer.concat([len, t, data, c]); };
const raw = Buffer.alloc((meta.width + 1) * meta.height);
for (let y = 0; y < meta.height; y++) buf.copy(raw, y * (meta.width + 1) + 1, y * meta.width, (y + 1) * meta.width);
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(meta.width, 0); ihdr.writeUInt32BE(meta.height, 4); ihdr[8] = 8; ihdr[9] = 0;
writeFileSync(dest ?? new URL(`../scratch-${name}-${frame}.png`, outDir), Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
