// Byte-level BPE tokenizer: a dependency-free port of the subset of HF `tokenizers` that Relay-2's tokenizer.json
// uses (added tokens, NFC normalizer, ByteLevel pre-tokenizer with the GPT-2 regex, BPE without dropout/unk).
// Offsets are returned like the Python binding returns them: Unicode code point indices into the original text.

// Oniguruma's Unicode \s (the regex engine HF tokenizers uses): U+0009-000D, U+0085 and the Z categories.
const WS = '\\t-\\r\\x85\\p{Z}';
const SPLIT = new RegExp(
  `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^${WS}\\p{L}\\p{N}]+|[${WS}]+(?![^${WS}])|[${WS}]+`, 'gu');
// Rust char::is_whitespace (Unicode White_Space), used by the lstrip option of added tokens.
const RUST_WS = /[\t-\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
// Python str.isspace.
const PY_SPACE = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f,
  0x205f, 0x3000]);
export const pyIsSpaceCp = cp => PY_SPACE.has(cp);

function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const out = new Array(256);
  bs.forEach((b, i) => { out[b] = String.fromCodePoint(cs[i]); });
  return out;
}
const BYTE_CHAR = bytesToUnicode();

/** Code points that attach to the previous character under NFC (marks, Hangul medial/final jamo). */
const ATTACH = /^[\p{M}\u1160-\u11ff\ud7b0-\ud7ff]$/u;

/**
 * NFC with alignments as HF tokenizers computes them: every output character maps to the original character it
 * comes from; a composed character keeps the alignment of its starter source character, characters merged
 * into it disappear, the pieces of a decomposed character all keep its alignment.
 * Returns {str, cps: [code points], al: [[start, end] UTF-16 ranges in `src`]}.
 */
function nfcAligned(src, base) {
  const cps = [], al = [];
  const norm = src.normalize('NFC');
  if (norm === src) {
    let i = 0;
    for (const ch of src) {
      const cp = ch.codePointAt(0);
      cps.push(cp);
      al.push([base + i, base + i + ch.length]);
      i += ch.length;
    }
    return {str: src, cps, al};
  }
  // cluster = starter + attached characters; normalise cluster by cluster
  const chars = [];
  let i = 0;
  for (const ch of src) { chars.push([ch, i]); i += ch.length; }
  let out = '';
  for (let s = 0; s < chars.length;) {
    let e = s + 1;
    while (e < chars.length && ATTACH.test(chars[e][0])) e++;
    const seg = chars.slice(s, e);
    const raw = seg.map(x => x[0]).join('');
    const n = raw.normalize('NFC');
    // canonical decomposition of every source character, each piece remembering its source
    const dec = [];
    seg.forEach(([ch], j) => { for (const d of ch.normalize('NFD')) dec.push([d, j]); });
    const used = new Array(dec.length).fill(false);
    let prev = 0;
    for (const ch of n) {
      // an output character comes from the source of its first decomposed piece (the starter it composed onto)
      const first = Array.from(ch.normalize('NFD'))[0];
      let q = dec.findIndex(([d], x) => !used[x] && d === first);
      let src_i = q >= 0 ? dec[q][1] : prev;
      for (const d of ch.normalize('NFD')) {
        q = dec.findIndex(([dd], x) => !used[x] && dd === d);
        if (q >= 0) used[q] = true;
      }
      const [c0, o] = seg[src_i];
      cps.push(ch.codePointAt(0));
      al.push([base + o, base + o + c0.length]);
      prev = src_i;
    }
    out += n;
    s = e;
  }
  return {str: out, cps, al};
}

class MinHeap {
  constructor() { this.a = []; }
  push(x) {
    const a = this.a;
    a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (MinHeap.lt(a[i], a[p])) { [a[i], a[p]] = [a[p], a[i]]; i = p; } else break;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && MinHeap.lt(a[l], a[m])) m = l;
        if (r < a.length && MinHeap.lt(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
  get size() { return this.a.length; }
  static lt(x, y) { return x.rank < y.rank || (x.rank === y.rank && x.pos < y.pos); }
}

export class Tokenizer {
  /** @param {object|string} json  parsed tokenizer.json (or its text) */
  constructor(json) {
    if (typeof json === 'string') json = JSON.parse(json);
    if (json instanceof Uint8Array || json instanceof ArrayBuffer) json = JSON.parse(new TextDecoder().decode(json));
    const m = json.model;
    if (m.type !== 'BPE') throw new Error('only BPE tokenizers are supported');
    if (json.normalizer && json.normalizer.type !== 'NFC') throw new Error('only the NFC normalizer is supported');
    this.vocab = new Map(Object.entries(m.vocab));
    this.byteId = BYTE_CHAR.map(c => this.vocab.get(c));
    this.merges = new Map();
    m.merges.forEach((mg, rank) => {
      const [a, b] = Array.isArray(mg) ? mg : mg.split(' ');
      const ia = this.vocab.get(a), ib = this.vocab.get(b), n = this.vocab.get(a + b);
      if (ia === undefined || ib === undefined || n === undefined) throw new Error(`merge ${a} ${b}: not in vocab`);
      const key = ia * 65536 + ib;
      if (!this.merges.has(key)) this.merges.set(key, {rank, id: n});
    });
    this.added = new Map();
    const special = [], normal = [];
    for (const t of json.added_tokens || []) {
      this.added.set(t.content, t.id);
      (t.normalized ? normal : special).push(t);
    }
    const index = list => {
      const byFirst = new Map();
      for (const t of list) {
        const f = t.content[0];
        if (!byFirst.has(f)) byFirst.set(f, []);
        byFirst.get(f).push(t);
      }
      for (const v of byFirst.values()) v.sort((x, y) => y.content.length - x.content.length);
      return byFirst;
    };
    this.special = index(special);
    this.normal = index(normal);
    this.id = name => this.added.get(name) ?? this.vocab.get(name);
    this.wordCache = new Map();
    this.cache = new Map();
  }

  /** Leftmost-longest matches of added tokens in `s` -> [[start, end, token]] (UTF-16 indices). */
  static matches(s, byFirst) {
    const out = [];
    let last = 0;
    for (let i = 0; i < s.length;) {
      const cands = byFirst.get(s[i]);
      let hit = null;
      if (cands) for (const t of cands) if (s.startsWith(t.content, i)) { hit = t; break; }
      if (!hit) { i++; continue; }
      let start = i;
      if (hit.lstrip) while (start > last && RUST_WS.test(s[start - 1])) start--;
      if (hit.rstrip) throw new Error('rstrip added tokens are not supported');
      if (hit.single_word) throw new Error('single_word added tokens are not supported');
      out.push([start, i + hit.content.length, hit]);
      i += hit.content.length;
      last = i;
    }
    return out;
  }

  /** BPE of one pre-token (byte string as byte-level chars) -> [[id, byteStart, byteEnd]]. */
  bpe(bytes) {
    const key = String.fromCharCode(...bytes);
    const hit = this.wordCache.get(key);
    if (hit) return hit;
    const sym = [];
    for (let i = 0; i < bytes.length; i++) {
      const id = this.byteId[bytes[i]];
      if (id === undefined) continue; // no unk token: unknown characters are dropped, as in HF BPE
      sym.push({id, s: i, e: i + 1, prev: sym.length - 1, next: sym.length + 1, len: 1});
    }
    if (sym.length) sym[sym.length - 1].next = -1;
    const heap = new MinHeap();
    for (let i = 0; i + 1 < sym.length; i++) {
      const m = this.merges.get(sym[i].id * 65536 + sym[i + 1].id);
      if (m) heap.push({pos: i, rank: m.rank, id: m.id});
    }
    while (heap.size) {
      const top = heap.pop();
      const cur = sym[top.pos];
      if (cur.len === 0 || cur.next === -1) continue;
      const right = sym[cur.next];
      const m = this.merges.get(cur.id * 65536 + right.id);
      if (!m || m.id !== top.id) continue;
      cur.id = top.id;
      cur.e = right.e;
      cur.len += right.len;
      right.len = 0;
      cur.next = right.next;
      if (cur.next >= 0) sym[cur.next].prev = top.pos;
      if (cur.prev >= 0) {
        const p = sym[cur.prev];
        const pm = this.merges.get(p.id * 65536 + cur.id);
        if (pm) heap.push({pos: cur.prev, rank: pm.rank, id: pm.id});
      }
      if (cur.next >= 0) {
        const nm = this.merges.get(cur.id * 65536 + sym[cur.next].id);
        if (nm) heap.push({pos: top.pos, rank: nm.rank, id: nm.id});
      }
    }
    const out = sym.filter(x => x.len > 0).map(x => [x.id, x.s, x.e]);
    if (this.wordCache.size > 200000) this.wordCache.clear();
    this.wordCache.set(key, out);
    return out;
  }

  /** Pre-tokenize and BPE one normalized stretch (cps/al slices) into `ids`/`offs` (UTF-16 offsets). */
  _model(str, cps, al, ids, offs) {
    // UTF-16 index -> code point index within str
    const u2c = new Int32Array(str.length + 1);
    let ci = 0;
    for (let u = 0; u < str.length; ci++) {
      const w = cps[ci] > 0xffff ? 2 : 1;
      u2c[u] = ci;
      if (w === 2) u2c[u + 1] = ci;
      u += w;
    }
    u2c[str.length] = ci;
    const pieces = [];
    let pos = 0;
    SPLIT.lastIndex = 0;
    for (const m of str.matchAll(SPLIT)) {
      if (m.index > pos) pieces.push([pos, m.index]);
      pieces.push([m.index, m.index + m[0].length]);
      pos = m.index + m[0].length;
    }
    if (pos < str.length) pieces.push([pos, str.length]);
    for (const [a, b] of pieces) {
      const c0 = u2c[a], c1 = u2c[b];
      const bytes = [], borig = [];
      for (let c = c0; c < c1; c++) {
        const cp = cps[c];
        let enc;
        if (cp < 0x80) enc = [cp];
        else if (cp < 0x800) enc = [0xc0 | (cp >> 6), 0x80 | (cp & 63)];
        else if (cp < 0x10000) enc = [0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63)];
        else enc = [0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63)];
        for (const x of enc) { bytes.push(x); borig.push(c); }
      }
      for (const [id, s, e] of this.bpe(bytes)) {
        ids.push(id);
        offs.push([al[borig[s]][0], al[borig[e - 1]][1]]);
      }
    }
  }

  /** Full encoding without special tokens: {ids, offsets} with code point offsets into `text`. */
  encode(text) {
    const ids = [], offs = [];
    let pos = 0;
    const segs = [];
    for (const [s, e, t] of Tokenizer.matches(text, this.special)) {
      if (s > pos) segs.push([pos, s, null]);
      segs.push([s, e, t]);
      pos = e;
    }
    if (pos < text.length) segs.push([pos, text.length, null]);
    for (const [s, e, t] of segs) {
      if (t) { ids.push(t.id); offs.push([s, e]); continue; }
      const n = nfcAligned(text.slice(s, e), s);
      // UTF-16 index in n.str -> code point index
      const u2c = [];
      let ci = 0;
      for (let u = 0; u < n.str.length;) { u2c[u] = ci; const w = n.cps[ci] > 0xffff ? 2 : 1; if (w === 2) u2c[u + 1] = ci; u += w; ci++; }
      u2c[n.str.length] = ci;
      let p = 0;
      const sub = (a, b) => {
        const c0 = u2c[a], c1 = u2c[b];
        this._model(n.str.slice(a, b), n.cps.slice(c0, c1), n.al.slice(c0, c1), ids, offs);
      };
      for (const [a, b, tok] of Tokenizer.matches(n.str, this.normal)) {
        if (a > p) sub(p, a);
        ids.push(tok.id);
        offs.push([n.al[u2c[a]][0], n.al[u2c[b] - 1][1]]);
        p = b;
      }
      if (p < n.str.length) sub(p, n.str.length);
    }
    // UTF-16 offsets -> code point offsets
    if (/[\ud800-\udfff]/.test(text)) {
      const map = new Int32Array(text.length + 1);
      let c = 0;
      for (let u = 0; u < text.length; u++) {
        map[u] = c;
        const cc = text.charCodeAt(u);
        if (cc >= 0xd800 && cc < 0xdc00 && u + 1 < text.length) {
          const d = text.charCodeAt(u + 1);
          if (d >= 0xdc00 && d < 0xe000) { map[u + 1] = c; u++; }
        }
        c++;
      }
      map[text.length] = c;
      for (const o of offs) { o[0] = map[o[0]]; o[1] = map[o[1]]; }
    }
    return {ids, offsets: offs};
  }

  /** Tok.ids: token ids without specials, truncated to n (cached). */
  ids(text, n) {
    const key = n + '\u0000' + text;
    let out = this.cache.get(key);
    if (out === undefined) {
      out = this.encode(text).ids.slice(0, n);
      if (this.cache.size > 300000) this.cache.clear();
      this.cache.set(key, out);
    }
    return out.slice();
  }

  /** Tok.request: ids truncated to n and code point offsets with leading whitespace skipped. */
  request(text, n) {
    const e = this.encode(text);
    const cps = Array.from(text, ch => ch.codePointAt(0));
    const ids = e.ids.slice(0, n);
    const offs = e.offsets.slice(0, n).map(([a, b]) => {
      while (a < b && PY_SPACE.has(cps[a])) a++;
      return [a, b];
    });
    return {ids, offs};
  }
}

export {SPLIT as PRE_TOKENIZE_RE};
