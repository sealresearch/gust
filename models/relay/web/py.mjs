// Python semantics the Relay host depends on: str()/repr() of JSON values, float formatting and parsing, str.strip,
// round, int(), and a translator from Python `re` patterns to equivalent JavaScript regular expressions.

/** A Python float whose value is integral (5.0): JavaScript numbers cannot tell it from the int 5, Python can. */
export class PyFloat {
  constructor(v) { this.v = v; }
  valueOf() { return this.v; }
  toJSON() { return this.v; }
  toString() { return floatRepr(this.v); }
}

/** A computed Python float: integral values are boxed so str() renders them as Python does ('5.0'). */
export const pyFloat = v => (Number.isInteger(v) ? new PyFloat(v) : v);
export const isPyFloat = v => v instanceof PyFloat || (typeof v === 'number' && !Number.isInteger(v));
export const num = v => (v instanceof PyFloat ? v.v : v);

// ---------------------------------------------------------------- whitespace, strip

/** Python str.isspace / re's Unicode \s, as a character-class body. */
export const PY_WS = '\\t-\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const WS_RE = new RegExp(`^[${PY_WS}]$`, 'u');
export const isSpace = ch => WS_RE.test(ch);
const STRIP_RE = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, 'gu');
/** str.strip() with no arguments. */
export const strip = s => s.replace(STRIP_RE, '');
/** str.strip(chars) / rstrip(chars). */
export function stripChars(s, chars) {
  const set = new Set(Array.from(chars));
  const a = Array.from(s);
  let i = 0, j = a.length;
  while (i < j && set.has(a[i])) i++;
  while (j > i && set.has(a[j - 1])) j--;
  return a.slice(i, j).join('');
}
export function rstripChars(s, chars) {
  const set = new Set(Array.from(chars));
  const a = Array.from(s);
  let j = a.length;
  while (j > 0 && set.has(a[j - 1])) j--;
  return a.slice(0, j).join('');
}

// ---------------------------------------------------------------- numbers

/** repr(float) / str(float). */
export function floatRepr(v) {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0.0' : '0.0';
  const sign = v < 0 ? '-' : '';
  const [mant, ex] = Math.abs(v).toExponential().split('e');
  const digits = mant.replace('.', '');
  const e = parseInt(ex, 10);
  if (e >= -5 + 1 && e < 16) {  // Python repr: fixed notation for 1e-4 <= |v| < 1e16
    let s;
    if (e >= 0) {
      s = digits.length > e + 1 ? digits.slice(0, e + 1) + '.' + digits.slice(e + 1) : digits.padEnd(e + 1, '0') + '.0';
    } else s = '0.' + '0'.repeat(-e - 1) + digits;
    return sign + s;
  }
  const m = digits.length > 1 ? digits[0] + '.' + digits.slice(1) : digits;
  const es = Math.abs(e) < 10 ? '0' + Math.abs(e) : String(Math.abs(e));
  return `${sign}${m}e${e < 0 ? '-' : '+'}${es}`;
}

/** repr(str). */
export function strRepr(s) {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (ch === '\\') out += '\\\\';
    else if (ch === q) out += '\\' + q;
    else if (ch === '\t') out += '\\t';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (cp < 0x20 || cp === 0x7f || (ch !== ' ' && /^[\p{C}\p{Z}]$/u.test(ch))) {
      out += cp < 0x100 ? '\\x' + cp.toString(16).padStart(2, '0')
        : cp < 0x10000 ? '\\u' + cp.toString(16).padStart(4, '0') : '\\U' + cp.toString(16).padStart(8, '0');
    } else out += ch;
  }
  return out + q;
}

/** repr() of a JSON value. */
export function pyRepr(v) {
  if (typeof v === 'string') return strRepr(v);
  if (Array.isArray(v)) return '[' + v.map(pyRepr).join(', ') + ']';
  if (v && typeof v === 'object' && !(v instanceof PyFloat))
    return '{' + Object.entries(v).map(([k, x]) => strRepr(k) + ': ' + pyRepr(x)).join(', ') + '}';
  return pyStr(v);
}

/** str() of a JSON value (None, True, 5, 5.0, 'text', lists and dicts with repr items). */
export function pyStr(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (v instanceof PyFloat) return floatRepr(v.v);
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : floatRepr(v);
  if (typeof v === 'string') return v;
  return pyRepr(v);
}

/** Python truthiness of a JSON value. */
export function truthy(v) {
  if (v === null || v === undefined || v === false || v === '' || v === 0) return false;
  if (v instanceof PyFloat) return v.v !== 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return true;
}

/** Python iteration over a value (list items, string characters, dict keys). */
export function pyIter(v) {
  if (Array.isArray(v)) return v;
  if (typeof v === 'string') return Array.from(v);
  if (v && typeof v === 'object') return Object.keys(v);
  throw new TypeError('not iterable');
}

/** Python indexing seq[i] (i >= 0) with IndexError. */
export function pyIndex(seq, i) {
  const a = typeof seq === 'string' ? Array.from(seq) : seq;
  if (!Array.isArray(a) || i < 0 || i >= a.length) throw new RangeError('index out of range');
  return a[i];
}

/** Unicode decimal digits (category Nd) -> ASCII, as Python's int() and float() accept them. */
export function asciiDigits(s) {
  return s.replace(/\p{Nd}/gu, ch => {
    let cp = ch.codePointAt(0), z = cp;
    while (/^\p{Nd}$/u.test(String.fromCodePoint(z - 1))) z--;
    return String((cp - z) % 10);
  });
}

/** int(str) for the digit strings the regexes capture. */
export function pyInt(s) {
  const t = strip(asciiDigits(s)).replace(/_/g, '');
  if (!/^[+-]?\d+$/.test(t)) throw new RangeError(`invalid literal for int(): ${s}`);
  return parseInt(t, 10);
}

/** float(str): returns the value or throws (ValueError). */
export function pyFloatParse(s) {
  const t = strip(asciiDigits(s));
  if (/^[+-]?(inf|infinity)$/i.test(t)) return t[0] === '-' ? -Infinity : Infinity;
  if (/^[+-]?nan$/i.test(t)) return NaN;
  const D = '\\d(?:_?\\d)*';
  if (!new RegExp(`^[+-]?(?:${D}(?:\\.(?:${D})?)?|\\.${D})(?:[eE][+-]?${D})?$`).test(t))
    throw new RangeError(`could not convert string to float: ${s}`);
  return Number(t.replace(/_/g, ''));
}

/** round(x) (half to even) for finite floats; raises like Python for inf/nan. */
export function pyRound(x) {
  if (!Number.isFinite(x)) throw new RangeError('cannot round a non-finite float');
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** int(float): truncation; raises for inf/nan. */
export function pyIntFloat(x) {
  if (!Number.isFinite(x)) throw new RangeError('cannot convert float to integer');
  return Math.trunc(x) + 0;
}

export const pyMod = (a, b) => ((a % b) + b) % b;

// ---------------------------------------------------------------- regular expressions

const W = '\\p{L}\\p{N}_';
const BOUNDARY = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`;
const NOT_BOUNDARY = `(?:(?<=[${W}])(?=[${W}])|(?<![${W}])(?![${W}]))`;

/**
 * Translates a Python `re` pattern (str pattern, Unicode semantics) to a JavaScript RegExp with flag `u`:
 * \w \d \s \b and their negations get Python's Unicode meaning. `flags` may add i/g/y.
 */
export function pyRe(src, flags = '') {
  let out = '', inClass = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (c === '\\') {
      const n = src[++i];
      if (n === 'w') out += inClass ? W : `[${W}]`;
      else if (n === 'W') { if (inClass) throw new Error('\\W in class'); out += `[^${W}]`; }
      else if (n === 'd') out += inClass ? '\\p{Nd}' : '\\p{Nd}';
      else if (n === 'D') { if (inClass) throw new Error('\\D in class'); out += '\\P{Nd}'; }
      else if (n === 's') out += inClass ? PY_WS : `[${PY_WS}]`;
      else if (n === 'S') { if (inClass) throw new Error('\\S in class'); out += `[^${PY_WS}]`; }
      else if (n === 'b') out += inClass ? '\\x08' : BOUNDARY;
      else if (n === 'B') out += NOT_BOUNDARY;
      else if (n === 'A') out += '^';
      else if (n === 'Z') out += '$';
      else if (/[A-Za-z0-9]/.test(n)) out += '\\' + n;  // \n \t \x.. \u....
      else if (/[\^$\\.*+?()[\]{}|/]/.test(n) || (inClass && n === '-')) out += '\\' + n;
      else out += n;  // identity escapes (\" \' \- outside a class) are plain characters
      continue;
    }
    if (inClass) {
      if (c === ']' ) inClass = false;
      out += c;
      continue;
    }
    if (c === '[') {
      inClass = true;
      out += c;
      if (src[i + 1] === '^') { out += '^'; i++; }
      if (src[i + 1] === ']') { out += '\\]'; i++; }
      continue;
    }
    out += c;
  }
  return new RegExp(out, 'u' + flags);
}

/** re.escape for the literal suffixes the number parser uses. */
export const reEscape = s => s.replace(/[\\^$.*+?()[\]{}|\-#&~ \t\n\r\v\f]/g, m => '\\' + m);

/** re.fullmatch(pattern, s) with a compiled pyRe source. */
export function fullmatcher(src, flags = '') {
  return pyRe(`^(?:${src})$`, flags);
}

// ---------------------------------------------------------------- JSON with Python float literals

/**
 * JSON.parse that keeps Python's int/float distinction: number literals written with a fraction or exponent whose
 * value is integral (1.0, 1e3) become PyFloat; also accepts NaN, Infinity and -Infinity like Python's json.
 */
export function parseJSON(text) {
  let i = 0;
  const ws = () => { while (i < text.length && ' \t\n\r'.includes(text[i])) i++; };
  const value = () => {
    ws();
    const c = text[i];
    if (c === '{') {
      i++;
      const o = {};
      ws();
      if (text[i] === '}') { i++; return o; }
      for (;;) {
        ws();
        const k = str();
        ws();
        if (text[i++] !== ':') throw new SyntaxError(`expected : at ${i}`);
        o[k] = value();
        ws();
        const d = text[i++];
        if (d === '}') return o;
        if (d !== ',') throw new SyntaxError(`expected , at ${i}`);
      }
    }
    if (c === '[') {
      i++;
      const a = [];
      ws();
      if (text[i] === ']') { i++; return a; }
      for (;;) {
        a.push(value());
        ws();
        const d = text[i++];
        if (d === ']') return a;
        if (d !== ',') throw new SyntaxError(`expected , at ${i}`);
      }
    }
    if (c === '"') return str();
    if (text.startsWith('true', i)) { i += 4; return true; }
    if (text.startsWith('false', i)) { i += 5; return false; }
    if (text.startsWith('null', i)) { i += 4; return null; }
    if (text.startsWith('NaN', i)) { i += 3; return NaN; }
    if (text.startsWith('Infinity', i)) { i += 8; return Infinity; }
    if (text.startsWith('-Infinity', i)) { i += 9; return -Infinity; }
    const m = /^-?\d+(\.\d+)?([eE][+-]?\d+)?/.exec(text.slice(i, i + 400));
    if (!m) throw new SyntaxError(`unexpected ${c} at ${i}`);
    i += m[0].length;
    const v = Number(m[0]);
    return (m[1] || m[2]) ? pyFloat(v) : v;
  };
  const str = () => {
    if (text[i] !== '"') throw new SyntaxError(`expected string at ${i}`);
    let j = i + 1;
    while (text[j] !== '"') { if (text[j] === '\\') j++; j++; }
    const s = JSON.parse(text.slice(i, j + 1));
    i = j + 1;
    return s;
  };
  const v = value();
  ws();
  if (i !== text.length) throw new SyntaxError(`trailing data at ${i}`);
  return v;
}
