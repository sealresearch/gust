// Value programs (port of relay/programs.py, inference side): `run(program, text, param, ctx)` executes one program
// and coerces the result to the parameter's declared type. Spans are UTF-16 [start, end) indices into `text` (the
// decoder converts the Python code point spans before calling run). Values follow Python's typing where it shows in
// the output: integral floats are PyFloat (str() -> '5.0'), ints are plain numbers.
import {TABLES, AFFIXES} from './tables.mjs';
import {FIX} from './hostfix.mjs';
import {
  PyFloat, fullmatcher, isPyFloat, num, pyFloat, pyFloatParse, pyIndex, pyInt, pyIntFloat, pyMod, pyRe, pyRound,
  pyStr, reEscape, rstripChars, strip, truthy,
} from './py.mjs';

export {TABLES};
export const TABLE_NAMES = Object.keys(TABLES);
export const affixes = () => AFFIXES;

// ---------------------------------------------------------------- numbers

const UNITS = new Map('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'
  .split(' ').map((w, i) => [w, i]));
const TENS = new Map([['twenty', 20], ['thirty', 30], ['forty', 40], ['fifty', 50], ['sixty', 60], ['seventy', 70],
  ['eighty', 80], ['ninety', 90]]);
const SCALES = new Map([['hundred', 100], ['thousand', 1000], ['million', 1e6], ['billion', 1e9], ['trillion', 1e12]]);
const SPECIAL = new Map([['a dozen', 12], ['dozen', 12], ['half', 0.5], ['a half', 0.5], ['quarter', 0.25],
  ['a quarter', 0.25], ['once', 1], ['twice', 2], ['thrice', 3], ['single', 1], ['a couple', 2], ['couple', 2],
  ['pair', 2], ['a pair', 2], ['first', 1], ['second', 2], ['third', 3], ['fourth', 4], ['fifth', 5], ['sixth', 6],
  ['seventh', 7], ['eighth', 8], ['ninth', 9], ['tenth', 10], ['noon', 12], ['midnight', 0], ['none', 0], ['no', 0],
  ['a', 1], ['an', 1]]);
const byLenDesc = keys => [...keys].sort((a, b) => b.length - a.length);  // stable, like Python's sorted
export const NUMWORD = '(?:' + byLenDesc([...UNITS.keys(), ...TENS.keys(), ...SCALES.keys()]).join('|') + ')';

export const NUM_SCALES = [1, 0.01, 100, 1000, 1e6, 1e9, -1, 60, 3600, 1 / 60, 1e-3, 86400, 7, 0.5, 2];

const SUFFIXES = [['percent', 0.01], ['%', 0.01], ['bn', 1e9], ['k', 1e3], ['m', 1e6]].map(([suf, m]) =>
  [suf, m, pyRe('^-?[\\d.]+\\s?' + reEscape(suf) + '$')]);
const SPLIT_WORDS = pyRe('[\\s-]+', 'g');

/**
 * FIX.locale: '€42.80', '¥500', '1.250,00', '23,40', '1 250,50', 'two and a half', 'twenty-one point five'.
 * Returns undefined when the extended rules do not apply (the original parser decides).
 */
const DIGIT_WORD = new Map([['zero', 0], ['oh', 0], ['one', 1], ['two', 2], ['three', 3], ['four', 4], ['five', 5],
  ['six', 6], ['seven', 7], ['eight', 8], ['nine', 9]]);
function parseLocale(s) {
  let t = strip(s.toLowerCase().replace(/[€£¥₹$]|\b(?:eur|usd|gbp|jpy|euros?|dollars?|pounds?|yen|bucks|quid)\b/g, ''));
  t = t.replace(/(\d)[\s  ](?=\d{3}\b)/g, '$1');            // 1 250,50 -> 1250,50
  if (/^-?\d{1,3}(\.\d{3})+,\d+$/.test(t)) return Number(t.replace(/\./g, '').replace(',', '.'));  // 1.250,00
  if (/^-?\d+,\d{1,2}$/.test(t)) return Number(t.replace(',', '.'));    // 23,40 / 1,5 (decimal comma)
  let m = t.match(/^(.+?)\s+and\s+a\s+half$/);
  if (m) { const v = parseNumber(m[1]); return v === null ? undefined : v + 0.5; }
  m = t.match(/^(.+?)\s+point\s+([a-z\s-]+)$/);
  if (m && m[2].split(/[\s-]+/).every(w => DIGIT_WORD.has(w))) {
    const v = parseNumber(m[1]);
    if (v === null) return undefined;
    return Number(`${v}.${m[2].split(/[\s-]+/).map(w => DIGIT_WORD.get(w)).join('')}`);
  }
  if (t !== strip(s.toLowerCase())) { const v = parseNumber(t); return v === null ? undefined : v; }
  return undefined;
}

/** parse_number: '1,500' -> 1500.0, '5k' -> 5000.0, 'twenty one' -> 21.0; null when unparseable. */
export function parseNumber(s) {
  if (FIX.locale && !parseNumber.inner) {
    parseNumber.inner = true;
    try { const v = parseLocale(s); if (v !== undefined && Number.isFinite(v)) return v; } finally { parseNumber.inner = false; }
  }
  let t = strip(s.toLowerCase().replaceAll(',', '').replaceAll('$', ''));
  let mult = 1;
  const money = s.includes('$') || s.includes('€') || s.includes('£');
  for (let [suf, m, rx] of SUFFIXES) {
    if (suf === 'm' && !money) m = 1;
    if (t.endsWith(suf) && rx.test(t)) {
      t = strip(t.slice(0, t.length - suf.length));
      mult = suf !== 'percent' && suf !== '%' ? m : 1;
      break;
    }
  }
  try {
    return pyFloatParse(t) * mult;
  } catch { /* ValueError */ }
  if (SPECIAL.has(t)) return SPECIAL.get(t);
  let total = 0, cur = 0;
  for (const w of t.split(SPLIT_WORDS)) {
    if (w === 'and') continue;
    if (UNITS.has(w)) cur += UNITS.get(w);
    else if (TENS.has(w)) cur += TENS.get(w);
    else if (w === 'hundred') cur = Math.max(cur, 1) * 100;
    else if (SCALES.has(w)) { total += Math.max(cur, 1) * SCALES.get(w); cur = 0; }
    else return null;
  }
  return total + cur;
}

// ---------------------------------------------------------------- dates and times

const MONTH_NAMES = 'january february march april may june july august september october november december'.split(' ');
const MONTHS = new Map(MONTH_NAMES.map((m, i) => [m, i + 1]));
for (const [m, i] of [...MONTHS]) if (!MONTHS.has(m.slice(0, 3))) MONTHS.set(m.slice(0, 3), i);
MONTHS.set('sept', 9);
const WEEKDAYS = new Map('monday tuesday wednesday thursday friday saturday sunday'.split(' ').map((d, i) => [d, i]));
for (const [d, i] of [...WEEKDAYS]) if (!WEEKDAYS.has(d.slice(0, 3))) WEEKDAYS.set(d.slice(0, 3), i);
export const MON = '(?:' + byLenDesc(MONTHS.keys()).join('|') + ')\\.?';
const WD = '(?:' + byLenDesc(WEEKDAYS.keys()).join('|') + ')';
const AP = '(a\\.?m\\.?|p\\.?m\\.?)';

export const DATE_SRC = [
  ['mdy', '\\b(' + MON + ')\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b', 'i'],
  ['dmy', '\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(' + MON + ')(?:,?\\s+(\\d{4}))?\\b', 'i'],
  ['iso', '\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})\\b', ''],
  ['slash', '\\b(\\d{1,2})/(\\d{1,2})/(\\d{2,4})\\b', ''],
  ['rel', '\\b(today|tonight|tomorrow|day after tomorrow|yesterday)\\b', 'i'],
  ['wd', '\\b(?:(this|next|coming|this coming|on)\\s+)?(' + WD + ')\\b', 'i'],
  ['in', '\\bin\\s+(\\w+)\\s+(day|days|week|weeks)\\b', 'i'],
];
export const TIME_SRC = [
  ['hm_ap', '\\b(\\d{1,2})(?::(\\d{2}))?\\s*' + AP + '(?![\\w])', 'i'],
  ['hm24', '\\b(\\d{1,2}):(\\d{2})(?::\\d{2})?\\b(?!\\s*(?:a\\.?m|p\\.?m))', 'i'],
  ['word', '\\b(noon|midday|midnight)\\b', 'i'],
  ['past', '\\b(half|quarter|\\d{1,2}|' + NUMWORD + ')\\s+(past|to|after|before)\\s+(\\d{1,2}|' + NUMWORD +
    ')(?:\\s*' + AP + ')?', 'i'],
  ['oclock', '\\b(\\d{1,2}|' + NUMWORD + ")\\s+o'?\\s?clock(?:\\s*" + AP + ')?', 'i'],
  ['wordhm', '\\b(' + NUMWORD + ')\\s+(' + NUMWORD + '(?:[\\s-]' + NUMWORD + ')?)\\s*' + AP, 'i'],
];
const DATE_FULL = DATE_SRC.map(([k, src, f]) => [k, fullmatcher(src, f)]);
const TIME_FULL = TIME_SRC.map(([k, src, f]) => [k, fullmatcher(src, f)]);
export const EVENING_SRC = '\\b(tonight|evening|this evening|in the evening|pm|night|afternoon|dinner)\\b';
export const MORNING_SRC = '\\b(morning|a\\.?m\\.?|breakfast)\\b';
const EVENING = pyRe(EVENING_SRC, 'i');
const MORNING = pyRe(MORNING_SRC, 'i');

function small(s) {
  const v = parseNumber(s);
  return v === null ? null : pyIntFloat(v);
}

/** resolve_time: '2 PM' -> [14, 0]; 'half past seven' + 'tonight' -> [19, 30]; null if unparseable. */
export function resolveTime(spanText, contextText = '') {
  const s = strip(spanText.toLowerCase());
  for (const [kind, rx] of TIME_FULL) {
    const m = rx.exec(s);
    if (!m) continue;
    let h, mi, ap = null;
    if (kind === 'hm_ap') { h = pyInt(m[1]); mi = m[2] !== undefined ? pyInt(m[2]) : 0; ap = m[3]; }
    else if (kind === 'hm24') { h = pyInt(m[1]); mi = pyInt(m[2]); }
    else if (kind === 'word') return m[1] === 'midnight' ? [0, 0] : [12, 0];
    else if (kind === 'past') {
      const a = m[1], rel = m[2], base = small(m[3]);
      ap = m[4] ?? null;
      const mins = a === 'half' ? 30 : a === 'quarter' ? 15 : small(a);
      if (base === null || mins === null) return null;
      [h, mi] = rel === 'past' || rel === 'after' ? [base, mins] : [pyMod(base - 1, 24), 60 - mins];
    } else if (kind === 'oclock') { h = small(m[1]); mi = 0; ap = m[2] ?? null; }
    else { h = small(m[1]); mi = small(m[2]); ap = m[3]; }
    if (h === null || mi === null || !(h >= 0 && h <= 24 && mi >= 0 && mi < 60)) return null;
    if (ap) h = (h % 12) + (ap.startsWith('p') ? 12 : 0);
    else if (h < 12 && kind !== 'hm24' && EVENING.test(contextText) && !MORNING.test(contextText)) h += 12;
    else if (h < 7 && kind === 'past' && !MORNING.test(contextText)) h += 12;
    return [h % 24, mi];
  }
  return null;
}

// proleptic Gregorian dates as {y, m, d}, like datetime.date
const isLeap = y => y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
const DIM = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const dim = (y, m) => (m === 2 && isLeap(y) ? 29 : DIM[m]);
function mkDate(y, m, d) {
  if (!(y >= 1 && y <= 9999) || !(m >= 1 && m <= 12) || !(d >= 1 && d <= dim(y, m))) throw new RangeError('date');
  return {y, m, d};
}
function ordinal({y, m, d}) {
  const py = y - 1;
  let n = py * 365 + Math.floor(py / 4) - Math.floor(py / 100) + Math.floor(py / 400);
  for (let k = 1; k < m; k++) n += dim(y, k);
  return n + d;
}
function fromOrdinal(n) {
  if (n < 1 || n > 3652059) throw new RangeError('date value out of range');
  let y = Math.max(1, Math.floor(n / 365.2425) - 1);
  while (ordinal({y: y + 1, m: 1, d: 1}) <= n) y++;
  while (ordinal({y, m: 1, d: 1}) > n) y--;
  let r = n - ordinal({y, m: 1, d: 1}) + 1, m = 1;
  while (r > dim(y, m)) { r -= dim(y, m); m++; }
  return {y, m, d: r};
}
const addDays = (date, n) => fromOrdinal(ordinal(date) + n);
const weekday = date => (ordinal(date) + 6) % 7;

/** resolve_date: month-name, ISO, slash, relative and weekday dates; missing year -> next occurrence from `now`. */
export function resolveDate(spanText, now) {
  const s = strip(spanText).toLowerCase();
  for (const [kind, rx] of DATE_FULL) {
    const m = rx.exec(s);
    if (!m) continue;
    try {
      if (kind === 'mdy' || kind === 'dmy') {
        let [mon, day] = kind === 'mdy' ? [m[1], m[2]] : [m[2], m[1]];
        const key = rstripChars(mon, '.');
        if (!MONTHS.has(key)) return null;  // KeyError
        mon = MONTHS.get(key);
        let year = m[3] !== undefined ? pyInt(m[3]) : null;
        if (year === null) {
          if (!now) return null;
          year = now.y;
          if (ordinal(mkDate(year, mon, pyInt(day))) < ordinal(now)) year += 1;
        }
        return mkDate(year, mon, pyInt(day));
      }
      if (kind === 'iso') return mkDate(pyInt(m[1]), pyInt(m[2]), pyInt(m[3]));
      if (kind === 'slash') {
        let y = pyInt(m[3]);
        y = y < 100 ? y + 2000 : y;
        return mkDate(y, pyInt(m[1]), pyInt(m[2]));
      }
      if (!now) return null;
      const today = {y: now.y, m: now.m, d: now.d};
      if (kind === 'rel') {
        const off = new Map([['today', 0], ['tonight', 0], ['tomorrow', 1], ['day after tomorrow', 2], ['yesterday', -1]])
          .get(m[1]);
        return off === undefined ? null : addDays(today, off);
      }
      if (kind === 'wd') {
        if (!WEEKDAYS.has(m[2])) return null;  // KeyError
        const target = WEEKDAYS.get(m[2]);
        let ahead = pyMod(target - weekday(today), 7);
        if (m[1] && m[1].startsWith('next') && ahead === 0) ahead = 7;
        return addDays(today, ahead);
      }
      if (kind === 'in') {
        const n = small(m[1]);
        return n === null ? null : addDays(today, n * (m[2].startsWith('w') ? 7 : 1));
      }
    } catch {
      return null;  // ValueError / KeyError -> None; OverflowError would raise, which run's caller also skips
    }
  }
  return null;
}

export const DATE_FORMATS = ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
  '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y', '%Y/%m/%d', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
  '%H:%M', '%H:%M:%S', '%I:%M %p', '%B %d, %Y', '%d %B %Y', '%Y%m%d', '%Y-%m'];

const MONTH_TITLE = MONTH_NAMES.map(m => m[0].toUpperCase() + m.slice(1));
const p2 = n => String(n).padStart(2, '0');
function strftime(fmt, y, mo, d, h, mi) {
  return fmt.replace(/%([YmdHMSIpB%])/g, (_, c) => ({
    Y: String(y).padStart(4, '0'), m: p2(mo), d: p2(d), H: p2(h), M: p2(mi), S: '00', I: p2(h % 12 || 12),
    p: h < 12 ? 'AM' : 'PM', B: MONTH_TITLE[mo - 1], '%': '%',
  })[c]);
}

/** parse_now: context 'now' (datetime.fromisoformat after dropping 'Z') -> {y, m, d} or null. */
export function parseNow(ctx) {
  const now = ctx && typeof ctx === 'object' ? ctx.now : undefined;
  if (!truthy(now)) return null;
  const s = pyStr(now).replaceAll('Z', '');
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:([^]))?([^]*)$/u.exec(s) || /^(\d{4})(\d{2})(\d{2})(?:([^]))?([^]*)$/u.exec(s);
  if (!m) return null;
  if (m[4] !== undefined) {
    const t = /^(\d{2})(?::?(\d{2})(?::?(\d{2})(?:[.,](\d+))?)?)?(?:[+-](\d{2})(?::?(\d{2})(?::?(\d{2})(?:[.,]\d+)?)?)?)?$/.exec(m[5]);
    if (!t || +t[1] > 23 || (t[2] && +t[2] > 59) || (t[3] && +t[3] > 59)) return null;
  }
  try {
    return mkDate(+m[1], +m[2], +m[3]);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- options in descriptions

const QUOTED = pyRe('["\'“‘`]([^"\'”’`\\n]{1,40})["\'”’`]', 'g');
const DASHED = pyRe('(?:^|\\s)-\\s*[\\"\']?([\\w/.*+\\-]{1,40})[\\"\']?\\s*:', 'g');

/** desc_options: values quoted or listed in the parameter description, in order of first appearance, at most 32. */
export function descOptions(param) {
  const raw = param.description;
  const d = truthy(raw) ? (typeof raw === 'string' ? raw : pyStr(raw)) : '';
  const out = [];
  for (const rx of [QUOTED, DASHED]) {
    rx.lastIndex = 0;
    for (const m of d.matchAll(rx)) {
      const v = strip(m[1]);
      if (v && !out.includes(v)) out.push(v);
    }
  }
  return out.slice(0, 32);
}

// ---------------------------------------------------------------- interpreter

const has = (o, k) => o !== null && typeof o === 'object' && Object.prototype.hasOwnProperty.call(o, k);
const get = (o, k) => (has(o, k) ? o[k] : undefined);

/** coerce(v, typ): the parameter's declared type decides ints, floats, booleans and strings. */
export function coerce(v, typ) {
  if (v === null || v === undefined || typ === null || typ === undefined || typ === 'any' || typ === 'array' ||
      typ === 'object') return v;
  if ((typ === 'integer' || typ === 'number') && typeof v === 'string') {
    const f = parseNumber(v);
    if (f === null) return v;
    if (typ === 'integer') return Math.abs(f - pyRound(f)) < 1e-9 ? pyRound(f) : v;
    return typ === 'number' ? pyFloat(f) : v;
  }
  if (typ === 'integer' && v instanceof PyFloat) return v.v;
  if (typ === 'boolean' && typeof v === 'string' && ['true', 'false'].includes(strip(v).toLowerCase()))
    return strip(v).toLowerCase() === 'true';
  if (typ === 'string' && typeof v !== 'string' && !Array.isArray(v) && !(typeof v === 'object' && !(v instanceof PyFloat)))
    return typeof v === 'boolean' ? String(v) : pyStr(v);
  return v;
}

const DIGITS_STRIP = pyRe('[^\\d+]', 'g');

function run_(prog, text, param, ctx) {
  const op = prog.op;
  const sl = sp => text.slice(sp[0], sp[1]);
  switch (op) {
    case 'default': return get(param, 'default');
    case 'bool': return prog.value;
    case 'enum': return pyIndex(param.enum, prog.index);
    case 'option': return pyIndex(descOptions(param), prog.index);
    case 'const': return prog.value;
    case 'digits': return sl(prog.span).replace(DIGITS_STRIP, '');
    case 'affix': { const [pre, suf] = pyIndex(AFFIXES, prog.index); return pre + sl(prog.span) + suf; }
    case 'ref': return `#${prog.call}`;
    case 'copy': return sl(prog.span);
    case 'number': {
      const p = parseNumber(sl(prog.span));
      if (p === null) throw new TypeError('NoneType * float');
      const v = p * prog.scale;
      return prog.int && Math.abs(v - pyRound(v)) < 1e-9 ? pyRound(v) : pyFloat(v);
    }
    case 'timepart': {
      const t = resolveTime(sl(prog.span), text);
      if (t === null) throw new TypeError('cannot unpack None');
      return prog.part === 'hour' ? t[0] : t[1];
    }
    case 'datetime': {
      const now = parseNow(ctx);
      const d = prog.date ? resolveDate(sl(prog.date), now) : now;
      const t = prog.time ? resolveTime(sl(prog.time), text) : [0, 0];
      if (d === null || t === null) return null;
      return strftime(prog.format, d.y, d.m, d.d, t[0], t[1]);
    }
    case 'lookup': {
      const table = TABLES[prog.table];
      const key = strip(sl(prog.span)).toLowerCase();
      if (!has(table, key)) throw new RangeError(`KeyError ${key}`);
      return table[key];
    }
    case 'list': return prog.items.map(p => run(p, text, truthy(get(param, 'items')) ? param.items : {}, ctx));
    default: throw new Error(`op ${op}`);
  }
}

/** Execute a program and coerce the result to the parameter's declared type. Throws where Python raises. */
export function run(prog, text, param = {}, ctx = null) {
  param = param || {};
  return coerce(run_(prog, text, param, ctx), has(param, 'type') ? param.type : undefined);
}

export {num, isPyFloat};
