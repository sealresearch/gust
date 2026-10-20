// AR glasses environment: a heads-up display over a night street. Relay's calls drive navigation, nearby pins,
// capture, saved places and calls. Same contract as phone.mjs; render(s) is a pure SVG string (560x460).

const HUD = '#7cf5d4';      // display ink
const WARN = '#ff6b6b';     // REC, call end
const MODES = {walking: {speed: 80, verb: 'walk'}, cycling: {speed: 250, verb: 'ride'}, driving: {speed: 520, verb: 'drive'}};
const POIS = {
  coffee: [['Ember Roasters', 140], ['Night Owl Café', 310], ['Bean & Barrel', 520]],
  food: [['Noodle Lantern', 180], ['Corner Bistro', 260], ['Taco Tram', 450]],
  pharmacy: [['Greenleaf Pharmacy', 220], ['24h Apotheek', 610]],
  transit: [['Tram stop Canal St', 90], ['Metro Central', 380], ['Bus 12 · Market', 240]],
  atm: [['ATM · City Bank', 70], ['ATM · Station hall', 330]],
  parking: [['P1 Harbour garage', 290], ['P2 Street parking', 120]],
  restroom: [['Public restroom', 160], ['Station toilets', 400]],
};
const POI_ALIAS = [[/coffee|caf|espresso|latte/, 'coffee'], [/food|eat|restaurant|lunch|dinner|bite|pizza|sushi/, 'food'],
  [/pharm|drug|chemist|medic/, 'pharmacy'], [/transit|tram|bus|metro|subway|train|stop|station/, 'transit'],
  [/atm|cash|bank|money/, 'atm'], [/park/, 'parking'], [/rest ?room|toilet|bathroom|wc/, 'restroom']];
const AVOID = ['stairs', 'hills', 'busy roads', 'tolls', 'highways', 'ferries'];
const SIGN = {jp: '東口 → 出口 · 4-7番線', en: 'East exit → Platforms 4-7'};
const SIGN_T = {english: 'East exit → Platforms 4-7', spanish: 'Salida este → Andenes 4-7', french: 'Sortie est → Quais 4-7',
  german: 'Ausgang Ost → Gleise 4-7', dutch: 'Oostuitgang → Perrons 4-7', italian: 'Uscita est → Binari 4-7'};
const PIN_SPOTS = [[112, 226], [446, 214], [212, 252], [362, 250]];

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});

export default {
  key: 'glasses',
  label: 'AR glasses',
  subtitle: 'a short request becomes a route',
  tagline: 'Turn-by-turn HUD, nearby places, capture and calls on smart glasses',
  placeholder: 'tell the glasses where to go...',
  tools: [
    tool('navigate_to', 'Start directions to a place (take me to, go to, get me to).', {
      destination: str('Where to go, as the user names it, e.g. "the central station".'),
      mode: en(['walking', 'cycling', 'driving'], 'How the user travels: walking (on foot), cycling (bike) or driving (car).')},
    ['destination']),
    tool('find_nearby', 'Show pins for nearby places of one kind.', {
      category: en(['coffee', 'food', 'pharmacy', 'transit', 'atm', 'parking', 'restroom'],
        'Kind of place: coffee, food (restaurants), pharmacy, transit (bus, tram, metro), atm (cash), parking, restroom.')},
    ['category']),
    tool('avoid_on_route', 'Make the route avoid something.', {
      avoid: en(AVOID, 'What to avoid: stairs, hills, busy roads, tolls, highways or ferries.')}, ['avoid']),
    tool('stop_navigation', 'Stop or cancel the current route.'),
    tool('take_photo', 'Take a photo of what the user sees.'),
    tool('record_video', 'Start or stop video recording.', {action: en(['start', 'stop'], 'start or stop recording.')}, ['action']),
    tool('remember_place', 'Save or remember this place under a name.', {label: str('Name to save this spot as, e.g. "my car".')}, ['label']),
    tool('describe_view', 'Describe out loud what the user is looking at.'),
    tool('translate_sign', 'Translate the sign in view.', {language: str('Language name to translate into, e.g. English.')}, ['language']),
    tool('call_contact', 'Call a person.', {name: str('Contact name.')}, ['name']),
    tool('end_call', 'Hang up the current call.'),
    tool('set_display_brightness', 'Set the display brightness.', {percent: {type: 'integer', description: 'Brightness percent, 5-100.'}}, ['percent']),
  ],

  init() {
    return {heading: 18, arrow: 0, dest: '', mode: 'walking', distance: 0, turn: 'straight', next: 0, avoid: [],
      category: '', pois: [], places: [], caption: '', recording: false, recStart: 0, call: '', flash: false,
      brightness: 85, photos: 0, note: 'heads-up display ready'};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const frame = () => ctx.render && ctx.render();
    switch (c.tool) {
      case 'navigate_to': {
        const dest = String(a.destination || '').trim() || 'destination';
        const h = hash(dest.toLowerCase());
        s.dest = dest;
        s.mode = pickMode(a.mode, s.mode);
        s.distance = 350 + (h % 2600);
        s.turn = ['left', 'right', 'straight'][h % 3];
        s.next = 40 + ((h >>> 4) % 180);
        const goal = s.turn === 'left' ? -55 : s.turn === 'right' ? 55 : 0;
        const fromH = s.heading, toH = (h >>> 2) % 360;
        for (let i = 1; i <= 8; i++) {
          s.arrow = s.arrow + (goal - s.arrow) * (i / 8);
          s.heading = Math.round(fromH + angDiff(fromH, toH) * (i / 8) + 360) % 360;
          frame(); await ctx.sleep(45);
        }
        s.note = `route to ${dest}`;
        return `navigating to ${dest} (${s.mode}, ${fmtDist(s.distance)}, ${eta(s)} min)`;
      }
      case 'find_nearby': {
        const cat = pickCat(a.category);
        s.category = cat; s.pois = POIS[cat].map(([name, d]) => ({name, d}));
        s.note = `${s.pois.length} ${cat} places nearby`;
        return `${s.pois.length} ${cat} places: ${s.pois.map(p => p.name).join(', ')}`;
      }
      case 'avoid_on_route': {
        const v = pickFrom(AVOID, a.avoid) || String(a.avoid || '').toLowerCase();
        if (v && !s.avoid.includes(v)) s.avoid.push(v);
        if (s.dest) s.distance = Math.round(s.distance * 1.08);
        s.note = `avoiding ${s.avoid.join(', ')}`;
        return `route avoids ${v}`;
      }
      case 'stop_navigation': {
        const was = s.dest; s.dest = ''; s.avoid = []; s.arrow = 0; s.note = 'directions ended';
        return was ? `stopped directions to ${was}` : 'no route was active';
      }
      case 'take_photo':
        s.flash = true; frame(); await ctx.sleep(220); s.flash = false; s.photos++; s.note = `photo ${s.photos} saved`;
        return 'photo taken';
      case 'record_video': {
        const stop = /stop|end|off|finish/.test(String(a.action || '').toLowerCase());
        if (stop) { const was = s.recording; s.recording = false; s.note = 'recording saved'; return was ? 'recording stopped and saved' : 'was not recording'; }
        s.recording = true; s.note = 'recording'; return 'recording started';
      }
      case 'remember_place': {
        const label = String(a.label || 'this spot').trim();
        s.places = [label, ...s.places.filter(p => p !== label)].slice(0, 4); s.note = `saved "${label}"`;
        return `saved this spot as "${label}"`;
      }
      case 'describe_view': {
        const text = 'A quiet street at night. A tram stop is ahead on the right, a café with its lights on to the left, and a crosswalk about thirty metres away.';
        s.caption = text; ctx.speak && ctx.speak(text); s.note = 'described the view';
        return 'described the view out loud';
      }
      case 'translate_sign': {
        const lang = String(a.language || 'English').trim();
        const code = {en: 'english', es: 'spanish', fr: 'french', de: 'german', nl: 'dutch', it: 'italian'}[lang.toLowerCase()];
        const t = SIGN_T[code || lang.toLowerCase()];
        s.caption = t ? `${SIGN.jp}  =  ${t}` : `${SIGN.jp}  =  ${SIGN.en}  (no ${lang} sample in this demo)`;
        s.note = `sign translated to ${lang}`;
        return `sign in ${lang}: ${t || SIGN.en}`;
      }
      case 'call_contact': s.call = String(a.name || 'contact'); s.note = `calling ${s.call}`; return `calling ${s.call} (sandbox: nothing is dialled)`;
      case 'end_call': { const was = s.call; s.call = ''; s.note = 'call ended'; return was ? `ended call with ${was}` : 'no call to end'; }
      case 'set_display_brightness': s.brightness = clamp(a.percent, 5, 100); s.note = `display ${s.brightness}%`; return `display brightness ${s.brightness}%`;
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    const dim = (1 - s.brightness / 100) * 0.8;
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs>
    <linearGradient id="gsky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#05070f"/><stop offset=".55" stop-color="#0d1426"/><stop offset="1" stop-color="#141a24"/></linearGradient>
    <linearGradient id="groad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#11151d"/><stop offset="1" stop-color="#1b2029"/></linearGradient>
    <radialGradient id="gvig" cx="50%" cy="50%" r="62%"><stop offset=".6" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".85"/></radialGradient>
    <radialGradient id="glamp" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#ffd58a" stop-opacity=".45"/><stop offset="1" stop-color="#ffd58a" stop-opacity="0"/></radialGradient>
    <clipPath id="glens"><rect x="8" y="8" width="544" height="444" rx="46"/></clipPath>
  </defs>
  <rect width="560" height="460" fill="#000"/>
  <g clip-path="url(#glens)">
    ${scene()}
    <g opacity="${(1 - dim).toFixed(2)}">${hud(s)}</g>
    ${s.flash ? '<rect width="560" height="460" fill="#fff" opacity=".92"/>' : ''}
    <rect width="560" height="460" fill="url(#gvig)" pointer-events="none"/>
  </g>
  <rect x="8" y="8" width="544" height="444" rx="46" fill="none" stroke="#2c3440" stroke-width="3"/>
</svg>`;
  },

  examples: [
    {title: 'Walk there', sub: 'destination + mode', q: 'Walk me to the central station'},
    {title: 'Coffee nearby', sub: 'enum from wording', q: 'Is there a coffee place around here?'},
    {title: 'Step-free route', sub: '2 calls', q: 'Take me to the city library and avoid stairs'},
    {title: 'Drive, no tolls', sub: '2 calls · mode + avoid', q: 'Drive to the airport and avoid tolls'},
    {title: 'Not on foot', sub: 'negation', q: "I don't want to walk, let's cycle to the science museum"},
    {title: 'Pin + photo', sub: '2 calls', q: 'Remember this spot as my bike and take a photo'},
    {title: 'Read the sign', sub: 'language span', q: 'Translate this sign into Spanish'},
    {title: 'Stop recording', sub: 'stop, not start', q: 'Stop the video recording'},
    {title: 'What is that?', sub: 'real speech', q: 'Describe what I am looking at'},
    {title: 'Arrived', sub: '2 calls', q: 'Stop navigation and call Sam'},
    {title: 'Trivia', sub: 'abstains', q: 'How tall is the Eiffel Tower?'},
  ],
  help: 'Try: "walk me to the central station", "find a pharmacy nearby", "drive to the airport and avoid tolls".',
};

// ---------------------------------------------------------------- helpers

function hash(t) { let h = 2166136261; for (const ch of t) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619) >>> 0; } return h; }
function clamp(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : hi; }
function angDiff(a, b) { return ((b - a + 540) % 360) - 180; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const cap = t => { const x = String(t); return x ? x[0].toUpperCase() + x.slice(1) : x; };
function wrap(t, n) { const out = ['']; for (const x of String(t).split(/\s+/)) { if ((out.at(-1) + ' ' + x).trim().length > n) out.push(x); else out[out.length - 1] = (out.at(-1) + ' ' + x).trim(); } return out; }
function pickFrom(list, v) { const x = String(v || '').toLowerCase(); return list.find(o => x === o) || list.find(o => x.includes(o) || o.includes(x) && x.length > 2) || ''; }
function pickMode(v, d) {
  const x = String(v || '').toLowerCase();
  if (/cycl|bike|bicycle|ride/.test(x)) return 'cycling';
  if (/driv|car/.test(x)) return 'driving';
  if (/walk|foot/.test(x)) return 'walking';
  return d || 'walking';
}
function pickCat(v) { const x = String(v || '').toLowerCase(); if (POIS[x]) return x; const hit = POI_ALIAS.find(([rx]) => rx.test(x)); return hit ? hit[1] : 'coffee'; }
const fmtDist = m => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m / 10) * 10} m`);
const eta = s => Math.max(1, Math.round(s.distance / MODES[s.mode].speed));
const T = (x, y, t, sz = 11, fill = HUD, extra = '') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;

// static night street (vanishing point 280,250)
function scene() {
  let stars = '';
  for (let i = 0; i < 46; i++) { const h = hash('star' + i); stars += `<circle cx="${20 + h % 520}" cy="${14 + (h >>> 9) % 130}" r="${(h >>> 3) % 3 === 0 ? 1.3 : 0.7}" fill="#cfd8ff" opacity="${0.3 + ((h >>> 5) % 6) / 10}"/>`; }
  const blds = [[0, 120, 96], [92, 150, 70], [158, 190, 54], [366, 186, 50], [412, 140, 74], [480, 108, 90]];
  let b = '';
  blds.forEach(([x, top, w], k) => {
    b += `<rect x="${x}" y="${top}" width="${w}" height="${270 - top}" fill="#0a0e17" stroke="#151b28"/>`;
    for (let wy = top + 12; wy < 255; wy += 16) for (let wx = x + 8; wx < x + w - 8; wx += 14) {
      const h = hash(`${k}:${wx}:${wy}`);
      if (h % 5 < 2) b += `<rect x="${wx}" y="${wy}" width="7" height="9" fill="${h % 7 === 0 ? '#9fd3ff' : '#ffcf7a'}" opacity="${0.35 + (h % 4) / 10}"/>`;
    }
  });
  const lane = [-3, -1, 1, 3].map(k => `<line x1="280" y1="250" x2="${280 + k * 150}" y2="460" stroke="#2a3140" stroke-width="1"/>`).join('');
  const dashes = [0, 1, 2, 3, 4].map(i => { const t = 0.15 + i * 0.19, y = 250 + 210 * t * t * 1.6; return y < 460 ? `<rect x="${280 - 1 - 3 * t}" y="${y}" width="${2 + 6 * t}" height="${4 + 22 * t}" fill="#3b4354"/>` : ''; }).join('');
  const lamps = [[150, 214], [410, 214], [60, 300], [500, 300]].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="34" fill="url(#glamp)"/><rect x="${x - 1}" y="${y}" width="2" height="${300 - y + 60}" fill="#232a36"/><circle cx="${x}" cy="${y}" r="3" fill="#ffe2a8"/>`).join('');
  return `<rect width="560" height="270" fill="url(#gsky)"/>${stars}<circle cx="470" cy="56" r="14" fill="#e9edf7" opacity=".85"/><circle cx="476" cy="52" r="12" fill="#0b1120"/>${b}
    <polygon points="0,460 280,250 560,460" fill="url(#groad)"/><polygon points="0,270 280,250 0,460" fill="#0d1118"/><polygon points="560,270 280,250 560,460" fill="#0d1118"/>${lane}${dashes}${lamps}`;
}

function compass(s) {
  const cx = 280, y = 40, span = 220, ppd = span / 90;  // pixels per degree, +-90 visible
  const names = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW'};
  let out = `<rect x="${cx - span - 10}" y="${y - 22}" width="${2 * span + 20}" height="34" rx="10" fill="#020611" opacity=".55"/>`;
  for (let d = -90; d <= 90; d += 5) {
    const deg = ((s.heading + d) % 360 + 360) % 360, x = cx + d * ppd;
    const r = Math.round(deg / 5) * 5 % 360;
    if (r % 45 === 0) out += T(x, y - 4, names[r], 11, HUD, 'text-anchor="middle" font-weight="700"');
    else if (r % 15 === 0) out += `<line x1="${x}" y1="${y + 2}" x2="${x}" y2="${y - 6}" stroke="${HUD}" stroke-opacity=".55"/>`;
    else out += `<line x1="${x}" y1="${y + 2}" x2="${x}" y2="${y - 1}" stroke="${HUD}" stroke-opacity=".3"/>`;
  }
  return out + `<polygon points="${cx - 6},${y + 14} ${cx + 6},${y + 14} ${cx},${y + 6}" fill="${HUD}"/>` + T(cx, y + 28, `${Math.round(s.heading)}°`, 9, HUD, 'text-anchor="middle" font-family="ui-monospace,monospace"');
}

function hud(s) {
  let o = compass(s);
  // route ribbon on the road
  if (s.dest) {
    const bend = s.turn === 'left' ? -1 : s.turn === 'right' ? 1 : 0;
    const ex = 280 + bend * 120, ey = bend ? 262 : 256;
    o += `<path d="M280 455 C280 360 ${280 + bend * 10} 300 ${ex} ${ey}" fill="none" stroke="${HUD}" stroke-width="10" stroke-opacity=".18" stroke-linecap="round"/>`
      + `<path d="M280 455 C280 360 ${280 + bend * 10} 300 ${ex} ${ey}" fill="none" stroke="${HUD}" stroke-width="3" stroke-dasharray="10 7" stroke-linecap="round"/>`
      + `<circle cx="${ex}" cy="${ey}" r="5" fill="${HUD}"/>`;
  }
  // direction arrow
  o += `<g transform="translate(280 352) rotate(${s.arrow.toFixed(1)})" opacity="${s.dest ? 1 : 0.35}">
      <path d="M0 -44 L28 -6 L11 -6 L11 30 L-11 30 L-11 -6 L-28 -6 Z" fill="${HUD}" fill-opacity=".16" stroke="${HUD}" stroke-width="2.5" stroke-linejoin="round"/></g>`;
  // destination card
  if (s.dest) {
    const m = MODES[s.mode];
    o += `<rect x="24" y="76" width="210" height="${s.avoid.length ? 86 : 70}" rx="12" fill="#020611" opacity=".72" stroke="${HUD}" stroke-opacity=".5"/>`
      + T(38, 96, 'TO', 9, HUD, 'opacity=".7" letter-spacing="1.5"') + T(60, 96, s.dest.length > 24 ? s.dest.slice(0, 23) + '…' : s.dest, 13, '#fff', 'font-weight="700"')
      + T(38, 122, fmtDist(s.distance), 20, HUD, 'font-weight="700"') + T(130, 122, `${eta(s)} min ${m.verb}`, 12, '#dfe', '')
      + T(38, 142, `${cap(s.mode)} · arrive ${arrive(eta(s))}`, 10, '#a8c', 'fill-opacity=".9"')
      + (s.avoid.length ? T(38, 156, `avoiding ${s.avoid.join(', ')}`, 9, '#ffcf7a') : '');
    const instr = s.turn === 'straight' ? `continue straight for ${fmtDist(s.next)}` : `in ${fmtDist(s.next)}, turn ${s.turn}`;
    o += `<rect x="${280 - 120}" y="404" width="240" height="30" rx="15" fill="#020611" opacity=".78" stroke="${HUD}" stroke-opacity=".6"/>` + T(280, 424, instr, 12, '#fff', 'text-anchor="middle" font-weight="600"');
  } else {
    o += `<rect x="190" y="404" width="180" height="30" rx="15" fill="#020611" opacity=".6"/>` + T(280, 424, 'say where you want to go', 11, HUD, 'text-anchor="middle" opacity=".8"');
  }
  // POI pins
  s.pois.forEach((p, i) => {
    const [x, y] = PIN_SPOTS[i % PIN_SPOTS.length];
    o += `<g><path d="M${x} ${y} l-7 -12 a8 8 0 1 1 14 0 z" fill="${HUD}" fill-opacity=".85"/><circle cx="${x}" cy="${y - 15}" r="3" fill="#020611"/>`
      + `<rect x="${x - 58}" y="${y - 46}" width="116" height="24" rx="7" fill="#020611" opacity=".8"/>` + T(x, y - 36, p.name, 9, '#fff', 'text-anchor="middle" font-weight="600"')
      + T(x, y - 26, `${p.d} m · ${Math.max(1, Math.round(p.d / 80))} min walk`, 8, HUD, 'text-anchor="middle"') + '</g>';
  });
  // saved places (right)
  s.places.forEach((p, i) => {
    o += `<rect x="${536 - 130}" y="${150 + i * 26}" width="130" height="20" rx="10" fill="#020611" opacity=".7" stroke="#ffcf7a" stroke-opacity=".6"/>` + T(536 - 120, 164 + i * 26, `★ ${p.length > 17 ? p.slice(0, 16) + '…' : p}`, 10, '#ffcf7a');
  });
  // REC + photos + call
  if (s.recording) o += `<circle cx="470" cy="88" r="6" fill="${WARN}"/>` + T(482, 92, 'REC', 11, WARN, 'font-weight="700" letter-spacing="1"');
  if (s.photos) o += T(536, 120, `◉ ${s.photos} photo${s.photos > 1 ? 's' : ''}`, 9, HUD, 'text-anchor="end" opacity=".8"');
  if (s.call) o += `<rect x="24" y="${s.dest ? 172 : 76}" width="180" height="30" rx="15" fill="#0f3d2e" opacity=".9"/><circle cx="42" cy="${s.dest ? 187 : 91}" r="7" fill="#3ddc97"/>` + T(56, s.dest ? 191 : 95, `on call · ${s.call}`, 11, '#fff', 'font-weight="600"');
  // caption
  if (s.caption) {
    const lines = wrap(s.caption, 62).slice(0, 3);
    o += `<rect x="60" y="248" width="440" height="${lines.length * 15 + 10}" rx="10" fill="#020611" opacity=".8"/>`
      + lines.map((l, i) => T(280, 263 + i * 15, l, 11, '#e8fff8', 'text-anchor="middle"')).join('');
  }
  // status line
  o += T(280, 449, `${s.note} · display ${s.brightness}%`, 9, HUD, 'text-anchor="middle" opacity=".7" font-family="ui-monospace,monospace"');
  return o;
}

function arrive(min) { const t = 9 * 60 + 41 + min; return `${String(Math.floor(t / 60) % 24).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`; }
