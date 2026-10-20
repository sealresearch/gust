// Car environment: an EV dashboard (instrument cluster, centre screen, climate strip) driven by Relay's calls.
// Same contract as phone.mjs; render(s) is a pure SVG string (560x460).

const ACC = '#4da3ff';
const HOT = '#ff8a3d';
const FAN = {off: 0, low: 1, medium: 3, high: 4, max: 5};
const FEATURES = ['defrost', 'headlights', 'sunroof', 'hazard lights', 'wipers', 'air conditioning'];
const WIN = ['front left', 'front right', 'rear left', 'rear right'];
const DESTS = ['work', 'home', 'airport'];

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});

export default {
  key: 'car',
  label: 'Car',
  subtitle: 'climate, media, navigation, calls',
  tagline: 'Climate zones, seats, windows, media, maps and calls on a car dashboard',
  placeholder: 'tell the car what you need...',
  tools: [
    tool('set_temperature', 'Set the cabin temperature in degrees.', {
      temperature: {type: 'number', description: 'Target temperature in °C, 16-28.'},
      zone: en(['both', 'driver', 'passenger'], 'Which side: both (whole car), driver or passenger.')}, ['temperature']),
    tool('set_seat_heater', 'Set a seat heater level.', {
      seat: en(['driver', 'passenger', 'both'], 'Which seat.'),
      level: en(['off', 'low', 'high'], 'Heat level: off, low or high.')}, ['seat', 'level']),
    tool('set_fan', 'Set the climate fan speed.', {speed: en(['off', 'low', 'medium', 'high', 'max'], 'Fan speed.')}, ['speed']),
    tool('set_windows', 'Open or close car windows.', {
      window: en(['all', 'driver', 'passenger', 'rear', 'left side', 'right side', 'front', 'rear left', 'rear right'],
        'Which windows: all, driver, passenger, rear, left side, right side, front, rear left or rear right.'),
      position: en(['open', 'closed', 'half'], 'open (down), closed (up) or half open.')}, ['window', 'position']),
    tool('lock_doors', 'Lock or unlock the doors.', {action: en(['lock', 'unlock'], 'lock or unlock.')}, ['action']),
    tool('play_music', 'Play a song, artist, playlist or genre.', {query: str('What to play.')}, ['query']),
    tool('control_media', 'Pause, resume or skip the current audio.', {
      action: en(['pause', 'resume', 'skip', 'previous'], 'pause, resume, skip to the next track, or go back to the previous one.')}, ['action']),
    tool('set_volume', 'Set the audio volume.', {percent: {type: 'integer', description: 'Volume percent, 0-100.'}}, ['percent']),
    tool('navigate_to', 'Get directions to a place.', {destination: str('Where to drive, e.g. "the airport".')}, ['destination']),
    tool('call_contact', 'Call a person.', {name: str('Contact name.')}, ['name']),
    tool('send_message', 'Send a text message.', {recipient: str('Contact name.'), text: str('The message text.')}, ['recipient', 'text']),
    tool('set_feature', 'Switch a car feature on or off.', {
      feature: en(FEATURES, 'defrost (foggy windshield), headlights, sunroof, hazard lights, wipers (rain) or air conditioning.'),
      state: en(['on', 'off'], 'on (or open) / off (or closed).')}, ['feature', 'state']),
    tool('cruise_control', 'Turn cruise control on at a speed, or off.', {
      state: en(['on', 'off'], 'on or off.'),
      speed: {type: 'integer', description: 'Cruise speed in km/h.'}}, ['state']),
    tool('check_range', 'Say the battery level and remaining driving range.'),
  ],

  init() {
    return {speed: 62, cruise: 0, temp: {driver: 21, passenger: 21}, seat: {driver: 'off', passenger: 'off'}, fan: 'low',
      windows: {'front left': 0, 'front right': 0, 'rear left': 0, 'rear right': 0}, locked: true,
      media: 'Morning Drive Mix', playing: true, volume: 40, dest: '', eta: 0, km: 0, call: '', message: null,
      features: {defrost: false, headlights: true, sunroof: false, 'hazard lights': false, wipers: false, 'air conditioning': true},
      battery: 68, range: 214, note: 'ready'};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const frame = () => ctx.render && ctx.render();
    switch (c.tool) {
      case 'set_temperature': {
        const t = clampF(a.temperature, 16, 28);
        const z = pick(['both', 'driver', 'passenger'], a.zone, 'both');
        if (z !== 'passenger') s.temp.driver = t;
        if (z !== 'driver') s.temp.passenger = t;
        s.note = `${z === 'both' ? 'cabin' : z} ${t}°`; return `${z === 'both' ? 'both zones' : z} set to ${t} °C`;
      }
      case 'set_seat_heater': {
        const seat = pick(['driver', 'passenger', 'both'], a.seat, 'driver');
        const lv = pick(['off', 'low', 'high'], a.level, /warm|heat|on/.test(String(a.level)) ? 'high' : 'off');
        if (seat !== 'passenger') s.seat.driver = lv;
        if (seat !== 'driver') s.seat.passenger = lv;
        s.note = `${seat} seat heat ${lv}`; return `${seat} seat heater ${lv}`;
      }
      case 'set_fan': {
        const x = String(a.speed ?? '').toLowerCase();
        s.fan = FAN[x] !== undefined ? x : /full|max|highest|top/.test(x) ? 'max' : /^\d$/.test(x) ? Object.keys(FAN).find(k => FAN[k] >= +x) : 'medium';
        s.note = `fan ${s.fan}`; return `fan ${s.fan}`;
      }
      case 'set_windows': {
        const w = pick(['all', 'driver', 'passenger', 'rear', 'left side', 'right side', 'front', 'rear left', 'rear right'], a.window, 'all');
        const p = String(a.position || '').toLowerCase();
        const v = /half|bit|crack|little/.test(p) ? 50 : /close|up|shut/.test(p) ? 0 : 100;
        const which = {all: WIN, driver: ['front left'], passenger: ['front right'], rear: ['rear left', 'rear right'],
          'left side': ['front left', 'rear left'], 'right side': ['front right', 'rear right'],
          front: ['front left', 'front right'], 'rear left': ['rear left'], 'rear right': ['rear right']}[w];
        for (let step = 1; step <= 5; step++) {
          for (const k of which) s.windows[k] = Math.round(s.windows[k] + (v - s.windows[k]) * step / 5);
          frame(); await ctx.sleep(70);
        }
        s.note = `${w} window${w === 'driver' || w === 'passenger' ? '' : 's'} ${v === 0 ? 'closed' : v === 50 ? 'half open' : 'open'}`;
        return s.note;
      }
      case 'lock_doors': s.locked = !/unlock|open/.test(String(a.action || '').toLowerCase()); s.note = s.locked ? 'doors locked' : 'doors unlocked'; return s.note;
      case 'play_music': s.media = String(a.query || 'radio'); s.playing = true; s.note = `playing ${s.media}`; return s.note;
      case 'control_media': {
        const x = String(a.action || '').toLowerCase();
        if (/pause|stop/.test(x)) s.playing = false;
        else if (/resume|play|continue/.test(x)) s.playing = true;
        else if (/prev|back/.test(x)) { s.media = 'Previous track'; s.playing = true; }
        else { s.media = 'Next track'; s.playing = true; }
        s.note = `media ${x}`; return `media: ${x}`;
      }
      case 'set_volume': s.volume = clampI(a.percent, 0, 100); s.note = `volume ${s.volume}%`; return s.note;
      case 'navigate_to': {
        s.dest = String(a.destination || 'destination').trim();
        const h = hash(s.dest.toLowerCase()); s.km = 4 + h % 60; s.eta = Math.round(s.km * 1.4 + 3);
        s.note = `route to ${s.dest}`; return `navigating to ${s.dest} (${s.km} km, ${s.eta} min)`;
      }
      case 'call_contact': s.call = String(a.name || 'contact'); s.message = null; s.note = `calling ${s.call}`; return `calling ${s.call} (sandbox: nothing is dialled)`;
      case 'send_message': s.message = {to: String(a.recipient || ''), text: String(a.text || '')}; s.note = `text to ${s.message.to}`; return `message to ${s.message.to}: "${s.message.text}"`;
      case 'set_feature': {
        const f = pick(FEATURES, a.feature, '') || alias(a.feature);
        if (!f) return `unknown feature ${a.feature}`;
        const on = !/off|close|shut/.test(String(a.state || 'on').toLowerCase());
        s.features[f] = on; s.note = `${f} ${f === 'sunroof' ? (on ? 'open' : 'closed') : on ? 'on' : 'off'}`; return s.note;
      }
      case 'cruise_control': {
        const off = /off|cancel|stop/.test(String(a.state || '').toLowerCase());
        if (off) { s.cruise = 0; s.note = 'cruise off'; return 'cruise control off'; }
        const target = a.speed !== undefined ? clampI(a.speed, 30, 160) : s.speed;
        for (let i = 1; i <= 8; i++) { s.speed = Math.round(s.speed + (target - s.speed) * i / 8); frame(); await ctx.sleep(60); }
        s.cruise = target; s.note = `cruise ${target} km/h`; return `cruise control set to ${target} km/h`;
      }
      case 'check_range': {
        const t = `Battery is at ${s.battery} percent, about ${s.range} kilometres of range.`;
        ctx.speak && ctx.speak(t); s.note = `${s.battery}% · ${s.range} km`; return t;
      }
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs>
    <linearGradient id="cbg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d0f14"/><stop offset="1" stop-color="#07080b"/></linearGradient>
    <linearGradient id="cmap" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#121a24"/><stop offset="1" stop-color="#0e141c"/></linearGradient>
    <clipPath id="cmapclip"><rect x="16" y="166" width="292" height="196" rx="12"/></clipPath>
  </defs>
  <rect width="560" height="460" fill="url(#cbg)"/>
  ${cluster(s)}${mapCard(s)}${mediaCard(s)}${phoneCard(s)}${climate(s)}
</svg>`;
  },

  examples: [
    {title: 'Warm + jazz', sub: '2 calls · number', q: 'Set the temperature to 22 degrees and play some jazz'},
    {title: 'Drive + call', sub: '2 calls', q: 'Navigate to the airport and call Lena'},
    {title: 'Hands-free text', sub: 'message span', q: "Text Omar that I'll be there in ten minutes"},
    {title: 'Cold seat', sub: 'seat + level enums', q: 'Turn the passenger seat heater on high'},
    {title: 'Fresh air', sub: '2 calls · enums', q: 'Put the fan on max and close all the windows'},
    {title: 'Only my side', sub: 'negation', q: 'Open the driver window but not the passenger one'},
    {title: 'Foggy glass', sub: 'feature from context', q: 'The windshield is fogged up, turn on the defrost'},
    {title: 'Highway', sub: '2 calls · speed', q: 'Set cruise control to 110 and skip this song'},
    {title: 'Lock + quiet', sub: '2 calls', q: 'Lock the doors and turn the volume down to 20 percent'},
    {title: 'Range', sub: 'real speech', q: 'How much range do I have left?'},
    {title: 'Off topic', sub: 'abstains', q: 'Who won the Formula 1 race last weekend?'},
  ],
  help: 'Try: "set the temperature to 22 and play some jazz", "navigate to the airport and call Lena", "fan on max".',
};

// ---------------------------------------------------------------- helpers

function hash(t) { let h = 2166136261; for (const ch of t) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619) >>> 0; } return h; }
function clampI(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : lo; }
function clampF(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n * 2) / 2)) : 21; }
function pick(list, v, d) { const x = String(v ?? '').toLowerCase().trim(); return list.find(o => o === x) || list.find(o => x && (x.includes(o) || o.includes(x))) || d; }
function alias(v) {
  const x = String(v || '').toLowerCase();
  return /fog|demist|defog|windshield|windscreen/.test(x) ? 'defrost' : /light|lamp|beam/.test(x) && !/hazard/.test(x) ? 'headlights'
    : /roof/.test(x) ? 'sunroof' : /hazard|blinker|flasher/.test(x) ? 'hazard lights' : /wiper|rain/.test(x) ? 'wipers'
    : /ac|a\/c|air|cool/.test(x) ? 'air conditioning' : '';
}
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const T = (x, y, t, sz = 11, fill = '#e8ecf3', extra = '') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;
const clip = (t, n) => { const x = String(t); return x.length > n ? x.slice(0, n - 1) + '…' : x; };
function wrap(t, n) { const out = ['']; for (const x of String(t).split(/\s+/)) { if ((out.at(-1) + ' ' + x).trim().length > n) out.push(x); else out[out.length - 1] = (out.at(-1) + ' ' + x).trim(); } return out; }
const card = (x, y, w, h) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#141821" stroke="#222836"/>`;
function arc(cx, cy, r, a0, a1) {
  const p = a => [cx + r * Math.cos(a * Math.PI / 180), cy + r * Math.sin(a * Math.PI / 180)];
  const [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
}

function cluster(s) {
  const cx = 104, cy = 96, r = 62, a0 = 135, a1 = 405, max = 180;
  const av = a0 + (a1 - a0) * Math.min(s.speed, max) / max;
  const col = s.cruise ? '#3ddc97' : ACC;
  let o = `<rect x="16" y="12" width="528" height="142" rx="16" fill="#10131a" stroke="#1f2430"/>`
    + `<path d="${arc(cx, cy, r, a0, a1)}" fill="none" stroke="#232a37" stroke-width="9" stroke-linecap="round"/>`
    + `<path d="${arc(cx, cy, r, a0, Math.max(a0 + 1, av))}" fill="none" stroke="${col}" stroke-width="9" stroke-linecap="round"/>`
    + T(cx, cy + 6, s.speed, 34, '#fff', 'text-anchor="middle" font-weight="700"') + T(cx, cy + 24, 'km/h', 10, '#8a93a6', 'text-anchor="middle"')
    + (s.cruise ? T(cx, cy + 50, `◎ CRUISE ${s.cruise}`, 10, '#3ddc97', 'text-anchor="middle" font-weight="700"') : T(cx, cy + 50, 'cruise off', 9, '#5c6578', 'text-anchor="middle"'));
  // centre: time, gear, indicators
  o += T(280, 48, '09:41', 22, '#fff', 'text-anchor="middle" font-weight="600"')
    + ['P', 'R', 'N', 'D'].map((g, i) => T(250 + i * 20, 74, g, 13, g === 'D' ? ACC : '#5c6578', `text-anchor="middle" font-weight="${g === 'D' ? 800 : 500}"`)).join('');
  const ind = [['headlights', '◐ lights'], ['defrost', '≋ defrost'], ['hazard lights', '⚠ hazard'], ['wipers', '⌒ wipers'], ['sunroof', '▭ roof']];
  o += ind.map(([k, l], i) => { const on = s.features[k]; const x = 196 + (i % 3) * 58, y = 102 + Math.floor(i / 3) * 22;
    return `<rect x="${x - 26}" y="${y - 12}" width="52" height="17" rx="8.5" fill="${on ? (k === 'hazard lights' ? '#5a2a10' : '#14304f') : '#171b24'}"/>` + T(x, y, l, 8.5, on ? (k === 'hazard lights' ? HOT : '#bcdcff') : '#5c6578', 'text-anchor="middle"'); }).join('')
    + `<rect x="${196 + 2 * 58 - 26}" y="${124 - 12}" width="52" height="17" rx="8.5" fill="${s.locked ? '#14304f' : '#3a1a1a'}"/>` + T(196 + 2 * 58, 124, s.locked ? '🔒 locked' : '🔓 open', 8.5, s.locked ? '#bcdcff' : '#ff9b9b', 'text-anchor="middle"');
  // right: battery / range
  const bw = 120, bx = 400;
  o += T(bx, 48, 'BATTERY', 9, '#8a93a6', 'letter-spacing="1.5"') + `<rect x="${bx}" y="58" width="${bw}" height="16" rx="5" fill="#1c212c"/><rect x="${bx}" y="58" width="${bw * s.battery / 100}" height="16" rx="5" fill="#3ddc97"/>`
    + `<rect x="${bx + bw + 2}" y="62" width="4" height="8" rx="1" fill="#1c212c"/>`
    + T(bx, 96, `${s.battery}%`, 20, '#fff', 'font-weight="700"') + T(bx + 58, 96, `${s.range} km left`, 11, '#a9b3c6')
    + T(bx, 122, s.features['air conditioning'] ? '❄ A/C on' : 'A/C off', 10, s.features['air conditioning'] ? '#9fd3ff' : '#5c6578')
    + T(bx + 66, 122, '☀ 14° out', 10, '#a9b3c6');
  return o;
}

function mapCard(s) {
  let o = card(16, 166, 292, 196) + `<g clip-path="url(#cmapclip)"><rect x="16" y="166" width="292" height="196" fill="url(#cmap)"/>`;
  for (let i = 0; i < 7; i++) o += `<line x1="${16 + i * 48}" y1="166" x2="${-20 + i * 48 + 60}" y2="362" stroke="#1b2633" stroke-width="${i % 3 ? 3 : 7}"/>`;
  for (let i = 0; i < 5; i++) o += `<line x1="16" y1="${186 + i * 42}" x2="308" y2="${176 + i * 44}" stroke="#1b2633" stroke-width="${i % 2 ? 3 : 6}"/>`;
  o += `<path d="M30 300 Q 90 270 140 290 T 260 230" fill="none" stroke="#18303f" stroke-width="14" opacity=".8"/>`;  // river
  if (s.dest) {
    o += `<path d="M150 340 L150 300 L196 286 L214 246 L262 214" fill="none" stroke="${ACC}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>`
      + `<circle cx="262" cy="214" r="9" fill="${ACC}"/><circle cx="262" cy="214" r="3.5" fill="#fff"/>`;
  }
  o += `<polygon points="150,328 143,345 150,340 157,345" fill="#fff"/></g>`;
  o += `<rect x="26" y="176" width="${s.dest ? 200 : 120}" height="${s.dest ? 44 : 26}" rx="9" fill="#0b0e14" opacity=".88"/>`
    + (s.dest ? T(36, 194, clip(s.dest, 26), 12, '#fff', 'font-weight="700"') + T(36, 211, `${s.eta} min · ${s.km} km · via fastest`, 10, '#9fc7ff')
      : T(36, 193, 'no route set', 10, '#8a93a6'));
  return o;
}

function mediaCard(s) {
  const x = 318, y = 166, w = 226;
  return card(x, y, w, 92) + `<rect x="${x + 12}" y="${y + 12}" width="48" height="48" rx="8" fill="${s.playing ? '#2b4f7a' : '#232a37'}"/>`
    + T(x + 36, y + 43, '♫', 22, '#fff', 'text-anchor="middle"')
    + T(x + 70, y + 30, clip(s.media, 22), 12, '#fff', 'font-weight="700"') + T(x + 70, y + 47, s.playing ? 'now playing' : 'paused', 10, s.playing ? '#9fc7ff' : '#8a93a6')
    + T(x + w - 14, y + 47, s.playing ? '⏸' : '▶', 14, '#fff', 'text-anchor="end"')
    + T(x + 12, y + 80, `vol ${s.volume}%`, 9, '#8a93a6') + `<rect x="${x + 62}" y="${y + 74}" width="${w - 76}" height="6" rx="3" fill="#232a37"/><rect x="${x + 62}" y="${y + 74}" width="${(w - 76) * s.volume / 100}" height="6" rx="3" fill="${ACC}"/>`;
}

function phoneCard(s) {
  const x = 318, y = 266, w = 226;
  let o = card(x, y, w, 96);
  if (s.call) {
    o += `<circle cx="${x + 32}" cy="${y + 34}" r="18" fill="#1f5d45"/>` + T(x + 32, y + 40, s.call[0].toUpperCase(), 16, '#fff', 'text-anchor="middle" font-weight="700"')
      + T(x + 60, y + 31, clip(s.call, 20), 13, '#fff', 'font-weight="700"') + T(x + 60, y + 47, 'calling… 00:04', 10, '#3ddc97')
      + T(x + 12, y + 82, 'sandbox: nothing is dialled', 9, '#5c6578');
  } else if (s.message) {
    const lines = wrap(s.message.text, 30).slice(0, 2);
    o += T(x + 12, y + 22, `to ${clip(s.message.to, 24)}`, 10, '#8a93a6')
      + `<rect x="${x + 12}" y="${y + 30}" width="${w - 24}" height="${lines.length * 14 + 12}" rx="10" fill="${ACC}"/>`
      + lines.map((l, i) => T(x + 22, y + 46 + i * 14, l, 10.5, '#fff')).join('') + T(x + w - 14, y + 88, 'sent', 9, '#8a93a6', 'text-anchor="end"');
  } else {
    o += T(x + 12, y + 24, 'PHONE', 9, '#8a93a6', 'letter-spacing="1.5"') + T(x + 12, y + 48, 'connected · no active call', 11, '#a9b3c6')
      + T(x + 12, y + 72, 'recent: Lena, Omar, Dad', 10, '#5c6578');
  }
  return o;
}

function climate(s) {
  const y = 374;
  let o = `<rect x="16" y="${y}" width="528" height="74" rx="14" fill="#10131a" stroke="#1f2430"/>`;
  const zone = (x, label, t, seat, anchor) => {
    const heat = seat === 'high' ? 3 : seat === 'low' ? 1 : 0;
    const waves = [0, 1, 2].map(i => `<path d="M${x + (anchor === 'end' ? -70 : 70) + i * 7} ${y + 54} q3 -5 0 -10 q-3 -5 0 -10" fill="none" stroke="${i < heat || (heat === 1 && i === 0) ? HOT : '#2a3140'}" stroke-width="2"/>`).join('');
    return T(x, y + 22, label, 9, '#8a93a6', `text-anchor="${anchor}" letter-spacing="1.5"`) + T(x, y + 52, `${t}°`, 26, t >= 23 ? '#ffb27a' : t <= 18 ? '#9fd3ff' : '#fff', `text-anchor="${anchor}" font-weight="700"`)
      + waves + T(x + (anchor === 'end' ? -60 : 60), y + 66, `seat ${seat}`, 8, heat ? HOT : '#5c6578', `text-anchor="middle"`);
  };
  o += zone(36, 'DRIVER', s.temp.driver, s.seat.driver, 'start') + zone(524, 'PASSENGER', s.temp.passenger, s.seat.passenger, 'end');
  // fan
  const lv = FAN[s.fan] ?? 1;
  o += T(176, y + 22, `FAN · ${s.fan}`, 9, '#8a93a6', 'letter-spacing="1.5"')
    + [0, 1, 2, 3, 4].map(i => `<rect x="${176 + i * 12}" y="${y + 58 - (i + 1) * 6}" width="8" height="${(i + 1) * 6}" rx="2" fill="${i < lv ? ACC : '#232a37'}"/>`).join('');
  // windows: top-down car with four window bars
  const wx = 266, wy = y + 12;
  o += T(wx + 6, y + 22, 'WINDOWS', 9, '#8a93a6', 'letter-spacing="1.5" text-anchor="start"')
    + `<rect x="${wx + 82}" y="${wy}" width="34" height="54" rx="10" fill="#1a1f2a" stroke="#2c3443"/>`
    + `<rect x="${wx + 88}" y="${wy + 16}" width="22" height="${s.features.sunroof ? 14 : 4}" rx="2" fill="${s.features.sunroof ? '#9fd3ff' : '#2c3443'}"/>`;
  WIN.forEach((k, i) => {
    const left = i % 2 === 0, rear = i >= 2, open = s.windows[k];
    const bx = left ? wx + 78 : wx + 116, by = wy + (rear ? 32 : 10);
    o += `<rect x="${bx}" y="${by}" width="4" height="16" rx="2" fill="#2c3443"/><rect x="${bx}" y="${by}" width="4" height="${16 * open / 100}" rx="2" fill="#9fd3ff"/>`;
  });
  const open = WIN.filter(k => s.windows[k] > 0);
  o += T(wx + 6, y + 40, open.length ? `${open.length} open` : 'all closed', 10, open.length ? '#9fd3ff' : '#a9b3c6')
    + T(wx + 6, y + 56, s.features.sunroof ? 'roof open' : 'roof shut', 9, '#5c6578');
  // note
  o += T(255, 147, `› ${clip(s.note, 30)}`, 9.5, '#cfe3ff', 'text-anchor="middle" font-family="ui-monospace,monospace"');
  return o;
}
