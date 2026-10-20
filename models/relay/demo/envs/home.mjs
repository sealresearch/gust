// Smart home environment: a floor plan whose rooms, doors, blinds, thermostat and appliances react to Relay's calls.
// Same contract as phone.mjs: default export {key, label, subtitle, tagline, placeholder, tools, init(), render(s),
// apply(call, s, ctx) -> note, examples, help}. render(s) is a pure SVG string (560x460).

const ROOMS = ['living room', 'kitchen', 'bedroom', 'bathroom', 'office'];
const ROOM_ALL = [...ROOMS, 'all'];
const APPLIANCES = ['coffee machine', 'kettle', 'tv', 'fan', 'dishwasher', 'washing machine', 'oven', 'garden sprinklers'];
const SCENES = ['good morning', 'good night', 'movie night', 'dinner party', 'away'];

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});

export default {
  key: 'home',
  label: 'Smart home',
  subtitle: 'from buttons to sentences',
  tagline: 'Lights, heating, locks, blinds, music, appliances and scenes in one flat',
  placeholder: 'tell the house what you want...',
  tools: [
    tool('set_lights', 'Switch room lights on or off, or set them to a brightness percent (e.g. dim to 20%).', {
      room: en(ROOM_ALL, 'Which room; "all" for every room or the whole house.'),
      state: en(['on', 'off'], 'On or off.'),
      brightness: {type: 'integer', description: 'Brightness percent 1-100, only if a number is said.'}}, ['room', 'state']),
    tool('dim_lights', 'Make a room dimmer or brighter when no number is given (too bright, too dark).', {
      room: en(ROOM_ALL, 'Which room.'),
      direction: en(['down', 'up'], 'down = dim it, it is too bright; up = brighten, it is too dark.')}, ['room', 'direction']),
    tool('set_thermostat', 'Set the heating to a target temperature in degrees.', {
      temperature: {type: 'number', description: 'Target temperature in °C.'}}, ['temperature']),
    tool('nudge_temperature', 'Make it a bit warmer or cooler without a target number.', {
      direction: en(['warmer', 'cooler'], 'Warmer (I am cold) or cooler (I am hot).'),
      degrees: {type: 'number', description: 'How many degrees, if said.'}}, ['direction']),
    tool('set_door', 'Lock or unlock a door.', {
      door: en(['front', 'back', 'garage', 'all doors'], 'Which door: front, back, garage, or all doors.'),
      action: en(['lock', 'unlock'], 'Lock or unlock.')}, ['door', 'action']),
    tool('set_blinds', 'Open, half-open or close the blinds or curtains in a room.', {
      room: en(ROOM_ALL, 'Which room.'),
      position: en(['open', 'half', 'closed'], 'Open (raise), half, or closed (lower, shut).')}, ['room', 'position']),
    tool('play_music', 'Play music in a room.', {
      query: str('What to play: a genre, artist or playlist as said.'),
      room: en(ROOMS, 'Room to play it in.')}, ['query']),
    tool('control_music', 'Pause, resume, skip or change the volume of the music that is playing.', {
      action: en(['pause', 'resume', 'next', 'louder', 'quieter'], 'What to do with the current music.')}, ['action']),
    tool('switch_appliance', 'Start or stop an appliance: switch it on or off.', {
      appliance: en(APPLIANCES, 'The appliance.'),
      state: en(['on', 'off'], 'on = start, run, switch on; off = stop, switch off.')}, ['appliance', 'state']),
    tool('run_scene', 'Run a preset scene that changes several things at once.', {
      scene: en(SCENES, 'good morning, good night (bedtime), movie night, dinner party, away (leaving the house).')}, ['scene']),
    tool('set_security', 'Arm or disarm the alarm system.', {
      mode: en(['away', 'home', 'disarmed'], 'away = armed while out, home = armed at night, disarmed = off.')}, ['mode']),
    tool('check_status', 'Answer a question such as is the door locked, what is the temperature, are the lights on.', {
      item: en(['temperature', 'lights', 'doors', 'blinds', 'music', 'alarm', 'appliances'], 'What to check.')}, ['item'],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also|possible|could|would|can|please)\\b)\\s*(is|are|did|what|what\'s|how|which)\\b[\\s\\S]*\\b(doors?|locked|lights?|temperature|thermostat|heating|warm|cold|alarm|blinds|music|playing|appliances?|running)\\b[\\s\\S]*\\?\\s*$']}),
  ],

  init() {
    const lights = {};
    for (const r of ROOMS) lights[r] = {on: false, b: 70};
    lights['living room'] = {on: true, b: 80};
    lights.kitchen = {on: true, b: 60};
    return {lights, temp: 20, target: 20, doors: {front: false, back: true, garage: true},
      blinds: {'living room': 'open', kitchen: 'open', bedroom: 'half', bathroom: 'closed', office: 'open'},
      music: null, musicRoom: 'living room', musicState: 'stopped', volume: 40,
      appliances: Object.fromEntries(APPLIANCES.map(a => [a, false])), security: 'disarmed', scene: '', note: 'all quiet'};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const rooms = r => { const k = normRoom(r); return k === 'all' ? ROOMS : k ? [k] : ROOMS; };
    const done = t => (s.note = t);
    switch (c.tool) {
      case 'set_lights': {
        const on = norm(a.state, {on: 'on', off: 'off'}) !== 'off';
        const b = a.brightness !== undefined && a.brightness !== null ? clamp(a.brightness, 1, 100) : null;
        for (const r of rooms(a.room)) { s.lights[r].on = on; if (b) s.lights[r].b = b; else if (on && s.lights[r].b < 10) s.lights[r].b = 70; }
        return done(`${label(a.room)} lights ${on ? 'on' : 'off'}${on && b ? ` at ${b}%` : ''}`);
      }
      case 'dim_lights': {
        const d0 = String(a.direction).toLowerCase(); const up = /^(up|bright|brighter|brighten|raise|more)$/.test(d0) || (/dark/.test(d0));
        for (const r of rooms(a.room)) { const L = s.lights[r]; if (!L.on) { L.on = true; L.b = up ? 60 : 30; } else L.b = clamp(L.b + (up ? 30 : -30), 10, 100); }
        return done(`${label(a.room)} lights ${up ? 'brighter' : 'dimmer'}`);
      }
      case 'set_thermostat': s.target = clampF(a.temperature, 10, 30); await warm(s, ctx); return done(`thermostat set to ${s.target}°`);
      case 'nudge_temperature': {
        const d = Number(a.degrees) > 0 ? Number(a.degrees) : 2;
        const up = !/cool|cold|down|low/i.test(String(a.direction));
        s.target = clampF(s.target + (up ? d : -d), 10, 30); await warm(s, ctx);
        return done(`${up ? 'warmer' : 'cooler'}: thermostat ${s.target}°`);
      }
      case 'set_door': {
        const lock = !/unlock|open/i.test(String(a.action));
        const k = String(a.door || 'front').toLowerCase();
        const ds = /all|both|every/.test(k) ? Object.keys(s.doors) : Object.keys(s.doors).filter(d => k.includes(d));
        for (const d of ds.length ? ds : ['front']) s.doors[d] = lock;
        return done(`${ds.length > 1 ? 'all doors' : `${ds[0] || 'front'} door`} ${lock ? 'locked' : 'unlocked'}`);
      }
      case 'set_blinds': {
        const p = /half/i.test(String(a.position)) ? 'half' : /clos|shut|down|lower/i.test(String(a.position)) ? 'closed' : 'open';
        for (const r of rooms(a.room)) s.blinds[r] = p;
        return done(`${label(a.room)} blinds ${p}`);
      }
      case 'play_music': s.music = a.query || 'music'; s.musicState = 'playing'; s.musicRoom = normRoom(a.room) && normRoom(a.room) !== 'all' ? normRoom(a.room) : s.musicRoom;
        return done(`playing ${s.music} in the ${s.musicRoom}`);
      case 'control_music': {
        if (!s.music) return done('nothing is playing');
        const act = String(a.action || '').toLowerCase();
        if (/pause|stop/.test(act)) s.musicState = 'paused';
        else if (/resume|play/.test(act)) s.musicState = 'playing';
        else if (/next|skip/.test(act)) s.musicState = 'playing';
        else if (/loud|up/.test(act)) s.volume = clamp(s.volume + 20, 0, 100);
        else if (/quiet|down|soft/.test(act)) s.volume = clamp(s.volume - 20, 0, 100);
        return done(`music: ${act}${/loud|quiet/.test(act) ? ` (volume ${s.volume}%)` : ''}`);
      }
      case 'switch_appliance': {
        const k = APPLIANCES.find(x => String(a.appliance || '').toLowerCase().includes(x.split(' ')[0])) || a.appliance;
        s.appliances[k] = !/off|stop/i.test(String(a.state));
        return done(`${k} ${s.appliances[k] ? 'on' : 'off'}`);
      }
      case 'run_scene': return done(scene(s, String(a.scene || '').toLowerCase()));
      case 'set_security': {
        const m = /disarm|off/i.test(String(a.mode)) ? 'disarmed' : /home|night|stay/i.test(String(a.mode)) ? 'home' : 'away';
        s.security = m; return done(m === 'disarmed' ? 'alarm disarmed' : `alarm armed (${m})`);
      }
      case 'check_status': return done(status(s, String(a.item || '')));
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    // plan: living room (big, left), kitchen (top middle), office (top right), bedroom (bottom right), bathroom (bottom middle)
    const R = {
      'living room': [30, 40, 220, 330],
      kitchen: [250, 40, 150, 150],
      office: [400, 40, 130, 150],
      bathroom: [250, 190, 120, 180],
      bedroom: [370, 190, 160, 180],
    };
    const defs = ROOMS.map((r, i) => `<radialGradient id="g${i}" cx="50%" cy="50%" r="65%"><stop offset="0" stop-color="#ffcf7a" stop-opacity="1"/><stop offset="1" stop-color="#ffb347" stop-opacity="0"/></radialGradient>`).join('');
    const rooms = ROOMS.map((r, i) => {
      const [x, y, w, h] = R[r], L = s.lights[r];
      const op = L.on ? (0.18 + 0.7 * L.b / 100).toFixed(2) : 0;
      return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#15161b" stroke="#3a3c46" stroke-width="3"/>
  <rect x="${x + 2}" y="${y + 2}" width="${w - 4}" height="${h - 4}" fill="url(#g${i})" opacity="${op}"/>
  <text x="${x + 10}" y="${y + 20}" fill="#e8e8ea" font-size="12" font-weight="600">${cap(r)}</text>
  <circle cx="${x + w - 16}" cy="${y + 16}" r="6" fill="${L.on ? '#ffd27a' : '#2c2d35'}" stroke="#555" stroke-width="1"/>
  <text x="${x + 10}" y="${y + 35}" fill="${L.on ? '#ffd9a0' : '#777'}" font-size="10" font-family="ui-monospace,monospace">${L.on ? `lights ${L.b}%` : 'lights off'}</text>
  ${blinds(x + w - 58, y + h - 18, s.blinds[r])}`;
    }).join('');
    // furniture hints
    const furn = `<rect x="60" y="250" width="110" height="40" rx="8" fill="#23242c"/><rect x="80" y="160" width="70" height="40" rx="4" fill="#1d1e24" stroke="#33343d"/>
  <rect x="262" y="150" width="70" height="16" rx="3" fill="#23242c"/><rect x="405" y="230" width="90" height="70" rx="6" fill="#23242c"/>
  <circle cx="300" cy="300" r="22" fill="#1d1e24" stroke="#33343d"/><rect x="420" y="80" width="80" height="30" rx="3" fill="#23242c"/>`;
    // doors
    const door = (x1, y1, x2, y2, locked, name, tx, ty, anchor = 'middle') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${locked ? '#e5484d' : '#46c37b'}" stroke-width="7" stroke-linecap="round"/>
  <text x="${tx}" y="${ty}" fill="${locked ? '#ff8a8d' : '#7fe0a6'}" font-size="10" text-anchor="${anchor}" font-family="ui-monospace,monospace">${name} ${locked ? '🔒 locked' : 'unlocked'}</text>`;
    const doors = door(100, 370, 160, 370, s.doors.front, 'front', 130, 388) +
      door(530, 250, 530, 310, s.doors.back, 'back', 524, 326, 'end') +
      door(30, 70, 30, 130, s.doors.garage, 'garage', 36, 146, 'start');
    // thermostat pill
    const heating = s.target > s.temp + 0.01;
    const thermo = `<rect x="165" y="60" width="72" height="44" rx="22" fill="#0e0f13" stroke="${heating ? '#ff8a4c' : s.target < s.temp - 0.01 ? '#5ab0ff' : '#444'}" stroke-width="2"/>
  <text x="201" y="82" fill="#fff" font-size="15" font-weight="700" text-anchor="middle">${fmt(s.temp)}°</text>
  <text x="201" y="96" fill="#aaa" font-size="9" text-anchor="middle">set ${fmt(s.target)}°</text>`;
    // music
    const [mx, my, mw] = R[s.musicRoom] || R['living room'];
    const music = s.music ? `<text x="${mx + mw / 2}" y="${my + 75}" fill="${s.musicState === 'playing' ? '#c9a6ff' : '#888'}" font-size="12" text-anchor="middle">${s.musicState === 'playing' ? '♪' : '❚❚'} ${esc(s.music)} · vol ${s.volume}%</text>` : '';
    // appliances
    const onA = Object.entries(s.appliances).filter(([, v]) => v).map(([k]) => k);
    const apps = onA.slice(0, 4).map((k, i) => `<rect x="${200 + i * 84}" y="378" width="80" height="18" rx="9" fill="#1f3a2b" stroke="#46c37b"/><text x="${240 + i * 84}" y="391" fill="#bff0d2" font-size="9" text-anchor="middle">${esc(k)} on</text>`).join('');
    const sec = s.security === 'disarmed' ? '#666' : '#e5484d';
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs>${defs}</defs>
  <rect width="560" height="460" fill="#0b0b0f"/>
  ${rooms}
  ${furn}
  ${apps}
  ${thermo}
  ${music}
  ${doors}
  <circle cx="40" cy="22" r="6" fill="${sec}"/><text x="52" y="26" fill="#ccc" font-size="11">alarm ${s.security === 'disarmed' ? 'off' : `armed · ${s.security}`}</text>
  ${s.scene ? `<text x="530" y="26" fill="#c9a6ff" font-size="11" text-anchor="end">scene · ${esc(s.scene)}</text>` : ''}
  <rect x="24" y="408" width="512" height="34" rx="17" fill="#15161b" stroke="#2a2b33"/>
  <text x="280" y="430" fill="#ddd" font-size="12" text-anchor="middle">${esc(s.note)}</text>
</svg>`;
  },

  examples: [
    {title: 'Dim + lock', sub: '2 calls', q: 'Dim the bedroom lights to 20% and lock the front door'},
    {title: 'Leaving home', sub: '3 calls, "everywhere" = all', q: 'Turn off the lights everywhere, lock all the doors and arm the alarm in away mode'},
    {title: 'Heat + jazz', sub: 'number grounding', q: 'Set the heating to 22 degrees and put on some jazz in the kitchen'},
    {title: 'Movie night', sub: 'one call, many changes', q: "It's movie night"},
    {title: 'Too bright', sub: 'relative, no number', q: 'The living room is too bright'},
    {title: 'Chilly', sub: 'warmth nudge', q: "I'm a bit cold, make it 3 degrees warmer"},
    {title: 'Back, not front', sub: 'polarity + exclusion', q: 'Unlock the back door, not the front one'},
    {title: 'Blinds only', sub: 'negation', q: "Close the bedroom blinds, don't touch the lights"},
    {title: 'Coffee + kettle', sub: 'appliances, 2 calls', q: 'Turn on the coffee machine and switch off the kettle'},
    {title: 'Volume', sub: 'controls what is playing', q: 'Turn the music up a bit'},
    {title: 'Status', sub: 'question about state', q: 'Is the back door locked?'},
    {title: 'Not a home thing', sub: 'abstains', q: 'Who won the football match yesterday?'},
  ],
  help: 'Try: "dim the bedroom and lock the front door", "it\'s movie night", "make it a bit warmer", "is the back door locked?".',
};

function scene(s, name) {
  const all = (on, b) => { for (const r of ROOMS) { s.lights[r].on = on; if (b) s.lights[r].b = b; } };
  if (/morning/.test(name)) { all(false); s.lights.kitchen = {on: true, b: 90}; for (const r of ROOMS) s.blinds[r] = 'open'; s.target = 21; s.appliances['coffee machine'] = true; s.security = 'disarmed'; s.scene = 'good morning'; return 'good morning: blinds up, kitchen lit, coffee on'; }
  if (/night|bed|sleep/.test(name) && !/movie/.test(name)) { all(false); s.lights.bedroom = {on: true, b: 15}; for (const d in s.doors) s.doors[d] = true; for (const r of ROOMS) s.blinds[r] = 'closed'; s.target = 18; s.security = 'home'; if (s.music) s.musicState = 'paused'; s.scene = 'good night'; return 'good night: doors locked, blinds down, 18°, alarm home'; }
  if (/movie|film|cinema/.test(name)) { all(false); s.lights['living room'] = {on: true, b: 15}; s.blinds['living room'] = 'closed'; s.appliances.tv = true; if (s.music) s.musicState = 'paused'; s.scene = 'movie night'; return 'movie night: lights low, blinds closed, tv on'; }
  if (/party|dinner/.test(name)) { all(true, 85); s.lights.kitchen.b = 100; s.music = s.music || 'party mix'; s.musicState = 'playing'; s.volume = 70; s.scene = 'dinner party'; return 'dinner party: lights up, music on'; }
  if (/away|leav|out/.test(name)) { all(false); for (const d in s.doors) s.doors[d] = true; s.security = 'away'; s.musicState = s.music ? 'paused' : s.musicState; for (const a in s.appliances) s.appliances[a] = false; s.target = 16; s.scene = 'away'; return 'away: everything off, doors locked, alarm armed'; }
  return `no scene called "${name}"`;
}

function status(s, item) {
  if (/temp|heat|therm|warm|cold/.test(item)) return `it is ${fmt(s.temp)}° inside, thermostat set to ${fmt(s.target)}°`;
  if (/light/.test(item)) { const on = ROOMS.filter(r => s.lights[r].on); return on.length ? `lights on in: ${on.join(', ')}` : 'all lights are off'; }
  if (/door|lock/.test(item)) return Object.entries(s.doors).map(([d, l]) => `${d} ${l ? 'locked' : 'unlocked'}`).join(', ');
  if (/blind|curtain/.test(item)) return ROOMS.map(r => `${r} ${s.blinds[r]}`).join(', ');
  if (/music/.test(item)) return s.music ? `${s.music} is ${s.musicState} in the ${s.musicRoom}` : 'no music playing';
  if (/alarm|secur/.test(item)) return s.security === 'disarmed' ? 'the alarm is off' : `the alarm is armed (${s.security})`;
  if (/appl/.test(item)) { const on = Object.keys(s.appliances).filter(k => s.appliances[k]); return on.length ? `running: ${on.join(', ')}` : 'no appliances running'; }
  return 'everything looks normal';
}

async function warm(s, ctx) {
  // animate the measured temperature creeping toward the target
  for (let i = 0; i < 6 && Math.abs(s.target - s.temp) > 0.05; i++) {
    s.temp = Math.round((s.temp + (s.target - s.temp) * 0.5) * 10) / 10;
    if (ctx && ctx.render) { ctx.render(); await ctx.sleep(120); }
  }
  s.temp = s.target;
}

function blinds(x, y, p) {
  const n = p === 'open' ? 2 : p === 'half' ? 5 : 9;
  const slats = Array.from({length: n}, (_, i) => `<line x1="${x}" y1="${y - 26 + i * 3}" x2="${x + 48}" y2="${y - 26 + i * 3}" stroke="#8a8fa3" stroke-width="1.6"/>`).join('');
  return `<rect x="${x - 2}" y="${y - 30}" width="52" height="30" rx="2" fill="#0f1016" stroke="#33343d"/>${slats}<text x="${x + 24}" y="${y + 10}" fill="#888" font-size="8" text-anchor="middle">blinds ${p}</text>`;
}

function normRoom(r) {
  const t = String(r || '').toLowerCase();
  if (!t) return '';
  if (/all|every|whole|house|flat|home/.test(t)) return 'all';
  if (/living|lounge|sitting/.test(t)) return 'living room';
  if (/kitchen/.test(t)) return 'kitchen';
  if (/bed/.test(t)) return 'bedroom';
  if (/bath|shower/.test(t)) return 'bathroom';
  if (/office|study/.test(t)) return 'office';
  return '';
}
const label = r => (normRoom(r) === 'all' ? 'all' : normRoom(r) || 'all');
function norm(v, map) { return map[String(v || '').toLowerCase()] || String(v || '').toLowerCase(); }
function clamp(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : lo; }
function clampF(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n * 2) / 2)) : 20; }
const fmt = n => (Number.isInteger(n) ? String(n) : n.toFixed(1));
const cap = t => t[0].toUpperCase() + t.slice(1);
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
