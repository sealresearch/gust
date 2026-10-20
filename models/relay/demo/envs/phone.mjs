// Phone environment: a handset whose screen reacts to Relay's calls.
// Contract (every env module): default export {key, label, subtitle, tagline, placeholder, tools, init(), render(s),
// apply(call, s, ctx) -> note string, examples[{title, sub, q}], help}. render returns an SVG/HTML string for a
// 560x460 stage; apply mutates s and may use ctx {render, sleep, speak, setAccent, openUrl, clipboard, toast}.

const APPS = ['photos', 'browser', 'settings', 'files', 'messages', 'phone', 'clock', 'camera', 'music', 'maps',
  'notes', 'calendar', 'weather', 'mail'];

const PHOTOS = [
  {tag: 'beach', label: 'beach'}, {tag: 'beach', label: 'beach'}, {tag: 'dog', label: 'dog'},
  {tag: 'last weekend', label: 'weekend'}, {tag: 'last weekend', label: 'weekend'}, {tag: 'screenshot', label: 'screenshot'},
  {tag: 'screenshot', label: 'screenshot'}, {tag: 'food', label: 'food'}, {tag: 'sunset', label: 'sunset'},
  {tag: 'birthday', label: 'birthday'}, {tag: 'dog', label: 'dog'}, {tag: 'receipt', label: 'receipt'},
];
const DOCS = ['Apartment lease 2026.pdf', 'Tax return 2025.pdf', 'Passport scan.jpg', 'Car insurance policy.pdf',
  'Wedding speech draft.docx', 'Gym membership contract.pdf', 'Recipe - lasagna.txt'];

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});

export default {
  key: 'phone',
  label: 'Phone',
  subtitle: 'an assistant that acts, not answers',
  tagline: 'Photos, browser, settings, messages, alarms and music on a phone',
  placeholder: 'ask the phone to do something...',
  tools: [
    tool('create_album', 'Create a new, empty photo album.', {name: str('Album name, e.g. "Summer 2026".')}, ['name']),
    tool('move_photos', 'Move existing photos into an album.', {
      album: str('The album to move them into.'),
      filter: str('Which photos, e.g. "beach", "dog", "last weekend", "screenshots".')}, ['album', 'filter']),
    tool('delete_photos', 'Delete photos that match a description.', {filter: str('Which photos, e.g. "screenshots".')}, ['filter']),
    tool('open_website', 'Open a website in the browser.', {url: str('Web address such as wikipedia.org.')}, ['url'],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(open|go to|visit)\\b[\\s\\S]*\\.(com|org|net|io|dev|app)\\b']}),
    tool('set_brightness', 'Set the screen brightness to a percentage.', {percent: {type: 'integer', description: 'Brightness percent, 1-100.'}}, ['percent']),
    tool('set_volume', 'Set the media volume to a percentage.', {percent: {type: 'integer', description: 'Volume percent, 0-100.'}}, ['percent']),
    tool('toggle_setting', 'Turn a phone setting on or off.', {
      setting: {type: 'string', enum: ['wifi', 'bluetooth', 'airplane mode', 'do not disturb', 'flashlight', 'dark mode'], description: 'The setting.'},
      state: {type: 'string', enum: ['on', 'off'], description: 'Turn it on or off.'}}, ['setting', 'state']),
    tool('set_accent_color', 'Change the theme accent color.', {color: str('A color name such as blue, green, purple, orange.')}, ['color']),
    tool('search_documents', 'Search the documents and files on the phone.', {query: str('What to look for, e.g. "lease".')}, ['query'],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(find|where is|look for|search)\\b[\\s\\S]*\\b(document|file|pdf|lease|contract|policy|passport|scan)\\b']}),
    tool('read_aloud', 'Say a text out loud with the phone speaker.', {text: str('The exact words to say.')}, ['text']),
    tool('send_message', 'Send a text message to a contact.', {recipient: str('Contact name.'), text: str('The message text.')}, ['recipient', 'text']),
    tool('call_contact', 'Start a phone call with a contact.', {name: str('Contact name.')}, ['name']),
    tool('set_alarm', 'Set an alarm.', {time: str('Alarm time, e.g. 6:45 or 7am.'), label: str('Optional alarm label.')}, ['time']),
    tool('set_timer', 'Start a countdown timer.', {duration: str('How long, e.g. "10 minutes".'), label: str('Optional label.')}, ['duration']),
    tool('play_music', 'Play music: a song, artist, album or genre.', {query: str('What to play.')}, ['query']),
    tool('take_photo', 'Take a photo with the camera.', {camera: {type: 'string', enum: ['back', 'front'], description: 'Which camera; front for a selfie.'}}),
    tool('open_app', 'Open an app on the phone.', {name: {type: 'string', enum: APPS, description: 'The app.'}}, ['name']),
    tool('show_map', 'Show a place on the map.', {query: str('Place, address or business.')}, ['query']),
  ],

  init() {
    return {app: 'home', brightness: 70, volume: 50, accent: '#ff4d4f', albums: {'Camera roll': 12}, moved: {}, deleted: {},
      url: '', query: '', found: [], spoken: '', message: null, calling: '', alarm: null, timer: null, music: '',
      map: '', toggles: {wifi: true, bluetooth: true, 'airplane mode': false, 'do not disturb': false, flashlight: false,
        'dark mode': true}, flash: false, toast: ''};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const go = app => { s.app = app; };
    switch (c.tool) {
      case 'create_album': s.albums[a.name] = s.albums[a.name] || 0; go('photos'); return `album "${a.name}" created`;
      case 'move_photos': {
        const f = String(a.filter || '').toLowerCase();
        let n = 0;
        PHOTOS.forEach((p, i) => { if (!s.deleted[i] && (p.tag.includes(f) || f.includes(p.tag) || f.includes(p.label))) { s.moved[i] = a.album; n++; } });
        s.albums[a.album] = (s.albums[a.album] || 0) + n; go('photos');
        return `${n} photo${n === 1 ? '' : 's'} moved to "${a.album}"`;
      }
      case 'delete_photos': {
        const f = String(a.filter || '').toLowerCase();
        let n = 0;
        PHOTOS.forEach((p, i) => { if (p.tag.includes(f) || f.includes(p.tag) || f.includes(p.label)) { s.deleted[i] = true; n++; } });
        go('photos'); return `${n} photo${n === 1 ? '' : 's'} deleted`;
      }
      case 'open_website': {
        let u = String(a.url || '').trim();
        if (!/\./.test(u)) u += '.com';
        s.url = u.replace(/^https?:\/\//, ''); go('browser'); return `opened ${s.url}`;
      }
      case 'set_brightness': s.brightness = clamp(a.percent, 1, 100); go('settings'); return `brightness ${s.brightness}%`;
      case 'set_volume': s.volume = clamp(a.percent, 0, 100); go('settings'); return `volume ${s.volume}%`;
      case 'toggle_setting': if (a.setting in s.toggles) s.toggles[a.setting] = a.state === 'on'; go('settings');
        return `${a.setting} ${a.state}`;
      case 'set_accent_color': s.accent = a.color; ctx.setAccent && ctx.setAccent(a.color); go('settings'); return `accent ${a.color}`;
      case 'search_documents': {
        const q = String(a.query || '').toLowerCase().split(/\W+/).filter(w => w.length > 2);
        s.query = a.query; s.found = DOCS.filter(d => q.some(w => d.toLowerCase().includes(w))); go('files');
        return s.found.length ? `found ${s.found[0]}` : 'no matching document';
      }
      case 'read_aloud': s.spoken = a.text; ctx.speak && ctx.speak(a.text); go('settings'); return `said "${a.text}"`;
      case 'send_message': s.message = {to: a.recipient, text: a.text}; go('messages'); return `message to ${a.recipient}`;
      case 'call_contact': s.calling = a.name; go('phone'); return `calling ${a.name} (nothing is dialled)`;
      case 'set_alarm': s.alarm = {time: a.time, label: a.label || ''}; go('clock'); return `alarm ${a.time}`;
      case 'set_timer': s.timer = {duration: a.duration, label: a.label || ''}; go('clock'); return `timer ${a.duration}`;
      case 'play_music': s.music = a.query; go('music'); return `playing ${a.query}`;
      case 'take_photo': go('camera'); s.flash = true; ctx.render(); await ctx.sleep(250); s.flash = false;
        return `photo taken (${a.camera || 'back'} camera)`;
      case 'open_app': go(a.name); return `opened ${a.name}`;
      case 'show_map': s.map = a.query; go('maps'); return `map: ${a.query}`;
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    const acc = s.accent;
    const dim = (1 - s.brightness / 100) * 0.75;
    const x0 = 180, y0 = 18, W = 200, H = 420;
    const scr = screen(s, x0 + 10, y0 + 40, W - 20, H - 70, acc);
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs><radialGradient id="pg" cx="50%" cy="30%" r="70%"><stop offset="0" stop-color="${acc}" stop-opacity=".35"/><stop offset="1" stop-color="#0b0b0f" stop-opacity="0"/></radialGradient></defs>
  <rect x="${x0}" y="${y0}" width="${W}" height="${H}" rx="30" fill="#111" stroke="#2a2a30" stroke-width="2"/>
  <rect x="${x0 + 8}" y="${y0 + 8}" width="${W - 16}" height="${H - 16}" rx="24" fill="#0b0b0f"/>
  <rect x="${x0 + 8}" y="${y0 + 8}" width="${W - 16}" height="${H - 16}" rx="24" fill="url(#pg)"/>
  <rect x="${x0 + 70}" y="${y0 + 14}" width="60" height="14" rx="7" fill="#000"/>
  <text x="${x0 + 26}" y="${y0 + 26}" fill="#ddd" font-size="10" font-weight="600">9:41</text>
  <text x="${x0 + W - 28}" y="${y0 + 26}" fill="#ddd" font-size="9" text-anchor="end">${s.toggles['airplane mode'] ? '✈' : '▮▮▮'} ${s.toggles.wifi ? 'wifi' : ''}</text>
  ${scr}
  ${s.toggles.flashlight ? `<circle cx="${x0 + W - 30}" cy="${y0 + 50}" r="7" fill="#fff8c4"/>` : ''}
  <rect x="${x0 + 8}" y="${y0 + 8}" width="${W - 16}" height="${H - 16}" rx="24" fill="#000" opacity="${dim.toFixed(2)}" pointer-events="none"/>
  ${s.flash ? `<rect x="${x0 + 8}" y="${y0 + 8}" width="${W - 16}" height="${H - 16}" rx="24" fill="#fff"/>` : ''}
  <rect x="${x0 + 70}" y="${y0 + H - 18}" width="60" height="4" rx="2" fill="#666"/>
  <text x="24" y="448" fill="#777" font-size="11" font-family="ui-monospace,monospace">brightness ${s.brightness}%  volume ${s.volume}%</text>
  <text x="536" y="448" fill="#777" font-size="11" text-anchor="end" font-family="ui-monospace,monospace">app · ${s.app}</text>
</svg>`;
  },

  examples: [
    {title: 'Album', sub: '2-step chain, ref to call 1', q: 'Make an album called Summer and move the beach photos into it'},
    {title: 'Open a site', sub: '1 call', q: 'Open wikipedia.org'},
    {title: 'Dim the screen', sub: 'number grounding', q: 'Turn the brightness down to 30 percent'},
    {title: 'Find the lease', sub: 'trigger routes to search', q: 'Where is my apartment lease?'},
    {title: 'Speak + theme', sub: 'real side effects', q: 'Read out loud: good morning. Then make the accent color blue'},
    {title: 'Delete', sub: 'only what was asked', q: 'Delete the screenshots'},
    {title: 'Text + alarm', sub: '2 calls', q: "Text Maya that I'm running late and set an alarm for 6:45"},
    {title: 'Call + music', sub: '2 calls', q: 'Call mom and play some jazz'},
    {title: 'Timer + setting', sub: '2 calls', q: 'Start a 25 minute timer and turn on do not disturb'},
    {title: 'Front camera', sub: 'enum option', q: 'Take a photo with the front camera'},
    {title: 'Map', sub: 'span copy', q: 'Show me the Eiffel Tower on the map'},
    {title: 'Not a phone thing', sub: 'abstains', q: 'What is the capital of Australia?'},
  ],
  help: 'Try: "make an album called Summer and put the beach photos in it", "dim the screen to 30%", "text Maya I\'m late".',
};

function clamp(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : lo; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
function wrap(t, n) { const w = String(t).split(/\s+/); const out = ['']; for (const x of w) { if ((out.at(-1) + ' ' + x).trim().length > n) out.push(x); else out[out.length - 1] = (out.at(-1) + ' ' + x).trim(); } return out; }

function screen(s, x, y, w, h, acc) {
  const T = (tx, ty, t, sz = 11, fill = '#eee', extra = '') => `<text x="${tx}" y="${ty}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;
  const title = t => T(x + 10, y + 22, t, 15, '#fff', 'font-weight="700"');
  switch (s.app) {
    case 'home': {
      const icons = ['Photos', 'Settings', 'Files', 'Clock', 'Music', 'Maps', 'Notes', 'Calendar', 'Mail', 'Messages', 'Weather', 'Camera'];
      return T(x + 14, y + 26, 'Saturday 17 October', 10, '#bbb') + T(x + 14, y + 50, '14° and clear', 18, '#fff', 'font-weight="600"') +
        icons.map((n, i) => { const cx = x + 22 + (i % 4) * 45, cy = y + 88 + Math.floor(i / 4) * 62;
          return `<rect x="${cx - 16}" y="${cy - 16}" width="32" height="32" rx="9" fill="${i % 3 === 0 ? acc : '#2a2a33'}" opacity="${i % 3 === 0 ? 0.85 : 1}"/>` + T(cx, cy + 30, n, 8, '#bbb', 'text-anchor="middle"'); }).join('') +
        T(x + w / 2, y + h - 18, 'say something to the phone', 9, '#888', 'text-anchor="middle"');
    }
    case 'photos': {
      const albums = Object.entries(s.albums).map(([n, c], i) => `<rect x="${x + 8 + i * 62}" y="${y + 34}" width="58" height="20" rx="10" fill="${n === 'Camera roll' ? '#2a2a33' : acc}"/>` + T(x + 37 + i * 62, y + 48, `${n} · ${c}`, 8, '#fff', 'text-anchor="middle"')).slice(0, 3).join('');
      const tiles = PHOTOS.map((p, i) => s.deleted[i] ? '' : `<rect x="${x + 8 + (i % 3) * 60}" y="${y + 64 + Math.floor(i / 3) * 62}" width="56" height="56" rx="6" fill="#23232b" stroke="${s.moved[i] ? acc : 'none'}" stroke-width="2"/>` + T(x + 36 + (i % 3) * 60, y + 96 + Math.floor(i / 3) * 62, p.label, 8, '#aaa', 'text-anchor="middle"')).join('');
      return title('Photos') + albums + tiles;
    }
    case 'browser': return `<rect x="${x + 8}" y="${y + 8}" width="${w - 16}" height="24" rx="12" fill="#22222a"/>` + T(x + 20, y + 24, s.url, 10, '#ddd') +
      T(x + 12, y + 60, s.url.split('.')[0].replace(/^www$/, ''), 20, '#fff', 'font-weight="700"') +
      [0, 1, 2, 3, 4, 5].map(i => `<rect x="${x + 12}" y="${y + 80 + i * 18}" width="${w - 30 - (i % 3) * 25}" height="8" rx="4" fill="#2a2a33"/>`).join('');
    case 'settings': {
      const bar = (ty, label, v) => T(x + 12, ty, `${label} ${v}%`, 10, '#ccc') + `<rect x="${x + 12}" y="${ty + 6}" width="${w - 24}" height="6" rx="3" fill="#2a2a33"/><rect x="${x + 12}" y="${ty + 6}" width="${(w - 24) * v / 100}" height="6" rx="3" fill="${acc}"/>`;
      const tg = Object.entries(s.toggles).map(([k, v], i) => T(x + 12, y + 128 + i * 24, k, 10, '#ddd') + `<rect x="${x + w - 44}" y="${y + 118 + i * 24}" width="30" height="14" rx="7" fill="${v ? acc : '#3a3a44'}"/><circle cx="${v ? x + w - 21 : x + w - 37}" cy="${y + 125 + i * 24}" r="5" fill="#fff"/>`).join('');
      return title('Settings') + bar(y + 50, 'brightness', s.brightness) + bar(y + 80, 'volume', s.volume) + T(x + 12, y + 110, `accent ${s.accent}`, 9, acc) + tg +
        (s.spoken ? wrap(`🔊 "${s.spoken}"`, 30).map((l, i) => T(x + 12, y + h - 30 + i * 12, l, 9, '#bbb')).join('') : '');
    }
    case 'files': return title('Files') + `<rect x="${x + 8}" y="${y + 32}" width="${w - 16}" height="22" rx="11" fill="#22222a"/>` + T(x + 18, y + 47, `🔎 ${s.query}`, 10, '#ddd') +
      (s.found.length ? s.found : DOCS).map((d, i) => `<rect x="${x + 8}" y="${y + 64 + i * 30}" width="${w - 16}" height="26" rx="6" fill="#1c1c22" stroke="${s.found.length && i === 0 ? acc : 'none'}"/>` + T(x + 16, y + 81 + i * 30, d, 9, '#ddd')).join('');
    case 'messages': return title(s.message ? s.message.to : 'Messages') + (s.message ? wrap(s.message.text, 26).map((l, i, a) => (i === 0 ? `<rect x="${x + w - 16 - 150}" y="${y + 50}" width="150" height="${a.length * 14 + 12}" rx="12" fill="${acc}"/>` : '') + T(x + w - 158, y + 66 + i * 14, l, 10, '#fff')).join('') + T(x + w - 16, y + 80 + wrap(s.message.text, 26).length * 14, 'delivered', 8, '#888', 'text-anchor="end"') : '');
    case 'phone': return T(x + w / 2, y + 90, s.calling ? s.calling[0].toUpperCase() : '', 40, '#fff', 'text-anchor="middle" font-weight="700"') + T(x + w / 2, y + 140, s.calling, 16, '#fff', 'text-anchor="middle"') + T(x + w / 2, y + 160, 'calling… (sandbox: nothing is dialled)', 8, '#999', 'text-anchor="middle"') + `<circle cx="${x + w / 2}" cy="${y + h - 50}" r="20" fill="#e5484d"/>`;
    case 'clock': return title('Clock') + (s.alarm ? `<rect x="${x + 8}" y="${y + 36}" width="${w - 16}" height="60" rx="10" fill="#1c1c22"/>` + T(x + 18, y + 72, s.alarm.time, 24, '#fff', 'font-weight="700"') + T(x + 18, y + 88, `alarm ${s.alarm.label}`, 9, acc) : '') +
      (s.timer ? `<rect x="${x + 8}" y="${y + 106}" width="${w - 16}" height="60" rx="10" fill="#1c1c22"/>` + T(x + 18, y + 142, s.timer.duration, 20, '#fff', 'font-weight="700"') + T(x + 18, y + 158, `timer running ${s.timer.label}`, 9, acc) : '');
    case 'camera': return `<rect x="${x + 4}" y="${y + 4}" width="${w - 8}" height="${h - 70}" rx="8" fill="#15151a"/>` + T(x + w / 2, y + h / 2 - 20, 'viewfinder', 10, '#555', 'text-anchor="middle"') + `<circle cx="${x + w / 2}" cy="${y + h - 32}" r="20" fill="none" stroke="#fff" stroke-width="3"/><circle cx="${x + w / 2}" cy="${y + h - 32}" r="15" fill="#fff"/>`;
    case 'music': return `<rect x="${x + 30}" y="${y + 30}" width="${w - 60}" height="${w - 60}" rx="12" fill="${acc}" opacity=".8"/>` + T(x + w / 2, y + w + 0, s.music || 'nothing playing', 12, '#fff', 'text-anchor="middle" font-weight="600"') + T(x + w / 2, y + w + 18, s.music ? 'now playing' : '', 9, '#aaa', 'text-anchor="middle"') + T(x + w / 2, y + w + 50, '⏮   ⏸   ⏭', 16, '#fff', 'text-anchor="middle"');
    case 'maps': return `<rect x="${x + 4}" y="${y + 4}" width="${w - 8}" height="${h - 8}" rx="8" fill="#16201a"/>` + [0, 1, 2, 3, 4].map(i => `<line x1="${x}" y1="${y + 40 + i * 60}" x2="${x + w}" y2="${y + 20 + i * 60}" stroke="#25352b" stroke-width="6"/>`).join('') + `<circle cx="${x + w / 2}" cy="${y + h / 2}" r="8" fill="${acc}"/>` + `<rect x="${x + 10}" y="${y + h - 50}" width="${w - 20}" height="34" rx="8" fill="#0b0b0f" opacity=".85"/>` + T(x + 18, y + h - 29, s.map, 10, '#fff');
    default: return title(s.app[0].toUpperCase() + s.app.slice(1)) + T(x + 12, y + 50, `${s.app} is open`, 10, '#aaa');
  }
}
