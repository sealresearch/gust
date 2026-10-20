// Laptop environment: a desktop (menu bar, dock, cascading windows) driven by Relay's calls, with REAL side effects
// through ctx: open_website -> ctx.openUrl (new tab), send_email -> ctx.mailto (mail app draft), copy_to_clipboard ->
// ctx.clipboard, speak -> ctx.speak, set_timer -> a live countdown (ctx.render every second, spoken when done).
// Same contract as phone.mjs; render(s) is a pure SVG string (560x460).

const ACC = '#8b7bff';
const APPS = ['finder', 'browser', 'notes', 'calendar', 'mail', 'music', 'timer', 'terminal', 'settings'];
const ICON = {finder: ['#3d8bfd', '☺'], browser: ['#1fb6ff', '◎'], notes: ['#f5c542', '✎'], calendar: ['#ff5d5d', '17'],
  mail: ['#4f8cff', '✉'], music: ['#ff4f8b', '♫'], timer: ['#ff9f43', '◷'], terminal: ['#2b2f3a', '>_'], settings: ['#8d96a8', '⚙']};
const FILES = ['Budget 2026.xlsx', 'Q3 report.pdf', 'Trip itinerary - Lisbon.pdf', 'Invoice 0142.pdf', 'Team photo.jpg',
  'Resume.docx', 'Meeting notes - Oct 12.txt', 'Logo final.png'];
const WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
const TODAY = 17;  // Saturday 17 October 2026 (Oct 1 is a Thursday)

let tick = null;   // live timer interval (one per page)

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});

export default {
  key: 'laptop',
  label: 'Laptop',
  subtitle: 'plain words, real actions',
  tagline: 'Windows, files, notes, timers and mail on a desktop, with real tabs, mail drafts, clipboard and speech',
  placeholder: 'ask the laptop to do something...',
  tools: [
    tool('open_website', 'Open a website in a new browser tab.', {url: str('Web address, e.g. github.com.')}, ['url'],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(open|go to|visit|load)\\b[\\s\\S]*\\.(com|org|net|io|dev|app|ai|edu)\\b']}),
    tool('send_email', 'Write an email draft in the mail app.', {
      to: str('Recipient name or email address.'), subject: str('Subject line.'), body: str('The message text.')}, ['to']),
    tool('create_note', 'Write a new note.', {title: str("The note's name, e.g. Groceries."), body: str("The note's content.")}),
    tool('create_event', 'Add an event to the calendar.', {
      title: str('What the event is.'), date: str('Day, e.g. Friday or October 20.'), time: str('Start time, e.g. 3pm.')}, ['title']),
    tool('set_timer', 'Start a countdown timer.', {
      minutes: {type: 'number', description: 'Minutes.'}, seconds: {type: 'integer', description: 'Seconds.'},
      label: str('What the timer is for, e.g. tea.')}),
    tool('control_timer', 'Pause, resume or cancel the running timer.', {action: en(['pause', 'resume', 'cancel'], 'pause, resume or cancel.')}, ['action']),
    tool('copy_to_clipboard', 'Copy text to the clipboard.', {text: str('The exact text to copy.')}, ['text']),
    tool('speak', 'Say text out loud.', {text: str('The exact words to say.')}, ['text']),
    tool('open_app', 'Open an app.', {app: en(APPS, 'The app: finder (files), browser, notes, calendar, mail, music, timer, terminal or settings.')}, ['app']),
    tool('close_app', 'Close an app window.', {app: en([...APPS, 'all'], 'The app to close, or all.')}, ['app']),
    tool('play_music', 'Play music: a song, artist, playlist or genre.', {query: str('What to play.')}, ['query']),
    tool('set_volume', 'Set the sound volume.', {percent: {type: 'integer', description: 'Volume percent, 0-100.'}}, ['percent']),
    tool('toggle_setting', 'Turn a system setting on or off.', {
      setting: en(['do not disturb', 'wifi', 'bluetooth', 'dark mode', 'mute'], 'The setting.'),
      state: en(['on', 'off'], 'on or off.')}, ['setting', 'state']),
    tool('search_files', 'Search files on the laptop.', {query: str('What to look for, e.g. "invoice".')}, ['query']),
    tool('lock_screen', 'Lock the screen.'),
  ],

  init() {
    if (tick) { clearInterval(tick); tick = null; }
    return {windows: [], volume: 60, toggles: {'do not disturb': false, wifi: true, bluetooth: true, 'dark mode': true, mute: false},
      timer: null, music: '', playing: false, notes: [], events: [{title: 'Standup', day: 19, time: '9:30'}], mail: null, url: '',
      clip: '', spoken: '', found: null, fquery: '', locked: false, toast: '', toastAt: 0};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    if (s.locked && c.tool !== 'lock_screen') s.locked = false;
    const toast = t => { s.toast = t; ctx.toast && ctx.toast(t); };
    switch (c.tool) {
      case 'open_website': {
        let u = String(a.url || '').trim().replace(/\s+/g, '');
        if (!u) return 'no address given';
        if (!/\./.test(u)) u += '.com';
        s.url = u.replace(/^https?:\/\//, '');
        front(s, 'browser');
        ctx.openUrl && ctx.openUrl(/^https?:\/\//.test(u) ? u : `https://${u}`);
        toast(`opened ${s.url} in a new tab`); return `opened ${s.url} in a real tab`;
      }
      case 'send_email': {
        const m = {to: String(a.to || ''), subject: String(a.subject || ''), body: String(a.body || '')};
        s.mail = m; front(s, 'mail');
        const q = [m.subject && `subject=${encodeURIComponent(m.subject)}`, m.body && `body=${encodeURIComponent(m.body)}`].filter(Boolean).join('&');
        const href = `mailto:${encodeURIComponent(m.to).replace(/%40/g, '@')}${q ? '?' + q : ''}`;
        ctx.mailto && ctx.mailto(href);
        toast('draft opened in your mail app'); return `email draft to ${m.to} opened in the mail app (nothing is sent)`;
      }
      case 'create_note': {
        const n = {title: String(a.title || '').trim() || firstWords(a.body), body: String(a.body || '')};
        s.notes.unshift(n); front(s, 'notes'); return `note "${n.title}" saved`;
      }
      case 'create_event': {
        const e = {title: String(a.title || 'Event'), day: parseDay(a.date), date: String(a.date || ''), time: String(a.time || '')};
        s.events.push(e); front(s, 'calendar');
        return `event "${e.title}"${e.date ? ' on ' + e.date : ''}${e.time ? ' at ' + e.time : ''}`;
      }
      case 'set_timer': {
        const mins = Number(a.minutes) || 0, secs = Number(a.seconds) || 0;
        const total = Math.round(mins * 60 + secs);
        if (!(total > 0)) return 'how long should the timer run?';
        const label = String(a.label || '').trim();
        s.timer = {total, left: total, label, state: 'running'}; front(s, 'timer');
        startTick(s, ctx);
        return `timer ${fmt(total)}${label ? ' for ' + label : ''} running`;
      }
      case 'control_timer': {
        if (!s.timer) return 'no timer is running';
        const x = String(a.action || '').toLowerCase();
        if (/pause|hold/.test(x)) s.timer.state = 'paused';
        else if (/resume|continue|start/.test(x)) { if (s.timer.state !== 'done') { s.timer.state = 'running'; startTick(s, ctx); } }
        else { s.timer.state = 'cancelled'; if (tick) { clearInterval(tick); tick = null; } }
        front(s, 'timer'); return `timer ${s.timer.state}`;
      }
      case 'copy_to_clipboard': {
        s.clip = String(a.text || '');
        let ok = true;
        try { if (ctx.clipboard) await ctx.clipboard(s.clip); } catch { ok = false; }
        toast(ok ? `copied "${short(s.clip, 28)}"` : 'clipboard blocked by the browser');
        return ok ? `copied "${s.clip}" to the real clipboard` : 'clipboard write was blocked';
      }
      case 'speak': s.spoken = String(a.text || ''); ctx.speak && ctx.speak(s.spoken); toast(`🔊 "${short(s.spoken, 30)}"`); return `said "${s.spoken}"`;
      case 'open_app': { const app = pickApp(a.app); if (!app) return `unknown app ${a.app}`; front(s, app); return `opened ${app}`; }
      case 'close_app': {
        const x = String(a.app || '').toLowerCase();
        if (/all|every/.test(x)) { const n = s.windows.length; s.windows = []; return `closed ${n} window${n === 1 ? '' : 's'}`; }
        const app = pickApp(x); const had = s.windows.some(w => w === app);
        s.windows = s.windows.filter(w => w !== app); return had ? `closed ${app}` : `${app || x} was not open`;
      }
      case 'play_music': s.music = String(a.query || 'something'); s.playing = true; front(s, 'music'); return `playing ${s.music}`;
      case 'set_volume': s.volume = clamp(a.percent, 0, 100); s.toggles.mute = s.volume === 0; toast(`volume ${s.volume}%`); return `volume ${s.volume}%`;
      case 'toggle_setting': {
        const k = Object.keys(s.toggles).find(t => t === String(a.setting).toLowerCase()) || (/disturb|dnd|focus/.test(String(a.setting)) ? 'do not disturb' : '');
        if (!k) return `unknown setting ${a.setting}`;
        s.toggles[k] = !/off|disable/.test(String(a.state || 'on').toLowerCase()); toast(`${k} ${s.toggles[k] ? 'on' : 'off'}`);
        return `${k} ${s.toggles[k] ? 'on' : 'off'}`;
      }
      case 'search_files': {
        const q = String(a.query || '').toLowerCase();
        const words = q.split(/[^a-z0-9]+/).filter(w => w.length > 2 && !['the', 'file', 'files', 'my', 'and'].includes(w));
        s.fquery = a.query || ''; s.found = FILES.filter(f => words.some(w => f.toLowerCase().includes(w) || (w === 'spreadsheet' && /xlsx/.test(f)) || (w === 'photo' && /jpg|png/.test(f))));
        front(s, 'finder'); return s.found.length ? `found ${s.found.join(', ')}` : 'no matching files';
      }
      case 'lock_screen': s.locked = true; return 'screen locked (any next request unlocks it)';
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    const top = s.windows.at(-1);
    const sx = 46, sy = 18, sw = 468, sh = 352;
    const wins = s.windows.map((w, i) => win(s, w, sx + 26 + i * 44, sy + 40 + i * 30, 300, 210, i === s.windows.length - 1)).join('');
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs>
    <linearGradient id="lwall" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2a1f5c"/><stop offset=".5" stop-color="#3b2a7a"/><stop offset="1" stop-color="#d0628a"/></linearGradient>
    <radialGradient id="lglow" cx="75%" cy="20%" r="60%"><stop offset="0" stop-color="#ffb37a" stop-opacity=".45"/><stop offset="1" stop-color="#ffb37a" stop-opacity="0"/></radialGradient>
    <linearGradient id="lbase" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3a3d46"/><stop offset="1" stop-color="#1d1f25"/></linearGradient>
    <clipPath id="lscr"><rect x="${sx}" y="${sy}" width="${sw}" height="${sh}" rx="6"/></clipPath>
  </defs>
  <rect width="560" height="460" fill="#0b0b0f"/>
  <rect x="${sx - 10}" y="${sy - 10}" width="${sw + 20}" height="${sh + 20}" rx="16" fill="#16171c" stroke="#2c2e36" stroke-width="2"/>
  <g clip-path="url(#lscr)">
    <rect x="${sx}" y="${sy}" width="${sw}" height="${sh}" fill="url(#lwall)"/><rect x="${sx}" y="${sy}" width="${sw}" height="${sh}" fill="url(#lglow)"/>
    <path d="M${sx} ${sy + sh - 90} C ${sx + 140} ${sy + sh - 150} ${sx + 300} ${sy + sh - 40} ${sx + sw} ${sy + sh - 120} L${sx + sw} ${sy + sh} L${sx} ${sy + sh} Z" fill="#1b1440" opacity=".55"/>
    ${menuBar(s, sx, sy, sw, top)}
    ${wins}
    ${dock(s, sx, sy, sw, sh)}
    ${s.toast ? toastPill(s.toast, sx + sw - 8, sy + 26) : ''}
    ${s.locked ? lockScreen(sx, sy, sw, sh) : ''}
  </g>
  <path d="M10 ${sy + sh + 12} L550 ${sy + sh + 12} L540 ${sy + sh + 40} Q 280 ${sy + sh + 50} 20 ${sy + sh + 40} Z" fill="url(#lbase)"/>
  <rect x="236" y="${sy + sh + 12}" width="88" height="6" rx="3" fill="#0f1014"/>
  <text x="24" y="448" fill="#777" font-size="11" font-family="ui-monospace,monospace">volume ${s.toggles.mute ? 'muted' : s.volume + '%'}${s.timer ? '  timer ' + fmt(s.timer.left) + ' ' + s.timer.state : ''}</text>
  <text x="536" y="448" fill="#777" font-size="11" text-anchor="end" font-family="ui-monospace,monospace">${top ? 'front · ' + top : 'desktop'}</text>
</svg>`;
  },

  examples: [
    {title: 'Draft an email', sub: 'real side effect · mail app', q: 'Email sam@example.com with the subject Lunch and say see you at noon'},
    {title: 'Focus timer', sub: 'live countdown + speech', q: 'Start a 25 minute timer for focus'},
    {title: 'Copy it', sub: 'real clipboard', q: 'Copy 221B Baker Street to the clipboard'},
    {title: 'Open a tab', sub: 'real new tab', q: 'Open github.com'},
    {title: 'Grocery note', sub: 'title + body spans', q: 'Make a note called Groceries: eggs, milk and bread'},
    {title: 'Not that app', sub: 'negation', q: 'Open Notes, not Mail'},
    {title: 'Desk setup', sub: '2 calls', q: 'Open the calendar and play some lo-fi'},
    {title: 'Say it', sub: 'quoted text verbatim', q: 'Say "the build is green" out loud'},
    {title: 'Focus mode', sub: '2 calls', q: 'Turn on do not disturb and find the budget spreadsheet'},
    {title: 'Hold on', sub: 'pause, not cancel', q: 'Pause the timer'},
    {title: 'Appointment', sub: 'date + time', q: 'Add a dentist appointment on Friday at 3pm'},
    {title: 'Science question', sub: 'abstains', q: 'Why is the sky blue?'},
  ],
  help: 'Try: "start a 25 minute timer", "email sam@example.com about lunch", "copy 221B Baker Street to the clipboard".',
};

// ---------------------------------------------------------------- helpers

function clamp(v, lo, hi) { const n = Number(v); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : lo; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const short = (t, n) => { const x = String(t ?? ''); return x.length > n ? x.slice(0, n - 1) + '…' : x; };
const fmt = sec => `${String(Math.floor(Math.max(0, sec) / 60)).padStart(2, '0')}:${String(Math.max(0, sec) % 60).padStart(2, '0')}`;
const firstWords = t => String(t || 'Untitled').split(/\s+/).slice(0, 3).join(' ');
function wrap(t, n) { const out = ['']; for (const x of String(t).split(/\s+/)) { if ((out.at(-1) + ' ' + x).trim().length > n) out.push(x); else out[out.length - 1] = (out.at(-1) + ' ' + x).trim(); } return out; }
const T = (x, y, t, sz = 10, fill = '#e9e9f2', extra = '') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;
function front(s, app) { s.windows = [...s.windows.filter(w => w !== app), app].slice(-3); }
function pickApp(v) {
  const x = String(v || '').toLowerCase();
  if (APPS.includes(x)) return x;
  const al = [[/finder|files?|folder/, 'finder'], [/browser|safari|chrome|web/, 'browser'], [/note/, 'notes'], [/calendar|agenda/, 'calendar'],
    [/mail|email|inbox/, 'mail'], [/music|spotify|song/, 'music'], [/timer|clock/, 'timer'], [/terminal|shell|console/, 'terminal'], [/setting|preference/, 'settings']];
  const hit = al.find(([rx]) => rx.test(x)); return hit ? hit[1] : '';
}
function parseDay(d) {
  const x = String(d || '').toLowerCase();
  if (!x) return 0;
  if (/today|tonight/.test(x)) return TODAY;
  if (/tomorrow/.test(x)) return TODAY + 1;
  const iso = x.match(/\d{4}-(\d{2})-(\d{2})/); if (iso) return +iso[1] === 10 ? +iso[2] : 0;
  const wd = WEEKDAYS.findIndex(w => x.includes(w));
  if (wd >= 0) { const diff = ((wd - 6 + 7) % 7) || 7; return TODAY + diff; }
  const n = x.match(/\b(\d{1,2})(st|nd|rd|th)?\b/); return n && +n[1] <= 31 ? +n[1] : 0;
}
function startTick(s, ctx) {
  if (tick) clearInterval(tick);
  if (!ctx.render) return;
  tick = setInterval(() => {
    const t = s.timer;
    if (!t || t.state !== 'running') return;
    t.left -= 1;
    if (t.left <= 0) {
      t.left = 0; t.state = 'done'; clearInterval(tick); tick = null;
      const msg = `${t.label ? t.label[0].toUpperCase() + t.label.slice(1) + ' timer' : 'Your timer'} is done`;
      s.toast = msg; ctx.speak && ctx.speak(msg); ctx.toast && ctx.toast(msg);
    }
    ctx.render();
  }, 1000);
  if (tick && tick.unref) tick.unref();
}

function menuBar(s, x, y, w, top) {
  const name = top ? top[0].toUpperCase() + top.slice(1) : 'Finder';
  const right = [s.playing && s.music ? `♫ ${short(s.music, 14)}` : '', s.toggles['do not disturb'] ? '☾' : '', s.toggles.wifi ? 'wifi' : 'no wifi',
    s.toggles.mute ? 'muted' : `vol ${s.volume}%`, '▮▮▮▯', 'Sat 17 Oct 09:41'].filter(Boolean).join('   ');
  return `<rect x="${x}" y="${y}" width="${w}" height="18" fill="#0d0b1c" opacity=".55"/>`
    + T(x + 10, y + 13, '◆', 10, '#fff') + T(x + 26, y + 13, name, 9.5, '#fff', 'font-weight="700"')
    + T(x + 30 + name.length * 6.2, y + 13, 'File   Edit   View   Window   Help', 9, '#e6e3ff', 'xml:space="preserve"')
    + T(x + w - 8, y + 13, right, 8.5, '#f2f0ff', 'text-anchor="end" xml:space="preserve"');
}

function dock(s, x, y, w, h) {
  const n = APPS.length, iw = 26, gap = 7, dw = n * (iw + gap) + gap, dx = x + (w - dw) / 2, dy = y + h - 40;
  return `<rect x="${dx}" y="${dy}" width="${dw}" height="36" rx="12" fill="#ffffff" fill-opacity=".14" stroke="#ffffff" stroke-opacity=".2"/>`
    + APPS.map((a, i) => { const ix = dx + gap + i * (iw + gap), [col, g] = ICON[a];
      return `<rect x="${ix}" y="${dy + 4}" width="${iw}" height="${iw}" rx="7" fill="${col}"/>` + T(ix + iw / 2, dy + 21, g, a === 'calendar' || a === 'terminal' ? 9 : 13, a === 'notes' ? '#5a4300' : '#fff', 'text-anchor="middle" font-weight="700"')
        + (s.windows.includes(a) ? `<circle cx="${ix + iw / 2}" cy="${dy + 33}" r="1.6" fill="#fff"/>` : ''); }).join('');
}

function toastPill(t, rx, y) {
  const w = Math.min(220, 26 + String(t).length * 5.6);
  return `<rect x="${rx - w}" y="${y}" width="${w}" height="24" rx="8" fill="#16131f" fill-opacity=".92" stroke="${ACC}" stroke-opacity=".6"/>` + T(rx - w + 12, y + 16, short(t, 38), 9.5, '#fff');
}

function lockScreen(x, y, w, h) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#0c0a18" opacity=".86"/>` + T(x + w / 2, y + 120, '09:41', 44, '#fff', 'text-anchor="middle" font-weight="300"')
    + T(x + w / 2, y + 146, 'Saturday 17 October', 12, '#cfcbe8', 'text-anchor="middle"') + `<circle cx="${x + w / 2}" cy="${y + 210}" r="22" fill="#2b2547"/>` + T(x + w / 2, y + 216, '🔒', 16, '#fff', 'text-anchor="middle"')
    + T(x + w / 2, y + 252, 'locked · say anything to unlock', 10, '#9d97c2', 'text-anchor="middle"');
}

function win(s, app, x, y, w, h, isTop) {
  const title = {finder: 'Finder', browser: s.url || 'New tab', notes: 'Notes', calendar: 'Calendar · October 2026', mail: 'New message',
    music: 'Music', timer: 'Timer', terminal: 'Terminal', settings: 'Settings'}[app];
  let o = `<g><rect x="${x + 3}" y="${y + 6}" width="${w}" height="${h}" rx="10" fill="#000" opacity=".3"/>`
    + `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="#1d1b26" stroke="${isTop ? '#4a4566' : '#2f2c3d'}"/>`
    + `<rect x="${x}" y="${y}" width="${w}" height="24" rx="10" fill="#2a2737"/><rect x="${x}" y="${y + 14}" width="${w}" height="10" fill="#2a2737"/>`
    + ['#ff5f57', '#febc2e', '#28c840'].map((c, i) => `<circle cx="${x + 14 + i * 14}" cy="${y + 12}" r="4.5" fill="${isTop ? c : '#55516a'}"/>`).join('')
    + T(x + w / 2, y + 16, short(title, 34), 9.5, '#d9d6ea', 'text-anchor="middle" font-weight="600"');
  const bx = x + 12, by = y + 36, bw = w - 24;
  switch (app) {
    case 'browser':
      o += `<rect x="${bx}" y="${by - 4}" width="${bw}" height="20" rx="10" fill="#14121c"/>` + T(bx + 12, by + 10, `🔒 ${s.url || 'search or type an address'}`, 9, '#bdb8d6')
        + (s.url ? T(bx, by + 46, short(s.url.split('/')[0].replace(/^www\./, '').split('.')[0], 18), 22, '#fff', 'font-weight="800"') + T(bx, by + 62, 'opened for real in a new browser tab', 9, ACC) : '')
        + [0, 1, 2, 3, 4].map(i => `<rect x="${bx}" y="${by + 76 + i * 16}" width="${bw - (i % 3) * 40}" height="7" rx="3.5" fill="#2e2b3d"/>`).join('');
      break;
    case 'notes': {
      o += `<rect x="${x}" y="${y + 24}" width="86" height="${h - 24}" fill="#24212f"/>`
        + (s.notes.length ? s.notes : [{title: 'Welcome', body: ''}]).slice(0, 6).map((n, i) => `<rect x="${x + 4}" y="${y + 30 + i * 26}" width="78" height="22" rx="5" fill="${i === 0 ? '#f5c542' : 'none'}" fill-opacity=".25"/>` + T(x + 10, y + 45 + i * 26, short(n.title, 12), 9, '#f2eedb')).join('');
      const n = s.notes[0];
      o += n ? T(x + 98, by + 8, short(n.title, 26), 14, '#fff', 'font-weight="700"') + wrap(n.body, 34).slice(0, 8).map((l, i) => T(x + 98, by + 30 + i * 15, l, 10, '#e7e2c9')).join('')
        : T(x + 98, by + 10, 'No notes yet', 11, '#999');
      break;
    }
    case 'timer': {
      const t = s.timer, cx = x + w / 2, cy = y + 112, r = 58;
      const frac = t ? t.left / t.total : 0, circ = 2 * Math.PI * r;
      const col = !t ? '#555' : t.state === 'done' ? '#28c840' : t.state === 'paused' ? '#febc2e' : t.state === 'cancelled' ? '#ff5f57' : '#ff9f43';
      o += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#2e2b3d" stroke-width="9"/>`
        + `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${col}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${(circ * frac).toFixed(1)} ${circ.toFixed(1)}" transform="rotate(-90 ${cx} ${cy})"/>`
        + T(cx, cy + 9, t ? fmt(t.left) : '00:00', 26, '#fff', 'text-anchor="middle" font-weight="700" font-family="ui-monospace,monospace"')
        + T(cx, cy + 28, t ? `${t.label ? t.label + ' · ' : ''}${t.state}` : 'no timer', 10, col, 'text-anchor="middle"');
      break;
    }
    case 'calendar': {
      const cw = bw / 7;
      o += ['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((d, i) => T(bx + cw * i + cw / 2, by + 4, d, 8, '#8f8aab', 'text-anchor="middle"')).join('');
      for (let d = 1; d <= 31; d++) {
        const idx = d + 3, col = idx % 7, row = Math.floor(idx / 7), cx = bx + cw * col + cw / 2, cy = by + 20 + row * 26;  // Oct 1 = Thursday
        const ev = s.events.filter(e => e.day === d);
        o += (d === TODAY ? `<circle cx="${cx}" cy="${cy - 3}" r="8" fill="#ff5d5d"/>` : '') + T(cx, cy, d, 8.5, d === TODAY ? '#fff' : '#cfcbe8', 'text-anchor="middle"')
          + ev.slice(0, 1).map(() => `<rect x="${cx - cw / 2 + 3}" y="${cy + 4}" width="${cw - 6}" height="5" rx="2.5" fill="${ACC}"/>`).join('');
      }
      const last = s.events.at(-1);
      if (last) o += `<rect x="${bx}" y="${y + h - 26}" width="${bw}" height="18" rx="6" fill="${ACC}" fill-opacity=".25"/>` + T(bx + 8, y + h - 13, short(`${last.title}${last.date ? ' · ' + last.date : ''}${last.time ? ' · ' + last.time : ''}`, 46), 9, '#fff');
      break;
    }
    case 'mail': {
      const m = s.mail || {to: '', subject: '', body: ''};
      o += [['To', m.to], ['Subject', m.subject]].map(([k, v], i) => T(bx, by + 6 + i * 20, `${k}:`, 9, '#8f8aab') + T(bx + 50, by + 6 + i * 20, short(v, 40), 10, '#fff') + `<line x1="${bx}" y1="${by + 12 + i * 20}" x2="${bx + bw}" y2="${by + 12 + i * 20}" stroke="#2e2b3d"/>`).join('')
        + wrap(m.body, 46).slice(0, 5).map((l, i) => T(bx, by + 56 + i * 14, l, 10, '#e9e9f2')).join('')
        + (s.mail ? `<rect x="${bx}" y="${y + h - 30}" width="${bw}" height="20" rx="6" fill="${ACC}" fill-opacity=".25"/>` + T(bx + 8, y + h - 16, 'draft handed to your mail app · nothing sent', 9, '#fff') : '');
      break;
    }
    case 'music':
      o += `<rect x="${bx}" y="${by}" width="96" height="96" rx="10" fill="#ff4f8b" opacity=".85"/>` + T(bx + 48, by + 60, '♫', 34, '#fff', 'text-anchor="middle"')
        + T(bx + 110, by + 30, short(s.music || 'Nothing playing', 22), 13, '#fff', 'font-weight="700"') + T(bx + 110, by + 48, s.playing ? 'now playing' : 'paused', 10, '#ff9fc0')
        + T(bx + 110, by + 80, '⏮   ⏸   ⏭', 16, '#fff', 'xml:space="preserve"') + `<rect x="${bx}" y="${by + 120}" width="${bw}" height="4" rx="2" fill="#2e2b3d"/><rect x="${bx}" y="${by + 120}" width="${bw * 0.35}" height="4" rx="2" fill="#ff4f8b"/>`;
      break;
    case 'finder': {
      const list = s.found && s.found.length ? s.found : FILES;
      o += `<rect x="${x}" y="${y + 24}" width="70" height="${h - 24}" fill="#24212f"/>` + ['Recents', 'Desktop', 'Documents', 'Downloads'].map((n, i) => T(x + 10, y + 44 + i * 18, n, 9, i === 2 ? '#fff' : '#a29dbd')).join('')
        + (s.fquery ? T(x + 82, by + 2, `🔎 "${short(s.fquery, 26)}" · ${s.found.length} found`, 9, ACC) : T(x + 82, by + 2, 'Documents', 9, '#a29dbd'))
        + list.slice(0, 7).map((f, i) => { const hit = s.found && s.found.includes(f); const ext = f.split('.').pop();
          const col = {pdf: '#ff6b6b', xlsx: '#28c840', jpg: '#ffb020', png: '#ffb020', docx: '#4f8cff', txt: '#aaa'}[ext] || '#aaa';
          return `<rect x="${x + 80}" y="${by + 10 + i * 22}" width="${w - 92}" height="19" rx="5" fill="${hit ? ACC : '#000'}" fill-opacity="${hit ? 0.3 : 0}"/><rect x="${x + 86}" y="${by + 14 + i * 22}" width="9" height="11" rx="2" fill="${col}"/>` + T(x + 102, by + 23 + i * 22, short(f, 30), 9, '#e9e9f2'); }).join('');
      break;
    }
    case 'terminal':
      o += `<rect x="${x + 1}" y="${y + 24}" width="${w - 2}" height="${h - 25}" fill="#0e0f13"/>` + T(bx, by + 6, '~ % relay --status', 10, '#7ee787', 'font-family="ui-monospace,monospace"')
        + T(bx, by + 22, 'relay 11.5 MB · on-device · ready', 10, '#c9d1d9', 'font-family="ui-monospace,monospace"') + T(bx, by + 38, '~ % █', 10, '#7ee787', 'font-family="ui-monospace,monospace"');
      break;
    case 'settings':
      o += Object.entries(s.toggles).map(([k, v], i) => T(bx, by + 10 + i * 24, k, 10, '#e9e9f2') + `<rect x="${bx + bw - 32}" y="${by + i * 24}" width="30" height="15" rx="7.5" fill="${v ? ACC : '#3a3650'}"/><circle cx="${v ? bx + bw - 10 : bx + bw - 24}" cy="${by + 7.5 + i * 24}" r="5.5" fill="#fff"/>`).join('')
        + T(bx, by + 136, `volume ${s.volume}%`, 10, '#e9e9f2') + `<rect x="${bx}" y="${by + 142}" width="${bw}" height="5" rx="2.5" fill="#3a3650"/><rect x="${bx}" y="${by + 142}" width="${bw * s.volume / 100}" height="5" rx="2.5" fill="${ACC}"/>`;
      break;
  }
  return o + '</g>';
}
