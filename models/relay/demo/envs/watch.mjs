// Watch environment: notifications turned into structured data (payments, sentiment, reminders, replies) plus workouts,
// timers and alarms. A watch face on the left, result cards and the notification stack on the right.
// Same contract as phone.mjs; render(s) is a pure SVG string (560x460).

const CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF'];
const WORKOUTS = ['run', 'walk', 'cycle', 'swim', 'yoga', 'strength'];

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});

const SEED = [
  {app: 'Bank', color: '#46c37b', text: 'Card payment of €42.80 at Albert Heijn on 16 October.'},
  {app: 'Messages', color: '#5ab0ff', from: 'Alex', text: 'Alex: are we still on for dinner tonight?'},
  {app: 'Calendar', color: '#f5a524', text: 'Dentist appointment tomorrow at 10:30.'},
];

export default {
  key: 'watch',
  label: 'Watch',
  subtitle: 'notifications into structured data',
  tagline: 'Payments, sentiment, reminders, replies, workouts and timers on the wrist',
  placeholder: 'paste a notification or ask the watch...',
  tools: [
    tool('record_payment', 'Log a payment from a bank or card notification into the budget.', {
      merchant: str('Shop or company that was paid.'),
      amount: {type: 'number', description: 'Amount paid.'},
      currency: en(CURRENCIES, 'Currency code: USD for $ or dollars, EUR for € or euros, GBP for £ or pounds.'),
      date: str('Date of the payment.')}, ['merchant', 'amount', 'currency']),
    tool('classify_sentiment', 'Label the tone of a message, review or comment the user quotes.', {
      sentiment: en(['positive', 'negative', 'neutral', 'mixed'], 'positive (happy, praise), negative (complaint, angry), neutral (plain fact), mixed (both).')}, ['sentiment']),
    tool('set_reminder', 'Create a reminder.', {
      text: str('What to be reminded about.'),
      time: str('When, as said, e.g. "at 6pm" or "tomorrow morning".')}, ['text']),
    tool('reply_message', 'Send a reply to a message.', {
      recipient: str('Who to reply to.'),
      text: str('The reply text.')}, ['text']),
    tool('start_workout', 'Start tracking a workout.', {
      activity: en(WORKOUTS, 'run (jog), walk (hike), cycle (bike ride), swim, yoga, strength (weights, gym).')}, ['activity']),
    tool('control_workout', 'Pause, resume or end the current workout.', {
      action: en(['pause', 'resume', 'stop'], 'pause (take a break), resume (continue), stop (end it, finish, done).')}, ['action']),
    tool('set_timer', 'Start a countdown timer.', {
      duration: str('How long, e.g. "10 minutes".'),
      label: str('What the timer is for, if said.')}, ['duration']),
    tool('control_timer', 'Pause, resume or cancel the running timer.', {
      action: en(['pause', 'resume', 'cancel'], 'What to do with the timer.')}, ['action']),
    tool('set_alarm', 'Set a wake-up alarm.', {time: str('Alarm time, e.g. 6:30 or 7am.')}, ['time']),
    tool('show_heart_rate', 'Show the current heart rate.', {}, [],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(heart ?rate|pulse|bpm)\\b']}),
    tool('show_activity', 'Show today\'s steps, calories and activity rings.'),
    tool('find_phone', 'Make the paired phone ring so it can be found.', {}, [],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(where\'?s|where is|find|lost|ring|can\'?t find)\\s+(my\\s+)?phone\\b']}),
    tool('dismiss_notification', 'Clear the newest notification.'),
  ],

  init() {
    return {cards: SEED.map(c => ({...c})), results: [], workout: null, timer: null, alarm: null, hr: 72, steps: 6420,
      ping: false, note: ''};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const push = r => { if (['workout', 'timer', 'phone', 'hr', 'activity'].includes(r.type)) s.results = s.results.filter(x => x.type !== r.type); s.results.unshift(r); s.results = s.results.slice(0, 4); };
    switch (c.tool) {
      case 'record_payment': {
        const cur = normCur(a.currency);
        push({type: 'payment', merchant: a.merchant || '?', amount: Number(a.amount), currency: cur, date: a.date || 'today'});
        return `logged ${sym(cur)}${money(a.amount)} at ${a.merchant}`;
      }
      case 'classify_sentiment': {
        const v = String(a.sentiment || '').toLowerCase();
        const k = /mix|both/.test(v) ? 'mixed' : /pos|happy|good|great|love/.test(v) ? 'positive' : /neg|bad|angry|complain|terrible|hate/.test(v) ? 'negative' : 'neutral';
        push({type: 'sentiment', value: k}); return `sentiment: ${k}`;
      }
      case 'set_reminder': push({type: 'reminder', text: a.text, time: a.time || ''}); return `reminder: ${a.text}${a.time ? ` (${a.time})` : ''}`;
      case 'reply_message': {
        const to = a.recipient || (s.cards.find(x => x.from) || {}).from || 'them';
        push({type: 'reply', to, text: a.text});
        s.cards = s.cards.filter(x => x.from !== to);
        return `replied to ${to}`;
      }
      case 'start_workout': {
        const w = String(a.activity || 'run').toLowerCase();
        const k = WORKOUTS.find(x => w.includes(x)) || (/jog/.test(w) ? 'run' : /bike|ride/.test(w) ? 'cycle' : /gym|weight|lift/.test(w) ? 'strength' : /hike/.test(w) ? 'walk' : 'run');
        s.workout = {kind: k, state: 'active', mins: 0};
        for (let i = 0; i < 4; i++) { s.hr += 9; s.workout.mins = i; ctx.render(); await ctx.sleep(150); }
        push({type: 'workout'}); return `${k} started`;
      }
      case 'control_workout': {
        if (!s.workout) return 'no workout running';
        const v = String(a.action || '').toLowerCase();
        s.workout.state = /pause|break/.test(v) ? 'paused' : /resume|continue/.test(v) ? 'active' : 'ended';
        if (s.workout.state === 'ended') s.hr = 80;
        push({type: 'workout'}); return `workout ${s.workout.state}`;
      }
      case 'set_timer': s.timer = {duration: a.duration, label: a.label || '', state: 'running'}; push({type: 'timer'}); return `timer ${a.duration}`;
      case 'control_timer': {
        if (!s.timer) return 'no timer running';
        const v = String(a.action || '').toLowerCase();
        s.timer.state = /pause|hold/.test(v) ? 'paused' : /resume|continue/.test(v) ? 'running' : 'cancelled';
        push({type: 'timer'}); return `timer ${s.timer.state}`;
      }
      case 'set_alarm': s.alarm = a.time; push({type: 'alarm', time: a.time}); return `alarm ${a.time}`;
      case 'show_heart_rate': push({type: 'hr', bpm: s.hr}); return `heart rate ${s.hr} bpm`;
      case 'show_activity': push({type: 'activity'}); return `${s.steps} steps today`;
      case 'find_phone':
        for (let i = 0; i < 3; i++) { s.ping = true; ctx.render(); await ctx.sleep(220); s.ping = false; ctx.render(); await ctx.sleep(140); }
        push({type: 'phone'}); return 'phone is ringing';
      case 'dismiss_notification': { const d = s.cards.shift(); return d ? `dismissed ${d.app}` : 'no notifications'; }
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    const cx = 140, cy = 220;
    // watch body
    const face = `<rect x="${cx - 46}" y="20" width="92" height="90" rx="14" fill="#1c1d22"/><rect x="${cx - 46}" y="330" width="92" height="110" rx="14" fill="#1c1d22"/>
  <rect x="${cx - 92}" y="${cy - 112}" width="184" height="224" rx="48" fill="#222329" stroke="#3a3c46" stroke-width="3"/>
  <rect x="${cx + 92}" y="${cy - 40}" width="10" height="34" rx="4" fill="#3a3c46"/>
  <rect x="${cx - 80}" y="${cy - 100}" width="160" height="200" rx="38" fill="#050507"/>`;
    const w = s.workout;
    let screen;
    if (s.ping) screen = `<circle cx="${cx}" cy="${cy}" r="44" fill="none" stroke="#5ab0ff" stroke-width="3"/><circle cx="${cx}" cy="${cy}" r="64" fill="none" stroke="#5ab0ff" stroke-opacity=".4" stroke-width="2"/>
  <text x="${cx}" y="${cy + 6}" fill="#fff" font-size="15" font-weight="700" text-anchor="middle">pinging</text><text x="${cx}" y="${cy + 24}" fill="#aaa" font-size="10" text-anchor="middle">phone</text>`;
    else if (w && w.state !== 'ended') screen = `<text x="${cx}" y="${cy - 52}" fill="${w.state === 'paused' ? '#f5a524' : '#46c37b'}" font-size="11" font-weight="600" text-anchor="middle">${w.kind.toUpperCase()} · ${w.state}</text>
  <text x="${cx}" y="${cy - 8}" fill="#fff" font-size="34" font-weight="700" text-anchor="middle">0:0${w.mins}</text>
  <text x="${cx}" y="${cy + 22}" fill="#ff6b6e" font-size="14" text-anchor="middle">♥ ${s.hr} bpm</text>
  <text x="${cx}" y="${cy + 44}" fill="#aaa" font-size="10" text-anchor="middle">${(w.mins * 0.18).toFixed(2)} km</text>`;
    else screen = `<text x="${cx}" y="${cy - 62}" fill="#ff6b6e" font-size="10" font-weight="600" text-anchor="middle">SAT 17</text>
  <text x="${cx}" y="${cy - 12}" fill="#fff" font-size="40" font-weight="700" text-anchor="middle">9:41</text>
  ${ring(cx - 38, cy + 30, 13, 0.72, '#ff4d6d')}${ring(cx, cy + 30, 13, 0.55, '#a3e635')}${ring(cx + 38, cy + 30, 13, 0.9, '#22d3ee')}
  <text x="${cx}" y="${cy + 66}" fill="#ff6b6e" font-size="10" text-anchor="middle">♥ ${s.hr}  ·  ${s.cards.length} new</text>
  ${s.timer && s.timer.state !== 'cancelled' ? `<text x="${cx}" y="${cy + 82}" fill="#f5a524" font-size="9" text-anchor="middle">⏱ ${esc(s.timer.duration)} ${s.timer.state}</text>` : s.alarm ? `<text x="${cx}" y="${cy + 82}" fill="#aaa" font-size="9" text-anchor="middle">⏰ ${esc(s.alarm)}</text>` : ''}`;
    // right column: results then notifications
    let y = 22;
    const X = 280, W = 260;
    const out = [];
    out.push(`<text x="${X}" y="${y}" fill="#888" font-size="10" letter-spacing="1">RESULTS</text>`); y += 8;
    if (!s.results.length) { out.push(`<text x="${X}" y="${y + 18}" fill="#555" font-size="11">nothing yet</text>`); y += 30; }
    for (const r of s.results) { const [svg, h] = resultCard(r, s, X, y, W); out.push(svg); y += h + 8; if (y > 300) break; }
    y += 6;
    out.push(`<text x="${X}" y="${y + 6}" fill="#888" font-size="10" letter-spacing="1">NOTIFICATIONS</text>`); y += 14;
    for (const n of s.cards) {
      if (y > 420) break;
      const lines = wrap(n.text, 40).slice(0, 2);
      const h = 22 + lines.length * 13;
      out.push(`<rect x="${X}" y="${y}" width="${W}" height="${h}" rx="10" fill="#15161b" stroke="#25262e"/><circle cx="${X + 12}" cy="${y + 12}" r="4" fill="${n.color}"/>
  <text x="${X + 22}" y="${y + 15}" fill="#ddd" font-size="10" font-weight="600">${n.app}</text><text x="${X + W - 10}" y="${y + 15}" fill="#666" font-size="9" text-anchor="end">now</text>
  ${lines.map((l, i) => `<text x="${X + 12}" y="${y + 30 + i * 13}" fill="#bbb" font-size="10">${esc(l)}</text>`).join('')}`);
      y += h + 6;
    }
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <rect width="560" height="460" fill="#0b0b0f"/>
  ${face}${screen}
  ${out.join('\n  ')}
</svg>`;
  },

  examples: [
    {title: 'Log a payment', sub: 'extraction: span, number, enum, date', q: 'Log this payment: You paid $23.50 at Blue Bottle Coffee on October 15'},
    {title: 'Euros', sub: 'currency enum from a word', q: 'Save this to my budget: Card payment of 18.20 euros at Starbucks on 14 October'},
    {title: 'Remind me', sub: 'span + time', q: 'Remind me to call the plumber at 6pm'},
    {title: 'Reply to Alex', sub: 'recipient + text', q: 'Reply to Alex: yes, see you at eight'},
    {title: 'Two at once', sub: '2 calls, 2 tools', q: 'Remind me to buy flowers tomorrow morning and reply to Sam: happy birthday!'},
    {title: 'Go for a run', sub: '2 calls', q: 'Start a run and set a timer for 20 minutes'},
    {title: 'Take a break', sub: 'pause, not end', q: 'Pause my workout, I need a breather'},
    {title: 'Weights', sub: 'enum from meaning', q: 'Track a gym session with weights'},
    {title: 'Lost phone', sub: 'trigger', q: "Where's my phone?"},
    {title: 'Clear it', sub: 'no-arg call', q: 'Dismiss that notification'},
    {title: 'Not a watch thing', sub: 'abstains', q: 'Write me a poem about the sea'},
  ],
  help: 'Try pasting a bank notification ("log this: you paid $23.50 at Blue Bottle Coffee"), "remind me to call the plumber at 6pm", "start a run".',
};

function resultCard(r, s, x, y, w) {
  const T = (tx, ty, t, sz = 11, fill = '#eee', extra = '') => `<text x="${tx}" y="${ty}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;
  const box = (h, col) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#17181e" stroke="${col}" stroke-opacity=".7"/>`;
  switch (r.type) {
    case 'payment': return [box(66, '#46c37b') + T(x + 12, y + 18, 'PAYMENT LOGGED', 9, '#46c37b', 'font-weight="700"') +
      T(x + 12, y + 40, r.merchant, 13, '#fff', 'font-weight="600"') + T(x + w - 12, y + 40, `${sym(r.currency)}${money(r.amount)}`, 16, '#fff', 'font-weight="700" text-anchor="end"') +
      T(x + 12, y + 57, `${r.currency} · ${r.date}`, 10, '#999'), 66];
    case 'sentiment': {
      const col = {positive: '#46c37b', negative: '#e5484d', neutral: '#8a8fa3', mixed: '#f5a524'}[r.value];
      return [box(44, col) + T(x + 12, y + 27, 'SENTIMENT', 9, '#aaa', 'font-weight="700"') +
        `<rect x="${x + w - 100}" y="${y + 12}" width="88" height="20" rx="10" fill="${col}"/>` + T(x + w - 56, y + 26, r.value, 11, '#fff', 'text-anchor="middle" font-weight="600"'), 44];
    }
    case 'reminder': return [box(50, '#f5a524') + T(x + 12, y + 18, `REMINDER${r.time ? ` · ${r.time}` : ''}`, 9, '#f5a524', 'font-weight="700"') + T(x + 12, y + 38, wrap(r.text, 36)[0], 12, '#fff'), 50];
    case 'reply': return [box(54, '#5ab0ff') + T(x + 12, y + 18, `REPLY TO ${String(r.to).toUpperCase()}`, 9, '#5ab0ff', 'font-weight="700"') +
      `<rect x="${x + 40}" y="${y + 25}" width="${w - 52}" height="20" rx="10" fill="#2563eb"/>` + T(x + 50, y + 39, wrap(r.text, 32)[0], 10, '#fff'), 54];
    case 'workout': { const wk = s.workout || {kind: '?', state: '?'}; return [box(44, '#46c37b') + T(x + 12, y + 27, `WORKOUT · ${wk.kind}`, 11, '#fff', 'font-weight="600"') + T(x + w - 12, y + 27, wk.state, 11, wk.state === 'paused' ? '#f5a524' : wk.state === 'ended' ? '#999' : '#46c37b', 'text-anchor="end"'), 44]; }
    case 'timer': { const t = s.timer || {}; return [box(44, '#f5a524') + T(x + 12, y + 27, `⏱ ${t.duration}${t.label ? ` · ${t.label}` : ''}`, 12, '#fff', 'font-weight="600"') + T(x + w - 12, y + 27, t.state, 11, '#f5a524', 'text-anchor="end"'), 44]; }
    case 'alarm': return [box(44, '#8a8fa3') + T(x + 12, y + 27, `⏰ alarm ${r.time}`, 12, '#fff', 'font-weight="600"'), 44];
    case 'hr': return [box(44, '#e5484d') + T(x + 12, y + 28, `♥ ${r.bpm} bpm`, 14, '#ff6b6e', 'font-weight="700"') + T(x + w - 12, y + 28, 'resting range', 9, '#888', 'text-anchor="end"'), 44];
    case 'activity': return [box(44, '#a3e635') + T(x + 12, y + 28, `${s.steps.toLocaleString('en-US')} steps · 410 kcal · 9 h standing`, 11, '#fff'), 44];
    case 'phone': return [box(44, '#5ab0ff') + T(x + 12, y + 28, '📱 phone ringing · about 4 m away', 11, '#fff'), 44];
    default: return ['', 0];
  }
}

function normCur(c) {
  const t = String(c || '').toUpperCase();
  if (CURRENCIES.includes(t)) return t;
  if (/€|EURO/.test(t)) return 'EUR';
  if (/£|POUND|GBP/.test(t)) return 'GBP';
  if (/¥|YEN/.test(t)) return 'JPY';
  if (/FRANC/.test(t)) return 'CHF';
  return 'USD';
}
const sym = c => ({EUR: '€', USD: '$', GBP: '£', JPY: '¥', CHF: 'CHF '}[c] || '');
const money = v => (Number.isFinite(Number(v)) ? Number(v).toFixed(2) : String(v));
function ring(x, y, r, f, col) {
  const C = 2 * Math.PI * r;
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${col}" stroke-opacity=".2" stroke-width="5"/><circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${col}" stroke-width="5" stroke-linecap="round" stroke-dasharray="${(C * f).toFixed(1)} ${C.toFixed(1)}" transform="rotate(-90 ${x} ${y})"/>`;
}
function wrap(t, n) { const w = String(t).split(/\s+/); const out = ['']; for (const x of w) { if ((out.at(-1) + ' ' + x).trim().length > n) out.push(x); else out[out.length - 1] = (out.at(-1) + ' ' + x).trim(); } return out; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
