// "Many tools" environment: one request routed across a large catalog of generic assistant tools, plus a small
// personal-notes store that search_notes matches lexically. Same contract as phone.mjs.

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const num = description => ({type: 'number', description});
const int = description => ({type: 'integer', description});
const CUR = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'INR', 'SEK'];
const cur = description => ({type: 'string', enum: CUR, description});

const NOTES = [
  {title: 'Wifi password', text: 'Home wifi password is maple-river-42, guest network is open'},
  {title: 'Passport renewal', text: 'Passport expires in March, book the renewal appointment at city hall'},
  {title: 'Lasagna recipe', text: 'Lasagna: beef ragu, bechamel, fresh pasta sheets, bake 45 minutes'},
  {title: 'Gift ideas', text: 'Birthday gift ideas for Maya: hiking boots, a film camera, concert tickets'},
  {title: 'Car service', text: 'Car service due at 60000 km, garage on Elm Street, ask about the brakes'},
  {title: 'Book list', text: 'Books to read: Project Hail Mary, The Overstory, Piranesi'},
];
const STOP = new Set(['the', 'and', 'for', 'my', 'about', 'note', 'notes', 'what', 'was', 'with', 'that', 'this', 'from']);
const words = t => String(t || '').toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 2 && !STOP.has(w));

const TOOLS = [
  tool('set_alarm', 'Set an alarm.', {time: str('Alarm time, e.g. 7am.')}, ['time']),
  tool('cancel_alarm', 'Cancel an existing alarm.', {time: str('Time of the alarm to cancel.')}),
  tool('set_timer', 'Start a countdown timer.', {minutes: int('Length in minutes.')}, ['minutes']),
  tool('stop_timer', 'Stop the running timer.'),
  tool('get_weather', 'Get the weather forecast for a city.', {city: str('City name.')}, ['city']),
  tool('set_lights', 'Turn the lights in a room on or off.', {room: str('Room name.'),
    state: {type: 'string', enum: ['on', 'off'], description: 'On or off.'}}, ['room', 'state']),
  tool('set_thermostat', 'Set the heating temperature.', {degrees: num('Target temperature.')}, ['degrees']),
  tool('lock_door', 'Lock a door.', {door: str('Which door, e.g. front.')}),
  tool('play_music', 'Play a song, artist or genre.', {query: str('What to play.')}, ['query']),
  tool('play_podcast', 'Play a podcast.', {title: str('Podcast name.')}, ['title']),
  tool('call_contact', 'Phone a contact.', {name: str('Contact name.')}, ['name']),
  tool('send_message', 'Send a text message.', {recipient: str('Contact name.'), text: str('Message text.')}, ['recipient', 'text']),
  tool('send_email', 'Send an email.', {to: str('Recipient.'), subject: str('Subject line.'), body: str('Email body.')}, ['to']),
  tool('create_event', 'Add a calendar event.', {title: str('Event name.'), date: str('Date.'), time: str('Start time.')}, ['title']),
  tool('set_reminder', 'Remind me to do something.', {text: str('What to remember.'), time: str('When.')}, ['text']),
  tool('create_note', 'Write down a new note.', {text: str('Note text.')}, ['text']),
  tool('search_notes', 'Look up something in my saved notes, like a password or recipe.', {query: str('Topic to look up, e.g. wifi password.')}, ['query']),
  tool('translate', 'Translate text into a language.', {text: str('Text to translate.'), language: str('Target language name, e.g. French.')}, ['text', 'language']),
  tool('convert_currency', 'Convert money between currencies. Dollars=USD, euros=EUR, pounds=GBP, yen=JPY.', {
    amount: num('Amount of money.'), from_currency: cur('Currency to convert from.'), to_currency: cur('Currency to convert to.')},
    ['amount', 'from_currency', 'to_currency']),
  tool('convert_units', 'Convert a length, weight or temperature to another unit.', {value: num('The number.'), from_unit: str('Unit to convert from.'),
    to_unit: str('Unit to convert to.')}, ['value', 'from_unit', 'to_unit']),
  tool('search_flights', 'Search for flights.', {destination: str('Arrival city.'), origin: str('Departure city.'), date: str('Travel date.')}, ['destination']),
  tool('book_hotel', 'Book a hotel.', {city: str('City.'), nights: int('Number of nights.')}, ['city']),
  tool('find_restaurant', 'Find a restaurant.', {cuisine: str('Type of food.'), near: str('Area or landmark.')}, ['cuisine']),
  tool('get_directions', 'Get directions to a place.', {destination: str('Where to go.')}, ['destination']),
  tool('order_ride', 'Order a taxi ride.', {destination: str('Where to go.')}, ['destination']),
  tool('order_food', 'Order food delivery.', {dish: str('What to order.')}, ['dish']),
  tool('calculate', 'Do arithmetic on plain numbers.', {expression: str('The arithmetic, e.g. 15% of 240.')}, ['expression']),
];

export default {
  key: 'search',
  label: 'Many tools',
  subtitle: 'route across 27 tools, search notes',
  tagline: 'One request, a 27-tool assistant catalog, and a personal notes store',
  placeholder: 'ask the assistant anything...',
  tools: TOOLS,

  init() { return {hits: {}, last: [], query: '', matched: [], searched: false, added: []}; },

  async apply(c, s) {
    const a = c.args || {};
    s.hits[c.tool] = true;
    s.last.push({tool: c.tool, args: a});
    if (s.last.length > 3) s.last.shift();
    if (c.tool === 'search_notes') {
      const q = new Set(words(a.query));
      s.query = a.query || ''; s.searched = true;
      s.matched = allNotes(s).map((n, i) => [i, words(n.title + ' ' + n.text).filter(w => q.has(w)).length]).filter(x => x[1] > 0)
        .sort((x, y) => y[1] - x[1]).map(x => x[0]);
      return s.matched.length ? `${s.matched.length} note${s.matched.length === 1 ? '' : 's'} match "${s.query}"` : `no note matches "${s.query}"`;
    }
    if (c.tool === 'create_note') { (s.added = s.added || []).push({title: 'New note', text: String(a.text || '')}); return 'note saved'; }
    return `${c.tool}(${Object.entries(a).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`;
  },

  render(s) {
    const acc = '#ff4d4f';
    const T = (x, y, t, sz = 11, fill = '#ddd', extra = '') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${sz}" ${extra}>${esc(t)}</text>`;
    // chip grid, wrapped
    let x = 20, y = 20, out = '';
    const lastSet = new Set(s.last.map(l => l.tool));
    for (const t of TOOLS) {
      const w = t.name.length * 6.1 + 16;
      if (x + w > 540) { x = 20; y += 24; }
      const on = s.hits[t.name], now = lastSet.has(t.name);
      out += `<rect x="${x}" y="${y}" width="${w.toFixed(1)}" height="19" rx="9.5" fill="${on ? acc : '#1c1c22'}" opacity="${on && !now ? 0.55 : 1}" stroke="${on ? acc : '#2e2e36'}"/>` +
        T(x + w / 2, y + 13, t.name, 10, on ? '#fff' : '#9a9aa4', 'text-anchor="middle" font-family="ui-monospace,monospace"');
      x += w + 6;
    }
    y += 34;
    // recent calls with args
    out += T(20, y, 'LAST CALLS', 9, '#777', 'letter-spacing="1.5" font-weight="600"');
    y += 6;
    const rows = s.last.length ? s.last : null;
    for (let i = 0; i < 3; i++) {
      const l = rows && rows[i];
      out += `<rect x="20" y="${y + i * 24}" width="520" height="20" rx="5" fill="#15151a"/>`;
      if (l) out += T(28, y + i * 24 + 14, cut(`${l.tool}(${Object.entries(l.args).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')})`, 84), 10, '#eee', 'font-family="ui-monospace,monospace"');
      else if (i === 0) out += T(28, y + 14, 'no calls yet', 10, '#555');
    }
    y += 84;
    // notes
    out += T(20, y, 'MY NOTES', 9, '#777', 'letter-spacing="1.5" font-weight="600"');
    if (s.searched) out += T(540, y, `search_notes: "${cut(s.query, 40)}" · ${s.matched.length} match`, 9, acc, 'text-anchor="end"');
    y += 8;
    const notes = allNotes(s);
    const shown = notes.slice(0, 8), step = Math.min(32, (436 - y) / shown.length), rh = step - 4;
    shown.forEach((n, i) => {
      const hit = s.matched.includes(i), fade = s.searched && !hit, ty = y + i * step + rh / 2 + 4;
      out += `<g opacity="${fade ? 0.35 : 1}"><rect x="20" y="${(y + i * step).toFixed(1)}" width="520" height="${rh.toFixed(1)}" rx="6" fill="#16161c" stroke="${hit ? acc : '#26262e'}" stroke-width="${hit ? 1.6 : 1}"/>` +
        T(30, ty.toFixed(1), n.title, 11, '#fff', 'font-weight="600"') + T(150, ty.toFixed(1), cut(n.text, 62), 10, '#aaa') + '</g>';
    });
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <rect width="560" height="460" fill="#0b0b0f"/>
  ${out}
  <text x="20" y="450" fill="#666" font-size="9.5">Note matching here is lexical (shared words), not semantic. Relay picks the tool and copies the query.</text>
</svg>`;
  },

  examples: [
    {title: 'Alarm', sub: '1 call of 27', q: 'Wake me up at 6:30 tomorrow'},
    {title: 'Currency', sub: 'names to codes', q: 'How much is 250 euros in dollars?'},
    {title: 'Search notes', sub: 'query extraction', q: 'What was the wifi password I wrote down?'},
    {title: 'Reminder', sub: 'text + time', q: 'Remind me to call the dentist at 4pm'},
    {title: 'Flights', sub: 'from, to, date', q: 'Find flights from Amsterdam to Lisbon on Friday'},
    {title: 'Trip', sub: '2 calls', q: 'Book a hotel in Rome for 3 nights and find an Italian restaurant near the Colosseum'},
    {title: 'Two conversions', sub: '2 calls, names to codes', q: 'Convert 100 dollars to yen and 50 euros to pounds'},
    {title: 'Text + call', sub: '2 calls', q: "Text Maya that I'm on my way and then call Tom"},
    {title: 'Cancel, not set', sub: 'contrast', q: 'Cancel my 7am alarm'},
    {title: 'Math', sub: 'expression span', q: 'What is 15% of 240?'},
    {title: 'Translate', sub: 'text + language', q: 'Translate good morning into Spanish'},
    {title: 'Not a tool', sub: 'abstains', q: 'Who wrote Pride and Prejudice?'},
  ],
  help: 'Try: "wake me up at 6:30", "how much is 250 euros in dollars?", "what was the wifi password I wrote down?".',
};

function allNotes(s) { return [...NOTES, ...(s.added || [])]; }
function cut(t, n) { t = String(t ?? ''); return t.length > n ? t.slice(0, n - 1) + '…' : t; }
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
