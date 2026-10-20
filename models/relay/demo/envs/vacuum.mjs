// Robot vacuum environment: a top-down flat map; the robot drives step by step (ctx.sleep + ctx.render) to rooms and
// dirt patches. Same contract as phone.mjs; render(s) is a pure SVG string (560x460).

const ROOMS = ['kitchen', 'living room', 'bedroom', 'bathroom', 'hallway'];
// room rectangles [x, y, w, h], interior centre, and the waypoints from the room to the hallway hub
const RECT = {
  kitchen: [30, 30, 170, 170], 'living room': [200, 30, 330, 220], bedroom: [30, 200, 170, 190],
  hallway: [200, 250, 180, 140], bathroom: [380, 250, 150, 140],
};
const CENTRE = {kitchen: [115, 115], 'living room': [365, 140], bedroom: [115, 295], hallway: [290, 320], bathroom: [455, 320]};
const CHAIN = {kitchen: [[200, 120], [290, 150], [290, 250]], 'living room': [[290, 250]], bedroom: [[200, 330]], hallway: [], bathroom: [[380, 330]]};
const HUB = [290, 320];
const DOCK = [240, 368];
const DIRT = [
  {room: 'kitchen', label: 'coffee spill', x: 80, y: 80},
  {room: 'kitchen', label: 'cereal crumbs', x: 155, y: 165},
  {room: 'living room', label: 'crumbs under the sofa', x: 330, y: 85},
  {room: 'living room', label: 'pet hair by the rug', x: 470, y: 200},
  {room: 'bedroom', label: 'dust under the bed', x: 95, y: 340},
  {room: 'hallway', label: 'muddy footprints', x: 330, y: 300},
  {room: 'bathroom', label: 'hair by the sink', x: 480, y: 290},
];
const PX_PER_M = 60;

const tool = (name, description, props = {}, required = [], extra = {}) =>
  ({name, description, parameters: {type: 'object', properties: props, required}, ...extra});
const str = description => ({type: 'string', description});
const en = (options, description) => ({type: 'string', enum: options, description});
const room = d => en(ROOMS, d);

export default {
  key: 'vacuum',
  label: 'Robot vacuum',
  subtitle: 'nuanced instructions, executed as moves',
  tagline: 'A robot vacuum that cleans rooms and spots, avoids rooms and drives where it is told',
  placeholder: 'tell the robot where to clean...',
  tools: [
    tool('clean_room', 'Clean (vacuum) a single room.', {room: room('The room to clean.')}, ['room']),
    tool('clean_everywhere', 'Vacuum the whole home, optionally skipping one room.', {
      skip: room('A room to leave out ("except the bedroom"), if any.')}),
    tool('go_to_room', 'Drive to a room without cleaning.', {room: room('Where to go.')}, ['room']),
    tool('clean_spot', 'Clean one specific mess or spot the user points out.', {
      spot: str('The mess as described, e.g. "coffee spill" or "crumbs under the sofa".')}, ['spot']),
    tool('avoid_room', 'Make a room off-limits so the robot stays out of it from now on.', {room: room('The room to keep away from.')}, ['room']),
    tool('move', 'Drive a set distance in a direction.', {
      direction: en(['forward', 'backward', 'left', 'right'], 'Direction relative to the robot.'),
      meters: {type: 'number', description: 'Distance in meters.'}}, ['direction', 'meters']),
    tool('set_suction', 'Change the suction power.', {
      level: en(['quiet', 'standard', 'max'], 'quiet (soft, low, silent), standard (normal) or max (strong, turbo, full power).')}, ['level']),
    tool('return_to_dock', 'Go back to the charging dock.'),
    tool('pause', 'Stop or pause what the robot is doing right now.', {},  [],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)\\W*(stop|pause|hold on|wait|halt|freeze)(\\s+(it|now|please|cleaning|vacuuming|right now|for a (sec|second|moment)))*\\W*$']}),
    tool('resume', 'Continue the cleaning that was paused.'),
    tool('report_status', 'Report battery, dust bin level and where the robot is.', {},  [],
      {triggers: ['^(?![\\s\\S]*\\b(and|then|also)\\b)[\\s\\S]*\\b(battery|charge left|how full|bin full|where are you|status)\\b[\\s\\S]*\\?\\s*$']}),
    tool('empty_bin', 'Empty the dust bin at the dock.'),
  ],

  init() {
    return {x: DOCK[0], y: DOCK[1], h: -90, suction: 'standard', status: 'docked', battery: 84, bin: 30,
      dirt: DIRT.map(d => ({...d, clean: false})), avoid: [], trail: [], pending: [], note: 'docked and charging', target: ''};
  },

  async apply(c, s, ctx) {
    const a = c.args || {};
    const done = t => (s.note = t);
    switch (c.tool) {
      case 'clean_room': {
        const r = normRoom(a.room);
        if (!r) return done(`I don't know a room called "${a.room}"`);
        if (s.avoid.includes(r)) return done(`${r} is a no-go zone, skipped`);
        s.pending = s.dirt.filter(d => d.room === r && !d.clean).map(d => d.label);
        s.target = r;
        await drive(s, ctx, CENTRE[r]);
        await sweep(s, ctx, r);
        const n = await work(s, ctx);
        return done(s.status === 'paused' ? `paused in the ${r}` : `${r} clean${n ? ` (${n} patch${n === 1 ? '' : 'es'})` : ''}`);
      }
      case 'clean_everywhere': {
        const skip = normRoom(a.skip);
        const order = ['hallway', 'bedroom', 'kitchen', 'living room', 'bathroom'].filter(r => r !== skip && !s.avoid.includes(r));
        s.pending = order.flatMap(r => s.dirt.filter(d => d.room === r && !d.clean).map(d => d.label));
        s.target = order.join(', ');
        const n = await work(s, ctx);
        return done(s.status === 'paused' ? 'paused mid-clean' : `whole home cleaned, ${n} patches${skip ? `, skipped the ${skip}` : ''}${s.avoid.length ? `, avoided ${s.avoid.join(', ')}` : ''}`);
      }
      case 'clean_spot': {
        const d = matchDirt(s, String(a.spot || ''));
        if (!d) return done(`no dirty spot like "${a.spot}" left`);
        s.pending = [d.label];
        await work(s, ctx);
        return done(`cleaned the ${d.label}`);
      }
      case 'avoid_room': {
        const r = normRoom(a.room);
        if (r && !s.avoid.includes(r)) s.avoid.push(r);
        return done(r ? `${r} marked as a no-go zone` : 'unknown room');
      }
      case 'go_to_room': {
        const r = normRoom(a.room);
        if (!r) return done('unknown room');
        if (s.avoid.includes(r)) return done(`${r} is a no-go zone`);
        await drive(s, ctx, CENTRE[r]);
        s.status = 'idle';
        return done(`arrived in the ${r}`);
      }
      case 'move': {
        const dir = String(a.direction || 'forward').toLowerCase();
        const m = Math.max(0.1, Math.min(5, Number(a.meters) || 1));
        const turn = /left/.test(dir) ? -90 : /right/.test(dir) ? 90 : /back/.test(dir) ? 180 : 0;
        const h = (s.h + turn) * Math.PI / 180;
        const tx = clamp(s.x + Math.cos(h) * m * PX_PER_M, 42, 518), ty = clamp(s.y + Math.sin(h) * m * PX_PER_M, 42, 378);
        s.docked = false;
        await line(s, ctx, tx, ty, /back/.test(dir));
        s.status = 'idle';
        return done(`moved ${dir} ${fmt(m)} m`);
      }
      case 'set_suction': {
        const l = String(a.level || '').toLowerCase();
        s.suction = /quiet|low|soft|silent|eco|gentle|down/.test(l) ? 'quiet' : /max|high|strong|turbo|full|up|power/.test(l) ? 'max' : 'standard';
        return done(`suction ${s.suction}`);
      }
      case 'return_to_dock':
        await drive(s, ctx, DOCK); s.h = -90; s.status = 'docked'; s.pending = [];
        return done('back on the dock, charging');
      case 'pause':
        s.status = 'paused'; return done(s.pending.length ? `paused, ${s.pending.length} spot${s.pending.length === 1 ? '' : 's'} left` : 'stopped');
      case 'resume': {
        if (!s.pending.length) { s.status = s.status === 'docked' ? 'docked' : 'idle'; return done('nothing to resume'); }
        s.status = 'cleaning';
        const n = await work(s, ctx);
        return done(`resumed, cleaned ${n} more`);
      }
      case 'report_status': {
        const r = roomAt(s.x, s.y);
        const left = s.dirt.filter(d => !d.clean).length;
        const t = `battery ${s.battery}%, bin ${s.bin}% full, ${s.status === 'docked' ? 'on the dock' : `in the ${r}`}, ${left} dirty spot${left === 1 ? '' : 's'} left`;
        ctx.speak && ctx.speak(t);
        return done(t);
      }
      case 'empty_bin':
        await drive(s, ctx, DOCK); s.h = -90; s.status = 'docked';
        for (let i = 0; i < 5 && s.bin > 0; i++) { s.bin = Math.max(0, s.bin - 10); ctx.render(); await ctx.sleep(120); }
        s.bin = 0; return done('dust bin emptied');
      default: return `${c.tool}: no visual for this tool`;
    }
  },

  render(s) {
    const floors = ROOMS.map(r => {
      const [x, y, w, h] = RECT[r];
      const wood = r === 'bathroom' || r === 'kitchen' ? 'tile' : 'wood';
      return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="url(#${wood})"/>` +
        (s.avoid.includes(r) ? `<rect x="${x + 4}" y="${y + 4}" width="${w - 8}" height="${h - 8}" fill="url(#nogo)" stroke="#e5484d" stroke-dasharray="6 4" stroke-width="2"/>` : '') +
        `<text x="${x + 10}" y="${y + 20}" fill="#d9d4c7" font-size="12" font-weight="600">${cap(r)}${s.avoid.includes(r) ? ' · no-go' : ''}</text>`;
    }).join('');
    // walls with door gaps
    const walls = `<g stroke="#5a5c66" stroke-width="5" stroke-linecap="square">
  <rect x="30" y="30" width="500" height="360" fill="none"/>
  <line x1="200" y1="30" x2="200" y2="100"/><line x1="200" y1="140" x2="200" y2="310"/><line x1="200" y1="350" x2="200" y2="390"/>
  <line x1="30" y1="200" x2="200" y2="200"/>
  <line x1="200" y1="250" x2="270" y2="250"/><line x1="310" y1="250" x2="530" y2="250"/>
  <line x1="380" y1="250" x2="380" y2="310"/><line x1="380" y1="350" x2="380" y2="390"/></g>`;
    const furniture = `<g fill="#2a2620" stroke="#3b362e">
  <rect x="40" y="172" width="110" height="20" rx="3"/><rect x="95" y="110" width="44" height="34" rx="4"/>
  <rect x="300" y="60" width="130" height="16" rx="5"/><rect x="420" y="130" width="70" height="44" rx="4"/>
  <rect x="45" y="225" width="90" height="110" rx="6"/><rect x="470" y="345" width="50" height="34" rx="12"/></g>`;
    const dirt = s.dirt.map(d => d.clean
      ? `<circle cx="${d.x}" cy="${d.y}" r="12" fill="none" stroke="#46c37b" stroke-width="1.5" stroke-dasharray="3 3"/><text x="${d.x}" y="${d.y + 24}" fill="#7fe0a6" font-size="8" text-anchor="middle">cleaned</text>`
      : `<g opacity=".9">${[0, 1, 2, 3, 4, 5].map(i => `<circle cx="${d.x + Math.cos(i * 1.1) * (5 + i * 1.5)}" cy="${d.y + Math.sin(i * 1.7) * (4 + i)}" r="${2.5 + (i % 3)}" fill="#8a6a3a"/>`).join('')}</g><text x="${d.x}" y="${d.y + 24}" fill="#e0b97a" font-size="8" text-anchor="middle">${esc(d.label)}</text>`).join('');
    const trail = s.trail.length > 1 ? `<polyline points="${s.trail.map(p => p.join(',')).join(' ')}" fill="none" stroke="#7fb7ff" stroke-opacity=".25" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>` : '';
    const dock = `<rect x="${DOCK[0] - 22}" y="${DOCK[1] + 8}" width="44" height="12" rx="3" fill="#1b1c22" stroke="#555"/><circle cx="${DOCK[0]}" cy="${DOCK[1] + 14}" r="3" fill="${s.status === 'docked' ? '#46c37b' : '#666'}"/><text x="${DOCK[0] + 28}" y="${DOCK[1] + 18}" fill="#999" font-size="8">dock</text>`;
    const hr = s.h * Math.PI / 180;
    const glow = s.status === 'cleaning' ? '#ff4d4f' : s.status === 'paused' ? '#f5a524' : s.status === 'moving' ? '#7fb7ff' : '#46c37b';
    const robot = `<g><circle cx="${s.x}" cy="${s.y}" r="21" fill="${glow}" opacity=".18"/>
  <circle cx="${s.x}" cy="${s.y}" r="15" fill="#eceef2" stroke="#1b1c22" stroke-width="2"/>
  <circle cx="${s.x}" cy="${s.y}" r="6" fill="#2b2d35"/>
  <line x1="${s.x}" y1="${s.y}" x2="${(s.x + Math.cos(hr) * 15).toFixed(1)}" y2="${(s.y + Math.sin(hr) * 15).toFixed(1)}" stroke="#ff4d4f" stroke-width="3" stroke-linecap="round"/></g>`;
    const bar = (x, label, v, col) => `<text x="${x}" y="410" fill="#aaa" font-size="10">${label}</text><rect x="${x}" y="416" width="70" height="6" rx="3" fill="#2a2b33"/><rect x="${x}" y="416" width="${(70 * v / 100).toFixed(1)}" height="6" rx="3" fill="${col}"/><text x="${x + 76}" y="423" fill="#ddd" font-size="10" font-family="ui-monospace,monospace">${v}%</text>`;
    return `<svg viewBox="0 0 560 460" xmlns="http://www.w3.org/2000/svg" font-family="Inter,system-ui,sans-serif">
  <defs>
    <pattern id="wood" width="40" height="14" patternUnits="userSpaceOnUse"><rect width="40" height="14" fill="#2b241b"/><line x1="0" y1="13.5" x2="40" y2="13.5" stroke="#211b14"/><line x1="18" y1="0" x2="18" y2="14" stroke="#241e16"/></pattern>
    <pattern id="tile" width="20" height="20" patternUnits="userSpaceOnUse"><rect width="20" height="20" fill="#24262b"/><path d="M20 0H0V20" fill="none" stroke="#1b1c20"/></pattern>
    <pattern id="nogo" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="12" height="12" fill="#e5484d" opacity=".08"/><line x1="0" y1="0" x2="0" y2="12" stroke="#e5484d" stroke-opacity=".35" stroke-width="3"/></pattern>
  </defs>
  <rect width="560" height="460" fill="#0b0b0f"/>
  ${floors}${furniture}${trail}${dirt}${walls}${dock}${robot}
  ${bar(30, 'battery', s.battery, s.battery < 20 ? '#e5484d' : '#46c37b')}${bar(150, 'bin', s.bin, s.bin > 80 ? '#e5484d' : '#e0b97a')}
  <text x="280" y="410" fill="#aaa" font-size="10">suction</text><text x="280" y="425" fill="#fff" font-size="12" font-weight="600">${s.suction}</text>
  <text x="350" y="410" fill="#aaa" font-size="10">state</text><text x="350" y="425" fill="${glow}" font-size="12" font-weight="600">${s.status}</text>
  <text x="530" y="410" fill="#aaa" font-size="10" text-anchor="end">${s.target ? `task · ${esc(s.target.length > 22 ? s.target.slice(0, 21) + '…' : s.target)}` : ''}</text>
  <rect x="30" y="434" width="500" height="20" rx="10" fill="#15161b"/><text x="280" y="448" fill="#ddd" font-size="11" text-anchor="middle">${esc(s.note)}</text>
</svg>`;
  },

  examples: [
    {title: 'This, not that', sub: 'exclusion', q: 'Forget the bedroom, clean the kitchen'},
    {title: 'All but one', sub: 'optional enum arg', q: 'Vacuum everywhere except the bathroom'},
    {title: 'Point at a mess', sub: 'span copy', q: 'There is a coffee spill in the kitchen, clean it up'},
    {title: 'Go + spot', sub: '2 calls, in order', q: 'Drive to the kitchen, then clean the cereal crumbs'},
    {title: 'Drive it', sub: 'numbers, 2 moves', q: 'Move forward 2 meters and then go right 1.5 meters'},
    {title: 'Full power', sub: 'polarity, 2 calls', q: 'Turn the suction up to max and clean the hallway'},
    {title: 'Baby asleep', sub: 'enum from meaning', q: 'Be quiet, the baby is sleeping'},
    {title: 'No-go zone', sub: 'negation as a rule', q: 'Stay out of the bedroom from now on'},
    {title: 'Hold on', sub: 'trigger routes to pause', q: 'Stop!'},
    {title: 'Battery?', sub: 'status question', q: 'How much battery do you have left?'},
    {title: 'Head home', sub: 'no-arg call', q: 'Go back to your dock'},
    {title: 'Not a vacuum thing', sub: 'abstains', q: 'Order me a pizza'},
  ],
  help: 'Try: "clean the kitchen but not the bedroom", "vacuum everywhere except the bathroom", "max suction and clean the hallway".',
};

// ---- motion ----
async function line(s, ctx, tx, ty, reverse = false) {
  const dx = tx - s.x, dy = ty - s.y, dist = Math.hypot(dx, dy);
  if (dist < 1) return;
  const ang = Math.atan2(dy, dx) * 180 / Math.PI;
  s.h = reverse ? ang + 180 : ang;
  const prev = s.status;
  if (s.status !== 'cleaning') s.status = 'moving';
  const steps = Math.max(2, Math.ceil(dist / 16));
  for (let i = 1; i <= steps; i++) {
    if (s.status === 'paused') return;
    s.x = s.x + dx / steps * 1; s.y = s.y + dy / steps * 1;
    if (s.status === 'cleaning') { s.trail.push([Math.round(s.x), Math.round(s.y)]); if (s.trail.length > 400) s.trail.shift(); }
    if (i % 12 === 0) s.battery = Math.max(5, s.battery - 1);
    ctx.render && ctx.render();
    await ctx.sleep(35);
  }
  s.x = tx; s.y = ty;
  if (prev === 'cleaning') s.status = 'cleaning';
}

async function drive(s, ctx, [tx, ty]) {
  const from = roomAt(s.x, s.y), to = roomAt(tx, ty);
  const path = [];
  if (from !== to) {
    path.push(...CHAIN[from]);
    if (from !== 'hallway' && to !== 'hallway') path.push(HUB);
    else if (from === 'hallway' && to !== 'hallway' && Math.hypot(s.x - HUB[0], s.y - HUB[1]) > 30) path.push(HUB);
    path.push(...[...CHAIN[to]].reverse());
  }
  path.push([tx, ty]);
  for (const [x, y] of dedupe(path)) { if (s.status === 'paused') return; await line(s, ctx, x, y); }
}

async function sweep(s, ctx, r) {
  // a short zig-zag across the room so a room clean looks like cleaning
  const [x, y, w, h] = RECT[r];
  s.status = 'cleaning';
  const rows = 3;
  for (let i = 0; i < rows; i++) {
    const yy = y + 45 + i * (h - 70) / (rows - 1);
    await line(s, ctx, i % 2 ? x + 30 : x + w - 30, yy);
    if (s.status === 'paused') return;
  }
}

async function work(s, ctx) {
  let n = 0;
  s.status = 'cleaning';
  while (s.pending.length) {
    if (s.status === 'paused') return n;
    const d = s.dirt.find(p => p.label === s.pending[0]);
    if (d && !d.clean) {
      await drive(s, ctx, [d.x, d.y]);
      if (s.status === 'paused') return n;
      s.status = 'cleaning';
      // a small circle over the patch
      for (let k = 0; k < 8; k++) {
        const t = k / 8 * Math.PI * 2;
        s.x = d.x + Math.cos(t) * 10; s.y = d.y + Math.sin(t) * 10; s.h = t * 180 / Math.PI + 90;
        s.trail.push([Math.round(s.x), Math.round(s.y)]);
        ctx.render && ctx.render(); await ctx.sleep(s.suction === 'max' ? 30 : 50);
      }
      d.clean = true; n++;
      s.bin = Math.min(100, s.bin + (s.suction === 'max' ? 12 : 9));
      s.battery = Math.max(5, s.battery - (s.suction === 'max' ? 3 : s.suction === 'quiet' ? 1 : 2));
    }
    s.pending.shift();
  }
  s.status = 'idle';
  ctx.render && ctx.render();
  return n;
}

function matchDirt(s, q) {
  const words = q.toLowerCase().split(/\W+/).filter(w => w.length > 2 && !['the', 'and', 'from', 'that', 'there'].includes(w));
  let best = null, score = 0;
  for (const d of s.dirt) {
    if (d.clean) continue;
    const sc = words.filter(w => d.label.includes(w) || d.room.includes(w) || (w.startsWith('crumb') && d.label.includes('crumb'))).length;
    if (sc > score) { best = d; score = sc; }
  }
  if (best) return best;
  // nothing matched: the nearest dirty patch
  return s.dirt.filter(d => !d.clean).sort((a, b) => Math.hypot(a.x - s.x, a.y - s.y) - Math.hypot(b.x - s.x, b.y - s.y))[0] || null;
}

function roomAt(x, y) {
  for (const r of ['hallway', 'bathroom', 'kitchen', 'bedroom', 'living room']) {
    const [rx, ry, w, h] = RECT[r];
    if (x >= rx && x <= rx + w && y >= ry && y <= ry + h) return r;
  }
  return 'hallway';
}
function normRoom(r) {
  const t = String(r || '').toLowerCase();
  if (!t) return '';
  if (/living|lounge|sitting/.test(t)) return 'living room';
  if (/kitchen/.test(t)) return 'kitchen';
  if (/bed/.test(t)) return 'bedroom';
  if (/bath|toilet|shower/.test(t)) return 'bathroom';
  if (/hall|entr|corridor/.test(t)) return 'hallway';
  return '';
}
function dedupe(p) { return p.filter((q, i) => i === 0 || q[0] !== p[i - 1][0] || q[1] !== p[i - 1][1]); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
const fmt = n => (Number.isInteger(n) ? String(n) : n.toFixed(1));
const cap = t => t[0].toUpperCase() + t.slice(1);
const esc = t => String(t ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
