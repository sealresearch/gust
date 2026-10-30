// Generates deterministic gray frame sequences for checking the C port of Squint against the JavaScript reference.
//   node make_sequences.mjs [--out DIR] [--only name1,name2] [--list]
// Output per sequence: <name>.gray (frames concatenated, width*height bytes each) and <name>.json (metadata + truth).
// Everything is seeded (Rng in render.mjs); no Math.random, no Date. See README.md for the formats.
import {createHash} from 'node:crypto';
import {closeSync, mkdirSync, openSync, readFileSync, writeFileSync, writeSync} from 'node:fs';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {Rng, projectPoint, renderJunk, renderScene, samplePoses} from './render.mjs';

try {
  os.setPriority(0, os.constants.priority.PRIORITY_LOW);
} catch {
  /* best effort */
}

const arg = (n, f) => {
  const i = process.argv.indexOf(n);
  return i < 0 ? f : process.argv[i + 1];
};
const REF = (process.env.SQUINT_REF ?? 'C:/Users/rickh/Documents/sandbox/squint-release/models/squint').replace(/\\/g, '/');
const OUT = fileURLToPath(arg('--out', null) ? new URL('file:///' + arg('--out').replace(/\\/g, '/') + '/') : new URL('../../../out/squint-seq/', import.meta.url));
const only = arg('--only', null)?.split(',');

// ---------------------------------------------------------------------------------------------------------------
// QR module grids: the reference repo's clean corpus (24 codes per version 1..10 and level L/M/Q/H, byte/numeric/
// alphanumeric, all masks), its demo codes, and a few Segno pairs for the switch sequences (tools/switch_codes.json).
const FIXTURES = new URL('../tests/fixtures/', import.meta.url);
const corpus = JSON.parse(readFileSync(new URL('clean-corpus.json', FIXTURES), 'utf8'));
const demo = JSON.parse(readFileSync(new URL('demo-codes.json', FIXTURES), 'utf8'));
const switchCodes = JSON.parse(readFileSync(new URL('./switch_codes.json', import.meta.url), 'utf8'));
const toBytes = hex => Buffer.from(hex, 'hex');
const fromGrid = (id, version, ecc, payloadHex, grid) => {
  const n = 17 + 4 * version;
  if (grid.length !== n * n) throw new Error('bad grid ' + id);
  return {id, version, ecc, n, grid: Uint8Array.from(grid, c => (c === '1' ? 1 : 0)), payloadHex};
};
const corp = (version, ecc, k) => {
  const list = corpus.filter(c => c.version === version && c.ecc === ecc);
  const c = list[((k % list.length) + list.length) % list.length];
  return fromGrid(`corpus:${version}${ecc}#${k}`, version, ecc, c.payloadHex, c.grid);
};
const demoCode = id => {
  const c = demo.find(x => x.id === id);
  const flat = c.modules.flat().join('');
  return fromGrid('demo:' + id, c.version, c.ecc, Buffer.from(c.payload, 'utf8').toString('hex'), flat);
};
const sw = id => {
  const c = switchCodes.find(x => x.id === id);
  return fromGrid('switch:' + id, c.version, c.ecc, c.payloadHex, c.grid);
};

// ---------------------------------------------------------------------------------------------------------------
// Pose helpers. A pose is {cx, cy, m (px per module), rot, yaw, pitch}; time t is in frames.
const still = (cx, cy, m, extra = {}) => () => ({cx, cy, m, rot: 0, yaw: 0, pitch: 0, ...extra});
/** Hand tremor like site SquintDemo.astro: a few slow sways plus a faster tremor. a = amplitude in px. */
const hand = (cx, cy, m, a, {phase = 0, rotAmp = 0.1, jitter = null, base = null} = {}) => t => {
  const s = t / 30 + phase;
  const p = base ? base(t) : {cx, cy, m, rot: 0, yaw: 0, pitch: 0};
  let dx = a * (Math.sin(2.1 * s) + 0.55 * Math.sin(5.3 * s + 1.3) + 0.25 * Math.sin(13.7 * s));
  let dy = a * (0.8 * Math.sin(1.7 * s + 0.4) + 0.5 * Math.sin(6.1 * s + 2.1) + 0.2 * Math.sin(11.9 * s + 0.7));
  const r = rotAmp * (a / 72) * 1.2 * Math.sin(1.3 * s + 0.9);
  if (jitter) {
    const j = jitter(t);
    dx += j[0];
    dy += j[1];
  }
  return {...p, cx: p.cx + dx, cy: p.cy + dy, rot: (p.rot ?? 0) + r};
};
/** Triangle wave of half-range A through 0 with slope 1. */
const fold = (s, A) => {
  const p = (((s + A) % (4 * A)) + 4 * A) % (4 * A);
  return (p < 2 * A ? p : 4 * A - p) - A;
};
/** Straight back-and-forth motion at `speed` px per frame along `angle` (radians), half-range A px. */
const straight = (cx, cy, m, speed, angle, A = 150) => t => {
  // still for 6 frames (so that the code can be acquired), then a speed ramp over 8 frames, then constant speed
  const g = t < 6 ? 0 : t < 14 ? ((t - 6) * (t - 6)) / 16 : 4 + (t - 14);
  const d = fold(g * speed, A);
  return {cx: cx + d * Math.cos(angle), cy: cy + d * Math.sin(angle), m, rot: 0, yaw: 0, pitch: 0};
};
/** Smoothed random jitter (px) per frame from the sequence rng, linearly interpolated. */
const makeJitter = (rng, frames, amp) => {
  const xs = [0], ys = [0];
  let vx = 0, vy = 0;
  for (let i = 1; i < frames + 2; i++) {
    vx = 0.8 * vx + rng.gauss() * amp * 0.6;
    vy = 0.8 * vy + rng.gauss() * amp * 0.6;
    xs.push(xs[i - 1] * 0.9 + vx);
    ys.push(ys[i - 1] * 0.9 + vy);
  }
  const at = (a, t) => {
    const k = Math.max(0, Math.min(a.length - 2, Math.floor(t)));
    return a[k] + (a[k + 1] - a[k]) * Math.max(0, Math.min(1, t - k));
  };
  return t => [at(xs, t), at(ys, t)];
};
const lerp = (a, b, u) => a + (b - a) * u;
const clamp01 = u => Math.max(0, Math.min(1, u));
const deg = d => (d * Math.PI) / 180;

/** Copy of a code with a fraction of modules flipped (finder patterns and separators kept). */
function damage(code, frac, rng, id) {
  const n = code.n, grid = code.grid.slice();
  const inFinder = (x, y) => (x < 8 && y < 8) || (x >= n - 8 && y < 8) || (x < 8 && y >= n - 8);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (!inFinder(x, y) && rng.float() < frac) grid[y * n + x] ^= 1;
  return {...code, id, grid};
}

// ---------------------------------------------------------------------------------------------------------------
// Sequence table. frame(i, ctx) returns either {junk: {...}} or a code frame:
//   {code, poseAt, t?, exposure?, truthCode? (default code, null = no code), ...render fields (defocus, noise, ...)}.
const sequences = [];
const add = def => sequences.push(def);
const W = 1280, H = 720;

// ----- sharp and static
add({
  name: 'static_sharp_v3', w: W, h: H, frames: 24, seed: 101,
  description: 'Sharp static version 3 code, light sensor noise.',
  setup: () => ({code: demoCode('website')}),
  frame: (i, c) => ({code: c.code, poseAt: still(640, 360, 7), exposure: 0, defocus: 0.5, noise: 2}),
});

// ----- hand-shake motion blur, straight paths (px per frame; exposure 0.85 frame)
for (const [tag, speed, angle, k] of [['light', 6, 20, 0], ['medium', 14, 0, 1], ['heavy', 28, 70, 2], ['extreme', 50, 135, 3]])
  add({
    name: 'straight_' + tag, w: W, h: H, frames: 36, seed: 110 + k,
    description: `Straight back-and-forth motion at ${speed} px/frame (blur about ${(speed * 0.85).toFixed(0)} px), module 5 px, version ${[3, 4, 5, 3][k]}.`,
    setup: () => ({code: [demoCode('website'), corp(4, 'M', 3), demoCode('digits'), corp(3, 'Q', 5)][k]}),
    frame: (i, c) => ({code: c.code, poseAt: straight(640, 360, [6, 5, 5.5, 6][k], speed, deg(angle)), exposure: 0.85, defocus: 0.7, noise: 3}),
  });

// ----- hand-shake, curved paths with rotation
for (const [tag, amp, k] of [['mild', 18, 0], ['hard', 85, 1]])
  add({
    name: 'curved_' + tag, w: W, h: H, frames: 48, seed: 120 + k,
    description: `Hand-tremor path (sum of sines, like the post demo), amplitude ${amp} px, with rotation, version 5 Q.`,
    setup: () => ({code: demoCode('digits')}),
    frame: (i, c) => ({code: c.code, poseAt: hand(640, 360, 5.5, amp, {phase: k}), exposure: 1, defocus: 0.75, noise: 3}),
  });

// ----- code moving into view from the edge
add({
  name: 'enter_from_left', w: W, h: H, frames: 48, seed: 130,
  description: 'Version 4 code slides in from beyond the left edge (about 24 px/frame, motion blur), then settles with tremor.',
  setup: () => ({code: corp(4, 'M', 7)}),
  frame: (i, c) => {
    const base = t => ({cx: lerp(-190, 640, clamp01(t / 32)) , cy: 360 + 30 * Math.sin(t / 9), m: 5.5, rot: 0, yaw: 0, pitch: 0});
    return {code: c.code, poseAt: hand(0, 0, 5.5, 8, {base, phase: 0.5}), exposure: 0.9, defocus: 0.7, noise: 3};
  },
});
add({
  name: 'enter_corner_640x480', w: 640, h: 480, frames: 40, seed: 131,
  description: 'Version 3 code enters diagonally from the bottom-right corner with blur.',
  setup: () => ({code: demoCode('website')}),
  frame: (i, c) => {
    const base = t => ({cx: lerp(900, 330, clamp01(t / 26)), cy: lerp(700, 250, clamp01(t / 26)), m: 4.5, rot: deg(-6) * (1 - clamp01(t / 26)), yaw: 0, pitch: 0});
    return {code: c.code, poseAt: hand(0, 0, 4.5, 6, {base}), exposure: 0.9, defocus: 0.6, noise: 3};
  },
});

// ----- distant small and close-up large
add({
  name: 'distant_small', w: W, h: H, frames: 48, seed: 140,
  description: 'Version 3 code about 75 px wide (module 2.6 px) with mild shake.',
  setup: () => ({code: corp(3, 'M', 9)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(700, 330, 2.6, 5, {rotAmp: 0.05}), exposure: 0.8, defocus: 0.6, noise: 2.5}),
});
add({
  name: 'closeup_large', w: W, h: H, frames: 40, seed: 141,
  description: 'Version 2 code filling most of the frame (module 14 to 24 px), breathing scale, slight shake.',
  setup: () => ({code: corp(2, 'Q', 4)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(640, 360, 19, 14, {base: t => ({cx: 640, cy: 360, m: 19 + 5 * Math.sin(t / 7), rot: 0, yaw: 0, pitch: 0})}), exposure: 0.8, defocus: 1.0, noise: 2.5}),
});

// ----- lighting, glare, sensor noise, gamma, perspective
add({
  name: 'lighting_gradient', w: W, h: H, frames: 48, seed: 150,
  description: 'Version 5 code under a lighting gradient that grows from 0 to 1.5 (with vignetting and reduced gain) and rotates, with mild shake.',
  setup: () => ({code: corp(5, 'M', 2)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(640, 360, 5, 10), exposure: 0.8, defocus: 0.6, noise: 3, grad: {angle: i / 10, amp: 1.5 * clamp01(i / 30)}, vignette: 0.5, gain: 0.8}),
});
add({
  name: 'glare_sweep', w: W, h: H, frames: 48, seed: 151,
  description: 'A bright glare blob sweeps across the code (partly washing it out), mild shake.',
  setup: () => ({code: corp(4, 'Q', 6)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(640, 360, 5.5, 10), exposure: 0.8, defocus: 0.6, noise: 3, gain: 0.9, glare: {x: lerp(380, 900, i / 47), y: 340 + 40 * Math.sin(i / 6), r: 70, amp: 2.2 * Math.sin((Math.PI * i) / 47) ** 2}}),
});
add({
  name: 'noise_then_gamma_1024x576', w: 1024, h: 576, frames: 72, seed: 152, resets: [36],
  description: 'Frames 0-35: version 4 H code, sensor noise sigma rising from 2 to 44 gray levels. Frames 36-71 (after reset): version 3 L code while gamma sweeps 0.45 to 2.4 and contrast drops to a washed-out look in the middle.',
  setup: () => ({a: corp(4, 'H', 8), b: corp(3, 'L', 11)}),
  frame: (i, c) => {
    if (i < 36) return {code: c.a, poseAt: hand(512, 288, 5, 8), exposure: 0.8, defocus: 0.6, noise: lerp(2, 44, i / 35)};
    const u = (i - 36) / 35, bump = Math.sin(Math.PI * u) ** 2;
    return {code: c.b, poseAt: hand(512, 288, 6, 8), exposure: 0.8, defocus: 0.7, noise: 3, gamma: lerp(0.45, 2.4, u), gain: 1 - 0.55 * bump, offset: 40 * bump};
  },
});
add({
  name: 'tilt_ramp_1024x576', w: 1024, h: 576, frames: 56, seed: 154,
  description: 'Version 5 code tilting from flat to about 65 degrees pitch and -50 degrees yaw, slow shake.',
  setup: () => ({code: corp(5, 'Q', 3)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(512, 288, 5.5, 6, {base: t => ({cx: 512, cy: 288, m: 5.5, rot: deg(8), pitch: deg(65 * clamp01(t / 50)), yaw: deg(-50 * clamp01(t / 50))})}), exposure: 0.8, defocus: 0.7, noise: 3}),
});
add({
  name: 'tilt_shake_640x480', w: 640, h: 480, frames: 40, seed: 155,
  description: 'Version 4 code tilted 35 degrees pitch / 25 degrees yaw, with heavy hand shake.',
  setup: () => ({code: corp(4, 'M', 12)}),
  frame: (i, c) => ({code: c.code, poseAt: hand(0, 0, 4.2, 40, {base: () => ({cx: 320, cy: 240, m: 4.2, rot: deg(-12), pitch: deg(35), yaw: deg(25)})}), exposure: 0.9, defocus: 0.8, noise: 3}),
});
add({
  name: 'screen_glow_640x480', w: 640, h: 480, frames: 32, seed: 156,
  description: 'Phone-screen style: bright white modules area, near-black code, dark surroundings, shake.',
  setup: () => ({code: demoCode('hello')}),
  frame: (i, c) => ({code: c.code, poseAt: hand(320, 240, 8, 28), exposure: 0.8, defocus: 1.0, noise: 2, palette: {dark: 6, paper: 248, desk: 22}, vignette: 0.2}),
});

// ----- beyond readable: blur that grows until no reader can read it, then recovers; focus blur likewise
add({
  name: 'blur_beyond_1024x576', w: 1024, h: 576, frames: 56, seed: 160,
  description: 'Version 4 code, 6 still frames, then speed ramps up to 130 px/frame (blur over 100 px, unreadable) and comes back down to a stop.',
  setup: () => ({code: corp(4, 'M', 19)}),
  frame: (i, c) => {
    // speed envelope: 0 for 6 frames, ramp up to 130 px/frame by frame 28, ramp down to 0 by frame 50
    const sp = t => (t < 6 ? 0 : t < 28 ? 130 * ((t - 6) / 22) : t < 50 ? 130 * (1 - (t - 28) / 22) : 0);
    const poseAt = t => {
      let x = 0;
      for (let k = 0; k < Math.floor(t); k++) x += sp(k + 0.5);
      x += sp(Math.floor(t) + 0.5) * (t - Math.floor(t));
      return {cx: 512 + 220 * Math.sin(x / 220), cy: 288, m: 5, rot: 0, yaw: 0, pitch: 0};
    };
    return {code: c.code, poseAt, exposure: 0.9, defocus: 0.7, noise: 3};
  },
});
add({
  name: 'defocus_ramp_1024x576', w: 1024, h: 576, frames: 56, seed: 161,
  description: 'Version 3 code, camera defocus sigma going from 0.5 to 14 px and back, static code.',
  setup: () => ({code: corp(3, 'Q', 13)}),
  frame: (i, c) => ({code: c.code, poseAt: still(512, 288, 6), exposure: 0, defocus: 0.5 + 13.5 * Math.sin((Math.PI * i) / 55) ** 2, noise: 3}),
});

// ----- versions 1..10 in pairs (error correction level cycles), each code: sharp start then moderate shake, then two
// empty frames and reset(). Tracking covers versions 1 to 6 only, so 7 to 10 must be read afresh every frame.
const ECC = ['L', 'M', 'Q', 'H'];
for (let v = 1; v <= 10; v += 2) {
  const big = v === 9, w = big ? 1280 : 640, h = big ? 720 : 480, side = big ? 420 : 250;
  add({
    name: `versions_${String(v).padStart(2, '0')}_${String(v + 1).padStart(2, '0')}_${w}x${h}`, w, h, frames: 32, seed: 200 + v, resets: [16],
    description: `Version ${v} (${ECC[(v - 1) % 4]}) in frames 0-13, version ${v + 1} (${ECC[v % 4]}) in frames 16-31; each starts with 4 sharp frames and then shakes; 2 empty frames and reset() between.${big ? ' At 1280x720, where untracked acquisition alternates the lighting-normalized view.' : ''}`,
    setup: () => ({codes: [corp(v, ECC[(v - 1) % 4], v), corp(v + 1, ECC[v % 4], v + 1)]}),
    frame: (i, c) => {
      if (i === 14 || i === 15) return {code: null, noise: 3};
      const k = i < 14 ? 0 : 1, j = i - 16 * k, code = c.codes[k], m = side / code.n;
      return {code, poseAt: hand(w / 2, h / 2, m, j < 4 ? 0 : 26 * (side / 250), {phase: k}), exposure: j < 4 ? 0 : 0.85, defocus: 0.6, noise: 3};
    },
  });
}
add({
  name: 'ecc_levels_v5_640x480', w: 640, h: 480, frames: 80, seed: 220, resets: [20, 40, 60],
  description: 'Version 5 with error correction L, M, Q, H in blocks of 20 frames, reset() between blocks, moderate shake.',
  setup: () => ({codes: ECC.map(e => corp(5, e, 15))}),
  frame: (i, c) => ({code: c.codes[Math.floor(i / 20)], poseAt: hand(320, 240, 5, 30, {phase: Math.floor(i / 20)}), exposure: 0.85, defocus: 0.7, noise: 3}),
});

// ----- payload switches in the middle of motion
add({
  name: 'switch_onechar_v3_640x480', w: 640, h: 480, frames: 48, seed: 300,
  description: 'Version 3 M: the displayed payload changes by one digit at frame 20 while the code is shaking and blurred.',
  setup: () => ({a: sw('sw_a3'), b: sw('sw_b3')}),
  frame: (i, c) => ({code: i < 20 ? c.a : c.b, poseAt: hand(320, 240, 6, 80), exposure: 1, defocus: 0.9, noise: 3}),
});
add({
  name: 'switch_numeric_v5_640x480', w: 640, h: 480, frames: 48, seed: 301,
  description: 'Version 5 Q numeric: last digit changes at frame 18 and again at 34 (blurred hand motion).',
  setup: () => ({a: sw('sw_n5a'), b: sw('sw_n5b')}),
  frame: (i, c) => ({code: i < 18 || i >= 34 ? c.a : c.b, poseAt: hand(320, 240, 4.8, 85, {phase: 1}), exposure: 1, defocus: 0.9, noise: 3}),
});
add({
  name: 'switch_version_2to4_640x480', w: 640, h: 480, frames: 48, seed: 302,
  description: 'Version 2 code switches to a version 4 code of about the same size at frame 22 (different payload, different version), then to version 6 at 36.',
  setup: () => ({a: sw('sw_c2'), b: sw('sw_c4'), c6: sw('sw_c6')}),
  frame: (i, c) => {
    const code = i < 22 ? c.a : i < 36 ? c.b : c.c6;
    return {code, poseAt: hand(320, 240, 220 / code.n, 70, {phase: 2}), exposure: 1, defocus: 0.9, noise: 3};
  },
});
add({
  name: 'switch_toggle_corpus_640x480', w: 640, h: 480, frames: 44, seed: 303,
  description: 'Two unrelated version 4 M codes alternate every 7 frames under moderate shake.',
  setup: () => ({a: corp(4, 'M', 1), b: corp(4, 'M', 2)}),
  frame: (i, c) => ({code: Math.floor(i / 7) % 2 ? c.b : c.a, poseAt: hand(320, 240, 5.5, 60), exposure: 1, defocus: 0.8, noise: 3}),
});
add({
  name: 'switch_static_sharp_640x480', w: 640, h: 480, frames: 24, seed: 304,
  description: 'Static sharp version 3 code switching payload (one digit) every 6 frames: the easy case that must not return stale data.',
  setup: () => ({a: sw('sw_a3'), b: sw('sw_b3')}),
  frame: (i, c) => ({code: Math.floor(i / 6) % 2 ? c.b : c.a, poseAt: still(320, 240, 7), exposure: 0, defocus: 0.5, noise: 2}),
});

// ----- vanish and reappear, junk frames, damage, occlusion
add({
  name: 'vanish_reappear_640x480', w: 640, h: 480, frames: 120, seed: 400,
  description: 'Version 4 code shakes for 24 frames, vanishes for 12, reappears elsewhere for 24, vanishes for 40 (more than 1.2 s), reappears, then a different code at the old spot.',
  setup: () => ({code: corp(4, 'Q', 9), other: corp(4, 'Q', 10)}),
  frame: (i, c) => {
    const here = hand(320, 240, 5.5, 24), there = hand(210, 300, 5.5, 24, {phase: 3});
    if (i < 24) return {code: c.code, poseAt: here, exposure: 0.85, defocus: 0.7, noise: 3};
    if (i < 36) return {code: null, noise: 3};
    if (i < 60) return {code: c.code, poseAt: there, exposure: 0.85, defocus: 0.7, noise: 3};
    if (i < 100) return {code: null, noise: 3};
    if (i < 108) return {code: c.code, poseAt: there, exposure: 0.85, defocus: 0.7, noise: 3};
    return {code: c.other, poseAt: there, exposure: 0.85, defocus: 0.7, noise: 3};
  },
});
add({
  name: 'vanish_long_gap_640x480', w: 640, h: 480, frames: 135, seed: 401,
  description: 'Static code for 20 frames, then 100 frames (3.3 s) of empty desk, then the same code returns at the same spot (state expiry after 3 s).',
  setup: () => ({code: corp(3, 'M', 6)}),
  frame: (i, c) => (i >= 20 && i < 120 ? {code: null, noise: 2.5} : {code: c.code, poseAt: hand(320, 240, 6, 6), exposure: 0.6, defocus: 0.6, noise: 2.5}),
});
add({
  name: 'junk_interleave_640x480', w: 640, h: 480, frames: 96, seed: 410,
  description: 'Shaking version 4 code with junk frames every few frames: blank 255/127/0, noise, stripes, gradient, checker, clutter, a forged code (half the modules random).',
  setup: (rng) => ({code: corp(4, 'M', 14), fake: damage(corp(4, 'M', 14), 0.5, rng, 'fake')}),
  frame: (i, c) => {
    const junk = [
      {kind: 'blank', value: 255}, {kind: 'blank', value: 127}, {kind: 'blank', value: 0}, {kind: 'noise'},
      {kind: 'stripes', period: 8}, {kind: 'stripes', period: 3, vertical: false}, {kind: 'gradient'}, {kind: 'gradient', vertical: true, lo: 40, hi: 200},
      {kind: 'checker', period: 6}, {kind: 'clutter'}, {kind: 'fake'},
    ];
    if (i % 4 === 3) {
      const j = junk[Math.floor(i / 4) % junk.length];
      if (j.kind === 'fake') return {code: c.fake, truthCode: null, poseAt: hand(320, 240, 5.5, 24), exposure: 0.85, defocus: 0.7, noise: 3};
      return {junk: j};
    }
    return {code: c.code, poseAt: hand(320, 240, 5.5, 24), exposure: 0.85, defocus: 0.7, noise: 3};
  },
});
add({
  name: 'damage_ramp_640x480', w: 640, h: 480, frames: 49, seed: 420,
  description: 'Static sharp version 5 Q code with an increasing fraction of random modules flipped (0, 1, 2, 3 percent are still the code; 15, 25, 40 percent are not).',
  setup: (rng) => {
    const code = corp(5, 'Q', 21), levels = [0, 0.01, 0.02, 0.03, 0.15, 0.25, 0.4];
    return {levels, variants: levels.map((f, k) => (f ? damage(code, f, rng, `flip${f}`) : code)), code};
  },
  frame: (i, c) => {
    const k = Math.floor(i / 7);
    return {code: c.variants[k], truthCode: c.levels[k] <= 0.03 ? c.code : null, poseAt: still(320, 240, 6), exposure: 0, defocus: 0.5, noise: 2};
  },
});
add({
  name: 'occlusion_640x480', w: 640, h: 480, frames: 48, seed: 430,
  description: 'Version 4 M code with a finger-coloured block sweeping across it (up to about 40 percent covered), mild shake.',
  setup: () => ({code: corp(4, 'M', 17)}),
  frame: (i, c) => {
    const x = lerp(160, 560, i / 47);
    return {code: c.code, poseAt: hand(320, 240, 5.5, 14), exposure: 0.8, defocus: 0.7, noise: 3, occluders: [{x0: x - 60, x1: x + 60, y0: 140 + 40 * Math.sin(i / 5), y1: 360, v: 150}]};
  },
});

// ----- reset in the middle, seed twins, timing
add({
  name: 'reset_mid_640x480', w: 640, h: 480, frames: 72, seed: 500, resets: [24, 25, 50],
  description: 'Shaking version 5 code with reset() before frames 24, 25 and 50.',
  setup: () => ({code: demoCode('digits')}),
  frame: (i, c) => ({code: c.code, poseAt: hand(320, 240, 5, 30), exposure: 0.85, defocus: 0.7, noise: 3}),
});
for (const tag of ['a', 'b'])
  add({
    name: `seed_twin_${tag}_640x480`, w: 640, h: 480, frames: 40, seed: tag === 'a' ? 600 : 601,
    description: 'Identical parameters in both twins; only the seed (sensor noise and hand jitter) differs.',
    setup: rng => ({code: corp(4, 'M', 20), jitter: makeJitter(rng, 40, 14)}),
    frame: (i, c) => ({code: c.code, poseAt: hand(320, 240, 5.5, 20, {jitter: c.jitter}), exposure: 0.85, defocus: 0.7, noise: 4}),
  });
add({
  name: 'timing_irregular_640x480', w: 640, h: 480, frames: 64, seed: 510,
  description: 'Shaking code with irregular frame times (+-9 ms jitter, dropped frames) and a 3.4 s pause in the clock before frame 40.',
  setup: rng => {
    const times = [];
    let t = 0;
    for (let i = 0; i < 64; i++) {
      t += 33.3333 + Math.round((rng.float() - 0.5) * 18 * 1000) / 1000 + (i % 13 === 12 ? 33.3333 : 0) + (i === 40 ? 3400 : 0);
      times.push(Math.round(t * 1e4) / 1e4);
    }
    return {code: corp(4, 'L', 5), times};
  },
  times: c => c.times,
  frame: (i, c) => ({code: c.code, poseAt: hand(320, 240, 5.5, 22), exposure: 0.85, defocus: 0.7, noise: 3}),
});
add({
  name: 'clock_restart_640x480', w: 640, h: 480, frames: 56, seed: 511,
  description: 'The frame clock jumps back to 0 before frame 28 without a reset() (the tracker restarts itself when time goes backward).',
  setup: () => ({code: corp(3, 'H', 2)}),
  times: () => Array.from({length: 56}, (_, i) => (i < 28 ? i * 33.3333 : (i - 28) * 33.3333)),
  frame: (i, c) => ({code: c.code, poseAt: hand(320, 240, 6, 20), exposure: 0.85, defocus: 0.7, noise: 3}),
});

// ----- other frame sizes, legal and illegal
const small = (name, w, h, frames, seed, description, version = 3, m = 5, amp = 10) =>
  add({
    name, w, h, frames, seed, description,
    setup: () => ({code: corp(version, 'M', seed % 24)}),
    frame: (i, c) => ({code: c.code, poseAt: hand(w / 2, h / 2, m, amp), exposure: 0.8, defocus: 0.6, noise: 3}),
  });
small('size_800x600', 800, 600, 24, 700, 'Legal size, height not a multiple of 16 (no tracking, acquisition only).', 3, 6, 12);
small('size_320x240', 320, 240, 24, 701, 'Small legal size.', 2, 4, 6);
small('size_1000x562', 1000, 562, 24, 702, 'Legal size, both sides not multiples of 16.', 4, 5, 12);
small('size_64x64_min', 64, 64, 16, 703, 'Smallest legal size with a version 1 code (module 2 px).', 1, 2, 1.5);
small('size_640x479_oddheight', 640, 479, 16, 704, 'Odd height: acquisition accepts it, the tracker does not (height % 16).', 3, 5, 8);
small('size_odd_641x480', 641, 480, 16, 705, 'ILLEGAL: odd width (refusal path).', 3, 5, 8);
small('size_big_1920x1080', 1920, 1080, 6, 706, 'ILLEGAL: larger than 1280x720.', 4, 8, 10);
small('size_narrow_62x64', 62, 64, 8, 708, 'ILLEGAL: narrower than 64.', 1, 1.5, 1);

// ---------------------------------------------------------------------------------------------------------------
// Rendering and writing.
function visibleFraction(poseAt, t, code, w, h) {
  const pose = poseAt(t), N = 10;
  let inside = 0;
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const X = ((i + 0.5) / N - 0.5) * code.n * pose.m, Y = ((j + 0.5) / N - 0.5) * code.n * pose.m;
      const [sx, sy] = projectPoint(pose, X, Y);
      if (sx >= 0 && sx < w && sy >= 0 && sy < h) inside++;
    }
  return inside / (N * N);
}

mkdirSync(OUT, {recursive: true});
if (process.argv.includes('--list')) {
  for (const s of sequences) console.log(s.name, s.w + 'x' + s.h, s.frames);
  process.exit(0);
}
const manifest = [];
for (const seq of sequences) {
  if (only && !only.includes(seq.name)) continue;
  const rng = new Rng(seq.seed);
  const ctx = seq.setup ? seq.setup(rng) : {};
  const {w, h, frames} = seq;
  const out = new Uint8Array(w * h);
  const fd = openSync(OUT + seq.name + '.gray', 'w');
  const hash = createHash('sha256');
  const truth = [], visible = [], kinds = [];
  const codesUsed = new Map();
  for (let i = 0; i < frames; i++) {
    const f = seq.frame(i, ctx);
    if (f.junk) {
      renderJunk(out, w, h, f.junk, rng);
      truth.push(null); visible.push(0); kinds.push('junk:' + f.junk.kind);
    } else {
      const spec = {...f};
      let vis = 0;
      if (f.code) {
        spec.poses = samplePoses(f.poseAt, f.t ?? i, f.exposure ?? 0, f.code.n, f.pm ?? 6);
        vis = visibleFraction(f.poseAt, f.t ?? i, f.code, w, h);
        codesUsed.set(f.code.id, f.code);
      }
      renderScene(out, w, h, spec, rng);
      const tc = 'truthCode' in f ? f.truthCode : f.code;
      truth.push(tc && vis >= 0.25 ? tc.payloadHex : null);
      visible.push(Math.round(vis * 100) / 100);
      kinds.push(f.code ? (tc ? 'code' : 'forged') : 'empty');
    }
    writeSync(fd, out);
    hash.update(out);
  }
  closeSync(fd);
  const times = seq.times ? seq.times(ctx) : Array.from({length: frames}, (_, i) => i * 33.3333);
  const meta = {
    name: seq.name, width: w, height: h, frames, times, resets: seq.resets ?? [], truth,
    seed: seq.seed, description: seq.description, kinds, visible,
    codes: [...codesUsed.values()].map(c => ({id: c.id, version: c.version, ecc: c.ecc, payloadHex: c.payloadHex})),
    sha256: hash.digest('hex'),
  };
  writeFileSync(OUT + seq.name + '.json', JSON.stringify(meta));
  manifest.push({name: seq.name, width: w, height: h, frames, sha256: meta.sha256});
  console.log(seq.name.padEnd(34), `${w}x${h}`.padEnd(10), String(frames).padStart(4), 'frames', meta.sha256.slice(0, 12));
}
if (!only) {
  writeFileSync(OUT + 'manifest.json', JSON.stringify(manifest, null, 1));
  console.log('total frames', manifest.reduce((a, m) => a + m.frames, 0));
}
