// Deterministic gray-frame renderer for the Squint test sequences. Pure JS, no canvas, no Math.random, no Date.
// A frame is: a scene (desk, a sheet of paper with a QR code on it, optional occluders), rendered by inverse mapping
// with supersampling and motion blur (the average of several poses across the exposure), then defocus, lighting
// gradient, glare, vignette, gamma and sensor noise. Same idea as site SquintDemo.astro (poses averaged over a
// 1/30 s exposure), but in plain arrays so that the same bytes come out every run.

/** xoshiro128** seeded with splitmix32. */
export class Rng {
  constructor(seed) {
    let s = seed >>> 0;
    const split = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = split();
    this.b = split();
    this.c = split();
    this.d = split();
  }
  /** Next uint32. */
  u32() {
    const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;
    const result = Math.imul(rotl(Math.imul(this.b, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.b << 9) >>> 0;
    this.c = (this.c ^ this.a) >>> 0;
    this.d = (this.d ^ this.b) >>> 0;
    this.b = (this.b ^ this.c) >>> 0;
    this.a = (this.a ^ this.d) >>> 0;
    this.c = (this.c ^ t) >>> 0;
    this.d = rotl(this.d, 11);
    return result;
  }
  /** Uniform [0, 1). */
  float() {
    return this.u32() / 4294967296;
  }
  /** Integer in [0, n). */
  int(n) {
    return Math.floor(this.float() * n);
  }
  /** Approximately normal (sum of 4 uniform bytes), mean 0, sd 1. */
  gauss() {
    const r = this.u32();
    return ((r & 255) + ((r >>> 8) & 255) + ((r >>> 16) & 255) + (r >>> 24) - 510) / 147.8;
  }
}

const DEFAULT_PALETTE = {dark: 22, paper: 232, desk: 112};

function rotationMatrix(rot, yaw, pitch) {
  // R = Rx(pitch) * Ry(yaw) * Rz(rot), row-major 3x3.
  const cz = Math.cos(rot), sz = Math.sin(rot);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cx = Math.cos(pitch), sx = Math.sin(pitch);
  const Rz = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
  const Ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
  const Rx = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
  const mul = (A, B) => {
    const C = new Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return C;
  };
  return mul(Rx, mul(Ry, Rz));
}

const FOCAL = 1000;

/** Screen position of a plane point (X, Y in pixels from the code centre) for a pose. */
export function projectPoint(pose, X, Y) {
  const R = rotationMatrix(pose.rot ?? 0, pose.yaw ?? 0, pose.pitch ?? 0);
  const px = R[0] * X + R[1] * Y, py = R[3] * X + R[4] * Y, pz = FOCAL + R[6] * X + R[7] * Y;
  return [pose.cx + (FOCAL * px) / pz, pose.cy + (FOCAL * py) / pz];
}

/** Poses across the exposure of frame time t (in frames). `poseAt(t)` -> {cx, cy, m, rot, yaw, pitch}. */
export function samplePoses(poseAt, t, exposure, n, pmModules = 6) {
  const a = poseAt(t - exposure / 2), b = poseAt(t + exposure / 2);
  const radius = ((n + 2 * pmModules) * Math.max(a.m, b.m)) / 2;
  const travel =
    Math.hypot(b.cx - a.cx, b.cy - a.cy) +
    radius * (Math.abs((b.rot ?? 0) - (a.rot ?? 0)) + 0.5 * Math.abs((b.m - a.m) / Math.max(a.m, 1e-6)));
  const mid = poseAt(t - exposure / 2 + exposure * 0.5);
  const midTravel = Math.hypot(mid.cx - (a.cx + b.cx) / 2, mid.cy - (a.cy + b.cy) / 2);
  const len = Math.max(travel, 2 * midTravel + travel * 0.5);
  const k = exposure <= 0 || len < 0.6 ? 1 : Math.min(40, Math.max(3, Math.ceil(len / 1.1) + 1));
  const poses = [];
  for (let i = 0; i < k; i++) poses.push(poseAt(k === 1 ? t : t - exposure / 2 + (exposure * i) / (k - 1)));
  return poses;
}

const pool = {acc: null, tmp: null};
function buffer(name, size) {
  if (!pool[name] || pool[name].length < size) pool[name] = new Float32Array(size);
  return pool[name];
}

function paintCode(F, w, h, code, poses, palette, pm, defocus) {
  const n = code.n;
  const T = n + 2 * pm;
  const tile = new Uint8Array(T * T).fill(1);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) tile[(y + pm) * T + x + pm] = code.grid[y * n + x] ? 0 : 1;
  const pal = Float32Array.of(palette.dark, palette.paper, palette.desk);
  // Bounding box of all poses.
  let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
  for (const pose of poses) {
    const half = (T / 2) * pose.m;
    for (const [X, Y] of [[-half, -half], [half, -half], [half, half], [-half, half]]) {
      const [sx, sy] = projectPoint(pose, X, Y);
      bx0 = Math.min(bx0, sx); bx1 = Math.max(bx1, sx);
      by0 = Math.min(by0, sy); by1 = Math.max(by1, sy);
    }
  }
  const pad = 2 + Math.ceil(3 * defocus);
  const x0 = Math.max(0, Math.floor(bx0) - pad), x1 = Math.min(w, Math.ceil(bx1) + pad);
  const y0 = Math.max(0, Math.floor(by0) - pad), y1 = Math.min(h, Math.ceil(by1) + pad);
  if (!(x1 > x0 && y1 > y0)) return null;
  const bw = x1 - x0, bh = y1 - y0;
  const k = poses.length;
  const ss = k >= 9 ? 1 : k >= 3 ? 2 : 3;
  const acc = buffer('acc', bw * bh);
  acc.fill(0, 0, bw * bh);
  for (const pose of poses) {
    const R = rotationMatrix(pose.rot ?? 0, pose.yaw ?? 0, pose.pitch ?? 0);
    const n0 = R[2], n1 = R[5], n2 = R[8]; // plane normal (third column)
    // local = t * (M q) - M (0,0,f), M = R^T, q = (dx, dy, f)
    const m00 = R[0], m01 = R[3], m02 = R[6], m10 = R[1], m11 = R[4], m12 = R[7];
    const f = FOCAL, inv = 1 / pose.m, half = n / 2 + pm;
    const offX = m02 * f, offY = m12 * f;
    for (let sj = 0; sj < ss; sj++) {
      const oy = (sj + 0.5) / ss;
      for (let si = 0; si < ss; si++) {
        const ox = (si + 0.5) / ss;
        for (let y = y0; y < y1; y++) {
          const dy = y + oy - pose.cy;
          let o = (y - y0) * bw;
          for (let x = x0; x < x1; x++, o++) {
            const dx = x + ox - pose.cx;
            const den = n0 * dx + n1 * dy + n2 * f;
            let v;
            if (den <= 1e-6) v = palette.desk;
            else {
              const t = (f * n2) / den;
              const u = (t * (m00 * dx + m01 * dy + m02 * f) - offX) * inv + half;
              const vv = (t * (m10 * dx + m11 * dy + m12 * f) - offY) * inv + half;
              v = u < 0 || vv < 0 || u >= T || vv >= T ? pal[2] : pal[tile[(vv | 0) * T + (u | 0)]];
            }
            acc[o] += v;
          }
        }
      }
    }
  }
  const S = k * ss * ss;
  for (let y = y0; y < y1; y++) {
    let o = (y - y0) * bw;
    let i = y * w + x0;
    for (let x = x0; x < x1; x++, o++, i++) F[i] = acc[o] / S;
  }
  return {x0, y0, x1, y1};
}

function blurRegion(F, w, r, sigma) {
  if (!(sigma > 0.05) || !r) return;
  const rad = Math.ceil(3 * sigma);
  const kernel = new Float32Array(2 * rad + 1);
  let sum = 0;
  for (let i = -rad; i <= rad; i++) sum += kernel[i + rad] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const bw = r.x1 - r.x0, bh = r.y1 - r.y0;
  const tmp = buffer('tmp', bw * bh);
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++) {
      let v = 0;
      for (let j = -rad; j <= rad; j++) {
        const xx = Math.min(bw - 1, Math.max(0, x + j));
        v += F[(r.y0 + y) * w + r.x0 + xx] * kernel[j + rad];
      }
      tmp[y * bw + x] = v;
    }
  for (let y = 0; y < bh; y++)
    for (let x = 0; x < bw; x++) {
      let v = 0;
      for (let j = -rad; j <= rad; j++) v += tmp[Math.min(bh - 1, Math.max(0, y + j)) * bw + x] * kernel[j + rad];
      F[(r.y0 + y) * w + r.x0 + x] = v;
    }
}

/**
 * Renders one code frame into `out` (Uint8Array, w*h).
 * spec: {code:{n,grid}|null, poses, palette, pm, defocus, gain, offset, grad:{angle,amp}, glare:{x,y,r,amp}, vignette,
 *        gamma, noise, occluders:[{x0,y0,x1,y1,v}], texture}
 */
export function renderScene(out, w, h, spec, rng) {
  const palette = spec.palette ?? DEFAULT_PALETTE;
  const F = buffer('frame', w * h);
  F.fill(palette.desk, 0, w * h);
  let region = null;
  if (spec.code && spec.poses && spec.poses.length)
    region = paintCode(F, w, h, spec.code, spec.poses, palette, spec.pm ?? 6, spec.defocus ?? 0);
  for (const o of spec.occluders ?? []) {
    const x0 = Math.max(0, Math.floor(o.x0)), x1 = Math.min(w, Math.ceil(o.x1));
    const y0 = Math.max(0, Math.floor(o.y0)), y1 = Math.min(h, Math.ceil(o.y1));
    for (let y = y0; y < y1; y++) F.fill(o.v, y * w + x0, y * w + Math.max(x0, x1));
    if (region) {
      region.x0 = Math.min(region.x0, x0); region.x1 = Math.max(region.x1, x1);
      region.y0 = Math.min(region.y0, y0); region.y1 = Math.max(region.y1, y1);
    }
  }
  if (region && spec.defocus > 0.05) {
    const pad = Math.ceil(3 * spec.defocus) + 1;
    region = {x0: Math.max(0, region.x0 - pad), x1: Math.min(w, region.x1 + pad), y0: Math.max(0, region.y0 - pad), y1: Math.min(h, region.y1 + pad)};
    blurRegion(F, w, region, spec.defocus);
  }
  postProcess(out, F, w, h, spec, rng);
}

/** Lighting, glare, vignette, gamma, noise and 8-bit quantization of the float image F into out. */
export function postProcess(out, F, w, h, spec, rng) {
  const gain = spec.gain ?? 1, offset = spec.offset ?? 0;
  const grad = spec.grad, glare = spec.glare, vig = spec.vignette ?? 0;
  const gamma = spec.gamma ?? 1, noise = spec.noise ?? 0;
  if (glare && glare.amp) {
    const r3 = Math.ceil(3 * glare.r);
    const gx0 = Math.max(0, Math.floor(glare.x - r3)), gx1 = Math.min(w, Math.ceil(glare.x + r3));
    const gy0 = Math.max(0, Math.floor(glare.y - r3)), gy1 = Math.min(h, Math.ceil(glare.y + r3));
    for (let y = gy0; y < gy1; y++)
      for (let x = gx0; x < gx1; x++) {
        const d2 = (x - glare.x) ** 2 + (y - glare.y) ** 2;
        F[y * w + x] += glare.amp * 255 * Math.exp(-d2 / (2 * glare.r * glare.r)) / Math.max(gain, 1e-3);
      }
  }
  let lut = null;
  if (gamma !== 1) {
    lut = new Float32Array(255 * 8 + 2);
    for (let i = 0; i < lut.length; i++) lut[i] = 255 * Math.pow(Math.min(1, i / 8 / 255), gamma);
  }
  const ca = grad ? Math.cos(grad.angle) : 0, sa = grad ? Math.sin(grad.angle) : 0;
  const halfDiag = 0.5 * Math.hypot(w, h);
  const gdx = grad ? (grad.amp * ca) / halfDiag : 0, gdy = grad ? (grad.amp * sa) / halfDiag : 0;
  const cx = w / 2, cy = h / 2, maxd2 = cx * cx + cy * cy;
  let state = rng ? rng.u32() | 1 : 1;
  let i = 0;
  for (let y = 0; y < h; y++) {
    const rowL = 1 + gdy * (y - cy) - gdx * cx;
    for (let x = 0; x < w; x++, i++) {
      let v = F[i] * gain + offset;
      if (grad) v *= rowL + gdx * x;
      if (vig) v *= 1 - vig * (((x - cx) ** 2 + (y - cy) ** 2) / maxd2);
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      if (lut) v = lut[(v * 8 + 0.5) | 0];
      if (noise) {
        state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
        const r = state >>> 0;
        v += (((r & 255) + ((r >>> 8) & 255) + ((r >>> 16) & 255) + (r >>> 24) - 510) / 147.8) * noise;
      }
      v = Math.floor(v + 0.5);
      out[i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
}

/** Non-code frames. kind: blank | noise | stripes | gradient | clutter | checker. Written straight to `out`. */
export function renderJunk(out, w, h, junk, rng, spec = {}) {
  const kind = junk.kind;
  if (kind === 'blank') out.fill(junk.value ?? 128);
  else if (kind === 'noise') {
    for (let i = 0; i < out.length; i += 4) {
      const r = rng.u32();
      out[i] = r & 255;
      if (i + 1 < out.length) out[i + 1] = (r >>> 8) & 255;
      if (i + 2 < out.length) out[i + 2] = (r >>> 16) & 255;
      if (i + 3 < out.length) out[i + 3] = r >>> 24;
    }
  } else if (kind === 'stripes') {
    const p = junk.period ?? 8, vertical = junk.vertical ?? true;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = Math.floor((vertical ? x : y) / p) % 2 ? 255 : 0;
  } else if (kind === 'gradient') {
    const lo = junk.lo ?? 0, hi = junk.hi ?? 255;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const t = junk.vertical ? y / (h - 1) : x / (w - 1);
        out[y * w + x] = Math.floor(lo + (hi - lo) * t + 0.5);
      }
  } else if (kind === 'checker') {
    const p = junk.period ?? 6;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) out[y * w + x] = (Math.floor(x / p) + Math.floor(y / p)) % 2 ? 235 : 20;
  } else if (kind === 'clutter') {
    // desk clutter: random dark and light rectangles and thin lines on a gray desk
    const F = buffer('frame', w * h);
    F.fill(junk.base ?? 112, 0, w * h);
    const count = junk.count ?? 60;
    for (let k = 0; k < count; k++) {
      const x0 = rng.int(w), y0 = rng.int(h), rw = 3 + rng.int(Math.max(4, w / 8)), rh = 3 + rng.int(Math.max(4, h / 8));
      const v = rng.int(2) ? 30 + rng.int(40) : 190 + rng.int(60);
      for (let y = y0; y < Math.min(h, y0 + rh); y++) F.fill(v, y * w + x0, y * w + Math.min(w, x0 + rw));
    }
    postProcess(out, F, w, h, {noise: junk.noise ?? 2}, rng);
    return;
  } else throw new Error('unknown junk kind ' + kind);
}
