/* Hypothesis-conditioned physical evidence verifier (saturation-aware, spatially varying motion blur).
   Static memory only; no allocator, no libm (WASM). Given the current registered crop and a payload
   hypothesis grid, it fits a forward model  obs = f( sum_t w_t(x) * (line_t * gauss_t * T)(x) )  where T is the
   hypothesis template, line_t/gauss_t are per-tile motion/defocus kernels and f is a monotone (isotonic) camera
   response that absorbs exposure, gamma and sensor clipping. It then computes, per module, the log-likelihood
   ratio of the hypothesis against flipping that module, and summarizes them into Bayes factors for "the displayed
   code is a different codeword" under two alternative models:
     S1  any single data-symbol change (Reed-Solomon MDS: all parity symbols change, symbol values uniform);
     S2  any multi-symbol change with uniformly random symbol values (parity symbols may coincide w.p. 1/256);
     R   a uniformly random codeword (independent module bits).
   Acceptance requires all three to be <= -threshold and the format bits to be confirmed. The scorer payload is
   never consulted; the hypothesis must come from an earlier fresh decode. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
#define MAXN  41
#define MAXS  ((MAXN + 8) * 2)
#define PATCH 40
#ifndef TILES
#define TILES 2
#endif
#define MAXT 4
static void tileBoxN(int t, int S, int nt, int *x0, int *y0, int *x1, int *y1) {
    int tx = t % nt, ty = t / nt;
    double w = (double)S / nt;
    *x0 = tx ? (int)((tx - .5) * w) : 0;
    *x1 = tx < nt - 1 ? (int)((tx + 1.5) * w) + 1 : S;
    *y0 = ty ? (int)((ty - .5) * w) : 0;
    *y1 = ty < nt - 1 ? (int)((ty + 1.5) * w) + 1 : S;
    if (*x1 > S) *x1 = S;
    if (*y1 > S) *y1 = S;
}
static void tileBox(int t, int S, int *x0, int *y0, int *x1, int *y1) {
    tileBoxN(t, S, TILES, x0, y0, x1, y1);
}
extern int sq_symbol_map(int, int);
extern uintptr_t sq_symbol_of(void), sq_block_of(void), sq_block_totals(void), sq_block_data(void),
    sq_format_positions(void);
static uint8_t grid[57 * 57];
static float obs[MAXS * MAXS], T[MAXS * MAXS], G[3][MAXS * MAXS], gtmp[MAXS * MAXS], Bt[MAXT * MAXT][MAXS * MAXS],
    B[MAXS * MAXS], Bf[MAXS * MAXS], W[MAXT * MAXT][MAXS * MAXS], W4[MAXT * MAXT][MAXS * MAXS], ones[MAXS * MAXS];
static float llr[57 * 57], patch[MAXT * MAXT][PATCH * PATCH], canvas[PATCH * PATCH], canvasG[PATCH * PATCH], tone[65];
static double stats[32];
static double threshold = 20, inflation = 4, formatThreshold = 10;
static int maxLength = 15, adaptiveInflation = 0, shiftSearch = 0;
static double outlierLog = 0, symbolCap = 0;
static int hetero = 0;
static float varBin[8];
API int se_hetero(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    hetero = enabled;
    return 0;
}
API int se_symbol_cap(double cap) {
    if (!(cap == 0 || (cap >= 1 && cap <= 20))) return -1;
    symbolCap = cap;
    return 0;
}
API int se_outlier(double logEps) {
    if (!(logEps == 0 || (logEps >= 2 && logEps <= 30))) return -1;
    outlierLog = logEps;
    return 0;
}
API int se_shift(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    shiftSearch = enabled;
    return 0;
}
API int se_adaptive(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    adaptiveInflation = enabled;
    return 0;
}
static const double sigmas[3] = {.5, .9, 1.6};
typedef struct {
    double L, theta;
    int s;
    double cost;
} Kernel;
static Kernel chosen[TILES * TILES], previous[TILES * TILES], fieldK[MAXT * MAXT];
static int fieldMode;
static double fieldMargin = .97;
API int se_field(int enabled, double margin) {
    if (enabled < 0 || enabled > 1 || !(margin > 0 && margin <= 1.2)) return -1;
    fieldMode = enabled;
    fieldMargin = margin;
    return 0;
}
static int warmStart, previousValid, coarseStep = 2;
API int se_warm(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    warmStart = enabled;
    previousValid = 0;
    return 0;
}
API int se_warm_reset(void) {
    previousValid = 0;
    return 0;
}
API int se_coarse_step(int step) {
    if (step < 1 || step > 4) return -1;
    coarseStep = step;
    return 0;
}
API uintptr_t se_grid(void) {
    return (uintptr_t)grid;
}
API uintptr_t se_llr(void) {
    return (uintptr_t)llr;
}
API uintptr_t se_stats(void) {
    return (uintptr_t)stats;
}
API int se_configure(double acceptThreshold, double noiseInflation, double formatConfirm, int length) {
    if (!(acceptThreshold > 0 && acceptThreshold < 1e4) || !(noiseInflation >= 1 && noiseInflation <= 64) ||
        !(formatConfirm >= 0 && formatConfirm < 1e4) || length < 0 || length > 24)
        return -1;
    threshold = acceptThreshold;
    inflation = noiseInflation;
    formatThreshold = formatConfirm;
    maxLength = length;
    return 0;
}
static int solve4(double a[4][5], double *out) {
    for (int c = 0; c < 4; c++) {
        int p = c;
        for (int r = c + 1; r < 4; r++)
            if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return 0;
        for (int j = 0; j < 5; j++) {
            double t = a[p][j];
            a[p][j] = a[c][j];
            a[c][j] = t;
        }
        double d = a[c][c];
        for (int j = c; j < 5; j++) a[c][j] /= d;
        for (int r = 0; r < 4; r++)
            if (r != c) {
                d = a[r][c];
                for (int j = c; j < 5; j++) a[r][j] -= d * a[c][j];
            }
    }
    for (int i = 0; i < 4; i++) out[i] = a[i][4];
    return 1;
}
static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
/* elementary functions without libm */
static double ex(double x) {
    if (x < -700) return 0;
    if (x > 700) x = 700;
    int k = (int)floor(x * 1.4426950408889634 + .5);
    double r = x - k * 0.6931471805599453,
           p = 1 + r * (1 + r * (.5 + r * (1. / 6 + r * (1. / 24 + r * (1. / 120 + r * (1. / 720 + r / 5040))))));
    union {
        double d;
        int64_t i;
    } u;
    u.d = p;
    u.i += (int64_t)k << 52;
    return u.d;
}
static double lg(double x) {
    if (!(x > 0)) return -1e300;
    union {
        double d;
        int64_t i;
    } u;
    u.d = x;
    int e = (int)((u.i >> 52) & 2047) - 1023;
    u.i = (u.i & 0xfffffffffffffLL) | 0x3ff0000000000000LL;
    double m = u.d;
    if (m > 1.4142135623730951) {
        m *= .5;
        e++;
    }
    double t = (m - 1) / (m + 1), t2 = t * t,
           s = t * (2 + t2 * (2. / 3 + t2 * (2. / 5 + t2 * (2. / 7 + t2 * (2. / 9 + t2 * (2. / 11 + t2 * 2. / 13))))));
    return s + e * 0.6931471805599453;
}
static double softplus(double x) {
    return x > 30 ? x : x < -30 ? ex(x) : lg(1 + ex(x));
} /* log(1+e^x) */
static double logsumexp2(double a, double b) {
    double m = a > b ? a : b;
    if (m < -1e299) return m;
    return m + lg(ex(a - m) + ex(b - m));
}
static double sn(double x) {
    while (x > 3.141592653589793) x -= 6.283185307179586;
    while (x < -3.141592653589793) x += 6.283185307179586;
    double x2 = x * x;
    return x * (1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42 * (1 - x2 / 72 * (1 - x2 / 110 * (1 - x2 / 156))))));
}
static double cs(double a) {
    return sn(a + 1.5707963267948966);
}
static double atan2a(double y, double x) {
    double ax = fabs(x), ay = fabs(y);
    if (ax < 1e-300 && ay < 1e-300) return 0;
    double a = (ax > ay ? ay / ax : ax / ay), s = a * a,
           r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if (ay > ax) r = 1.5707963267948966 - r;
    if (x < 0) r = 3.141592653589793 - r;
    if (y < 0) r = -r;
    return r;
}
static float sampleF(const float *a, int S, double x, double y, float outside) {
    if (!(x >= 0 && y >= 0 && x <= S - 1 && y <= S - 1)) return outside;
    int ix = (int)x, iy = (int)y;
    if (ix >= S - 1) ix = S - 2;
    if (iy >= S - 1) iy = S - 2;
    double fx = x - ix, fy = y - iy;
    return (float)((a[iy * S + ix] * (1 - fx) + a[iy * S + ix + 1] * fx) * (1 - fy) +
                   (a[(iy + 1) * S + ix] * (1 - fx) + a[(iy + 1) * S + ix + 1] * fx) * fy);
}
static void gauss(const float *in, float *out, float *tmp, int S, double sigma, float outside) {
    int R = (int)(3 * sigma + .999);
    if (R > 6) R = 6;
    double k[13], sum = 0;
    for (int i = -R; i <= R; i++) {
        k[i + R] = ex(-(double)(i * i) / (2 * sigma * sigma));
        sum += k[i + R];
    }
    for (int i = 0; i <= 2 * R; i++) k[i] /= sum;
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) {
            double acc = 0;
            for (int i = -R; i <= R; i++) {
                int uu = u + i;
                acc += k[i + R] * (uu < 0 || uu >= S ? outside : in[v * S + uu]);
            }
            tmp[v * S + u] = (float)acc;
        }
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) {
            double acc = 0;
            for (int i = -R; i <= R; i++) {
                int vv = v + i;
                acc += k[i + R] * (vv < 0 || vv >= S ? outside : tmp[vv * S + u]);
            }
            out[v * S + u] = (float)acc;
        }
}
/* line blur along (cos,sin) of length L. Tap-major: every tap has a constant offset, so its bilinear weights are
   identical for all pixels; sources are white-padded so the inner loop has no bounds checks (vectorizable). */
#define PAD 12
#define PS  (MAXS + 2 * PAD)
static float padded[3][PS * PS];
static void padSource(const float *src, int S, float *dst, float outside) {
    int P = S + 2 * PAD;
    for (int i = 0; i < P * P; i++) dst[i] = outside;
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) dst[(v + PAD) * P + u + PAD] = src[v * S + u];
}
static void tileWeights(float (*w)[MAXS * MAXS], int S, int nt) {
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) {
            double cy = (v + .5) / S * nt - .5, cx = (u + .5) / S * nt - .5;
            if (cy < 0) cy = 0;
            if (cx < 0) cx = 0;
            if (cy > nt - 1) cy = nt - 1;
            if (cx > nt - 1) cx = nt - 1;
            int ly = (int)cy, lx = (int)cx;
            if (ly > nt - 2) ly = nt - 2;
            if (lx > nt - 2) lx = nt - 2;
            double fy = cy - ly, fx = cx - lx;
            for (int t = 0; t < nt * nt; t++) w[t][v * S + u] = 0;
            w[ly * nt + lx][v * S + u] = (float)((1 - fy) * (1 - fx));
            w[ly * nt + lx + 1][v * S + u] = (float)((1 - fy) * fx);
            w[(ly + 1) * nt + lx][v * S + u] = (float)(fy * (1 - fx));
            w[(ly + 1) * nt + lx + 1][v * S + u] = (float)(fy * fx);
        }
}
static void lineBlurPadded(const float *src, int S, float *dst, double L, double theta, int x0, int y0, int x1, int y1,
                           int step) {
    int P = S + 2 * PAD, K = L < .5 ? 1 : (int)(L + 1.999);
    if (K > 32) K = 32;
    double c = cs(theta), s = sn(theta);
    float inv = 1.f / K;
    for (int v = y0; v < y1; v += step)
        for (int u = x0; u < x1; u += step) dst[v * S + u] = 0;
    for (int k = 0; k < K; k++) {
        double t = K == 1 ? 0 : (k / (double)(K - 1) - .5) * L, ox = t * c, oy = t * s;
        int ix = (int)floor(ox), iy = (int)floor(oy);
        float fx = (float)(ox - ix), fy = (float)(oy - iy);
        if (ix < -PAD + 1) ix = -PAD + 1;
        if (iy < -PAD + 1) iy = -PAD + 1;
        if (ix > PAD - 2) ix = PAD - 2;
        if (iy > PAD - 2) iy = PAD - 2;
        float w00 = (1 - fx) * (1 - fy) * inv, w01 = fx * (1 - fy) * inv, w10 = (1 - fx) * fy * inv,
              w11 = fx * fy * inv;
        for (int v = y0; v < y1; v += step) {
            const float *r0 = src + (v + PAD + iy) * P + PAD + ix, *r1 = r0 + P;
            float *o = dst + v * S;
            if (step == 1)
                for (int u = x0; u < x1; u++) o[u] += w00 * r0[u] + w01 * r0[u + 1] + w10 * r1[u] + w11 * r1[u + 1];
            else
                for (int u = x0; u < x1; u += step)
                    o[u] += w00 * r0[u] + w01 * r0[u + 1] + w10 * r1[u] + w11 * r1[u + 1];
        }
    }
}
/* generic (unpadded) version kept for the small single-module canvas */
static void lineBlur(const float *src, float *dst, int S, double L, double theta, int x0, int y0, int x1, int y1,
                     int step, float outside) {
    int K = L < .5 ? 1 : (int)(L + 1.999);
    if (K > 32) K = 32;
    float c = (float)cs(theta), s = (float)sn(theta), ox[32], oy[32], inv = 1.f / K;
    for (int k = 0; k < K; k++) {
        float t = K == 1 ? 0 : (float)((k / (double)(K - 1) - .5) * L);
        ox[k] = t * c;
        oy[k] = t * s;
    }
    for (int v = y0; v < y1; v += step)
        for (int u = x0; u < x1; u += step) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += sampleF(src, S, u + ox[k], v + oy[k], outside);
            dst[v * S + u] = acc * inv;
        }
}
/* censored affine fit obs ~ clip(a+b*X) over weighted pixels; returns weighted SSE */
static double affineCost(const float *X, int S, const float *w, int x0, int y0, int x1, int y1, int step) {
    double a = 0, b = 1;
    for (int it = 0; it < 3; it++) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int v = y0; v < y1; v += step)
            for (int u = x0; u < x1; u += step) {
                int i = v * S + u;
                double ww = w[i];
                if (ww < .05) continue;
                double x = X[i], y = obs[i];
                if (it && a + b * x >= 1 && y >= .98) continue;
                sw += ww;
                sx += ww * x;
                sy += ww * y;
                sxx += ww * x * x;
                sxy += ww * x * y;
            }
        double den = sw * sxx - sx * sx;
        if (sw < 8 || fabs(den) < 1e-12) break;
        b = (sw * sxy - sx * sy) / den;
        a = (sy - b * sx) / sw;
        if (b < .05) b = .05;
    }
    double cost = 0;
    for (int v = y0; v < y1; v += step)
        for (int u = x0; u < x1; u += step) {
            int i = v * S + u;
            double ww = w[i];
            if (ww < .05) continue;
            double r = obs[i] - clampd(a + b * X[i], 0, 1);
            cost += ww * r * r;
        }
    return cost;
}
static int eccOf(int n) {
    const uint16_t *pos = (const uint16_t *)sq_format_positions();
    int best = -1, bestDistance = 99;
    for (int format = 0; format < 32; format++) {
        unsigned data = format << 10, value = data;
        for (int i = 14; i >= 10; i--)
            if (value & (1u << i)) value ^= 0x537u << (i - 10);
        unsigned bits = (data | value) ^ 0x5412;
        int d0 = 0, d1 = 0;
        for (int i = 0; i < 15; i++) {
            d0 += ((bits >> i) & 1) != grid[pos[i]];
            d1 += ((bits >> i) & 1) != grid[pos[15 + i]];
        }
        int d = d0 < d1 ? d0 : d1;
        if (d < bestDistance) {
            bestDistance = d;
            best = format;
        }
    }
    return bestDistance == 0 ? best >> 3 : -1;
}
static double toneAt(double x) {
    if (x <= 0) return tone[0];
    if (x >= 1) return tone[64];
    double p = x * 64;
    int i = (int)p;
    if (i > 63) i = 63;
    double f = p - i;
    return tone[i] * (1 - f) + tone[i + 1] * f;
}
/* isotonic (non-decreasing) regression of obs on B with 64 bins, pool-adjacent-violators */
static void fitTone(int S) {
    double sum[64] = {0}, cnt[64] = {0};
    for (int i = 0; i < S * S; i++) {
        int b = (int)(B[i] * 64);
        if (b < 0) b = 0;
        if (b > 63) b = 63;
        sum[b] += obs[i];
        cnt[b] += 1;
    }
    double val[64], wt[64];
    int first[64], len = 0;
    for (int b = 0; b < 64; b++) {
        if (cnt[b] < 1) continue;
        val[len] = sum[b] / cnt[b];
        wt[len] = cnt[b];
        first[len] = b;
        len++;
        while (len > 1 && val[len - 2] > val[len - 1]) {
            double w = wt[len - 2] + wt[len - 1];
            val[len - 2] = (val[len - 2] * wt[len - 2] + val[len - 1] * wt[len - 1]) / w;
            wt[len - 2] = w;
            len--;
        }
    }
    if (!len) {
        for (int k = 0; k <= 64; k++) tone[k] = k / 64.f;
        return;
    }
    double kx[64], ky[64];
    int m = 0;
    for (int j = 0; j < len; j++) {
        int lo = first[j], hi = j + 1 < len ? first[j + 1] : 64;
        kx[m] = (lo + hi) * .5 / 64;
        ky[m] = val[j];
        m++;
    }
    for (int k = 0; k <= 64; k++) {
        double x = k / 64.;
        double y;
        if (x <= kx[0])
            y = ky[0];
        else if (x >= kx[m - 1])
            y = ky[m - 1];
        else {
            int j = 0;
            while (j + 1 < m && kx[j + 1] < x) j++;
            double f = (x - kx[j]) / (kx[j + 1] - kx[j]);
            y = ky[j] * (1 - f) + ky[j + 1] * f;
        }
        tone[k] = (float)y;
    }
}
API int se_run(const float *crop, int size, int version, double deadlineSeconds, double (*now)(void)) {
    int n = 17 + 4 * version, S = (n + 8) * 2;
    if (version < 1 || n > MAXN || size != (n + 8) * 4 || !crop) return -1;
    for (int i = 0; i < 32; i++) stats[i] = 0;
    double tStart = now ? now() : 0;
    if (sq_symbol_map(version, 0) < 1) return -1;
    int ecc = eccOf(n);
    stats[17] = ecc;
    if (ecc < 0) return 0;
    int blocks = sq_symbol_map(version, ecc);
    if (blocks < 1) return -1;
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) {
            obs[v * S + u] = (crop[(v * 2) * size + u * 2] + crop[(v * 2) * size + u * 2 + 1] +
                              crop[(v * 2 + 1) * size + u * 2] + crop[(v * 2 + 1) * size + u * 2 + 1]) *
                             .25f;
            int mx = u / 2 - 4, my = v / 2 - 4;
            T[v * S + u] = (mx >= 0 && my >= 0 && mx < n && my < n) ? (float)(1 - grid[my * n + mx]) : 1.f;
        }
    for (int s = 0; s < 3; s++) {
        gauss(T, G[s], gtmp, S, sigmas[s], 1.f);
        padSource(G[s], S, padded[s], 1.f);
    }
    tileWeights(W, S, TILES);
    int evaluations = 0;
    for (int t = 0; t < TILES * TILES; t++) {
        if (now && now() >= deadlineSeconds) return 0;
        int x0, y0, x1, y1;
        tileBox(t, S, &x0, &y0, &x1, &y1);
        if (x0 < 2) x0 = 2;
        if (y0 < 2) y0 = 2;
        if (x1 > S - 2) x1 = S - 2;
        if (y1 > S - 2) y1 = S - 2;
        double jxx = 0, jyy = 0, jxy = 0;
        for (int v = y0; v < y1; v++)
            for (int u = x0; u < x1; u++) {
                int i = v * S + u;
                double w = W[t][i];
                if (w < .05) continue;
                double gx = (obs[i + 1] - obs[i - 1]) * .5, gy = (obs[i + S] - obs[i - S]) * .5;
                jxx += w * gx * gx;
                jyy += w * gy * gy;
                jxy += w * gx * gy;
            }
        double tr = jxx + jyy, coh = tr > 1e-12 ? sqrt((jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy) / tr : 0,
               edge = .5 * atan2a(2 * jxy, jxx - jyy), theta = edge + 1.5707963267948966;
        if (t < 4) stats[22 + t] = coh;
        Kernel best = {0, 0, 0, 1e30};
        static const double coarse[] = {0, 1, 2, 4, 6, 9, 12, 15};
        if (warmStart &&
            previousValid) { /* temporal warm start from the previous frame's kernel for this tile (blur state only) */
            Kernel p = previous[t];
            double cand[5][2] = {
                {p.L, p.theta}, {p.L, theta}, {p.L > 2 ? p.L - 2 : 0, p.theta}, {p.L + 2, p.theta}, {0, theta}};
            for (int k = 0; k < 5; k++) {
                double L = cand[k][0];
                if (L > maxLength) L = maxLength;
                lineBlurPadded(padded[p.s], S, Bt[t], L, cand[k][1], x0, y0, x1, y1, coarseStep);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, coarseStep);
                evaluations++;
                if (c < best.cost) {
                    best.L = L;
                    best.theta = cand[k][1];
                    best.s = p.s;
                    best.cost = c;
                }
            }
            {
                int s_ = p.s ? 0 : 1;
                lineBlurPadded(padded[s_], S, Bt[t], best.L, best.theta, x0, y0, x1, y1, coarseStep);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, coarseStep);
                evaluations++;
                if (c < best.cost) {
                    best.s = s_;
                    best.cost = c;
                }
            }
        } else {
            int ncoarse = coh > .15 ? 8 : 3;
            for (int k = 0; k < ncoarse; k++) {
                double L = coarse[k];
                if (L > maxLength) break;
                lineBlurPadded(padded[0], S, Bt[t], L, theta, x0, y0, x1, y1, coarseStep);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, coarseStep);
                evaluations++;
                if (c < best.cost) {
                    best.L = L;
                    best.theta = theta;
                    best.s = 0;
                    best.cost = c;
                }
            }
            for (int s_ = 1; s_ < (coh <= .15 ? 3 : 2); s_++) {
                lineBlurPadded(padded[s_], S, Bt[t], best.L, best.theta, x0, y0, x1, y1, coarseStep);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, coarseStep);
                evaluations++;
                if (c < best.cost) {
                    best.s = s_;
                    best.cost = c;
                }
            }
        }
        /* refinement costs are compared on the refinement grid */
        {
            lineBlurPadded(padded[best.s], S, Bt[t], best.L, best.theta, x0, y0, x1, y1, 2);
            best.cost = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, 2);
        }
        if (best.L >= 1) {
            static const double dth[] = {-.21, .21, -.1, .1};
            static const double dl[] = {-1, 1, -.5, .5};
            for (int r = 0; r < 4; r++) {
                if (r == 2 && best.theta != theta) break;
                double th = (r < 2 ? theta : best.theta) + dth[r];
                lineBlurPadded(padded[best.s], S, Bt[t], best.L, th, x0, y0, x1, y1, 2);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, 2);
                evaluations++;
                if (c < best.cost) {
                    best.theta = th;
                    best.cost = c;
                }
            }
            double base = best.L;
            for (int r = 0; r < 4; r++) {
                double L = base + dl[r];
                if (r >= 2) L = best.L + dl[r];
                if (L < .5 || L > maxLength) continue;
                lineBlurPadded(padded[best.s], S, Bt[t], L, best.theta, x0, y0, x1, y1, 2);
                double c = affineCost(Bt[t], S, W[t], x0, y0, x1, y1, 2);
                evaluations++;
                if (c < best.cost) {
                    best.L = L;
                    best.cost = c;
                }
            }
        }
        chosen[t] = best;
        previous[t] = best;
    }
    previousValid = 1;
    if (now && now() >= deadlineSeconds) return 0;
    if (now) stats[26] = (now() - tStart) * 1000;
    for (int i = 0; i < S * S; i++) B[i] = 0;
    for (int t = 0; t < TILES * TILES; t++) {
        int x0, y0, x1, y1;
        tileBox(t, S, &x0, &y0, &x1, &y1);
        lineBlurPadded(padded[chosen[t].s], S, Bt[t], chosen[t].L, chosen[t].theta, x0, y0, x1, y1, 1);
        for (int v = y0; v < y1; v++)
            for (int u = x0; u < x1; u++) {
                int i = v * S + u;
                B[i] += W[t][i] * Bt[t][i];
            }
    }

    int nt = TILES;
    Kernel *K = chosen;
    float (*Wuse)[MAXS * MAXS] = W;
    stats[29] = 0;
    if (fieldMode) { /* rigid-motion blur field d(x)=a+w*J(x-c)+z*(x-c), fitted to the 2x2 tile kernels (sign-ambiguous), rendered 4x4 */
        for (int i = 0; i < S * S; i++) ones[i] = 1.f;
        double cost2 = affineCost(B, S, ones, 2, 2, S - 2, S - 2, 2), c = S * .5, bestRes = 1e30, par[4] = {0, 0, 0, 0};
        double vx[4], vy[4], px_[4], py_[4];
        for (int t = 0; t < 4; t++) {
            vx[t] = chosen[t].L * cs(chosen[t].theta);
            vy[t] = chosen[t].L * sn(chosen[t].theta);
            px_[t] = ((t % 2) + .5) * S / 2 - c;
            py_[t] = ((t / 2) + .5) * S / 2 - c;
        }
        for (int signs = 0; signs < 8; signs++) {
            double A[4][5] = {{0}};
            for (int t = 0; t < 4; t++) {
                double sg = (t && (signs >> (t - 1)) & 1) ? -1 : 1, dx = sg * vx[t], dy = sg * vy[t];
                double r1[4] = {1, 0, -py_[t], px_[t]}, r2[4] = {0, 1, px_[t], py_[t]};
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) A[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
                    A[i][4] += r1[i] * dx + r2[i] * dy;
                }
            }
            for (int i = 0; i < 4; i++) A[i][i] += 1e-6;
            double sol[4];
            if (!solve4(A, sol)) continue;
            double res = 0;
            for (int t = 0; t < 4; t++) {
                double sg = (t && (signs >> (t - 1)) & 1) ? -1 : 1,
                       ex_ = sol[0] - sol[2] * py_[t] + sol[3] * px_[t] - sg * vx[t],
                       ey_ = sol[1] + sol[2] * px_[t] + sol[3] * py_[t] - sg * vy[t];
                res += ex_ * ex_ + ey_ * ey_;
            }
            if (res < bestRes) {
                bestRes = res;
                for (int i = 0; i < 4; i++) par[i] = sol[i];
            }
        }
        int sMajor = 0;
        {
            int cnt = 0;
            for (int t = 0; t < 4; t++) cnt += chosen[t].s;
            sMajor = cnt >= 2 ? 1 : 0;
        }
        tileWeights(W4, S, 4);
        double bestF = 1e30, bestScale = 1;
        static const double scales[3] = {.85, 1, 1.15};
        for (int si = 0; si < 3; si++) {
            double sc = scales[si];
            for (int i = 0; i < S * S; i++) Bf[i] = 0;
            for (int t = 0; t < 16; t++) {
                double cxp = ((t % 4) + .5) * S / 4 - c, cyp = ((t / 4) + .5) * S / 4 - c,
                       dx = sc * (par[0] - par[2] * cyp + par[3] * cxp),
                       dy = sc * (par[1] + par[2] * cxp + par[3] * cyp), L = sqrt(dx * dx + dy * dy);
                if (L > maxLength) L = maxLength;
                fieldK[t].L = L;
                fieldK[t].theta = atan2a(dy, dx);
                fieldK[t].s = sMajor;
                int x0, y0, x1, y1;
                tileBoxN(t, S, 4, &x0, &y0, &x1, &y1);
                lineBlurPadded(padded[sMajor], S, Bt[t], L, fieldK[t].theta, x0, y0, x1, y1, 1);
                for (int v = y0; v < y1; v++)
                    for (int u = x0; u < x1; u++) {
                        int i = v * S + u;
                        Bf[i] += W4[t][i] * Bt[t][i];
                    }
            }
            double cf = affineCost(Bf, S, ones, 2, 2, S - 2, S - 2, 2);
            if (cf < bestF) {
                bestF = cf;
                bestScale = sc;
            }
        }
        stats[29] = bestF / (cost2 > 1e-12 ? cost2 : 1e-12);
        if (bestF < cost2 * fieldMargin) {
            double sc = bestScale;
            for (int i = 0; i < S * S; i++) Bf[i] = 0;
            for (int t = 0; t < 16; t++) {
                double cxp = ((t % 4) + .5) * S / 4 - c, cyp = ((t / 4) + .5) * S / 4 - c,
                       dx = sc * (par[0] - par[2] * cyp + par[3] * cxp),
                       dy = sc * (par[1] + par[2] * cxp + par[3] * cyp), L = sqrt(dx * dx + dy * dy);
                if (L > maxLength) L = maxLength;
                fieldK[t].L = L;
                fieldK[t].theta = atan2a(dy, dx);
                fieldK[t].s = sMajor;
                int x0, y0, x1, y1;
                tileBoxN(t, S, 4, &x0, &y0, &x1, &y1);
                lineBlurPadded(padded[sMajor], S, Bt[t], L, fieldK[t].theta, x0, y0, x1, y1, 1);
                for (int v = y0; v < y1; v++)
                    for (int u = x0; u < x1; u++) {
                        int i = v * S + u;
                        Bf[i] += W4[t][i] * Bt[t][i];
                    }
            }
            for (int i = 0; i < S * S; i++) B[i] = Bf[i];
            nt = 4;
            K = fieldK;
            Wuse = W4;
            stats[30] = 1;
        }
        stats[31] = par[2];
    }
    if (shiftSearch) { /* global sub-pixel registration refinement: resample the observation by (dx,dy) */
        static float baseObs[MAXS * MAXS];
        for (int i = 0; i < S * S; i++) {
            baseObs[i] = obs[i];
            W[0][i] = 1.f;
        }
        static float ones[MAXS * MAXS];
        for (int i = 0; i < S * S; i++) ones[i] = 1.f;
        double bestc = affineCost(B, S, ones, 2, 2, S - 2, S - 2, 1), bdx = 0, bdy = 0;
        for (int it = 0; it < 2; it++) {
            double step = it ? .25 : .5, cx = bdx, cy = bdy;
            for (int sy = -1; sy <= 1; sy++)
                for (int sx = -1; sx <= 1; sx++) {
                    if (!sx && !sy) continue;
                    double dx = cx + sx * step, dy = cy + sy * step;
                    for (int v = 0; v < S; v++)
                        for (int u = 0; u < S; u++) obs[v * S + u] = sampleF(baseObs, S, u + dx, v + dy, 1.f);
                    double c = affineCost(B, S, ones, 2, 2, S - 2, S - 2, 1);
                    if (c < bestc) {
                        bestc = c;
                        bdx = dx;
                        bdy = dy;
                    }
                }
        }
        for (int v = 0; v < S; v++)
            for (int u = 0; u < S; u++) obs[v * S + u] = sampleF(baseObs, S, u + bdx, v + bdy, 1.f);
        stats[20] = bdx;
        stats[21] = bdy;
    }
    fitTone(S);
    if (now) stats[27] = (now() - tStart) * 1000;
    double sse = 0;
    int informative = 0;
    for (int v = 1; v < S - 1; v++)
        for (int u = 1; u < S - 1; u++) {
            int i = v * S + u;
            double p = toneAt(B[i]);
            if (p >= .97 && obs[i] >= .97) continue;
            double r = obs[i] - p;
            sse += r * r;
            informative++;
        }
    double sig2 = informative > 16 ? sse / informative : 1e-2;
    if (sig2 < 2e-4) sig2 = 2e-4;
    double k = inflation;
    if (adaptiveInflation &&
        informative > 64) { /* variance inflation of a patch sum under correlated residuals: sum of normalized
    residual autocorrelation over lags |dx|,|dy|<=2 (effective-sample-size correction); clamped to [1,inflation*2] */
        double acc = 0;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                double c = 0;
                int m = 0;
                for (int v = 3; v < S - 3; v += 1)
                    for (int u = 3; u < S - 3; u += 1) {
                        int i = v * S + u, j = (v + dy) * S + u + dx;
                        double p0 = toneAt(B[i]), p1 = toneAt(B[j]);
                        if ((p0 >= .97 && obs[i] >= .97) || (p1 >= .97 && obs[j] >= .97)) continue;
                        c += (obs[i] - p0) * (obs[j] - p1);
                        m++;
                    }
                if (m) acc += c / m / sig2;
            }
        k = acc;
        if (k < 1) k = 1;
        if (k > inflation * 2) k = inflation * 2;
    }
    stats[18] = k;
    double scale = 1 / (2 * sig2 * k);
    if (hetero) { /* signal-dependent noise: residual variance per predicted-intensity bin (Poissonian-Gaussian style) */
        double sb[8] = {0}, cb[8] = {0};
        for (int v = 1; v < S - 1; v++)
            for (int u = 1; u < S - 1; u++) {
                int i = v * S + u;
                double p = toneAt(B[i]);
                if (p >= .97 && obs[i] >= .97) continue;
                int b = (int)(p * 8);
                if (b > 7) b = 7;
                if (b < 0) b = 0;
                double r = obs[i] - p;
                sb[b] += r * r;
                cb[b]++;
            }
        for (int b = 0; b < 8; b++) {
            double v = cb[b] >= 20 ? sb[b] / cb[b] : sig2;
            if (v < sig2 * .25) v = sig2 * .25;
            if (v < 2e-4) v = 2e-4;
            varBin[b] = (float)(1 / (2 * v * k));
        }
    }
    for (int t = 0; t < nt * nt; t++) {
        for (int i = 0; i < PATCH * PATCH; i++) canvas[i] = 0;
        int c0 = PATCH / 2 - 1;
        for (int v = 0; v < 2; v++)
            for (int u = 0; u < 2; u++) canvas[(c0 + v) * PATCH + c0 + u] = 1;
        gauss(canvas, canvasG, gtmp, PATCH, sigmas[K[t].s], 0.f);
        lineBlur(canvasG, patch[t], PATCH, K[t].L, K[t].theta, 0, 0, PATCH, PATCH, 1, 0.f);
    }
    if (now && now() >= deadlineSeconds) return 0;
    int bx0 = PATCH, by0 = PATCH, bx1 = 0, by1 = 0;
    for (int t = 0; t < nt * nt; t++)
        for (int v = 0; v < PATCH; v++)
            for (int u = 0; u < PATCH; u++)
                if (patch[t][v * PATCH + u] >= 1e-3) {
                    if (u < bx0) bx0 = u;
                    if (u + 1 > bx1) bx1 = u + 1;
                    if (v < by0) by0 = v;
                    if (v + 1 > by1) by1 = v + 1;
                }
    static float p0cache[MAXS * MAXS];
    for (int i = 0; i < S * S; i++) p0cache[i] = (float)toneAt(B[i]);
    for (int my = 0; my < n; my++) {
        if (now && (my & 7) == 7 && now() >= deadlineSeconds) return 0;
        for (int mx = 0; mx < n; mx++) {
            int px = (mx + 4) * 2, py = (my + 4) * 2, c0 = PATCH / 2 - 1;
            double sign = grid[my * n + mx] ? 1 : -1, delta = 0;
            int centre = (py + 1) * S + px + 1;
            float wt[MAXT * MAXT];
            int single = -1;
            for (int t = 0; t < nt * nt; t++) {
                wt[t] = Wuse[t][centre];
                if (wt[t] > .999f) single = t;
            }
            for (int pv = by0; pv < by1; pv++) {
                int v = py - c0 + pv;
                if (v < 1 || v >= S - 1) continue;
                for (int pu = bx0; pu < bx1; pu++) {
                    int u = px - c0 + pu;
                    if (u < 1 || u >= S - 1) continue;
                    int q = pv * PATCH + pu;
                    float d;
                    if (single >= 0)
                        d = patch[single][q];
                    else {
                        d = 0;
                        for (int t = 0; t < nt * nt; t++) d += wt[t] * patch[t][q];
                    }
                    if (d < 1e-3f) continue;
                    int i = v * S + u;
                    float o = obs[i], e0 = o - p0cache[i], e1 = o - (float)toneAt(B[i] + sign * d);
                    if (hetero) {
                        int b = (int)(p0cache[i] * 8);
                        if (b > 7) b = 7;
                        if (b < 0) b = 0;
                        delta += (e1 * e1 - e0 * e0) * varBin[b] / scale;
                    } else
                        delta += e1 * e1 - e0 * e0;
                }
            }
            llr[my * n + mx] = (float)(delta * scale);
        }
    }
    if (now) stats[28] = (now() - tStart) * 1000;
    const int16_t *symbolOf = (const int16_t *)sq_symbol_of();
    const int8_t *blockOf = (const int8_t *)sq_block_of();
    const int *totals = (const int *)sq_block_totals(), *data = (const int *)sq_block_data();
    static double lsum[8][255];
    for (int b = 0; b < blocks; b++)
        for (int s = 0; s < 255; s++) lsum[b][s] = 0;
    double random = 0, contradictions = 0, minLLR = 1e30;
    for (int i = 0; i < n * n; i++) {
        double l = llr[i];
        if (l < minLLR) minLLR = l;
        if (l < -3) contradictions++;
        if (outlierLog > 0) { /* module-level outlier mixture: LR=(1-eps)e^{-l}+eps, eps=e^-outlierLog */
            double a = -l, b = -outlierLog, m = a > b ? a : b;
            l = -(m + lg(ex(a - m) * (1 - ex(-outlierLog)) + ex(b - m)));
            llr[i] = (float)l;
        }
        if (symbolOf[i] < 0) continue;
        double sp = softplus(-l);
        lsum[blockOf[i]][symbolOf[i]] += sp;
        random += sp - 0.6931471805599453;
    }
    /* Per-class Bayes factors (log). A class fixes which symbols change; symbol values are uniform. For each class
    the expected likelihood ratio is a product of per-symbol means, and Markov's inequality bounds the probability
    that an alternative from that class yields BF <= e^-T by e^-T. The certificate takes the maximum over classes. */
    double s1max = -1e300, s2max = -1e300;
    for (int b = 0; b < blocks; b++) {
        double best1 = -1e300, best2 = -1e300, positive = 0, parity1 = 0, parity2 = 0;
        for (int s = 0; s < totals[b]; s++) {
            double t = lsum[b][s];
            double Bs = t > 1e-9 ? t + lg(1 - ex(-t)) - lg(255) : -60;
            if (symbolCap > 0)
                Bs = Bs -
                     softplus(
                         Bs -
                         symbolCap); /* symbol-level corruption prior: no single symbol contributes more than ~cap */
            if (s < data[b]) {
                if (Bs > best1) {
                    best2 = best1;
                    best1 = Bs;
                } else if (Bs > best2)
                    best2 = Bs;
                if (Bs > 0) positive += Bs;
            } else {
                parity1 += Bs;
                parity2 += lg((1 + 255 * ex(Bs)) / 256);
            }
        }
        double s1 = best1 + parity1; /* one data symbol changed: all parity symbols change (MDS) */
        double extra = positive - (best1 > 0 ? best1 : 0) -
                       (best2 > 0 ? best2 : 0);      /* further data symbols only if they raise the ratio */
        double s2 = best1 + best2 + extra + parity2; /* two or more data symbols: parity symbols uniform */
        if (s1 > s1max) s1max = s1;
        if (s2 > s2max) s2max = s2;
    }
    double formatSum = 0;
    {
        const uint16_t *fp = (const uint16_t *)sq_format_positions();
        for (int k = 0; k < 30; k++) {
            double v = llr[fp[k]];
            formatSum += v > 20 ? 20 : v;
        }
    }
    double s1 = s1max, s2 = s2max;
    int accept = s1 <= -threshold && s2 <= -threshold && random <= -threshold && formatSum >= formatThreshold;
    stats[0] = informative ? sse / informative : 0;
    stats[1] = chosen[0].L;
    stats[2] = chosen[TILES > 2 ? 2 : 1].L;
    stats[3] = chosen[TILES * TILES - TILES].L;
    stats[4] = chosen[TILES * TILES - 1].L;
    stats[5] = sig2;
    stats[6] = s1;
    stats[7] = s2;
    stats[8] = random;
    stats[9] = formatSum;
    stats[10] = contradictions;
    stats[11] = minLLR;
    stats[12] = evaluations;
    stats[13] = blocks;
    stats[14] = chosen[0].theta;
    stats[15] = chosen[0].s;
    stats[16] = informative;
    stats[19] = accept;
    return accept;
}
/* Cheap physical alignment score for candidate crops: fraction of crop variance left unexplained by the
   hypothesis template under one global motion kernel (direction from the structure tensor) and a clipped
   affine response. Lower is better. Geometry selection only; never used for acceptance. */
API double se_align_score(const float *crop, int size, int version) {
    int n = 17 + 4 * version, S = (n + 8) * 2;
    if (version < 1 || n > MAXN || size != (n + 8) * 4 || !crop) return 1e9;
    for (int v = 0; v < S; v++)
        for (int u = 0; u < S; u++) {
            obs[v * S + u] = (crop[(v * 2) * size + u * 2] + crop[(v * 2) * size + u * 2 + 1] +
                              crop[(v * 2 + 1) * size + u * 2] + crop[(v * 2 + 1) * size + u * 2 + 1]) *
                             .25f;
            int mx = u / 2 - 4, my = v / 2 - 4;
            T[v * S + u] = (mx >= 0 && my >= 0 && mx < n && my < n) ? (float)(1 - grid[my * n + mx]) : 1.f;
        }
    gauss(T, G[1], gtmp, S, sigmas[1], 1.f);
    padSource(G[1], S, padded[1], 1.f);
    double jxx = 0, jyy = 0, jxy = 0, mean = 0, var = 0;
    for (int v = 2; v < S - 2; v++)
        for (int u = 2; u < S - 2; u++) {
            int i = v * S + u;
            double gx = (obs[i + 1] - obs[i - 1]) * .5, gy = (obs[i + S] - obs[i - S]) * .5;
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
            mean += obs[i];
        }
    int count = (S - 4) * (S - 4);
    mean /= count;
    for (int v = 2; v < S - 2; v++)
        for (int u = 2; u < S - 2; u++) {
            double d = obs[v * S + u] - mean;
            var += d * d;
        }
    if (var < 1e-6) return 1e9;
    double theta = .5 * atan2a(2 * jxy, jxx - jyy) + 1.5707963267948966, best = 1e30;
    static const double lengths[] = {0, 3, 6, 10};
    for (int t = 0; t < TILES * TILES; t++)
        for (int i = 0; i < S * S; i++) W[t][i] = 1.f;
    for (int k = 0; k < 4; k++) {
        lineBlurPadded(padded[1], S, Bt[0], lengths[k], theta, 2, 2, S - 2, S - 2, 2);
        double c = affineCost(Bt[0], S, W[0], 2, 2, S - 2, S - 2, 2) * 4;
        if (c < best) best = c;
    }
    return best / var;
}
/* Bounded re-localization for drifted tracking: normalized cross-correlation of the hypothesis template
   (1 px per module, lightly blurred) against the frame resampled around the predicted quiet-zone quad,
   over translations of +/-RM modules and five scales. Produces a geometry candidate only. */
#define RM    12
#define RSIDE (MAXN + 8 + 2 * RM)
static float region[RSIDE * RSIDE], tmpl[(MAXN + 8) * (MAXN + 8)], tmplG[(MAXN + 8) * (MAXN + 8)],
    tmplTmp[(MAXN + 8) * (MAXN + 8)];
static int quadMap(const double *q, double edge, double *h) {
    double dx1 = q[2] - q[4], dx2 = q[6] - q[4], dx3 = q[0] - q[2] + q[4] - q[6], dy1 = q[3] - q[5], dy2 = q[7] - q[5],
           dy3 = q[1] - q[3] + q[5] - q[7], den = dx1 * dy2 - dx2 * dy1;
    if (fabs(den) < 1e-12) return 0;
    double g = (dx3 * dy2 - dx2 * dy3) / den, r = (dx1 * dy3 - dx3 * dy1) / den;
    h[0] = (q[2] - q[0] + g * q[2]) / edge;
    h[1] = (q[6] - q[0] + r * q[6]) / edge;
    h[2] = q[0];
    h[3] = (q[3] - q[1] + g * q[3]) / edge;
    h[4] = (q[7] - q[1] + r * q[7]) / edge;
    h[5] = q[1];
    h[6] = g / edge;
    h[7] = r / edge;
    return 1;
}
static void quadPoint(const double *h, double x, double y, double *o) {
    double d = h[6] * x + h[7] * y + 1;
    o[0] = (h[0] * x + h[1] * y + h[2]) / d;
    o[1] = (h[3] * x + h[4] * y + h[5]) / d;
}
static double frameSample(const uint8_t *f, int w, int hgt, double x, double y) {
    if (!(x >= 0 && y >= 0 && x < w - 1 && y < hgt - 1)) return 255;
    int ix = (int)x, iy = (int)y;
    double fx = x - ix, fy = y - iy;
    const uint8_t *p = f + iy * w + ix;
    return (p[0] * (1 - fx) + p[1] * fx) * (1 - fy) + (p[w] * (1 - fx) + p[w + 1] * fx) * fy;
}
static double relocStats[8];
API uintptr_t se_reloc_stats(void) {
    return (uintptr_t)relocStats;
}
API int se_relocalize(const uint8_t *frame, int width, int height, const double *quiet, int version, double *out) {
    int n = 17 + 4 * version, E = n + 8, R = E + 2 * RM;
    double h[8];
    if (version < 1 || n > MAXN || !quadMap(quiet, E, h)) return 0;
    for (int y = 0; y < E; y++)
        for (int x = 0; x < E; x++) {
            int mx = x - 4, my = y - 4;
            tmpl[y * E + x] = (mx >= 0 && my >= 0 && mx < n && my < n) ? (float)(1 - grid[my * n + mx]) : 1.f;
        }
    gauss(tmpl, tmplG, tmplTmp, E, .8, 1.f);
    double tm = 0, ts = 0;
    for (int i = 0; i < E * E; i++) tm += tmplG[i];
    tm /= E * E;
    for (int i = 0; i < E * E; i++) {
        tmplG[i] -= (float)tm;
        ts += tmplG[i] * tmplG[i];
    }
    if (ts < 1e-9) return 0;
    static const double scales[5] = {.86, .93, 1, 1.07, 1.15};
    double best = -2, bs = 1, bx = 0, by = 0, c = E * .5;
    for (int si = 0; si < 5; si++) {
        double s = scales[si];
        for (int gy = 0; gy < R; gy++)
            for (int gx = 0; gx < R; gx++) {
                double acc = 0;
                for (int k = 0; k < 4; k++) {
                    double ux = c + (gx - RM + .25 + .5 * (k & 1) - c) * s,
                           uy = c + (gy - RM + .25 + .5 * (k >> 1) - c) * s, p[2];
                    quadPoint(h, ux, uy, p);
                    acc += frameSample(frame, width, height, p[0], p[1]);
                }
                region[gy * R + gx] = (float)(acc * .25);
            }
        for (int pass = 0; pass < 2; pass++) {
            int step = pass ? 1 : 2, cx0 = pass ? (int)bx + RM - 1 : 0, cx1 = pass ? (int)bx + RM + 1 : 2 * RM,
                cy0 = pass ? (int)by + RM - 1 : 0, cy1 = pass ? (int)by + RM + 1 : 2 * RM;
            if (pass && bs != s) break;
            for (int oy = cy0; oy <= cy1; oy += step)
                for (int ox = cx0; ox <= cx1; ox += step) {
                    if (ox < 0 || oy < 0 || ox > 2 * RM || oy > 2 * RM) continue;
                    double sr = 0, srr = 0, srt = 0;
                    for (int y = 0; y < E; y++) {
                        const float *row = region + (oy + y) * R + ox, *t = tmplG + y * E;
                        for (int x = 0; x < E; x++) {
                            double v = row[x];
                            sr += v;
                            srr += v * v;
                            srt += v * t[x];
                        }
                    }
                    double var = srr - sr * sr / (E * E);
                    if (var < 1e-6) continue;
                    double ncc = srt / sqrt(var * ts);
                    if (ncc > best) {
                        best = ncc;
                        bs = s;
                        bx = ox - RM;
                        by = oy - RM;
                    }
                }
        }
    }
    double corners[8] = {0, 0, E, 0, E, E, 0, E};
    for (int k = 0; k < 4; k++) {
        double ux = c + (corners[k * 2] + bx - c) * bs, uy = c + (corners[k * 2 + 1] + by - c) * bs;
        quadPoint(h, ux, uy, out + k * 2);
    }
    relocStats[0] = best;
    relocStats[1] = bs;
    relocStats[2] = bx;
    relocStats[3] = by;
    return best > .2;
}
