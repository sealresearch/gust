/* Squint acquisition without third-party readers: finder-pattern detection, triplet geometry, version
   hypotheses, finder/alignment refinement, soft sampling and Squint's own Reed-Solomon decoder.
   Static memory only. Accepts only payloads that pass the decoder's RS, re-encoding and parity-reserve checks. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
extern uintptr_t sg_frame(void), sg_quad(void), sg_result(void), sg_soft_grid(void), sg_visible(void),
    sg_known_mask(void), sg_known_values(void);
extern int sg_fit(int, int, int, int, int), sg_sample_grid(int, int, int, int, double, double),
    sg_partial_sampling(int), sg_normalized(int), sg_search_radius(int);
extern uintptr_t sq_input(void), sq_visible(void), sq_result(void);
extern int sq_decode(int, int, double, int, int, double, double, int, double), sq_gmd(int), sq_reserve(int),
    sq_padding(int), sq_visibility(int), sq_reliability(int), sq_known_layout(int, uint8_t *, uint8_t *);
#define MAXW 1280
#define MAXH 720
static uint8_t dark[MAXW * MAXH], half[(MAXW / 2) * (MAXH / 2)];
static uint32_t integ[(MAXW + 1) * (MAXH + 1)];
typedef struct {
    float x, y, module;
    int count;
} Finder;
static Finder finders[96];
static int finderCount;
static double acquiredQuad[8], acquireStats[16];
static int acquiredVersion;
API uintptr_t sa_acquired_quad(void) {
    return (uintptr_t)acquiredQuad;
}
API uintptr_t sa_acquire_stats(void) {
    return (uintptr_t)acquireStats;
}
API int sa_acquired_version(void) {
    return acquiredVersion;
}
static float absf(float v) {
    return v < 0 ? -v : v;
}
static int ratioOK(const int *r) {
    int total = r[0] + r[1] + r[2] + r[3] + r[4];
    if (total < 7) return 0;
    {
        float o = (r[0] + r[1] + r[3] + r[4]) * .25f;
        return r[2] >= 1.6f * o && r[2] <= 6.f * o && absf(r[0] - o) < o * .8f && absf(r[1] - o) < o * .8f &&
               absf(r[3] - o) < o * .8f && absf(r[4] - o) < o * .8f;
    }
}
/* Runs through (x,y) along (dx,dy); returns centre offset and module size if the 1:1:3:1:1 pattern holds. */
static int crossCheck(int w, int h, int x, int y, int dx, int dy, float *centre, float *module) {
    if (!dark[y * w + x]) return 0;
    int r[5] = {0, 0, 0, 0, 0}, px = x, py = y;
    while (px >= 0 && py >= 0 && px < w && py < h && dark[py * w + px]) {
        r[2]++;
        px -= dx;
        py -= dy;
    }
    int back = r[2];
    while (px >= 0 && py >= 0 && px < w && py < h && !dark[py * w + px] && r[1] <= back * 2) {
        r[1]++;
        px -= dx;
        py -= dy;
    }
    while (px >= 0 && py >= 0 && px < w && py < h && dark[py * w + px] && r[0] <= back * 2) {
        r[0]++;
        px -= dx;
        py -= dy;
    }
    px = x + dx;
    py = y + dy;
    int fwd = 0;
    while (px >= 0 && py >= 0 && px < w && py < h && dark[py * w + px]) {
        r[2]++;
        fwd++;
        px += dx;
        py += dy;
    }
    while (px >= 0 && py >= 0 && px < w && py < h && !dark[py * w + px] && r[3] <= r[2] * 2) {
        r[3]++;
        px += dx;
        py += dy;
    }
    while (px >= 0 && py >= 0 && px < w && py < h && dark[py * w + px] && r[4] <= r[2] * 2) {
        r[4]++;
        px += dx;
        py += dy;
    }
    if (!ratioOK(r)) return 0;
    *centre = (fwd - (back - 1)) * .5f;
    *module = (r[0] + r[1] + r[2] + r[3] + r[4]) / 7.f;
    return 1;
}
static void addFinder(float x, float y, float m) {
    for (int i = 0; i < finderCount; i++) {
        Finder *f = finders + i;
        float d = (f->x - x) * (f->x - x) + (f->y - y) * (f->y - y), lim = (f->module > m ? f->module : m) * 2.5f;
        if (d < lim * lim && m < f->module * 1.6f && f->module < m * 1.6f) {
            f->x = (f->x * f->count + x) / (f->count + 1);
            f->y = (f->y * f->count + y) / (f->count + 1);
            f->module = (f->module * f->count + m) / (f->count + 1);
            f->count++;
            return;
        }
    }
    if (finderCount < 96) {
        finders[finderCount].x = x;
        finders[finderCount].y = y;
        finders[finderCount].module = m;
        finders[finderCount].count = 1;
        finderCount++;
    }
}
/* Adaptive threshold: dark if below 93% of the local mean (integral image). */
static int fullTick = 0, lightingMode = 0;
extern int sg_lighting(int, int);
extern int sg_glare(int), sg_local_levels(int);
API int sa_local_levels(int enabled) {
    return sg_local_levels(enabled);
}
API int sa_glare(int enabled) {
    return sg_glare(enabled);
}
API int sa_lighting(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    lightingMode = enabled;
    return sg_lighting(enabled, enabled);
}
static void binarize(const uint8_t *g, int w, int h, int window, int percent) {
    for (int x = 0; x <= w; x++) integ[x] = 0;
    for (int y = 1; y <= h; y++) {
        uint32_t row = 0;
        integ[y * (w + 1)] = 0;
        for (int x = 1; x <= w; x++) {
            row += g[(y - 1) * w + x - 1];
            integ[y * (w + 1) + x] = integ[(y - 1) * (w + 1) + x] + row;
        }
    }
    int r = window / 2;
    for (int y = 0; y < h; y++) {
        int y0 = y - r < 0 ? 0 : y - r, y1 = y + r + 1 > h ? h : y + r + 1;
        for (int x = 0; x < w; x++) {
            int x0 = x - r < 0 ? 0 : x - r, x1 = x + r + 1 > w ? w : x + r + 1;
            uint32_t s = integ[y1 * (w + 1) + x1] - integ[y0 * (w + 1) + x1] - integ[y1 * (w + 1) + x0] +
                         integ[y0 * (w + 1) + x0];
            uint32_t area = (uint32_t)((y1 - y0) * (x1 - x0));
            dark[y * w + x] = (uint32_t)g[y * w + x] * area * 100u < s * (uint32_t)percent;
        }
    }
}

/* Offset-invariant binarization: dark if below the midpoint of the local (block, 3x3-dilated) min and max.
   Robust to veiling glare and flat lighting where a mean-ratio threshold shifts. */
static uint8_t blockMin[(MAXW / 8 + 2) * (MAXH / 8 + 2)], blockMax[(MAXW / 8 + 2) * (MAXH / 8 + 2)];
static void binarizeMidrange(const uint8_t *g, int w, int h, int block) {
    int bw = (w + block - 1) / block, bh = (h + block - 1) / block;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            int mn = 255, mx = 0;
            for (int y = by * block; y < (by + 1) * block && y < h; y++)
                for (int x = bx * block; x < (bx + 1) * block && x < w; x++) {
                    int v = g[y * w + x];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            blockMin[by * bw + bx] = (uint8_t)mn;
            blockMax[by * bw + bx] = (uint8_t)mx;
        }
    for (int y = 0; y < h; y++) {
        int by = y / block;
        for (int x = 0; x < w; x++) {
            int bx = x / block, mn = 255, mx = 0;
            for (int dy = -2; dy <= 2; dy++) {
                int yy = by + dy;
                if (yy < 0 || yy >= bh) continue;
                for (int dx = -2; dx <= 2; dx++) {
                    int xx = bx + dx;
                    if (xx < 0 || xx >= bw) continue;
                    int a = blockMin[yy * bw + xx], b = blockMax[yy * bw + xx];
                    if (a < mn) mn = a;
                    if (b > mx) mx = b;
                }
            }
            dark[y * w + x] = (mx - mn >= 10) && (g[y * w + x] * 2 < mn + mx);
        }
    }
}
static void detect(int w, int h, int step) {
    for (int y = 0; y < h; y += step) {
        int runs[5] = {0}, colors[5] = {0}, count = 0, start = 0, color = dark[y * w];
        for (int x = 1; x <= w; x++) {
            int c = x < w ? dark[y * w + x] : !color;
            if (c == color) continue;
            for (int k = 0; k < 4; k++) {
                runs[k] = runs[k + 1];
                colors[k] = colors[k + 1];
            }
            runs[4] = x - start;
            colors[4] = color;
            count++;
            if (count >= 5 && colors[0] && colors[2] && colors[4] && ratioOK(runs)) {
                int cx = x - runs[4] - runs[3] - runs[2] / 2 - 1;
                float cy, mv, cxo, mh;
                if (cx >= 0 && crossCheck(w, h, cx, y, 0, 1, &cy, &mv)) {
                    int yc = (int)(y + cy + .5f);
                    if (yc >= 0 && yc < h && crossCheck(w, h, cx, yc, 1, 0, &cxo, &mh) && mh < mv * 1.6f &&
                        mv < mh * 1.6f) {
                        int xc = (int)(cx + cxo + .5f);
                        if (xc >= 0 && xc < w) addFinder(cx + cxo, y + cy, (mh + mv) * .5f);
                    }
                }
            }
            color = c;
            start = x;
        }
    }
}
static int tryDecode(int width, int height, int version, const double *quiet, double deadline, double (*now)(void)) {
    int n = 17 + 4 * version;
    if (sq_known_layout(version, (uint8_t *)sg_known_mask(), (uint8_t *)sg_known_values()) != n) return 0;
    double *q = (double *)sg_quad(), *res = (double *)sg_result(), fitted[8];
    int haveFit = 0;
    for (int k = 0; k < 8; k++) q[k] = quiet[k];
    if (now && now() >= deadline) return 0;
    if (!sg_fit(width, height, width, version, 1) && res[0]) {
        for (int k = 0; k < 8; k++) fitted[k] = res[5 + k];
        haveFit = 1;
    }
    for (int attempt = 0; attempt < 2; attempt++) {
        const double *Q = (attempt == 0 && haveFit) ? fitted : quiet;
        if (attempt == 1 && !haveFit) break;
        if (now && now() >= deadline) return 0;
        double h[8], dx1 = Q[2] - Q[4], dx2 = Q[6] - Q[4], dx3 = Q[0] - Q[2] + Q[4] - Q[6], dy1 = Q[3] - Q[5],
                     dy2 = Q[7] - Q[5], dy3 = Q[1] - Q[3] + Q[5] - Q[7], den = dx1 * dy2 - dx2 * dy1;
        if (fabs(den) < 1e-12) continue;
        double gg = (dx3 * dy2 - dx2 * dy3) / den, rr = (dx1 * dy3 - dx3 * dy1) / den;
        h[0] = Q[2] - Q[0] + gg * Q[2];
        h[1] = Q[6] - Q[0] + rr * Q[6];
        h[2] = Q[0];
        h[3] = Q[3] - Q[1] + gg * Q[3];
        h[4] = Q[7] - Q[1] + rr * Q[7];
        h[5] = Q[1];
        h[6] = gg;
        h[7] = rr;
        double lo = 4. / (n + 8), hi = 1 - lo, uv[8] = {lo, lo, hi, lo, hi, hi, lo, hi}, code[8];
        for (int k = 0; k < 4; k++) {
            double u = uv[k * 2], v = uv[k * 2 + 1], d = h[6] * u + h[7] * v + 1;
            code[k * 2] = (h[0] * u + h[1] * v + h[2]) / d;
            code[k * 2 + 1] = (h[3] * u + h[4] * v + h[5]) / d;
        }
        sg_partial_sampling(1);
        static const double offs[3][2] = {{0, 0}, {.15, .15}, {-.15, -.15}};
        for (int o = 0; o < 3; o++) {
            if (now && now() >= deadline) return 0;
            for (int k = 0; k < 8; k++) q[k] = code[k];
            if (sg_sample_grid(width, height, width, version, offs[o][0], offs[o][1]) != 1) continue;
            double *in = (double *)sq_input();
            const double *sg = (const double *)sg_soft_grid();
            uint8_t *vis = (uint8_t *)sq_visible();
            const uint8_t *sv = (const uint8_t *)sg_visible();
            for (int i = 0; i < n * n; i++) {
                in[i] = sg[i];
                vis[i] = sv[i];
            }
            sq_visibility(1);
            sq_padding(1);
            sq_gmd(1);
            sq_reserve(2);
            for (int rel = 0; rel < 3; rel++) {
                if (rel && now && now() >= deadline) return 0;
                sq_reliability(rel);
                if (sq_decode(n, 1, 1., 30, 1, .12, .10, 1, .05) == 1) {
                    for (int k = 0; k < 8; k++) acquiredQuad[k] = code[k];
                    acquiredVersion = version;
                    return 1;
                }
            }
        }
    }
    return 0;
}
/* Returns 1 when a payload was decoded (decoder result buffers hold it), 0 otherwise. The frame must already be in sg_frame. */
API int sa_acquire(int width, int height, double deadline, double (*now)(void)) {
    if (width < 64 || height < 64 || width > MAXW || height > MAXH) return -1;
    const uint8_t *g = (const uint8_t *)sg_frame();
    for (int i = 0; i < 16; i++) acquireStats[i] = 0;
    finderCount = 0;
    sg_normalized(1);
    sg_search_radius(2);
    double t0 = now ? now() : 0;
    /* Stage 1: half-resolution detection (codes with modules >= ~3 px). Coordinates are mapped back to full resolution. */
    int hw = width / 2, hh = height / 2;
    for (int y = 0; y < hh; y++)
        for (int x = 0; x < hw; x++) {
            const uint8_t *p = g + (y * 2) * width + x * 2;
            half[y * hw + x] = (uint8_t)((p[0] + p[1] + p[width] + p[width + 1] + 2) >> 2);
        }
    binarize(half, hw, hh, 21, 93);
    detect(hw, hh, 1);
    for (int i = 0; i < finderCount; i++) {
        finders[i].x = finders[i].x * 2 + .5f;
        finders[i].y = finders[i].y * 2 + .5f;
        finders[i].module *= 2;
    }
    {
        int c3 = 0;
        for (int i = 0; i < finderCount; i++) c3 += finders[i].count >= 2;
        if (c3 < 3 && lightingMode) {
            int keep = finderCount;
            binarizeMidrange(half, hw, hh, 8);
            detect(hw, hh, 1);
            for (int i = keep; i < finderCount; i++) {
                finders[i].x = finders[i].x * 2 + .5f;
                finders[i].y = finders[i].y * 2 + .5f;
                finders[i].module *= 2;
            }
        }
    }
    int good[24], ng = 0;
    for (int i = 0; i < finderCount && ng < 24; i++)
        if (finders[i].count >= 2 && finders[i].module >= 2.5f) good[ng++] = i;
    acquireStats[5] = now ? (now() - t0) * 1000 : 0;
    /* Stage 2: full-resolution detection only when stage 1 found fewer than three usable finders. */
    int halfFound = 0;
    for (int i = 0; i < finderCount; i++) halfFound += finders[i].count >= 2;
    fullTick++;
    if (ng < 3 && (halfFound > 0 || (fullTick & 1)) && !(now && now() >= deadline)) {
        int keep = finderCount;
        Finder saved[96];
        for (int i = 0; i < keep; i++) saved[i] = finders[i];
        finderCount = 0;
        binarize(g, width, height, width >= 960 ? 41 : 25, 93);
        detect(width, height, 1);
        if (!(now && now() >= deadline)) {
            int count3 = 0;
            for (int i = 0; i < finderCount; i++) count3 += finders[i].count >= 2;
            if (count3 > 0 && count3 < 3) {
                binarize(g, width, height, width >= 960 ? 81 : 49, 85);
                detect(width, height, 2);
            }
        }
        for (int i = 0; i < keep; i++) addFinder(saved[i].x, saved[i].y, saved[i].module);
        ng = 0;
        for (int i = 0; i < finderCount && ng < 24; i++)
            if (finders[i].count >= 2 && finders[i].module >= 1.5f) good[ng++] = i;
    }
    acquireStats[6] = now ? (now() - t0) * 1000 : 0;
    acquireStats[0] = finderCount;
    acquireStats[1] = ng;
    typedef struct {
        int a, b, c;
        double score;
    } Tri;
    static Tri tris[64];
    int nt = 0;
    for (int i = 0; i < ng; i++)
        for (int j = i + 1; j < ng; j++)
            for (int k = j + 1; k < ng; k++) {
                Finder *F[3] = {finders + good[i], finders + good[j], finders + good[k]};
                float m0 = F[0]->module, m1 = F[1]->module, m2 = F[2]->module,
                      mx = m0 > m1 ? (m0 > m2 ? m0 : m2) : (m1 > m2 ? m1 : m2),
                      mn = m0 < m1 ? (m0 < m2 ? m0 : m2) : (m1 < m2 ? m1 : m2);
                if (mx > mn * 1.7f) continue;
                double d[3];
                for (int e = 0; e < 3; e++) {
                    Finder *p = F[(e + 1) % 3], *r = F[(e + 2) % 3];
                    d[e] = sqrt((p->x - r->x) * (p->x - r->x) + (p->y - r->y) * (p->y - r->y));
                }
                int tl = d[0] > d[1] ? (d[0] > d[2] ? 0 : 2) : (d[1] > d[2] ? 1 : 2);
                Finder *A = F[tl], *B = F[(tl + 1) % 3], *C = F[(tl + 2) % 3];
                double ux = B->x - A->x, uy = B->y - A->y, vx = C->x - A->x, vy = C->y - A->y,
                       lu = sqrt(ux * ux + uy * uy), lv = sqrt(vx * vx + vy * vy);
                if (lu < 7 * mn || lv < 7 * mn) continue;
                double cosang = fabs((ux * vx + uy * vy) / (lu * lv)), ratio = lu > lv ? lu / lv : lv / lu;
                if (cosang > .45 || ratio > 1.6) continue;
                if (nt < 64) {
                    int ids[3] = {good[i], good[j], good[k]}, bi = ids[(tl + 1) % 3], ci = ids[(tl + 2) % 3];
                    if (ux * vy - uy * vx < 0) {
                        int t = bi;
                        bi = ci;
                        ci = t;
                    }
                    tris[nt].a = ids[tl];
                    tris[nt].b = bi;
                    tris[nt].c = ci;
                    tris[nt].score = cosang + (ratio - 1);
                    nt++;
                }
            }
    for (int i = 1; i < nt; i++) {
        Tri t = tris[i];
        int j = i - 1;
        while (j >= 0 && tris[j].score > t.score) {
            tris[j + 1] = tris[j];
            j--;
        }
        tris[j + 1] = t;
    }
    acquireStats[2] = nt;
    for (int t = 0; t < nt && t < 4; t++) {
        Finder *A = finders + tris[t].a, *B = finders + tris[t].b, *C = finders + tris[t].c;
        double m = (A->module + B->module + C->module) / 3,
               dist = (sqrt((B->x - A->x) * (B->x - A->x) + (B->y - A->y) * (B->y - A->y)) +
                       sqrt((C->x - A->x) * (C->x - A->x) + (C->y - A->y) * (C->y - A->y))) *
                      .5;
        int v0 = (int)floor((dist / m + 7 - 17) / 4 + .5);
        for (int dv = 0; dv < 3; dv++) {
            int version = v0 + (dv == 0 ? 0 : dv == 1 ? 1 : -1);
            if (version < 1 || version > 10) continue;
            if (now && now() >= deadline) return 0;
            int n = 17 + 4 * version;
            double ex = (B->x - A->x) / (n - 7), ey = (B->y - A->y) / (n - 7), fx = (C->x - A->x) / (n - 7),
                   fy = (C->y - A->y) / (n - 7), ox = A->x - 3.5 * (ex + fx), oy = A->y - 3.5 * (ey + fy);
            double cu[4] = {-4, n + 4, n + 4, -4}, cv[4] = {-4, -4, n + 4, n + 4}, quiet[8];
            for (int k = 0; k < 4; k++) {
                quiet[k * 2] = ox + cu[k] * ex + cv[k] * fx;
                quiet[k * 2 + 1] = oy + cu[k] * ey + cv[k] * fy;
            }
            if (tryDecode(width, height, version, quiet, deadline, now)) {
                acquireStats[3] = t;
                acquireStats[4] = version;
                return 1;
            }
        }
    }
    if (now && now() >= deadline) return 0;
    /* Two-finder hypotheses: the third finder lies a right angle away along either side, or across the diagonal. */
    int pairs = 0;
    for (int i = 0; i < ng && pairs < 3; i++)
        for (int j = i + 1; j < ng && pairs < 3; j++) {
            Finder *P = finders + good[i], *Q = finders + good[j];
            float mn = P->module < Q->module ? P->module : Q->module,
                  mx = P->module > Q->module ? P->module : Q->module;
            if (mx > mn * 1.7f) continue;
            double dx = Q->x - P->x, dy = Q->y - P->y, dist = sqrt(dx * dx + dy * dy), m = (P->module + Q->module) * .5;
            if (dist < 7 * m) continue;
            pairs++;
            double cand[6][2] = {{P->x - dy, P->y + dx},
                                 {P->x + dy, P->y - dx},
                                 {Q->x - dy, Q->y + dx},
                                 {Q->x + dy, Q->y - dx},
                                 {(P->x + Q->x) * .5 - dy * .5, (P->y + Q->y) * .5 + dx * .5},
                                 {(P->x + Q->x) * .5 + dy * .5, (P->y + Q->y) * .5 - dx * .5}};
            for (int c = 0; c < 6; c++) {
                if (now && now() >= deadline) return 0;
                double X[3][2] = {{P->x, P->y}, {Q->x, Q->y}, {cand[c][0], cand[c][1]}};
                double d[3];
                for (int e = 0; e < 3; e++) {
                    int a = (e + 1) % 3, b = (e + 2) % 3;
                    d[e] = sqrt((X[a][0] - X[b][0]) * (X[a][0] - X[b][0]) + (X[a][1] - X[b][1]) * (X[a][1] - X[b][1]));
                }
                int tl = d[0] > d[1] ? (d[0] > d[2] ? 0 : 2) : (d[1] > d[2] ? 1 : 2);
                double *A = X[tl], *B = X[(tl + 1) % 3], *C = X[(tl + 2) % 3];
                double ux = B[0] - A[0], uy = B[1] - A[1], vx = C[0] - A[0], vy = C[1] - A[1];
                if (ux * vy - uy * vx < 0) {
                    double *t = B;
                    B = C;
                    C = t;
                }
                double side = (sqrt((B[0] - A[0]) * (B[0] - A[0]) + (B[1] - A[1]) * (B[1] - A[1])) +
                               sqrt((C[0] - A[0]) * (C[0] - A[0]) + (C[1] - A[1]) * (C[1] - A[1]))) *
                              .5;
                int v0 = (int)floor((side / m + 7 - 17) / 4 + .5);
                for (int dv = 0; dv < 3; dv++) {
                    int version = v0 + (dv == 0 ? 0 : dv == 1 ? 1 : -1);
                    if (version < 1 || version > 10) continue;
                    int n = 17 + 4 * version;
                    double ex = (B[0] - A[0]) / (n - 7), ey = (B[1] - A[1]) / (n - 7), fx = (C[0] - A[0]) / (n - 7),
                           fy = (C[1] - A[1]) / (n - 7), ox = A[0] - 3.5 * (ex + fx), oy = A[1] - 3.5 * (ey + fy);
                    double cu[4] = {-4, n + 4, n + 4, -4}, cv[4] = {-4, -4, n + 4, n + 4}, quiet[8];
                    for (int k = 0; k < 4; k++) {
                        quiet[k * 2] = ox + cu[k] * ex + cv[k] * fx;
                        quiet[k * 2 + 1] = oy + cu[k] * ey + cv[k] * fy;
                    }
                    if (tryDecode(width, height, version, quiet, deadline, now)) {
                        acquireStats[3] = 100 + c;
                        acquireStats[4] = version;
                        return 1;
                    }
                }
            }
        }
    return 0;
}
static double wasmNow(void) {
    return seconds();
}
API int sa_acquire_wasm(int width, int height, double deadlineSeconds) {
    return sa_acquire(width, height, deadlineSeconds, wasmNow);
}
/* Enhanced view for difficult lighting (used on alternate frames): local contrast normalization from dilated/eroded
   block extrema (removes veiling, exposure offsets and flat low-light gain) followed by a light binomial denoise.
   Rewrites sg_frame in place; the original frame is no longer needed by acquisition afterwards. */
static uint8_t lmin[160 * 90], lmax[160 * 90], tmpb[160 * 90];
static float fmin_[161 * 91], fmax_[161 * 91];
API int sa_enhance_frame(int width, int height) {
    if (width != 1280 || height != 720) return -1;
    uint8_t *g = (uint8_t *)sg_frame();
    const int bw = 160, bh = 90;
    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++) {
            int mn = 255, mx = 0;
            for (int y = by * 8; y < by * 8 + 8; y++)
                for (int x = bx * 8; x < bx * 8 + 8; x++) {
                    int v = g[y * width + x];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            lmin[by * bw + bx] = (uint8_t)mn;
            lmax[by * bw + bx] = (uint8_t)mx;
        }
    /* 5x5 block erosion/dilation (separable) */
    for (int pass = 0; pass < 2; pass++) {
        uint8_t *a = pass ? lmax : lmin;
        for (int y = 0; y < bh; y++)
            for (int x = 0; x < bw; x++) {
                int v = a[y * bw + x];
                for (int d = -2; d <= 2; d++) {
                    int xx = x + d;
                    if (xx < 0 || xx >= bw) continue;
                    int w = a[y * bw + xx];
                    if (pass ? w > v : w < v) v = w;
                }
                tmpb[y * bw + x] = (uint8_t)v;
            }
        for (int y = 0; y < bh; y++)
            for (int x = 0; x < bw; x++) {
                int v = tmpb[y * bw + x];
                for (int d = -2; d <= 2; d++) {
                    int yy = y + d;
                    if (yy < 0 || yy >= bh) continue;
                    int w = tmpb[yy * bw + x];
                    if (pass ? w > v : w < v) v = w;
                }
                a[y * bw + x] = (uint8_t)v;
            }
    }
    /* 3x3 smoothing of the extrema grids */
    for (int y = 0; y < bh; y++)
        for (int x = 0; x < bw; x++) {
            float s0 = 0, s1 = 0;
            int c = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= bh || xx < 0 || xx >= bw) continue;
                    s0 += lmin[yy * bw + xx];
                    s1 += lmax[yy * bw + xx];
                    c++;
                }
            fmin_[y * bw + x] = s0 / c;
            fmax_[y * bw + x] = s1 / c;
        }
    /* per-row interpolation of offset/gain at block resolution, then fixed-point per pixel; 1-2-1 vertical+horizontal denoise */
    static int16_t offRow[160];
    static int32_t gainRow[160];
    static uint8_t prev[1280], cur[1280];
    for (int x = 0; x < width; x++) prev[x] = g[x];
    for (int y = 0; y < height; y++) {
        float gy = (y + .5f) / 8 - .5f;
        if (gy < 0) gy = 0;
        if (gy > bh - 1) gy = bh - 1;
        int y0 = (int)gy, y1 = y0 + 1 < bh ? y0 + 1 : y0;
        float fy = gy - y0;
        for (int bx = 0; bx < bw; bx++) {
            float mn = fmin_[y0 * bw + bx] * (1 - fy) + fmin_[y1 * bw + bx] * fy,
                  mx = fmax_[y0 * bw + bx] * (1 - fy) + fmax_[y1 * bw + bx] * fy, range = mx - mn;
            if (range < 6) range = 6;
            offRow[bx] = (int16_t)(mn * 16);
            gainRow[bx] = (int32_t)(255.f * 4096.f / range);
        }
        const uint8_t *r1 = g + y * width, *r2 = g + (y < height - 1 ? y + 1 : y) * width;
        for (int x = 0; x < width; x++) cur[x] = r1[x];
        for (int x = 0; x < width; x++) {
            int xl = x > 0 ? x - 1 : 0, xr = x < width - 1 ? x + 1 : x;
            int v = (prev[xl] + 2 * prev[x] + prev[xr] + 2 * (cur[xl] + 2 * cur[x] + cur[xr]) + r2[xl] + 2 * r2[x] +
                     r2[xr]); /* 16x */
            int bx = x >> 3;
            int o = ((v - offRow[bx]) * gainRow[bx]) >> 16;
            g[y * width + x] = (uint8_t)(o < 0 ? 0 : o > 255 ? 255 : o);
        }
        for (int x = 0; x < width; x++) prev[x] = cur[x];
    }
    return 0;
}
