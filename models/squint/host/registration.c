/* Bounded inverse-compositional photometric homography. Static memory only.
   Registration proposes geometry; acceptance belongs to a separate verifier. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
#define MAXSIDE 260
#define MAXPIX  (MAXSIDE * MAXSIDE)
static uint8_t knownMask[3249];
static int knownSide;
API uintptr_t sa_known_mask(void) {
    return (uintptr_t)knownMask;
}
API int sa_known(int side) {
    if (side != 0 && (side < 21 || side > 41 || (side - 17) % 4)) return -1;
    knownSide = side;
    return 0;
}
static int included(int x, int y) {
    if (!knownSide) return 1;
    int mx = x / 4 - 4, my = y / 4 - 4;
    if (mx < 0 || my < 0 || mx >= knownSide || my >= knownSide) return 1;
    return knownMask[my * knownSide + mx] != 0;
}
static float current[MAXPIX], target[MAXPIX], aligned[MAXPIX], jac[MAXPIX * 8];
static double warp[9], result[8], normal[8][8];
API uintptr_t sa_current(void) {
    return (uintptr_t)current;
}
API uintptr_t sa_target(void) {
    return (uintptr_t)target;
}
API uintptr_t sa_aligned(void) {
    return (uintptr_t)aligned;
}
API uintptr_t sa_warp(void) {
    return (uintptr_t)warp;
}
API uintptr_t sa_result(void) {
    return (uintptr_t)result;
}
static double absd(double x) {
    return x < 0 ? -x : x;
}
static float sample(float *a, int n, double x, double y) {
    if (!(x >= 0 && y >= 0 && x < n - 1 && y < n - 1)) return 1;
    int ix = (int)x, iy = (int)y;
    double fx = x - ix, fy = y - iy;
    return (float)((a[iy * n + ix] * (1 - fx) + a[iy * n + ix + 1] * fx) * (1 - fy) +
                   (a[(iy + 1) * n + ix] * (1 - fx) + a[(iy + 1) * n + ix + 1] * fx) * fy);
}
static int solve(double a[8][9], double *out) {
    for (int c = 0; c < 8; c++) {
        int p = c;
        for (int r = c + 1; r < 8; r++)
            if (absd(a[r][c]) > absd(a[p][c])) p = r;
        if (absd(a[p][c]) < 1e-10) return 0;
        for (int j = c; j < 9; j++) {
            double tmp = a[p][j];
            a[p][j] = a[c][j];
            a[c][j] = tmp;
        }
        double d = a[c][c];
        for (int j = c; j < 9; j++) a[c][j] /= d;
        for (int r = 0; r < 8; r++)
            if (r != c) {
                d = a[r][c];
                for (int j = c; j < 9; j++) a[r][j] -= d * a[c][j];
            }
    }
    for (int i = 0; i < 8; i++) out[i] = a[i][8];
    return 1;
}
static int inverse(double *a, double *b) {
    double d =
        a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (absd(d) < 1e-10) return 0;
    b[0] = (a[4] * a[8] - a[5] * a[7]) / d;
    b[1] = (a[2] * a[7] - a[1] * a[8]) / d;
    b[2] = (a[1] * a[5] - a[2] * a[4]) / d;
    b[3] = (a[5] * a[6] - a[3] * a[8]) / d;
    b[4] = (a[0] * a[8] - a[2] * a[6]) / d;
    b[5] = (a[2] * a[3] - a[0] * a[5]) / d;
    b[6] = (a[3] * a[7] - a[4] * a[6]) / d;
    b[7] = (a[1] * a[6] - a[0] * a[7]) / d;
    b[8] = (a[0] * a[4] - a[1] * a[3]) / d;
    return 1;
}
API int sa_register(int n, int iterations, int stride) {
    if (n < 32 || n > MAXSIDE || iterations < 1 || iterations > 40 || stride < 1 || stride > 4) return -1;
    double radius = (n - 1) * .5, tm = 0, ts = 0, jm[8] = {0}, jt[8] = {0};
    int count = 0;
    for (int y = 2; y < n - 2; y += stride)
        for (int x = 2; x < n - 2; x += stride) {
            if (!included(x, y)) continue;
            tm += target[y * n + x];
            count++;
        }
    tm /= count;
    for (int y = 2; y < n - 2; y += stride)
        for (int x = 2; x < n - 2; x += stride) {
            if (!included(x, y)) continue;
            double v = target[y * n + x] - tm;
            ts += v * v;
        }
    if (ts < count * .0001) return 0;
    double tstd = __builtin_sqrt(ts / count);
    for (int y = 2; y < n - 2; y += stride)
        for (int x = 2; x < n - 2; x += stride) {
            if (!included(x, y)) continue;
            int i = y * n + x;
            double px = (x - radius) / radius, py = (y - radius) / radius;
            double gx = (target[i + 1] - target[i - 1]) * .5 * radius / tstd,
                   gy = (target[i + n] - target[i - n]) * .5 * radius / tstd;
            double j[8] = {
                gx * px, gx * py, gx, gy * px, gy * py, gy, -(gx * px + gy * py) * px, -(gx * px + gy * py) * py};
            double v = (target[i] - tm) / tstd;
            for (int k = 0; k < 8; k++) {
                jac[i * 8 + k] = (float)j[k];
                jm[k] += j[k];
                jt[k] += j[k] * v;
            }
        }
    for (int k = 0; k < 8; k++) {
        jm[k] /= count;
        jt[k] /= count;
        for (int l = 0; l < 8; l++) normal[k][l] = 0;
    }
    for (int y = 2; y < n - 2; y += stride)
        for (int x = 2; x < n - 2; x += stride) {
            if (!included(x, y)) continue;
            int i = y * n + x;
            double v = (target[i] - tm) / tstd;
            for (int k = 0; k < 8; k++) jac[i * 8 + k] -= (float)(jm[k] + jt[k] * v);
            for (int k = 0; k < 8; k++)
                for (int l = 0; l <= k; l++) normal[k][l] += jac[i * 8 + k] * jac[i * 8 + l];
        }
    for (int k = 0; k < 8; k++) {
        for (int l = 0; l < k; l++) normal[l][k] = normal[k][l];
        normal[k][k] += .001 * count;
    }
    for (int k = 0; k < 9; k++) warp[k] = (k % 4 == 0);
    double best = 1e30, bestwarp[9];
    for (int k = 0; k < 9; k++) bestwarp[k] = warp[k];
    int used = 0;
    for (int iteration = 0; iteration < iterations; iteration++) {
        double im = 0, is = 0;
        int visible = 0;
        for (int y = 2; y < n - 2; y += stride)
            for (int x = 2; x < n - 2; x += stride) {
                if (!included(x, y)) continue;
                double px = (x - radius) / radius, py = (y - radius) / radius,
                       d = warp[6] * px + warp[7] * py + warp[8];
                double u = (warp[0] * px + warp[1] * py + warp[2]) / d * radius + radius,
                       v = (warp[3] * px + warp[4] * py + warp[5]) / d * radius + radius;
                visible += (u >= 0 && u < n - 1 && v >= 0 && v < n - 1);
                float z = sample(current, n, u, v);
                aligned[y * n + x] = z;
                im += z;
            }
        if (visible < count * .85) break;
        im /= count;
        for (int y = 2; y < n - 2; y += stride)
            for (int x = 2; x < n - 2; x += stride) {
                if (!included(x, y)) continue;
                double v = aligned[y * n + x] - im;
                is += v * v;
            }
        if (is < count * .0001) break;
        double istd = __builtin_sqrt(is / count), rhs[8] = {0}, loss = 0;
        for (int y = 2; y < n - 2; y += stride)
            for (int x = 2; x < n - 2; x += stride) {
                if (!included(x, y)) continue;
                int i = y * n + x;
                double e = (aligned[i] - im) / istd - (target[i] - tm) / tstd;
                loss += e * e;
                for (int k = 0; k < 8; k++) rhs[k] += jac[i * 8 + k] * e;
            }
        used = iteration + 1;
        if (loss < best) {
            best = loss;
            for (int k = 0; k < 9; k++) bestwarp[k] = warp[k];
        }
        double a[8][9], delta[8];
        for (int k = 0; k < 8; k++) {
            for (int l = 0; l < 8; l++) a[k][l] = normal[k][l];
            a[k][8] = rhs[k];
        }
        if (!solve(a, delta)) break;
        double magnitude = 0;
        for (int k = 0; k < 8; k++) magnitude += delta[k] * delta[k];
        if (magnitude < 1e-10) break;
        double scale = magnitude > .02 ? __builtin_sqrt(.02 / magnitude) : 1.;
        double update[9] = {1 + delta[0] * scale, delta[1] * scale,     delta[2] * scale,
                            delta[3] * scale,     1 + delta[4] * scale, delta[5] * scale,
                            delta[6] * scale,     delta[7] * scale,     1},
               inv[9], next[9];
        if (!inverse(update, inv)) break;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                next[r * 3 + c] = 0;
                for (int k = 0; k < 3; k++) next[r * 3 + c] += warp[r * 3 + k] * inv[k * 3 + c];
            }
        for (int k = 0; k < 9; k++) warp[k] = next[k] / next[8];
    }
    for (int k = 0; k < 9; k++) warp[k] = bestwarp[k];
    double maximum = 0;
    for (int y = -1; y <= 1; y += 2)
        for (int x = -1; x <= 1; x += 2) {
            double d = warp[6] * x + warp[7] * y + warp[8], u = (warp[0] * x + warp[1] * y + warp[2]) / d,
                   v = (warp[3] * x + warp[4] * y + warp[5]) / d;
            double distance = __builtin_sqrt((u - x) * (u - x) + (v - y) * (v - y)) * radius;
            if (distance > maximum) maximum = distance;
        }
    result[0] = best / count;
    result[1] = maximum;
    result[2] = used;
    if (!(maximum < n * .10 && best < count * 2)) return 0;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            double px = (x - radius) / radius, py = (y - radius) / radius, d = warp[6] * px + warp[7] * py + warp[8];
            aligned[y * n + x] = sample(current, n, (warp[0] * px + warp[1] * py + warp[2]) / d * radius + radius,
                                        (warp[3] * px + warp[4] * py + warp[5]) / d * radius + radius);
        }
    return 1;
}

/* Small smooth displacement field, fitted to payload-independent QR structure. */
static int solve12(double a[12][13], double *out) {
    for (int c = 0; c < 12; c++) {
        int p = c;
        for (int r = c + 1; r < 12; r++)
            if (absd(a[r][c]) > absd(a[p][c])) p = r;
        if (absd(a[p][c]) < 1e-9) return 0;
        for (int j = c; j < 13; j++) {
            double t = a[p][j];
            a[p][j] = a[c][j];
            a[c][j] = t;
        }
        double d = a[c][c];
        for (int j = c; j < 13; j++) a[c][j] /= d;
        for (int r = 0; r < 12; r++)
            if (r != c) {
                d = a[r][c];
                for (int j = c; j < 13; j++) a[r][j] -= d * a[c][j];
            }
    }
    for (int i = 0; i < 12; i++) out[i] = a[i][12];
    return 1;
}
static void polynomial(double x, double y, double *b) {
    b[0] = 1;
    b[1] = x;
    b[2] = y;
    b[3] = x * x;
    b[4] = x * y;
    b[5] = y * y;
}
API int sa_polynomial(int n, int iterations) {
    if (n < 116 || n > 196 || iterations < 1 || iterations > 12 || !knownSide) return -1;
    double coefficients[12] = {0}, bestCoefficients[12] = {0}, best = 1e30, radius = (n - 1) * .5;
    for (int iteration = 0; iteration < iterations; iteration++) {
        double normal12[12][13] = {{0}}, loss = 0;
        int count = 0;
        for (int y = 2; y < n - 2; y += 4)
            for (int x = 2; x < n - 2; x += 4) {
                if (!included(x, y)) continue;
                double b[6], u = x, v = y;
                polynomial((x - radius) / radius, (y - radius) / radius, b);
                for (int k = 0; k < 6; k++) {
                    u += coefficients[k] * b[k];
                    v += coefficients[k + 6] * b[k];
                }
                double value = sample(current, n, u, v), error = value - target[y * n + x],
                       gx = (sample(current, n, u + 1, v) - sample(current, n, u - 1, v)) * .5,
                       gy = (sample(current, n, u, v + 1) - sample(current, n, u, v - 1)) * .5, j[12];
                for (int k = 0; k < 6; k++) {
                    j[k] = gx * b[k];
                    j[k + 6] = gy * b[k];
                }
                double weight = absd(error) > .15 ? .15 / absd(error) : 1;
                for (int k = 0; k < 12; k++) {
                    for (int l = 0; l < 12; l++) normal12[k][l] += weight * j[k] * j[l];
                    normal12[k][12] -= weight * j[k] * error;
                }
                loss += error * error;
                count++;
            }
        if (!count) return 0;
        if (loss < best) {
            best = loss;
            for (int k = 0; k < 12; k++) bestCoefficients[k] = coefficients[k];
        }
        for (int k = 0; k < 12; k++) normal12[k][k] += .0005 * count;
        double delta[12];
        if (!solve12(normal12, delta)) break;
        double step = 0;
        for (int k = 0; k < 12; k++) step += delta[k] * delta[k];
        if (step < 1e-7) break;
        double scale = step > 2 ? __builtin_sqrt(2 / step) : 1;
        for (int k = 0; k < 12; k++) coefficients[k] += delta[k] * scale;
        double maximum = 0;
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                double b[6], dx = 0, dy = 0;
                polynomial(x, y, b);
                for (int k = 0; k < 6; k++) {
                    dx += coefficients[k] * b[k];
                    dy += coefficients[k + 6] * b[k];
                }
                double d = dx * dx + dy * dy;
                if (d > maximum) maximum = d;
            }
        if (maximum > 9) break;
    }
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            double b[6], u = x, v = y;
            polynomial((x - radius) / radius, (y - radius) / radius, b);
            for (int k = 0; k < 6; k++) {
                u += bestCoefficients[k] * b[k];
                v += bestCoefficients[k + 6] * b[k];
            }
            aligned[y * n + x] = sample(current, n, u, v);
        }
    return 1;
}
