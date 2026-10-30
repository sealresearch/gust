/* Isolated bounded homography/marker host experiment. No allocator. */
#include <stdint.h>
#include <stddef.h>
#include "psf_templates.h"
#include "squint_platform.h"
#ifndef SQUINT_FRAME_CAPACITY
#define SQUINT_FRAME_CAPACITY (4096 * 4096)
#endif
static uint8_t frame[SQUINT_FRAME_CAPACITY], crop[260 * 260];
static double quad[8], result[16], finder[49];
static int fw, fh, fs, markerSparse, fastMapping, searchRadius = 2, normalizedMarkers, markerExact;
static uint8_t knownMask[3249], knownValues[3249], sampleVisible[3249];
static int partialSampling;
API uintptr_t sg_visible(void) {
    return (uintptr_t)sampleVisible;
}
API int sg_partial_sampling(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    partialSampling = enabled;
    return 0;
}
static double softGrid[3249], blackLevels[3249], whiteLevels[3249];
API uintptr_t sg_known_mask(void) {
    return (uintptr_t)knownMask;
}
API uintptr_t sg_known_values(void) {
    return (uintptr_t)knownValues;
}
API uintptr_t sg_soft_grid(void) {
    return (uintptr_t)softGrid;
}
API int sg_marker_exact(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    markerExact = enabled;
    return 0;
}
API int sg_normalized(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    normalizedMarkers = enabled;
    return 0;
}
API int sg_search_radius(int radius) {
    if (radius < 1 || radius > 4) return -1;
    searchRadius = radius;
    return 0;
}
API int sg_fast_mapping(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    fastMapping = enabled;
    return 0;
}
API uintptr_t sg_frame(void) {
    return (uintptr_t)frame;
}
API uintptr_t sg_quad(void) {
    return (uintptr_t)quad;
}
API uintptr_t sg_result(void) {
    return (uintptr_t)result;
}
API uintptr_t sg_crop(void) {
    return (uintptr_t)crop;
}
API uintptr_t sg_template(void) {
    return (uintptr_t)finder;
}
static double absolute(double x) {
    return x < 0 ? -x : x;
}
static int floorint(double x) {
    int i = (int)x;
    return x < i ? i - 1 : i;
}
static int valid(const double *q) {
    double area = 0, sign = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4, k = (i + 2) % 4;
        double cross = (q[j * 2] - q[i * 2]) * (q[k * 2 + 1] - q[j * 2 + 1]) -
                       (q[j * 2 + 1] - q[i * 2 + 1]) * (q[k * 2] - q[j * 2]);
        if (!(absolute(cross) > 1e-9)) return 0;
        if (sign && cross * sign < 0) return 0;
        sign = cross;
        area += q[i * 2] * q[j * 2 + 1] - q[j * 2] * q[i * 2 + 1];
    }
    return absolute(area) >= 2;
}
static int mapping(const double *from, const double *to, double *h) {
    double a[8][9];
    for (int i = 0; i < 4; i++) {
        double x = from[i * 2], y = from[i * 2 + 1], u = to[i * 2], v = to[i * 2 + 1];
        double row0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u}, row1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        for (int j = 0; j < 9; j++) {
            a[i * 2][j] = row0[j];
            a[i * 2 + 1][j] = row1[j];
        }
    }
    for (int c = 0; c < 8; c++) {
        int p = c;
        for (int r = c + 1; r < 8; r++)
            if (absolute(a[r][c]) > absolute(a[p][c])) p = r;
        if (absolute(a[p][c]) < 1e-12) return 0;
        for (int j = 0; j < 9; j++) {
            double t = a[p][j];
            a[p][j] = a[c][j];
            a[c][j] = t;
        }
        double scale = a[c][c];
        for (int j = c; j < 9; j++) a[c][j] /= scale;
        for (int r = 0; r < 8; r++)
            if (r != c) {
                double factor = a[r][c];
                for (int j = c; j < 9; j++) a[r][j] -= factor * a[c][j];
            }
    }
    for (int r = 0; r < 8; r++) h[r] = a[r][8];
    h[8] = 1;
    return 1;
}
static int homography(const double *q, int size, double *h) {
    double edge = size - 1, from[8] = {0, 0, edge, 0, edge, edge, 0, edge};
    if (!fastMapping) return valid(q) && mapping(from, q, h);
    if (!valid(q)) return 0;
    double dx1 = q[2] - q[4], dx2 = q[6] - q[4], dx3 = q[0] - q[2] + q[4] - q[6], dy1 = q[3] - q[5], dy2 = q[7] - q[5],
           dy3 = q[1] - q[3] + q[5] - q[7], den = dx1 * dy2 - dx2 * dy1;
    if (absolute(den) < 1e-12) return 0;
    double g = (dx3 * dy2 - dx2 * dy3) / den, r = (dx1 * dy3 - dx3 * dy1) / den;
    h[0] = (q[2] - q[0] + g * q[2]) / edge;
    h[1] = (q[6] - q[0] + r * q[6]) / edge;
    h[2] = q[0];
    h[3] = (q[3] - q[1] + g * q[3]) / edge;
    h[4] = (q[7] - q[1] + r * q[7]) / edge;
    h[5] = q[1];
    h[6] = g / edge;
    h[7] = r / edge;
    h[8] = 1;
    return 1;
}
static double pixel(int x, int y) {
    return x < 0 || y < 0 || x >= fw || y >= fh ? 255 : frame[y * fs + x];
}
static double sample(double x, double y) {
    if (!(x > -100000 && x < 100000 && y > -100000 && y < 100000)) return 255;
    int a = floorint(x), b = floorint(y);
    double fx = x - a, fy = y - b;
    if (a >= 0 && b >= 0 && a + 1 < fw && b + 1 < fh) {
        const uint8_t *p = frame + b * fs + a;
        return (1 - fx) * (1 - fy) * p[0] + fx * (1 - fy) * p[1] + (1 - fx) * fy * p[fs] + fx * fy * p[fs + 1];
    }
    return (1 - fx) * (1 - fy) * pixel(a, b) + fx * (1 - fy) * pixel(a + 1, b) + (1 - fx) * fy * pixel(a, b + 1) +
           fx * fy * pixel(a + 1, b + 1);
}
static void map(const double *h, double x, double y, double *out) {
    double d = h[6] * x + h[7] * y + 1;
    out[0] = (h[0] * x + h[1] * y + h[2]) / d;
    out[1] = (h[3] * x + h[4] * y + h[5]) / d;
}
static int setup(int width, int height, int stride) {
    if (width < 1 || height < 1 || width > 4096 || height > 4096 || stride < width ||
        (int64_t)stride * height > sizeof(frame))
        return 0;
    fw = width;
    fh = height;
    fs = stride;
    return 1;
}
static double median_level(double *a, int length) {
    int low = 0, high = length - 1, k = length / 2;
    while (low < high) {
        double pivot = a[(low + high) / 2];
        int left = low, right = high;
        while (left <= right) {
            while (a[left] < pivot) left++;
            while (a[right] > pivot) right--;
            if (left <= right) {
                double t = a[left];
                a[left++] = a[right];
                a[right--] = t;
            }
        }
        if (k <= right)
            high = right;
        else if (k >= left)
            low = left;
        else
            break;
    }
    return a[k];
}
/* Lighting robustness (off by default): area sampling averages a 5-point cross per module (noise);
   clip-aware confidence lowers the confidence of modules that sample at the sensor ceiling when the code's
   white level is itself clipped, so clipped (possibly dark) modules become low-reliability instead of confident white. */
static int areaSampling, clipAware, glareErasure, localLevels;
API int sg_local_levels(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    localLevels = enabled;
    return 0;
}
API int sg_glare(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    glareErasure = enabled;
    return 0;
}
API int sg_lighting(int area, int clip) {
    if (area < 0 || area > 1 || clip < 0 || clip > 1) return -1;
    areaSampling = area;
    clipAware = clip;
    return 0;
}
static double sampleModule(const double *h, double mx, double my, double radius) {
    double p[2];
    map(h, mx, my, p);
    double v = sample(p[0], p[1]);
    if (!areaSampling) return v;
    static const double dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
    double acc = v * 2;
    for (int k = 0; k < 4; k++) {
        map(h, mx + dx[k] * radius, my + dy[k] * radius, p);
        acc += sample(p[0], p[1]);
    }
    return acc / 6;
}
static double rawGrid[3249];
API int sg_sample_grid(int width, int height, int stride, int version, double offsetX, double offsetY) {
    double h[9];
    if (!setup(width, height, stride) || version < 1 || version > 10 ||
        !(offsetX >= -1 && offsetX <= 1 && offsetY >= -1 && offsetY <= 1) || !homography(quad, 128, h))
        return -1;
    int n = 17 + 4 * version, black = 0, white = 0, supported = 0;
    double radius = 127. / n * .22;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int at = y * n + x;
            if (knownMask[at] > 1 || knownValues[at] > 1) return -1;
            double p[2];
            map(h, (x + .5 + offsetX) * 127 / n, (y + .5 + offsetY) * 127 / n, p);
            int a = floorint(p[0]), b = floorint(p[1]);
            sampleVisible[at] = !partialSampling || (a >= 0 && b >= 0 && a + 1 < fw && b + 1 < fh);
            supported += sampleVisible[at];
            double value = sampleModule(h, (x + .5 + offsetX) * 127 / n, (y + .5 + offsetY) * 127 / n, radius);
            softGrid[at] = value;
            rawGrid[at] = value;
            if (sampleVisible[at] && knownMask[at]) {
                if (knownValues[at])
                    blackLevels[black++] = value;
                else
                    whiteLevels[white++] = value;
            }
        }
    if (glareErasure &&
        partialSampling) { /* glare: a clipped module whose 3x3 neighbourhood is (almost) all clipped carries no information; mark it unknown */
        int erased = 0;
        static uint8_t clip[3249];
        for (int i = 0; i < n * n; i++) clip[i] = rawGrid[i] >= 250;
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                int at = y * n + x;
                if (!clip[at] || !sampleVisible[at]) continue;
                int c = 0, t = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;
                        t++;
                        c += clip[yy * n + xx];
                    }
                if (c * 9 >= t * 8 && !knownMask[at]) {
                    sampleVisible[at] = 0;
                    supported--;
                    erased++;
                }
            }
        result[13] = erased;
    }
    if (partialSampling && supported * 2 < n * n) {
        result[0] = 0;
        return 0;
    }
    if (!black || !white) return -1;
    double lo = median_level(blackLevels, black), hi = median_level(whiteLevels, white);
    result[14] = lo;
    result[15] = hi;
    if (hi - lo < 20) {
        result[0] = 0;
        return 0;
    }
    static double localLo[3249], localHi[3249];
    if (localLevels) { /* local black/white levels: 2nd-darkest and 2nd-brightest visible module in a 7x7 neighbourhood,
    blended with the global levels; neighbourhoods without contrast become unknown */
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                double a1 = 1e9, a2 = 1e9, b1 = -1e9, b2 = -1e9;
                int c = 0;
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy < 0 || xx < 0 || yy >= n || xx >= n) continue;
                        int k = yy * n + xx;
                        if (partialSampling && !sampleVisible[k]) continue;
                        double v = rawGrid[k];
                        c++;
                        if (v < a1) {
                            a2 = a1;
                            a1 = v;
                        } else if (v < a2)
                            a2 = v;
                        if (v > b1) {
                            b2 = b1;
                            b1 = v;
                        } else if (v > b2)
                            b2 = v;
                    }
                int at = y * n + x;
                if (c < 6) {
                    localLo[at] = lo;
                    localHi[at] = hi;
                    continue;
                }
                double l = a2, u = b2;
                if (u - l < 15 && partialSampling && sampleVisible[at] && !knownMask[at]) {
                    sampleVisible[at] = 0;
                    supported--;
                }
                localLo[at] = (l * 2 + lo) / 3;
                localHi[at] = (u * 2 + hi) / 3;
            }
        if (partialSampling && supported * 2 < n * n) {
            result[0] = 0;
            return 0;
        }
    }
    int clipped = clipAware && hi >= 244;
    double cost = 0;
    int known = 0;
    for (int i = 0; i < n * n; i++) {
        if (partialSampling && !sampleVisible[i]) {
            softGrid[i] = .5;
            continue;
        }
        double L = localLevels ? localLo[i] : lo, U = localLevels ? localHi[i] : hi;
        if (U - L < 10) {
            U = hi;
            L = lo;
        }
        double p = (U - softGrid[i]) / (U - L);
        p = p < .001 ? .001 : p > .999 ? .999 : p;
        if (clipped && rawGrid[i] >= 250 && p < .2) p = .2; /* clipped sample: white is likely but not trustworthy */
        softGrid[i] = p;
        if (knownMask[i]) {
            cost += absolute(softGrid[i] - knownValues[i]);
            known++;
        }
    }
    result[0] = 1;
    result[1] = cost / known;
    return 1;
}
API int sg_rectify(int width, int height, int stride, int size) {
    double h[9];
    if (!setup(width, height, stride) || size < 2 || size > 260 || !homography(quad, size, h)) return -1;
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            double p[2];
            map(h, x, y, p);
            double v = sample(p[0], p[1]);
            crop[y * size + x] = (uint8_t)(v <= 0 ? 0 : v >= 255 ? 255 : v + .5);
        }
    return 0;
}
static double localX[81], localY[81], localH[9], localCx, localCy, localPpm;
static uint8_t localExpected[81];
static int localCount, localWidth;
static void prepare_location(const double *h, double ppm, double cx, double cy, int alignment) {
    int sparse = markerSparse && !alignment, radius = alignment ? 2 : 4, width = sparse ? 5 : 2 * radius + 1,
        offsets[5] = {-3, -2, 0, 2, 3};
    localCount = 0;
    localWidth = width;
    for (int row = 0; row < width; row++)
        for (int col = 0; col < width; col++) {
            int y = sparse ? offsets[row] : row - radius, x = sparse ? offsets[col] : col - radius,
                m = absolute(x) > absolute(y) ? absolute(x) : absolute(y);
            localX[localCount] = x;
            localY[localCount] = y;
            localExpected[localCount++] = alignment ? !(m == 2 || m == 0) : !(m == 3 || m <= 1);
        }
    for (int k = 0; k < 9; k++) localH[k] = h[k];
    localCx = cx;
    localCy = cy;
    localPpm = ppm;
}
static double location_cost(double du, double dv) {
    double s = 0, xx[9], xy[9], xd[9], yx[9], yy[9], yd[9];
    for (int i = 0; i < localWidth; i++) {
        double tx = (localCx + localX[i] + du) * localPpm + (markerExact ? 0 : -.5),
               ty = (localCy + localY[i * localWidth] + dv) * localPpm + (markerExact ? 0 : -.5);
        xx[i] = localH[0] * tx;
        xy[i] = localH[3] * tx;
        xd[i] = localH[6] * tx;
        yx[i] = localH[1] * ty;
        yy[i] = localH[4] * ty;
        yd[i] = localH[7] * ty;
    }
    for (int row = 0, k = 0; row < localWidth; row++)
        for (int col = 0; col < localWidth; col++, k++) {
            double d = xd[col] + yd[row] + 1, px = (xx[col] + yx[row] + localH[2]) / d,
                   py = (xy[col] + yy[row] + localH[5]) / d;
            double value = sample(px, py) / 255;
            double delta = value - localExpected[k];
            s += delta * delta;
        }
    return s / localCount;
}
static double normalized_location_cost(double du, double dv) {
    double values[81], sum[2] = {0, 0};
    int count[2] = {0, 0};
    for (int k = 0; k < localCount; k++) {
        double point[2];
        map(localH, (localCx + localX[k] + du) * localPpm + (markerExact ? 0 : -.5),
            (localCy + localY[k] + dv) * localPpm + (markerExact ? 0 : -.5), point);
        values[k] = sample(point[0], point[1]);
        int expected = localExpected[k];
        sum[expected] += values[k];
        count[expected]++;
    }
    if (!count[0] || !count[1]) return 1e6;
    double low = sum[0] / count[0], high = sum[1] / count[1], contrast = high - low;
    if (contrast < 20) return 1e6;
    double residual = 0;
    for (int k = 0; k < localCount; k++) {
        double d = (values[k] - low) / contrast - localExpected[k];
        residual += d * d;
    }
    return residual / localCount;
}
static double marker_cost(double du, double dv) {
    return normalizedMarkers ? normalized_location_cost(du, dv) : location_cost(du, dv);
}
static double locate(const double *h, double ppm, double cx, double cy, int alignment, int fine, double *out,
                     int *evaluations) {
    prepare_location(h, ppm, cx, cy, alignment);
    double bx = 0, by = 0, best = marker_cost(0, 0);
    (*evaluations)++;
    for (int y = -searchRadius; y <= searchRadius; y++)
        for (int x = -searchRadius; x <= searchRadius; x++) {
            double cost = marker_cost(x, y);
            (*evaluations)++;
            if (cost < best) {
                best = cost;
                bx = x;
                by = y;
            }
        }
    double steps[4] = {.5, .25, .125, .0625};
    for (int k = 0; k < (fine ? 4 : 2); k++) {
        double ox = bx, oy = by;
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                double du = ox + x * steps[k], dv = oy + y * steps[k], cost = marker_cost(du, dv);
                (*evaluations)++;
                if (cost < best) {
                    best = cost;
                    bx = du;
                    by = dv;
                }
            }
    }
    map(h, (cx + bx) * ppm + (markerExact ? 0 : -.5), (cy + by) * ppm + (markerExact ? 0 : -.5), out);
    return best;
}
static void perspective(const double *m, double p, double r, double u, double v, double *out) {
    double d = 1 + p * u + r * v;
    out[0] = (m[0] + (m[2] * (1 + p) - m[0]) * u + (m[4] * (1 + r) - m[0]) * v) / d;
    out[1] = (m[1] + (m[3] * (1 + p) - m[1]) * u + (m[5] * (1 + r) - m[1]) * v) / d;
}
static double perspectiveU[243], perspectiveV[243];
static uint8_t perspectiveExpected[243];
static int perspectiveCount;
static void prepare_perspective(int n) {
    double centres[6] = {7.5, 7.5, n + .5, 7.5, 7.5, n + .5};
    int radius = markerSparse ? 3 : 4;
    perspectiveCount = 0;
    for (int k = 0; k < 3; k++)
        for (int y = -radius; y <= radius; y++)
            for (int x = -radius; x <= radius; x++) {
                int maximum = absolute(x) > absolute(y) ? absolute(x) : absolute(y);
                perspectiveU[perspectiveCount] = (centres[k * 2] + x - 7.5) / (n - 7);
                perspectiveV[perspectiveCount] = (centres[k * 2 + 1] + y - 7.5) / (n - 7);
                perspectiveExpected[perspectiveCount++] = !(maximum == 3 || maximum <= 1);
            }
}
static double perspective_cost(const double *m, int n, double p, double r) {
    (void)n;
    double s = 0;
    for (int k = 0; k < perspectiveCount; k++) {
        double u = perspectiveU[k], v = perspectiveV[k];
        if (1 + p * u + r * v < .3) return 1e30;
        double point[2];
        perspective(m, p, r, u, v, point);
        double d = sample(point[0], point[1]) / 255 - perspectiveExpected[k];
        s += d * d;
    }
    return s / perspectiveCount;
}
static int fit_inner(int width, int height, int stride, int version, int fine, int sparseMode) {
    markerSparse = sparseMode;
    double h[9], m[8], q[8], cost = 0;
    int evaluations = 0, n = 17 + 4 * version;
    if (!setup(width, height, stride) || version < 1 || version > 10 || (fine != 0 && fine != 1) ||
        !homography(quad, markerExact ? 128 : 64, h))
        return -1;
    double edge = markerExact ? 127 : 63, origin = markerExact ? 0 : -.5, ppm = (markerExact ? 127. : 64.) / (n + 8);
    cost += locate(h, ppm, 7.5, 7.5, 0, fine, m, &evaluations);
    cost += locate(h, ppm, n + .5, 7.5, 0, fine, m + 2, &evaluations);
    cost += locate(h, ppm, 7.5, n + .5, 0, fine, m + 4, &evaluations);
    if (version > 1) {
        cost += locate(h, ppm, n - 2.5, n - 2.5, 1, fine, m + 6, &evaluations);
        double from[8] = {7.5 * ppm + origin, 7.5 * ppm + origin,       (n + .5) * ppm + origin,
                          7.5 * ppm + origin, (n - 2.5) * ppm + origin, (n - 2.5) * ppm + origin,
                          7.5 * ppm + origin, (n + .5) * ppm + origin},
               to[8] = {m[0], m[1], m[2], m[3], m[6], m[7], m[4], m[5]}, fitted[9];
        if (!mapping(from, to, fitted)) return -1;
        double corners[8] = {0, 0, edge, 0, edge, edge, 0, edge};
        for (int k = 0; k < 4; k++) map(fitted, corners[k * 2], corners[k * 2 + 1], q + k * 2);
        cost /= 4;
    } else {
        prepare_perspective(n);
        double bp = 0, br = 0, best = 1e30;
        for (int i = -3; i <= 3; i++)
            for (int j = -3; j <= 3; j++) {
                double p = i * .1, r = j * .1, s = perspective_cost(m, n, p, r);
                evaluations++;
                if (s < best) {
                    best = s;
                    bp = p;
                    br = r;
                }
            }
        double steps[4] = {.05, .025, .0125, .00625};
        for (int k = 0; k < (fine ? 4 : 2); k++) {
            double op = bp, orr = br;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++) {
                    double p = op + i * steps[k], r = orr + j * steps[k], s = perspective_cost(m, n, p, r);
                    evaluations++;
                    if (s < best) {
                        best = s;
                        bp = p;
                        br = r;
                    }
                }
        }
        double lo = 7.5 * ppm + origin, span = (n - 7) * ppm, corners[8] = {0, 0, edge, 0, edge, edge, 0, edge};
        for (int k = 0; k < 4; k++)
            perspective(m, bp, br, (corners[k * 2] - lo) / span, (corners[k * 2 + 1] - lo) / span, q + k * 2);
        cost /= 3;
    }
    double timing = 1;
    if (homography(q, markerExact ? 128 : 64, h)) {
        double s = 0;
        int count = 0;
        for (int i = 8; i < n - 8; i++)
            for (int axis = 0; axis < 2; axis++) {
                double p[2];
                map(h, (axis ? 10.5 : i + 4.5) * ppm + origin, (axis ? i + 4.5 : 10.5) * ppm + origin, p);
                double d = sample(p[0], p[1]) / 255 - ((version >= 7 && absolute(i - (22 + 2 * (version - 7))) <= 2)
                                                           ? (absolute(i - (22 + 2 * (version - 7))) == 1)
                                                           : (i % 2 != 0));
                s += d * d;
                count++;
            }
        timing = s / count;
    }
    result[0] = valid(q);
    result[1] = cost;
    result[2] = timing;
    result[3] = (cost + timing) / 2;
    result[4] = evaluations;
    for (int k = 0; k < 8; k++) result[5 + k] = valid(q) ? q[k] : quad[k];
    return 0;
}
static double soft_cost(const double *q, int n) {
    double h[9], s = 0;
    if (!homography(q, 64, h)) return 1e30;
    double centres[6] = {7.5, 7.5, n + .5, 7.5, 7.5, n + .5}, ppm = 64. / (n + 8);
    for (int k = 0; k < 3; k++)
        for (int y = -3; y <= 3; y++)
            for (int x = -3; x <= 3; x++) {
                double p[2];
                map(h, (centres[k * 2] + x) * ppm - .5, (centres[k * 2 + 1] + y) * ppm - .5, p);
                double d = sample(p[0], p[1]) / 255 - finder[(y + 3) * 7 + x + 3];
                s += d * d;
            }
    return s / 147;
}
API int sg_refine(int width, int height, int stride, int version) {
    if (!setup(width, height, stride) || version < 1 || version > 10 || !valid(quad)) return -1;
    for (int k = 0; k < 49; k++)
        if (!(finder[k] >= 0 && finder[k] <= 1)) return -1;
    double q[8];
    for (int k = 0; k < 8; k++) q[k] = quad[k];
    double best = soft_cost(q, 17 + 4 * version), steps[4] = {2, 1, .5, .25};
    for (int step = 0; step < 4; step++)
        for (int k = 0; k < 8; k++) {
            double old = q[k], chosen = old;
            for (int sign = -1; sign <= 1; sign += 2) {
                q[k] = old + sign * steps[step];
                double score = soft_cost(q, 17 + 4 * version);
                if (score < best) {
                    best = score;
                    chosen = q[k];
                }
            }
            q[k] = chosen;
        }
    result[0] = 1;
    result[1] = best;
    result[4] = 65;
    for (int k = 0; k < 8; k++) result[5 + k] = q[k];
    return 0;
}
API int sg_rectify_fast(int width, int height, int stride, int size) {
    double hd[9];
    float h[9];
    if (!setup(width, height, stride) || size < 2 || size > 260 || !homography(quad, size, hd)) return -1;
    for (int j = 0; j < 9; j++) h[j] = (float)hd[j];
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            float d = h[6] * x + h[7] * y + 1, sx = (h[0] * x + h[1] * y + h[2]) / d,
                  sy = (h[3] * x + h[4] * y + h[5]) / d;
            if (!(sx > -100000 && sx < 100000 && sy > -100000 && sy < 100000)) {
                crop[y * size + x] = 255;
                continue;
            }
            int a = floorint(sx), b = floorint(sy);
            float fx = sx - a, fy = sy - b, v;
            if (a >= 0 && b >= 0 && a + 1 < fw && b + 1 < fh) {
                const uint8_t *p = frame + b * fs + a;
                v = (p[0] * (1 - fx) + p[1] * fx) * (1 - fy) + (p[fs] * (1 - fx) + p[fs + 1] * fx) * fy;
            } else
                v = (float)sample(sx, sy);
            crop[y * size + x] = (uint8_t)(v <= 0 ? 0 : v >= 255 ? 255 : v + .5f);
        }
    return 0;
}
API int sg_fit(int width, int height, int stride, int version, int fine) {
    return fit_inner(width, height, stride, version, fine, 0);
}
API int sg_fit_sparse(int width, int height, int stride, int version, int fine) {
    return fit_inner(width, height, stride, version, fine, 1);
}
/* Cheap ranking only; final fits always use the dense marker path. */
API int sg_partial_fit(int width, int height, int stride, int version) {
    double h[9], m[4];
    int evaluations = 0, n = 17 + 4 * version;
    if (!setup(width, height, stride) || version < 1 || version > 10 || !homography(quad, 128, h)) return -1;
    double ppm = 127. / (n + 8), shift = markerExact ? 0 : .5 / ppm;
    double first = locate(h, ppm, 7.5 + shift, 7.5 + shift, 0, 1, m, &evaluations),
           second = locate(h, ppm, n + .5 + shift, 7.5 + shift, 0, 1, m + 2, &evaluations);
    result[0] = first < 1e5 && second < 1e5;
    result[1] = (first + second) / 2;
    result[4] = evaluations;
    for (int k = 0; k < 4; k++) result[5 + k] = m[k];
    return 0;
}
API int sg_scout(int width, int height, int stride, int version) {
    double h[9];
    if (!setup(width, height, stride) || version < 1 || version > 10 || !homography(quad, markerExact ? 128 : 64, h))
        return -1;
    markerSparse = 1;
    int n = 17 + 4 * version, count = version > 1 ? 4 : 3, evaluations = 0;
    double ppm = (markerExact ? 127. : 64.) / (n + 8),
           centres[8] = {7.5, 7.5, n + .5, 7.5, 7.5, n + .5, n - 2.5, n - 2.5}, cost = 0;
    for (int k = 0; k < count; k++) {
        prepare_location(h, ppm, centres[k * 2], centres[k * 2 + 1], k == 3);
        double best = 1e30;
        for (int y = -2; y <= 2; y += 2)
            for (int x = -2; x <= 2; x += 2) {
                double candidate = marker_cost(x, y);
                evaluations++;
                if (candidate < best) best = candidate;
            }
        cost += best;
    }
    result[0] = 1;
    result[1] = cost / count;
    result[4] = evaluations;
    return 0;
}
static int psf_fit(int version, int fine, int size) {
    if (version < 1 || version > 10 || (size != 96 && size != 128)) return -1;
    int n = 17 + 4 * version, index = 0;
    double observed[147], sum = 0, centres[6] = {7.5, 7.5, n + .5, 7.5, 7.5, n + .5};
    for (int k = 0; k < 3; k++)
        for (int y = -3; y <= 3; y++)
            for (int x = -3; x <= 3; x++) {
                double sx = (float)((x + centres[k * 2]) * size / (n + 8) - .5),
                       sy = (float)((y + centres[k * 2 + 1]) * size / (n + 8) - .5);
                int a = floorint(sx), b = floorint(sy);
                double fx = floorint((sx - a) * 32 + .5) / 32., fy = floorint((sy - b) * 32 + .5) / 32.;
                if (fx == 1) {
                    a++;
                    fx = 0;
                }
                if (fy == 1) {
                    b++;
                    fy = 0;
                }
                double v =
                    floorint((1 - fx) * (1 - fy) * crop[b * size + a] + fx * (1 - fy) * crop[b * size + a + 1] +
                             (1 - fx) * fy * crop[(b + 1) * size + a] + fx * fy * crop[(b + 1) * size + a + 1] + .5) /
                    255.;
                observed[index++] = v;
                sum += v;
            }
    double mean = sum / 147, first = 1e30, second = 1e30;
    int fi = -1, si = -1;
    const double (*templates)[49] = fine ? psfFineValues : psfValues;
    for (int t = 0; t < (fine ? 39 : 21); t++) {
        double expectedMean = 0;
        for (int j = 0; j < 49; j++) expectedMean += templates[t][j];
        expectedMean /= 49;
        double covariance = 0, variance = 0;
        for (int j = 0; j < 147; j++) {
            double centred = templates[t][j % 49] - expectedMean;
            covariance += centred * (observed[j] - mean);
            variance += centred * centred;
        }
        double beta = covariance / (variance + 1e-8);
        if (beta < .3) beta = .3;
        if (beta > 2) beta = 2;
        double alpha = mean - beta * expectedMean, error = 0;
        for (int j = 0; j < 147; j++) {
            double d = alpha + beta * templates[t][j % 49] - observed[j];
            error += d * d;
        }
        error /= 147;
        if (error < first) {
            second = first;
            si = fi;
            first = error;
            fi = t;
        } else if (error < second) {
            second = error;
            si = t;
        }
    }
    result[0] = fi;
    result[1] = first;
    result[2] = si;
    result[3] = second;
    return 0;
}
API int sg_psf(int version) {
    return psf_fit(version, 0, 96);
}
API int sg_psf_fine(int version) {
    return psf_fit(version, 1, 96);
}

API int sg_psf_size(int version, int size, int fine) {
    if (fine != 0 && fine != 1) return -1;
    return psf_fit(version, fine, size);
}
