/* Dependency-free bounded white-patch proposals. Development-only classical host. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
#define PIXELS (640 * 360)
static uint8_t input[1280 * 720], preview[PIXELS], a[PIXELS], b[PIXELS], seen[PIXELS];
static uint32_t queue[PIXELS];
static int32_t quads[16 * 8];
static double scores[16];
static int count;
static int profiles = 12;
API int sqw_profiles(int budget) {
    if (budget < 1 || budget > 12) return -1;
    profiles = budget;
    return 0;
}
static int refine = 0;
API int sqw_refine(int value) {
    if (value < 0 || value > 1) return -1;
    refine = value;
    return 0;
}
API uintptr_t sqw_input(void) {
    return (uintptr_t)input;
}
API uintptr_t sqw_quads(void) {
    return (uintptr_t)quads;
}
static int absint(int v) {
    return v < 0 ? -v : v;
}
/* Fit straight outer edges, ignoring corners and interior QR holes. */
static void refine_edges(int *q, unsigned length, int w, int h) {
    double lines[4][3];
    int updated[8];
    for (int edge = 0; edge < 4; edge++) {
        int next = (edge + 1) % 4;
        double x0 = q[2 * edge], y0 = q[2 * edge + 1], dx = q[2 * next] - x0, dy = q[2 * next + 1] - y0,
               l2 = dx * dx + dy * dy;
        if (l2 < 100) return;
        double st = 0, sz = 0, stt = 0, stz = 0;
        unsigned count = 0;
        for (unsigned k = 0; k < length; k++) {
            int at = queue[k], x = at % w, y = at / w;
            if (x && x + 1 < w && y && y + 1 < h && a[at - 1] && a[at + 1] && a[at - w] && a[at + w]) continue;
            double px = x - x0, py = y - y0, t = (dx * px + dy * py) / l2, z = (dx * py - dy * px) / l2;
            if (t < .12 || t > .88 || z * z * l2 > 9) continue;
            st += t;
            sz += z;
            stt += t * t;
            stz += t * z;
            count++;
        }
        double denom = count * stt - st * st;
        if (count < 6 || denom < .0001) return;
        double slope = (count * stz - st * sz) / denom, intercept = (sz - slope * st) / count;
        lines[edge][0] = -dy - slope * dx;
        lines[edge][1] = dx - slope * dy;
        lines[edge][2] = l2 * intercept + lines[edge][0] * x0 + lines[edge][1] * y0;
    }
    for (int i = 0; i < 4; i++) {
        double *p = lines[(i + 3) % 4], *r = lines[i], det = p[0] * r[1] - r[0] * p[1];
        if (det > -1e-6 && det < 1e-6) return;
        double x = (p[2] * r[1] - r[2] * p[1]) / det, y = (p[0] * r[2] - r[0] * p[2]) / det;
        if (x < 0 || y < 0 || x >= w || y >= h || x - q[i * 2] > 10 || q[i * 2] - x > 10 || y - q[i * 2 + 1] > 10 ||
            q[i * 2 + 1] - y > 10)
            return;
        updated[i * 2] = (int)(x + .5);
        updated[i * 2 + 1] = (int)(y + .5);
    }
    for (int k = 0; k < 8; k++) q[k] = updated[k];
}
static void line_filter(const uint8_t *from, uint8_t *to, int w, int h, int radius, int dilation, int vertical) {
    int lines = vertical ? w : h, length = vertical ? h : w, step = vertical ? w : 1;
    for (int line = 0; line < lines; line++) {
        int base = vertical ? line : line * w, sum = 0;
        for (int j = 0; j <= radius && j < length; j++) sum += from[base + j * step] != 0;
        for (int i = 0; i < length; i++) {
            int lo = i - radius < 0 ? 0 : i - radius, hi = i + radius >= length ? length - 1 : i + radius;
            to[base + i * step] = dilation ? (sum > 0) : (sum == hi - lo + 1);
            if (i - radius >= 0) sum -= from[base + (i - radius) * step] != 0;
            if (i + radius + 1 < length) sum += from[base + (i + radius + 1) * step] != 0;
        }
    }
}
static void add_quad(const int *q, int area) {
    int64_t twice = 0;
    int sign = 0, minLength = 0x7fffffff, maxLength = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4, k = (i + 2) % 4;
        int dx = q[j * 2] - q[i * 2], dy = q[j * 2 + 1] - q[i * 2 + 1],
            cross = dx * (q[k * 2 + 1] - q[j * 2 + 1]) - dy * (q[k * 2] - q[j * 2]);
        if (!cross || (sign && ((cross > 0) != (sign > 0)))) return;
        sign = cross;
        int length = dx * dx + dy * dy;
        if (length < minLength) minLength = length;
        if (length > maxLength) maxLength = length;
        twice += (int64_t)q[i * 2] * q[j * 2 + 1] - (int64_t)q[j * 2] * q[i * 2 + 1];
    }
    if (twice < 0) twice = -twice;
    if (twice < 200 || minLength < 36 || maxLength > minLength * 9 || area * 2 < twice * .7) return;
    for (int i = 0; i < count; i++) {
        int distance = 0;
        for (int k = 0; k < 8; k++) distance += absint(quads[i * 8 + k] / 2 - q[k]);
        if (distance < 20) return;
    }
    double score = (double)area * 2 / twice;
    if (score > 1) score = 1;
    int slot = count;
    if (slot == 16) {
        slot = 0;
        for (int i = 1; i < count; i++)
            if (scores[i] < scores[slot]) slot = i;
        if (score <= scores[slot]) return;
    } else
        count++;
    scores[slot] = score;
    for (int k = 0; k < 8; k++) quads[slot * 8 + k] = q[k] * 2;
}
API int sqw_detect(int width, int height) {
    if (width < 2 || height < 2 || width > 1280 || height > 720 || width % 2 || height % 2) return -1;
    int w = width / 2, h = height / 2, pixels = w * h;
    if (pixels > PIXELS) return -1;
    count = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int at = y * 2 * width + x * 2;
            preview[y * w + x] = (input[at] + input[at + 1] + input[at + width] + input[at + width + 1] + 2) / 4;
        }
    const int thresholds[4] = {140, 180, 100, 210}, radii[3] = {3, 1, 5};
    for (int t = 0; t < 4; t++)
        for (int r = 0; r < 3; r++) {
            if (t * 3 + r >= profiles) break;
            for (int i = 0; i < pixels; i++) a[i] = preview[i] > thresholds[t];
            line_filter(a, b, w, h, radii[r], 1, 0);
            line_filter(b, a, w, h, radii[r], 1, 1);
            line_filter(a, b, w, h, radii[r], 0, 0);
            line_filter(b, a, w, h, radii[r], 0, 1);
            for (int i = 0; i < pixels; i++) seen[i] = 0;
            for (int seed = 0; seed < pixels; seed++)
                if (a[seed] && !seen[seed]) {
                    unsigned head = 0, tail = 1;
                    queue[0] = seed;
                    seen[seed] = 1;
                    int extrema[4] = {0x7fffffff, -0x7fffffff, -0x7fffffff, 0x7fffffff}, q[8] = {0};
                    while (head < tail) {
                        int at = queue[head++], x = at % w, y = at / w, sum = x + y, difference = x - y;
                        if (sum < extrema[0]) {
                            extrema[0] = sum;
                            q[0] = x;
                            q[1] = y;
                        }
                        if (difference > extrema[1]) {
                            extrema[1] = difference;
                            q[2] = x;
                            q[3] = y;
                        }
                        if (sum > extrema[2]) {
                            extrema[2] = sum;
                            q[4] = x;
                            q[5] = y;
                        }
                        if (difference < extrema[3]) {
                            extrema[3] = difference;
                            q[6] = x;
                            q[7] = y;
                        }
                        int neighbors[4] = {x ? at - 1 : -1, x + 1 < w ? at + 1 : -1, y ? at - w : -1,
                                            y + 1 < h ? at + w : -1};
                        for (int k = 0; k < 4; k++) {
                            int n = neighbors[k];
                            if (n >= 0 && a[n] && !seen[n]) {
                                seen[n] = 1;
                                queue[tail++] = n;
                            }
                        }
                    }
                    if (tail > 400 && tail < (unsigned)(pixels * .4)) {
                        if (refine) refine_edges(q, tail, w, h);
                        add_quad(q, tail);
                    }
                }
        }
    return count;
}
