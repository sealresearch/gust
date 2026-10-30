/* Bounded pyramidal Lucas-Kanade geometry proposals. Never accepts payloads. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
#define W 640
#define H 360
#define P (W * H + W * H / 4 + W * H / 16 + W * H / 64)
static uint8_t input[1280 * 720], before[P], now[P];
static double quad[8], output[16];
static float points[160][2], scores[W * H], patch[441], gx[441], gy[441];
typedef struct {
    float score;
    int at;
} Feature;
static Feature heap[W * H / 4];
static int better(Feature a, Feature b) {
    return a.score > b.score || (a.score == b.score && a.at < b.at);
}
static void sift(int at, int length) {
    Feature value = heap[at];
    for (int child = at * 2 + 1; child < length; child = at * 2 + 1) {
        if (child + 1 < length && better(heap[child + 1], heap[child])) child++;
        if (!better(heap[child], value)) break;
        heap[at] = heap[child];
        at = child;
    }
    heap[at] = value;
}
static int pw, ph, ready, count, maxPoints = 80, downscale = 2, reverseCheck = 1, windowRadius = 7;
API uintptr_t sf_input(void) {
    return (uintptr_t)input;
}
API uintptr_t sf_quad(void) {
    return (uintptr_t)quad;
}
API uintptr_t sf_result(void) {
    return (uintptr_t)output;
}
API int sf_reset(void) {
    ready = count = 0;
    return 0;
}
API int sf_budget(int n) {
    if (n < 24 || n > 160) return -1;
    maxPoints = n;
    return 0;
}
API int sf_scale(int n) {
    if (n != 2 && n != 4) return -1;
    downscale = n;
    ready = count = 0;
    return 0;
}
API int sf_window(int radius) {
    if (radius < 3 || radius > 7) return -1;
    windowRadius = radius;
    return 0;
}
static int motionModel;
API int sf_model(int m) {
    if (m < 0 || m > 1) return -1;
    motionModel = m;
    return 0;
}
/* similarity x' = a*x - b*y + tx, y' = b*x + a*y + ty (normalized coordinates) */
static int similarity(float from[][2], float to[][2], int *valid, int length, double *h) {
    double sx = 0, sy = 0, su = 0, sv = 0;
    int m = 0;
    for (int i = 0; i < length; i++)
        if (valid[i]) {
            sx += from[i][0] / pw;
            sy += from[i][1] / ph;
            su += to[i][0] / pw;
            sv += to[i][1] / ph;
            m++;
        }
    if (m < 2) return 0;
    sx /= m;
    sy /= m;
    su /= m;
    sv /= m;
    double num_a = 0, num_b = 0, den = 0;
    for (int i = 0; i < length; i++)
        if (valid[i]) {
            double x = from[i][0] / pw - sx, y = from[i][1] / ph - sy, u = to[i][0] / pw - su, v = to[i][1] / ph - sv;
            num_a += x * u + y * v;
            num_b += x * v - y * u;
            den += x * x + y * y;
        }
    if (den < 1e-12) return 0;
    double a = num_a / den, b = num_b / den;
    h[0] = a;
    h[1] = -b;
    h[2] = su - a * sx + b * sy;
    h[3] = b;
    h[4] = a;
    h[5] = sv - b * sx - a * sy;
    h[6] = 0;
    h[7] = 0;
    return 1;
}
API int sf_reverse(int n) {
    if (n != 0 && n != 1) return -1;
    reverseCheck = n;
    return 0;
}
static float absf(float a) {
    return a < 0 ? -a : a;
}
static float sample(uint8_t *a, int w, int h, float x, float y) {
    if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) return 255;
    int ix = (int)x, iy = (int)y;
    float fx = x - ix, fy = y - iy;
    return (a[iy * w + ix] * (1 - fx) + a[iy * w + ix + 1] * fx) * (1 - fy) +
           (a[(iy + 1) * w + ix] * (1 - fx) + a[(iy + 1) * w + ix + 1] * fx) * fy;
}
static void pyramid(uint8_t *out, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int dy = 0; dy < downscale; dy++)
                for (int dx = 0; dx < downscale; dx++)
                    sum += input[(y * downscale + dy) * w * downscale + x * downscale + dx];
            out[y * w + x] = (sum + downscale * downscale / 2) / (downscale * downscale);
        }
    int offset = 0;
    for (int level = 1; level < 4; level++) {
        int length = w * h;
        uint8_t *a = out + offset, *b = a + length;
        int nw = w / 2, nh = h / 2;
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++) {
                int i = y * 2 * w + x * 2;
                b[y * nw + x] = (a[i] + a[i + 1] + a[i + w] + a[i + w + 1] + 2) / 4;
            }
        offset += length;
        w = nw;
        h = nh;
    }
}
static int lk(uint8_t *a, uint8_t *b, float x, float y, float *outx, float *outy) {
    int offsets[4] = {0}, width[4] = {pw}, height[4] = {ph};
    for (int l = 1; l < 4; l++) {
        offsets[l] = offsets[l - 1] + width[l - 1] * height[l - 1];
        width[l] = width[l - 1] / 2;
        height[l] = height[l - 1] / 2;
    }
    float mx = x / 8, my = y / 8;
    for (int l = 3; l >= 0; l--) {
        int w = width[l], h = height[l];
        uint8_t *prev = a + offsets[l], *next = b + offsets[l];
        float px = x / (1 << l), py = y / (1 << l);
        if (l < 3) {
            mx *= 2;
            my *= 2;
        }
        double aa = 0, ab = 0, bb = 0;
        int i = 0;
        for (int dy = -windowRadius; dy <= windowRadius; dy++)
            for (int dx = -windowRadius; dx <= windowRadius; dx++, i++) {
                float xx = px + dx, yy = py + dy;
                patch[i] = sample(prev, w, h, xx, yy);
                gx[i] = (sample(prev, w, h, xx + 1, yy) - sample(prev, w, h, xx - 1, yy)) * .5f;
                gy[i] = (sample(prev, w, h, xx, yy + 1) - sample(prev, w, h, xx, yy - 1)) * .5f;
                aa += gx[i] * gx[i];
                ab += gx[i] * gy[i];
                bb += gy[i] * gy[i];
            }
        double determinant = aa * bb - ab * ab;
        if (determinant < 1e-4) return 0;
        for (int iteration = 0; iteration < 12; iteration++) {
            if (mx < -2 || my < -2 || mx > w + 2 || my > h + 2) return 0;
            double rx = 0, ry = 0;
            i = 0;
            for (int dy = -windowRadius; dy <= windowRadius; dy++)
                for (int dx = -windowRadius; dx <= windowRadius; dx++, i++) {
                    float e = sample(next, w, h, mx + dx, my + dy) - patch[i];
                    rx += gx[i] * e;
                    ry += gy[i] * e;
                }
            float tx = (float)((bb * rx - ab * ry) / determinant), ty = (float)((aa * ry - ab * rx) / determinant);
            mx -= tx;
            my -= ty;
            if (tx * tx + ty * ty < .0001) break;
        }
    }
    if (mx < 0 || my < 0 || mx >= pw - 1 || my >= ph - 1) return 0;
    *outx = mx;
    *outy = my;
    return 1;
}
static int solve(double a[8][9], double *out) {
    for (int c = 0; c < 8; c++) {
        int p = c;
        for (int r = c + 1; r < 8; r++)
            if (absf((float)a[r][c]) > absf((float)a[p][c])) p = r;
        if (absf((float)a[p][c]) < 1e-8) return 0;
        for (int j = c; j < 9; j++) {
            double t = a[p][j];
            a[p][j] = a[c][j];
            a[c][j] = t;
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
static int homography(float from[][2], float to[][2], int *valid, int length, double *h) {
    double a[8][9] = {{0}};
    int used = 0;
    for (int i = 0; i < length; i++)
        if (valid[i]) {
            double x = from[i][0] / pw, y = from[i][1] / ph, u = to[i][0] / pw, v = to[i][1] / ph;
            double rows[2][9] = {{x, y, 1, 0, 0, 0, -u * x, -u * y, u}, {0, 0, 0, x, y, 1, -v * x, -v * y, v}};
            for (int r = 0; r < 2; r++)
                for (int k = 0; k < 8; k++)
                    for (int j = 0; j < 9; j++) a[k][j] += rows[r][k] * rows[r][j];
            used++;
        }
    if (used < 8) return 0;
    return solve(a, h);
}
static float reprojection(float *x, float *y, double *h) {
    double px = x[0] / pw, py = x[1] / ph, d = h[6] * px + h[7] * py + 1;
    float u = (float)((h[0] * px + h[1] * py + h[2]) / d * pw), v = (float)((h[3] * px + h[4] * py + h[5]) / d * ph);
    return (u - y[0]) * (u - y[0]) + (v - y[1]) * (v - y[1]);
}
static int inside(int x, int y, double *q) {
    double sign = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        double cross = (q[j * 2] - q[i * 2]) * (y - q[i * 2 + 1]) - (q[j * 2 + 1] - q[i * 2 + 1]) * (x - q[i * 2]);
        if (sign && cross * sign < 0) return 0;
        if (cross) sign = cross;
    }
    return 1;
}
static void features(void) {
    double q[8], cx = 0, cy = 0;
    for (int i = 0; i < 4; i++) {
        cx += quad[i * 2] / (4 * downscale);
        cy += quad[i * 2 + 1] / (4 * downscale);
    }
    for (int i = 0; i < 4; i++) {
        q[i * 2] = cx + (quad[i * 2] / downscale - cx) * 1.5;
        q[i * 2 + 1] = cy + (quad[i * 2 + 1] / downscale - cy) * 1.5;
    }
    int left = pw - 4, right = 4, top = ph - 4, bottom = 4;
    for (int i = 0; i < 4; i++) {
        if (q[i * 2] < left) left = (int)q[i * 2];
        if (q[i * 2] > right) right = (int)q[i * 2] + 2;
        if (q[i * 2 + 1] < top) top = (int)q[i * 2 + 1];
        if (q[i * 2 + 1] > bottom) bottom = (int)q[i * 2 + 1] + 2;
    }
    if (left < 4) left = 4;
    if (top < 4) top = 4;
    if (right > pw - 4) right = pw - 4;
    if (bottom > ph - 4) bottom = ph - 4;
    left = (left + 1) & ~1;
    top = (top + 1) & ~1;
    float maximum = 0;
    for (int y = top; y < bottom; y += 2)
        for (int x = left; x < right; x += 2) scores[y * pw + x] = 0;
    for (int y = top; y < bottom; y += 2)
        for (int x = left; x < right; x += 2)
            if (inside(x, y, q)) {
                float aa = 0, ab = 0, bb = 0;
                for (int dy = -2; dy <= 2; dy++)
                    for (int dx = -2; dx <= 2; dx++) {
                        int i = (y + dy) * pw + x + dx;
                        float vx = (now[i + 1] - now[i - 1]) * .5f, vy = (now[i + pw] - now[i - pw]) * .5f;
                        aa += vx * vx;
                        ab += vx * vy;
                        bb += vy * vy;
                    }
                float value = (aa + bb - __builtin_sqrtf((aa - bb) * (aa - bb) + 4 * ab * ab)) * .5f;
                scores[y * pw + x] = value;
                if (value > maximum) maximum = value;
            }
#ifdef SQUINT_FLOW_REFERENCE
    count = 0;
    while (count < maxPoints) {
        float best = maximum * .01f;
        int at = -1;
        for (int y = top; y < bottom; y += 2)
            for (int x = left; x < right; x += 2)
                if (scores[y * pw + x] > best) {
                    best = scores[y * pw + x];
                    at = y * pw + x;
                }
        if (at < 0) break;
        int x = at % pw, y = at / pw;
        points[count][0] = (float)x;
        points[count][1] = (float)y;
        count++;
        for (int dy = -4; dy <= 4; dy++)
            for (int dx = -4; dx <= 4; dx++)
                if (dx * dx + dy * dy < 16 && x + dx >= 0 && x + dx < pw && y + dy >= 0 && y + dy < ph)
                    scores[(y + dy) * pw + x + dx] = 0;
    }
#else
    int length = 0;
    float threshold = maximum * .01f;
    for (int y = top; y < bottom; y += 2)
        for (int x = left; x < right; x += 2)
            if (scores[y * pw + x] > threshold) {
                heap[length].score = scores[y * pw + x];
                heap[length++].at = y * pw + x;
            }
    for (int i = length / 2 - 1; i >= 0; i--) sift(i, length);
    count = 0;
    while (count < maxPoints && length) {
        int at = heap[0].at;
        heap[0] = heap[--length];
        if (length) sift(0, length);
        if (scores[at] <= threshold) continue;
        int x = at % pw, y = at / pw;
        points[count][0] = (float)x;
        points[count][1] = (float)y;
        count++;
        for (int dy = -4; dy <= 4; dy++)
            for (int dx = -4; dx <= 4; dx++)
                if (dx * dx + dy * dy < 16 && x + dx >= 0 && x + dx < pw && y + dy >= 0 && y + dy < ph)
                    scores[(y + dy) * pw + x + dx] = 0;
    }
#endif
}
API int sf_frame(int width, int height, int anchor) {
    if (width < 64 || height < 64 || width > 1280 || height > 720 || width % 16 || height % 16) return -1;
    if (pw != width / downscale || ph != height / downscale) {
        ready = count = 0;
        pw = width / downscale;
        ph = height / downscale;
    }
    pyramid(now, pw, ph);
    int accepted = 0, inliers = 0;
    if (ready && !anchor && count >= 8) {
        float moved[160][2];
        int valid[160], total = 0;
        for (int i = 0; i < count; i++) {
            float x = points[i][0], y = points[i][1], bx, by;
            valid[i] =
                lk(before, now, points[i][0], points[i][1], &x, &y) &&
                (!reverseCheck ||
                 (lk(now, before, x, y, &bx, &by) &&
                  ((bx - points[i][0]) * (bx - points[i][0]) + (by - points[i][1]) * (by - points[i][1]) < 2.25f)));
            moved[i][0] = x;
            moved[i][1] = y;
            total += valid[i];
        }
        double best[8];
        int bestcount = 0;
        uint32_t random = 12345;
        if (total >= 8) {
            for (int trial = 0; trial < 32; trial++) {
                int need = motionModel ? 2 : 8, subset[160] = {0}, selected = 0;
                for (int j = 0; j < 512 && selected < need; j++) {
                    random = random * 1664525u + 1013904223u;
                    int at = (int)(random % count);
                    if (valid[at] && !subset[at]) {
                        subset[at] = 1;
                        selected++;
                    }
                }
                double h[8];
                if (selected < need || !(motionModel ? similarity(points, moved, subset, count, h)
                                                     : homography(points, moved, subset, count, h)))
                    continue;
                int found = 0;
                for (int i = 0; i < count; i++) found += valid[i] && reprojection(points[i], moved[i], h) < 2.25f;
                if (found > bestcount) {
                    bestcount = found;
                    for (int k = 0; k < 8; k++) best[k] = h[k];
                }
            }
        }
        if (bestcount >= 8) {
            for (int i = 0; i < count; i++) valid[i] = valid[i] && reprojection(points[i], moved[i], best) < 2.25f;
            double h[8];
            if (motionModel ? similarity(points, moved, valid, count, h) : homography(points, moved, valid, count, h)) {
                double candidate[8];
                int good = 1;
                for (int i = 0; i < 4; i++) {
                    double x = quad[i * 2] / width, y = quad[i * 2 + 1] / height, d = h[6] * x + h[7] * y + 1;
                    candidate[i * 2] = (h[0] * x + h[1] * y + h[2]) / d * width;
                    candidate[i * 2 + 1] = (h[3] * x + h[4] * y + h[5]) / d * height;
                    good &= candidate[i * 2] >= -width && candidate[i * 2] < width * 2 &&
                            candidate[i * 2 + 1] >= -height && candidate[i * 2 + 1] < height * 2;
                }
                if (good) {
                    for (int i = 0; i < 8; i++) quad[i] = candidate[i];
                    accepted = 1;
                    inliers = bestcount;
                }
            }
        }
    }
    features();
    int length = 0, levelWidth = pw, levelHeight = ph;
    for (int level = 0; level < 4; level++) {
        length += levelWidth * levelHeight;
        levelWidth /= 2;
        levelHeight /= 2;
    }
    for (int i = 0; i < length; i++) before[i] = now[i];
    ready = 1;
    for (int i = 0; i < 8; i++) output[i] = quad[i];
    output[8] = count;
    output[9] = inliers;
    return accepted;
}
