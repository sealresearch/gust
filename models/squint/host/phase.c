/* Fixed-memory current-image Fourier phase identity experiment. No payload decoding. */
#include <stddef.h>
#include "phase_tables.h"
typedef struct {
    double r, i;
} Complex;
static Complex currentPhase[64 * 64], candidatePhase[64 * 64];
static int preparedSide;
static void fft(Complex *x) {
    for (int i = 1, j = 0; i < 64; i++) {
        int bit = 32;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            Complex t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }
    for (int length = 2; length <= 64; length *= 2)
        for (int start = 0; start < 64; start += length)
            for (int j = 0; j < length / 2; j++) {
                int k = j * 128 / length;
                Complex a = x[start + j], b = x[start + j + length / 2];
                double r = b.r * phaseCos[k] + b.i * phaseSin[k], i = b.i * phaseCos[k] - b.r * phaseSin[k];
                x[start + j] = (Complex){a.r + r, a.i + i};
                x[start + j + length / 2] = (Complex){a.r - r, a.i - i};
            }
}
static void fft2(Complex *x) {
    Complex column[64];
    for (int y = 0; y < 64; y++) fft(x + y * 64);
    for (int col = 0; col < 64; col++) {
        for (int y = 0; y < 64; y++) column[y] = x[y * 64 + col];
        fft(column);
        for (int y = 0; y < 64; y++) x[y * 64 + col] = column[y];
    }
}
static void transform(const float *image, int side, Complex *out) {
    double ratio = (double)side / 64, mean = 0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) {
            double x0 = x * ratio, x1 = (x + 1) * ratio, y0 = y * ratio, y1 = (y + 1) * ratio, sum = 0;
            for (int yy = (int)y0; yy < y1; yy++)
                for (int xx = (int)x0; xx < x1; xx++) {
                    double left = x0 > xx ? x0 : xx, right = x1 < xx + 1 ? x1 : xx + 1, top = y0 > yy ? y0 : yy,
                           bottom = y1 < yy + 1 ? y1 : yy + 1;
                    sum += image[yy * side + xx] * (right - left) * (bottom - top);
                }
            out[y * 64 + x] = (Complex){sum / (ratio * ratio), 0};
            mean += out[y * 64 + x].r;
        }
    mean /= 4096;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) out[y * 64 + x].r = (out[y * 64 + x].r - mean) * phaseWindow[x] * phaseWindow[y];
    fft2(out);
}
int sp_phase_prepare(const float *candidate, int side) {
    if (side < 116 || side > 196 || side % 4) return -1;
    transform(candidate, side, candidatePhase);
    preparedSide = side;
    return 0;
}
double sp_phase_score(const float *image, int side) {
    if (side != preparedSide) return -1;
    transform(image, side, currentPhase);
    double scores[9] = {0}, weightSum = 0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x <= 32; x++) {
            int fy = y < 32 ? y : y - 64, radius = x * x + fy * fy;
            if (radius < 9 || radius > 256) continue;
            Complex a = currentPhase[y * 64 + x], b = candidatePhase[y * 64 + x];
            double r = a.r * b.r + a.i * b.i, i = a.i * b.r - a.r * b.i, magnitude = __builtin_sqrt(r * r + i * i),
                   weight = __builtin_sqrt(__builtin_sqrt(magnitude));
            if (magnitude < 1e-10) {
                r /= 1e-10;
                i /= 1e-10;
            } else {
                r /= magnitude;
                i /= magnitude;
            }
            weightSum += weight;
            for (int dy = -1, index = 0; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++, index++) {
                    int k = (x * dx + fy * dy + 256) % 128;
                    scores[index] += weight * (r * phaseCos[k] - i * phaseSin[k]);
                }
        }
    if (weightSum < 1e-10) return -1;
    double best = -1;
    for (int i = 0; i < 9; i++)
        if (scores[i] / weightSum > best) best = scores[i] / weightSum;
    return best;
}
