/* float32 graph kernels: float32 / float64-accurate convolution tiles. Part of the src/gust.c translation unit. */
/* ---- accurate convolution (p0 bit 1): float64 products and sums of the float32 coefficients ---- */

/* a[r][j] += sum_k x_r[k] * w[k][j] for rows r = 0 .. R-1 (x_r = x + r * xrow), k increasing. Each product of two
   float32 values is exact in float64; only the sums round. */
static void gf_tile64_4(const float *x, size_t xrow, const float *w, uint32_t kl, double a[4][4]) {
    const float *xs0 = x, *xs1 = x + xrow, *xs2 = x + 2 * xrow, *xs3 = x + 3 * xrow;
#ifdef GUST_F32_SIMD
    gv2d lo0 = {a[0][0], a[0][1]}, hi0 = {a[0][2], a[0][3]};
    gv2d lo1 = {a[1][0], a[1][1]}, hi1 = {a[1][2], a[1][3]};
    gv2d lo2 = {a[2][0], a[2][1]}, hi2 = {a[2][2], a[2][3]};
    gv2d lo3 = {a[3][0], a[3][1]}, hi3 = {a[3][2], a[3][3]};
    for (uint32_t k = 0; k < kl; k++) {
        const gv4f wf = *(const gv4f *)(w + (size_t)k * 4);
        const gv2d wl = __builtin_convertvector(__builtin_shufflevector(wf, wf, 0, 1), gv2d);
        const gv2d wh = __builtin_convertvector(__builtin_shufflevector(wf, wf, 2, 3), gv2d);
        const double x0 = (double)xs0[k], x1 = (double)xs1[k], x2 = (double)xs2[k], x3 = (double)xs3[k];
        const gv2d v0 = {x0, x0}, v1 = {x1, x1}, v2 = {x2, x2}, v3 = {x3, x3};
        lo0 = lo0 + v0 * wl;
        hi0 = hi0 + v0 * wh;
        lo1 = lo1 + v1 * wl;
        hi1 = hi1 + v1 * wh;
        lo2 = lo2 + v2 * wl;
        hi2 = hi2 + v2 * wh;
        lo3 = lo3 + v3 * wl;
        hi3 = hi3 + v3 * wh;
    }
    a[0][0] = lo0[0], a[0][1] = lo0[1], a[0][2] = hi0[0], a[0][3] = hi0[1];
    a[1][0] = lo1[0], a[1][1] = lo1[1], a[1][2] = hi1[0], a[1][3] = hi1[1];
    a[2][0] = lo2[0], a[2][1] = lo2[1], a[2][2] = hi2[0], a[2][3] = hi2[1];
    a[3][0] = lo3[0], a[3][1] = lo3[1], a[3][2] = hi3[0], a[3][3] = hi3[1];
#else
    for (uint32_t k = 0; k < kl; k++) {
        const double x0 = (double)xs0[k], x1 = (double)xs1[k], x2 = (double)xs2[k], x3 = (double)xs3[k];
        for (uint32_t j = 0; j < 4; j++) {
            const double wj = (double)w[(size_t)k * 4 + j];
            a[0][j] = a[0][j] + x0 * wj;
            a[1][j] = a[1][j] + x1 * wj;
            a[2][j] = a[2][j] + x2 * wj;
            a[3][j] = a[3][j] + x3 * wj;
        }
    }
#endif
}

static void gf_tile64_1(const float *x, const float *w, uint32_t kl, double a[4]) {
#ifdef GUST_F32_SIMD
    gv2d lo = {a[0], a[1]}, hi = {a[2], a[3]};
    for (uint32_t k = 0; k < kl; k++) {
        const gv4f wf = *(const gv4f *)(w + (size_t)k * 4);
        const gv2d wl = __builtin_convertvector(__builtin_shufflevector(wf, wf, 0, 1), gv2d);
        const gv2d wh = __builtin_convertvector(__builtin_shufflevector(wf, wf, 2, 3), gv2d);
        const double x0 = (double)x[k];
        const gv2d v0 = {x0, x0};
        lo = lo + v0 * wl;
        hi = hi + v0 * wh;
    }
    a[0] = lo[0], a[1] = lo[1], a[2] = hi[0], a[3] = hi[1];
#else
    for (uint32_t k = 0; k < kl; k++) {
        const double x0 = (double)x[k];
        for (uint32_t j = 0; j < 4; j++) a[j] = a[j] + x0 * (double)w[(size_t)k * 4 + j];
    }
#endif
}

/* ---- float32 convolution (p0 bit 1 clear): acc = float32(acc + float32(x * w)), k increasing ---- */

static void gf_tile32_4(const float *x, size_t xrow, const float *w, uint32_t kl, float a[4][4]) {
    const float *xs0 = x, *xs1 = x + xrow, *xs2 = x + 2 * xrow, *xs3 = x + 3 * xrow;
#ifdef GUST_F32_SIMD
    gv4f c0 = *(const gv4f *)a[0], c1 = *(const gv4f *)a[1], c2 = *(const gv4f *)a[2], c3 = *(const gv4f *)a[3];
    for (uint32_t k = 0; k < kl; k++) {
        const gv4f wv = *(const gv4f *)(w + (size_t)k * 4);
        const float x0 = xs0[k], x1 = xs1[k], x2 = xs2[k], x3 = xs3[k];
        const gv4f b0 = {x0, x0, x0, x0}, b1 = {x1, x1, x1, x1}, b2 = {x2, x2, x2, x2}, b3 = {x3, x3, x3, x3};
        c0 = c0 + b0 * wv;
        c1 = c1 + b1 * wv;
        c2 = c2 + b2 * wv;
        c3 = c3 + b3 * wv;
    }
    *(gv4f *)a[0] = c0, *(gv4f *)a[1] = c1, *(gv4f *)a[2] = c2, *(gv4f *)a[3] = c3;
#else
    for (uint32_t k = 0; k < kl; k++) {
        const float x0 = xs0[k], x1 = xs1[k], x2 = xs2[k], x3 = xs3[k];
        for (uint32_t j = 0; j < 4; j++) {
            const float wj = w[(size_t)k * 4 + j];
            a[0][j] = a[0][j] + x0 * wj;
            a[1][j] = a[1][j] + x1 * wj;
            a[2][j] = a[2][j] + x2 * wj;
            a[3][j] = a[3][j] + x3 * wj;
        }
    }
#endif
}

static void gf_tile32_1(const float *x, const float *w, uint32_t kl, float a[4]) {
#ifdef GUST_F32_SIMD
    gv4f c0 = *(const gv4f *)a;
    for (uint32_t k = 0; k < kl; k++) {
        const gv4f wv = *(const gv4f *)(w + (size_t)k * 4);
        const float x0 = x[k];
        const gv4f b0 = {x0, x0, x0, x0};
        c0 = c0 + b0 * wv;
    }
    *(gv4f *)a = c0;
#else
    for (uint32_t k = 0; k < kl; k++) {
        const float x0 = x[k];
        for (uint32_t j = 0; j < 4; j++) a[j] = a[j] + x0 * w[(size_t)k * 4 + j];
    }
#endif
}
