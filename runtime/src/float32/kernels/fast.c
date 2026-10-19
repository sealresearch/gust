/* float32 graph kernels: fast float32 kernels: wide panels, attention tiles, softmax, LayerNorm. Part of the src/gust.c translation unit. */
/* ---- fast float32 kernels (Relay encoder): wide convolution panels, float32 attention and LayerNorm ----
   Same contract as the float32 convolution above (acc = float32(acc + float32(x * w)), k increasing, seeded with
   the bias; lanes are output channels), so the bits equal gf_tile32_*; only the tile is wider: GF_TW output
   columns x GF_TR rows. The panel of GF_TW columns x GF_KB weights fits the 64 KiB scratch. */
#ifndef GF_TW
#if defined(__AVX512F__)
#define GF_TW 16u
#define GF_TR 12u /* one zmm per row, 12 rows: 15.7 vs 21.5 ms of scores at 600 tokens (Zen 4) */
#elif defined(__AVX__)
#define GF_TW 16u
#define GF_TR 6u
#elif defined(__wasm_simd128__)
#define GF_TW 8u
#define GF_TR 4u /* V8 keeps 4 x 8 accumulators in registers: 25.7 vs 15.4 GMAC/s for 6 x 8 (Node 24, x64) */
#else
#define GF_TW 8u
#define GF_TR 6u /* 6 x 8 measured fastest with SSE2 */
#endif
#endif
#define GF_NV (GF_TW / 8u)
#define GF_KB (16384u / GF_TW)
#ifdef GUST_F32_SIMD
typedef float gv8f __attribute__((vector_size(32), aligned(4), may_alias));
#endif

/* Restores a [kl][GF_TW] panel for outputs n0 .. n0+nc-1 (unused columns are zero). */
static void gf_decode_panel_w(const GfWeights *w, uint32_t n0, uint32_t nc, uint32_t k0, uint32_t kl, float *dst) {
    for (uint32_t j = 0; j < GF_TW; j++) {
        if (j >= nc) {
            for (uint32_t k = 0; k < kl; k++) dst[(size_t)k * GF_TW + j] = 0.0f;
            continue;
        }
        switch (w->bits) {
        case 2: gf_decode_row(w, 2, n0 + j, k0, kl, dst + j, GF_TW); break;
        case 3: gf_decode_row(w, 3, n0 + j, k0, kl, dst + j, GF_TW); break;
        case 4: gf_decode_row(w, 4, n0 + j, k0, kl, dst + j, GF_TW); break;
        case 5: gf_decode_row(w, 5, n0 + j, k0, kl, dst + j, GF_TW); break;
        case 6: gf_decode_row(w, 6, n0 + j, k0, kl, dst + j, GF_TW); break;
        case 7: gf_decode_row(w, 7, n0 + j, k0, kl, dst + j, GF_TW); break;
#ifdef GUST_FLOAT32_TEMPORAL
        case 32: gf_decode_row(w, 32, n0 + j, k0, kl, dst + j, GF_TW); break;
#endif
        default: gf_decode_row(w, 8, n0 + j, k0, kl, dst + j, GF_TW); break;
        }
    }
}

/* a[r][j] = a[r][j] + x_r[k] * w[k][j] for k = 0 .. kl-1 in order, rows r < R (a compile-time constant). */
static inline __attribute__((always_inline)) void gf_tile32w(const float *x, size_t xrow, const float *w,
                                                             uint32_t kl, float *a, const uint32_t R) {
#if defined(GUST_F32_SIMD) && defined(__AVX512F__) && GF_TW == 16u
    /* one 512-bit vector per row: the same lane arithmetic (acc = acc + x * w, k increasing), half the instructions */
    typedef float gv16f __attribute__((vector_size(64), aligned(4), may_alias));
    gv16f acc[GF_TR];
    for (uint32_t r = 0; r < R; r++) acc[r] = *(const gv16f *)(a + r * GF_TW);
    for (uint32_t k = 0; k < kl; k++) {
        const gv16f wv = *(const gv16f *)(w + (size_t)k * GF_TW);
        for (uint32_t r = 0; r < R; r++) {
            const float xv = x[(size_t)r * xrow + k];
            acc[r] = acc[r] + (gv16f){xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv, xv} * wv;
        }
    }
    for (uint32_t r = 0; r < R; r++) *(gv16f *)(a + r * GF_TW) = acc[r];
#elif defined(GUST_F32_SIMD)
    gv8f acc[GF_TR][GF_NV];
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t v = 0; v < GF_NV; v++) acc[r][v] = *(const gv8f *)(a + r * GF_TW + 8u * v);
    for (uint32_t k = 0; k < kl; k++) {
        gv8f wv[GF_NV];
        for (uint32_t v = 0; v < GF_NV; v++) wv[v] = *(const gv8f *)(w + (size_t)k * GF_TW + 8u * v);
        for (uint32_t r = 0; r < R; r++) {
            const float xv = x[(size_t)r * xrow + k];
            const gv8f b = {xv, xv, xv, xv, xv, xv, xv, xv};
            for (uint32_t v = 0; v < GF_NV; v++) acc[r][v] = acc[r][v] + b * wv[v];
        }
    }
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t v = 0; v < GF_NV; v++) *(gv8f *)(a + r * GF_TW + 8u * v) = acc[r][v];
#else
    for (uint32_t k = 0; k < kl; k++)
        for (uint32_t r = 0; r < R; r++) {
            const float xv = x[(size_t)r * xrow + k];
            for (uint32_t j = 0; j < GF_TW; j++) a[r * GF_TW + j] = a[r * GF_TW + j] + xv * w[(size_t)k * GF_TW + j];
        }
#endif
}

/* float32 dot product with eight partial sums (lane = d mod 8, d increasing), combined as
   ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7)). */
static inline float gf_dot32(const float *q, const float *k, uint32_t D) {
    float l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t d = 0;
#ifdef GUST_F32_SIMD
    gv8f s = {0, 0, 0, 0, 0, 0, 0, 0};
    for (; d + 8 <= D; d += 8) s = s + *(const gv8f *)(q + d) * *(const gv8f *)(k + d);
    for (uint32_t i = 0; i < 8; i++) l[i] = s[i];
#else
    for (; d + 8 <= D; d += 8)
        for (uint32_t i = 0; i < 8; i++) l[i] = l[i] + q[d + i] * k[d + i];
#endif
    for (uint32_t i = 0; d < D; d++, i++) l[i] = l[i] + q[d] * k[d];
    return ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
}

/* acc[d] = acc[d] + e * v[d] (lanes are dimensions; callers apply keys in increasing order). */
static inline void gf_axpy32(float *acc, float e, const float *v, uint32_t D) {
    uint32_t d = 0;
#ifdef GUST_F32_SIMD
    const gv8f ev = {e, e, e, e, e, e, e, e};
    for (; d + 8 <= D; d += 8) *(gv8f *)(acc + d) = *(const gv8f *)(acc + d) + ev * *(const gv8f *)(v + d);
#endif
    for (; d < D; d++) acc[d] = acc[d] + e * v[d];
}

/* float32 sum of n values with eight partial sums (lane = i mod 8), combined like gf_dot32. `sq`: sum of squared
   deviations from `mean` instead. */
static inline float gf_sum32(const float *x, uint32_t n, int sq, float mean) {
    float l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t i = 0;
#ifdef GUST_F32_SIMD
    gv8f s = {0, 0, 0, 0, 0, 0, 0, 0};
    const gv8f m = {mean, mean, mean, mean, mean, mean, mean, mean};
    if (sq)
        for (; i + 8 <= n; i += 8) {
            const gv8f t = *(const gv8f *)(x + i) - m;
            s = s + t * t;
        }
    else
        for (; i + 8 <= n; i += 8) s = s + *(const gv8f *)(x + i);
    for (uint32_t j = 0; j < 8; j++) l[j] = s[j];
#else
    for (; i + 8 <= n; i += 8)
        for (uint32_t j = 0; j < 8; j++) {
            const float t = sq ? x[i + j] - mean : x[i + j];
            l[j] = l[j] + (sq ? t * t : t);
        }
#endif
    for (uint32_t j = 0; i < n; i++, j++) {
        const float t = sq ? x[i] - mean : x[i];
        l[j] = l[j] + (sq ? t * t : t);
    }
    return ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
}

/* Vector exp and GELU of the fast paths. Pure float32 operations in a fixed order (no FMA, no contraction), so the
   SIMD and scalar spellings give the same bits on every build.
     exp(x), x clamped to [-87, 88]: n = round-to-even(x * log2 e) via the 1.5 * 2^23 trick, r = (x - n * ln2_hi) -
     n * ln2_lo, p = Cephes expf polynomial ((((((c5 r + c4) r + c3) r + c2) r + c1) r + c0) * r^2 + r) + 1,
     result p * 2^n with 2^n built from its exponent bits. Relative error about 1e-7.
     gelu(x) = (x * 0.5) * (1 + erf(x * 0.70710678)), erf by Abramowitz-Stegun 7.1.26 (absolute error 1.5e-7):
     t = 1 / (1 + 0.3275911 |u|), erf = sign(u) * (1 - (((((a5 t + a4) t + a3) t + a2) t + a1) t) * exp(-u * u)). */
#define GF_EXP_C5 1.9875691500e-4f
#define GF_EXP_C4 1.3981999507e-3f
#define GF_EXP_C3 8.3334519073e-3f
#define GF_EXP_C2 4.1665795894e-2f
#define GF_EXP_C1 1.6666665459e-1f
#define GF_EXP_C0 5.0000001201e-1f
#define GF_LOG2E 1.44269504088896341f
#define GF_LN2HI 0.693359375f
#define GF_LN2LO -2.12194440e-4f
#define GF_MAGIC 12582912.0f

static inline float gf_exp_s(float x) {
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    const float t = x * GF_LOG2E + GF_MAGIC;
    const float n = t - GF_MAGIC;
    float r = x - n * GF_LN2HI;
    r = r - n * GF_LN2LO;
    const float r2 = r * r;
    float p = GF_EXP_C5 * r + GF_EXP_C4;
    p = p * r + GF_EXP_C3;
    p = p * r + GF_EXP_C2;
    p = p * r + GF_EXP_C1;
    p = p * r + GF_EXP_C0;
    p = p * r2 + r;
    p = p + 1.0f;
    const int32_t k = (int32_t)gf_asuint(t) - (int32_t)gf_asuint(GF_MAGIC);
    return p * gf_asfloat((uint32_t)(k + 127) << 23);
}

static inline float gf_gelu_s(float x) {
    const float u = x * 0.70710678118654752440f;
    const float au = gf_asfloat(gf_asuint(u) & 0x7fffffffu);
    const float t = 1.0f / (1.0f + 0.3275911f * au);
    float p = 1.061405429f * t + -1.453152027f;
    p = p * t + 1.421413741f;
    p = p * t + -0.284496736f;
    p = p * t + 0.254829592f;
    p = p * t;
    const float y = 1.0f - p * gf_exp_s(-(au * au));
    const float e = gf_asfloat(gf_asuint(y) | (gf_asuint(u) & 0x80000000u));
    return (x * 0.5f) * (1.0f + e);
}

#ifdef GUST_F32_SIMD
typedef int32_t gv8i __attribute__((vector_size(32), aligned(4), may_alias));
#define GF_V8(c) ((gv8f){(c), (c), (c), (c), (c), (c), (c), (c)})
static inline gv8f gf_sel8(gv8i m, gv8f a, gv8f b) { /* m ? a : b, per lane */
    return (gv8f)(((gv8i)a & m) | ((gv8i)b & ~m));
}
static inline gv8f gf_exp8(gv8f x) {
    x = gf_sel8(x < GF_V8(-87.0f), GF_V8(-87.0f), x);
    x = gf_sel8(x > GF_V8(88.0f), GF_V8(88.0f), x);
    const gv8f t = x * GF_V8(GF_LOG2E) + GF_V8(GF_MAGIC);
    const gv8f n = t - GF_V8(GF_MAGIC);
    gv8f r = x - n * GF_V8(GF_LN2HI);
    r = r - n * GF_V8(GF_LN2LO);
    const gv8f r2 = r * r;
    gv8f p = GF_V8(GF_EXP_C5) * r + GF_V8(GF_EXP_C4);
    p = p * r + GF_V8(GF_EXP_C3);
    p = p * r + GF_V8(GF_EXP_C2);
    p = p * r + GF_V8(GF_EXP_C1);
    p = p * r + GF_V8(GF_EXP_C0);
    p = p * r2 + r;
    p = p + GF_V8(1.0f);
    const gv8i k = (gv8i)t - (gv8i)GF_V8(GF_MAGIC);
    return p * (gv8f)((k + 127) << 23);
}
static inline gv8f gf_gelu8(gv8f x) {
    const gv8f u = x * GF_V8(0.70710678118654752440f);
    const gv8f au = (gv8f)((gv8i)u & 0x7fffffff);
    const gv8f t = GF_V8(1.0f) / (GF_V8(1.0f) + GF_V8(0.3275911f) * au);
    gv8f p = GF_V8(1.061405429f) * t + GF_V8(-1.453152027f);
    p = p * t + GF_V8(1.421413741f);
    p = p * t + GF_V8(-0.284496736f);
    p = p * t + GF_V8(0.254829592f);
    p = p * t;
    const gv8f y = GF_V8(1.0f) - p * gf_exp8(-(au * au));
    const gv8f e = (gv8f)((gv8i)y | ((gv8i)u & (int32_t)0x80000000u));
    return (x * GF_V8(0.5f)) * (GF_V8(1.0f) + e);
}
#endif

/* z[i] = gelu(x[i]) for n values. */
static void gf_gelu_n(const float *x, float *z, size_t n) {
    size_t i = 0;
#ifdef GUST_F32_SIMD
    for (; i + 8 <= n; i += 8) *(gv8f *)(z + i) = gf_gelu8(*(const gv8f *)(x + i));
#endif
    for (; i < n; i++) z[i] = gf_gelu_s(x[i]);
}

/* e[j] = exp(s[j] - m) in place, returns the sum with eight partial lanes (lane j mod 8) and the tree of gf_dot32. */
/* max of s[0 .. n-1] (n > 0; 0 if n == 0). Scores are finite, so the maximum is one value whatever the order (only
   the sign of a zero maximum can depend on it, and exp(x - (+0)) == exp(x - (-0)) bit for bit). */
static inline float gf_maxf(const float *s, uint32_t n) {
    if (!n) return 0.0f;
    float m = s[0];
    uint32_t j = 1;
#ifdef GUST_F32_SIMD
    if (n >= 8) {
        gv8f mv = *(const gv8f *)s;
        for (j = 8; j + 8 <= n; j += 8) mv = __builtin_elementwise_max(mv, *(const gv8f *)(s + j));
        m = mv[0];
        for (uint32_t i = 1; i < 8; i++) m = mv[i] > m ? mv[i] : m;
    }
#endif
    for (; j < n; j++) m = s[j] > m ? s[j] : m;
    return m;
}

static float gf_softmax_exp(float *s, uint32_t n, float m) {
    float l[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t j = 0;
#ifdef GUST_F32_SIMD
    gv8f acc = GF_V8(0.0f);
    for (; j + 8 <= n; j += 8) {
        const gv8f e = gf_exp8(*(const gv8f *)(s + j) - GF_V8(m));
        *(gv8f *)(s + j) = e;
        acc = acc + e;
    }
    for (uint32_t i = 0; i < 8; i++) l[i] = acc[i];
#else
    for (; j + 8 <= n; j += 8)
        for (uint32_t i = 0; i < 8; i++) {
            const float e = gf_exp_s(s[j + i] - m);
            s[j + i] = e;
            l[i] = l[i] + e;
        }
#endif
    for (uint32_t i = 0; j < n; j++, i++) {
        const float e = gf_exp_s(s[j] - m);
        s[j] = e;
        l[i] = l[i] + e;
    }
    return ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
}

/* gf_dot32 of one query against four keys (rows k, k + stride, ...): the same arithmetic per key, four independent
   chains instead of one, so the adds overlap. */
static inline void gf_dot32x4(const float *q, const float *k, size_t stride, uint32_t D, float out[4]) {
#ifdef GUST_F32_SIMD
    if (!(D & 7u)) {
        gv8f s0 = GF_V8(0.0f), s1 = GF_V8(0.0f), s2 = GF_V8(0.0f), s3 = GF_V8(0.0f);
        const float *k1 = k + stride, *k2 = k + 2 * stride, *k3 = k + 3 * stride;
        for (uint32_t d = 0; d < D; d += 8) {
            const gv8f qv = *(const gv8f *)(q + d);
            s0 = s0 + qv * *(const gv8f *)(k + d);
            s1 = s1 + qv * *(const gv8f *)(k1 + d);
            s2 = s2 + qv * *(const gv8f *)(k2 + d);
            s3 = s3 + qv * *(const gv8f *)(k3 + d);
        }
        out[0] = ((s0[0] + s0[4]) + (s0[2] + s0[6])) + ((s0[1] + s0[5]) + (s0[3] + s0[7]));
        out[1] = ((s1[0] + s1[4]) + (s1[2] + s1[6])) + ((s1[1] + s1[5]) + (s1[3] + s1[7]));
        out[2] = ((s2[0] + s2[4]) + (s2[2] + s2[6])) + ((s2[1] + s2[5]) + (s2[3] + s2[7]));
        out[3] = ((s3[0] + s3[4]) + (s3[2] + s3[6])) + ((s3[1] + s3[5]) + (s3[3] + s3[7]));
        return;
    }
#endif
    for (uint32_t i = 0; i < 4; i++) out[i] = gf_dot32(q, k + i * stride, D);
}
