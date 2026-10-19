/* float32 graph kernels: int16-activation convolution (F_CONV p0 = 4). Part of the src/gust.c translation unit. */
/* ---- int16-activation convolution (F_CONV p0 bit 4, Relay encoder): weight codes times int16 activations ----
   Each input row is quantized with its own scale s = max|x| / L to q = round-half-even(x * (L / max|x|)), L = 32767
   for codes of at most 4 bits (|code| <= 8, K <= 2048) and smaller for 5- to 8-bit codes (gq_levels, ops.c), so the
   products code * q sum exactly in int32 without overflow; the output is
   float32(sum) * (s * weight_scale) (+ bias). Integer sums are exact, so every build returns the same bits. */
#ifdef GUST_F32_SIMD
typedef short gv8s __attribute__((vector_size(16), aligned(4), may_alias));
typedef int gv4i __attribute__((vector_size(16), aligned(4), may_alias));
typedef int32_t GqI32 __attribute__((may_alias, aligned(2)));
static inline __attribute__((always_inline)) gv4i gq_dot(gv8s a, gv8s b) { /* lane i: a[2i]*b[2i] + a[2i+1]*b[2i+1] */
#if defined(__wasm_simd128__)
    return (gv4i)__builtin_wasm_dot_s_i32x4_i16x8(a, b);
#elif defined(__aarch64__)
    const int16x8_t x = (int16x8_t)a, y = (int16x8_t)b; /* exact int32 products, pairwise sums: same as pmaddwd */
    return (gv4i)vpaddq_s32(vmull_s16(vget_low_s16(x), vget_low_s16(y)), vmull_high_s16(x, y));
#else
    return (gv4i)__builtin_ia32_pmaddwd128(a, b);
#endif
}
#endif
#if defined(__wasm_simd128__)
#define GQ_TR 4u /* 4 rows x 8 columns: 47.3 vs 34.9 GMAC/s for 6 x 8 in V8 */
#elif defined(__AVX2__) && !defined(__aarch64__)
#define GQ_TR 12u /* 256-bit tile: one vpmaddwd covers a pair of k for all 8 columns */
#else
#define GQ_TR 6u
#endif

/* max |x[k]| over k < K (exact, so any order gives the same value). */
static inline float gq_absmax(const float *x, uint32_t K) {
    float m = 0.0f;
    uint32_t k = 0;
#ifdef GUST_F32_SIMD
    gv4f mv = {0.0f, 0.0f, 0.0f, 0.0f};
    for (; k + 4 <= K; k += 4) { /* |x| is never NaN, so the pseudo-max (one instruction) is the exact max */
        const gv4f v = __builtin_elementwise_abs(*(const gv4f *)(x + k));
#if defined(__wasm_simd128__)
        mv = __builtin_wasm_pmax_f32x4(mv, v);
#elif defined(__aarch64__)
        mv = (gv4f)vmaxq_f32((float32x4_t)mv, (float32x4_t)v); /* no NaN here, so max is exact */
#else
        mv = __builtin_ia32_maxps(mv, v);
#endif
    }
    for (uint32_t i = 0; i < 4; i++)
        if (mv[i] > m) m = mv[i];
#endif
    for (; k < K; k++) {
        const float v = x[k] < 0.0f ? -x[k] : x[k];
        if (v > m) m = v;
    }
    return m;
}

/* out[j] = float32(acc[j]) * (s * ws[j]) (+ b[j]) for j < n: the same operations per element in every build. */
static inline void gq_epilogue(const int32_t *acc, float s, const float *ws, const float *b, float *out, uint32_t n) {
    uint32_t j = 0;
#ifdef GUST_F32_SIMD
    if (n == 8u) {
        const gv4f sv = {s, s, s, s};
        for (; j < 8u; j += 4) {
            gv4f y = __builtin_convertvector(*(const gv4i *)(acc + j), gv4f) * (sv * *(const gv4f *)(ws + j));
            if (b) y = y + *(const gv4f *)(b + j);
            *(gv4f *)(out + j) = y;
        }
        return;
    }
#endif
    for (; j < n; j++) {
        float y = (float)acc[j] * (s * ws[j]);
        if (b) y = y + b[j];
        out[j] = y;
    }
}

/* Signed codes of weights 0 .. K-1 of output row n into dst[16 * (k / 2) + 2 * j + k % 2]: column j < 8 of an
   8-column slab whose pairs of k are 16 values apart. bits <= 8, per-row scales. */
static void gq_codes_row(const GfWeights *w, uint32_t n, uint32_t K, int16_t *dst, uint32_t j) {
    const uint32_t W = 8u;
    const uint32_t bits = w->bits, mask = (1u << bits) - 1u, sign = 1u << (bits - 1u);
    const uint64_t pos = (uint64_t)n * w->K * bits;
    if (bits == 4 && !(pos & 7u)) { /* byte-aligned 4-bit row: two codes per byte, low nibble first */
        const uint8_t *b = w->code + (pos >> 3);
        int16_t *d = dst + 2u * j;
        const size_t step = 2u * W;
        uint32_t k = 0;
        for (; k + 2 <= K; k += 2, b++, d += step) {
            d[0] = (int16_t)((int32_t)((*b & 15u) ^ 8u) - 8);
            d[1] = (int16_t)((int32_t)((*b >> 4) ^ 8u) - 8);
        }
        if (k < K) d[0] = (int16_t)((int32_t)((*b & 15u) ^ 8u) - 8);
        return;
    }
    if (bits == 8 && !(pos & 7u)) { /* byte codes (float32-temporal-v1): the bit reader's values, one load each */
        const int8_t *b = (const int8_t *)(w->code + (pos >> 3));
        for (uint32_t k = 0; k < K; k++) dst[(size_t)(k >> 1) * 2u * W + 2u * j + (k & 1u)] = b[k];
        return;
    }
    const uint8_t *p = w->code + (pos >> 3);
    uint32_t acc = 0, nb = 0;
    if (pos & 7u) {
        acc = (uint32_t)*p++ >> (pos & 7u);
        nb = 8u - (uint32_t)(pos & 7u);
    }
    for (uint32_t k = 0; k < K; k++) {
        while (nb < bits) {
            acc |= (uint32_t)*p++ << nb;
            nb += 8u;
        }
        const uint32_t v = acc & mask;
        acc >>= bits;
        nb -= bits;
        dst[(size_t)(k >> 1) * 2u * W + 2u * j + (k & 1u)] = (int16_t)((int32_t)(v ^ sign) - (int32_t)sign);
    }
}

/* a[r][j] += sum over pairs p < kp of x_r[2p] w[p][j][0] + x_r[2p+1] w[p][j][1], j < 8 (int32, exact).
   x_r = x + r * xk; w is an 8-column slab (gq_codes_row), its pairs 16 values apart. */
static inline __attribute__((always_inline)) void gq_tile(const int16_t *x, uint32_t xk, const int16_t *w,
                                                          uint32_t kp, int32_t *a, const uint32_t R) {
    const uint32_t W = 8u;
#if defined(__aarch64__)
    /* AArch64: one vld2q splits a pair's 16 codes into the k = 2p and k = 2p + 1 rows of 8 columns; widening
       multiply-accumulates by the scalar activation (4 exact int32 MACs per instruction, no shuffles) suit
       in-order cores like Cortex-A53. Integer sums are exact, so the bits equal every other build. */
    int32x4_t acc[GQ_TR][2];
    for (uint32_t r = 0; r < R; r++) {
        acc[r][0] = vld1q_s32(a + 8u * r);
        acc[r][1] = vld1q_s32(a + 8u * r + 4u);
    }
    for (uint32_t p = 0; p < kp; p++) {
        const int16x8x2_t wv = vld2q_s16(w + (size_t)p * 2u * W);
        const int16x4_t w0l = vget_low_s16(wv.val[0]), w1l = vget_low_s16(wv.val[1]);
        for (uint32_t r = 0; r < R; r++) {
            /* Load the activation pair once and use indexed MACs. Scalar broadcasts otherwise become two
               LD1R loads per row plus copies of the high weight lanes on Cortex-A53. The alias type permits
               a two-byte-aligned pair, including an odd input-row stride rounded up by the caller. */
            const int16x4_t xv = vcreate_s16((uint32_t)*(const GqI32 *)(x + (size_t)r * xk + 2u * p));
            acc[r][0] = vmlal_lane_s16(vmlal_lane_s16(acc[r][0], w0l, xv, 0), w1l, xv, 1);
            acc[r][1] = vmlal_high_lane_s16(vmlal_high_lane_s16(acc[r][1], wv.val[0], xv, 0), wv.val[1], xv, 1);
        }
    }
    for (uint32_t r = 0; r < R; r++) {
        vst1q_s32(a + 8u * r, acc[r][0]);
        vst1q_s32(a + 8u * r + 4u, acc[r][1]);
    }
#elif defined(GUST_F32_SIMD) && defined(__AVX2__)
    /* x86 AVX2: a pair's 16 codes (8 columns x 2) are one 256-bit vector; vpmaddwd of the broadcast activation pair
       gives the 8 columns' exact int32 pair sums (the same integers as the 128-bit path). */
    typedef short gv16s __attribute__((vector_size(32), aligned(4), may_alias));
    typedef int gv8i __attribute__((vector_size(32), aligned(4), may_alias));
    gv8i acc[GQ_TR];
    for (uint32_t r = 0; r < R; r++) acc[r] = *(const gv8i *)(a + 8u * r);
    for (uint32_t p = 0; p < kp; p++) {
        const gv16s wv = *(const gv16s *)(w + (size_t)p * 2u * W);
        for (uint32_t r = 0; r < R; r++) {
            const int32_t xv = *(const GqI32 *)(x + (size_t)r * xk + 2u * p);
            const gv16s b = (gv16s)(gv8i){xv, xv, xv, xv, xv, xv, xv, xv};
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            acc[r] = (gv8i)__builtin_ia32_vpdpwssd256((__attribute__((vector_size(32))) int)acc[r], (__attribute__((vector_size(32))) int)b, (__attribute__((vector_size(32))) int)wv);
#else
            acc[r] = acc[r] + (gv8i)__builtin_ia32_pmaddwd256(b, wv);
#endif
        }
    }
    for (uint32_t r = 0; r < R; r++) *(gv8i *)(a + 8u * r) = acc[r];
#elif defined(GUST_F32_SIMD)
    gv4i acc[GQ_TR][2];
    for (uint32_t r = 0; r < R; r++) {
        acc[r][0] = *(const gv4i *)(a + 8u * r);
        acc[r][1] = *(const gv4i *)(a + 8u * r + 4u);
    }
    for (uint32_t p = 0; p < kp; p++) {
        const gv8s w0 = *(const gv8s *)(w + (size_t)p * 2u * W), w1 = *(const gv8s *)(w + (size_t)p * 2u * W + 8u);
        for (uint32_t r = 0; r < R; r++) {
            const int32_t xv = *(const GqI32 *)(x + (size_t)r * xk + 2u * p);
            const gv8s b = (gv8s)(gv4i){xv, xv, xv, xv};
            acc[r][0] = acc[r][0] + gq_dot(b, w0);
            acc[r][1] = acc[r][1] + gq_dot(b, w1);
        }
    }
    for (uint32_t r = 0; r < R; r++) {
        *(gv4i *)(a + 8u * r) = acc[r][0];
        *(gv4i *)(a + 8u * r + 4u) = acc[r][1];
    }
#else
    for (uint32_t p = 0; p < kp; p++)
        for (uint32_t r = 0; r < R; r++) {
            const int32_t x0 = x[(size_t)r * xk + 2u * p], x1 = x[(size_t)r * xk + 2u * p + 1u];
            for (uint32_t j = 0; j < 8u; j++)
                a[8u * r + j] += x0 * w[(size_t)p * 2u * W + 2u * j] + x1 * w[(size_t)p * 2u * W + 2u * j + 1u];
        }
#endif
}

/* q[k] = round-half-even(x[k] * inv) for k < K, zero up to kk (even). |x * inv| <= 32767 < 2^22, so adding and
   subtracting 1.5 * 2^23 rounds to the nearest integer, ties to even, in every build. */
static inline void gq_quantize(const float *x, uint32_t K, uint32_t kk, float inv, int16_t *q) {
    const float magic = 12582912.0f;
    uint32_t k = 0;
#ifdef GUST_F32_SIMD
    const gv4f iv = {inv, inv, inv, inv}, mg = {magic, magic, magic, magic};
    for (; k + 8 <= K; k += 8) { /* integral values in int16 range: int32 truncation and saturating narrowing are exact */
        const gv4i a = __builtin_convertvector((*(const gv4f *)(x + k) * iv + mg) - mg, gv4i);
        const gv4i b = __builtin_convertvector((*(const gv4f *)(x + k + 4) * iv + mg) - mg, gv4i);
#if defined(__wasm_simd128__)
        *(gv8s *)(q + k) = (gv8s)__builtin_wasm_narrow_s_i16x8_i32x4(a, b);
#elif defined(__aarch64__)
        *(gv8s *)(q + k) = (gv8s)vcombine_s16(vqmovn_s32((int32x4_t)a), vqmovn_s32((int32x4_t)b));
#else
        *(gv8s *)(q + k) = (gv8s)__builtin_ia32_packssdw128(a, b);
#endif
    }
#endif
    for (; k < K; k++) q[k] = (int16_t)(int32_t)((x[k] * inv + magic) - magic);
    for (uint32_t k = K; k < kk; k++) q[k] = 0;
}
