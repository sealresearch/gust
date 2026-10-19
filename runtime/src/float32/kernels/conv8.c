/* float32 graph kernels: int8-activation convolution (F_CONV p0 = 12). Part of the src/gust.c translation unit. */
/* ---- int8-activation convolution (F_CONV p0 = 4 | 8): weight codes (<= 4 bits) times per-row int8 activations ----
   q = round-half-even(x * (127 / max|x|)), |q| <= 127; sums of code * q are exact in int32, so every build (and
   every instruction choice below) returns the same integers, hence the same bits. Row stride kq = K rounded up to a
   multiple of 4 (zero padded). Slab of 8 output columns, layout per ISA:
     GQ8_QUAD (x86 AVX2, AArch64 dotprod, wasm relaxed SIMD): [kq / 4][8][4]   one 4-byte group of k per column
     A53-class NEON, plain wasm SIMD128:                       [8][kq]        each column's codes contiguous
     otherwise (scalar):                                       [kq][8]        8 columns per k */
#if (defined(__x86_64__) && defined(__AVX2__)) || (defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)) || \
    (defined(__wasm_simd128__) && defined(__wasm_relaxed_simd__))
#define GQ8_QUAD 1
#else
#define GQ8_QUAD 0
#endif
#ifdef GQ8_TR
#elif defined(__x86_64__) && defined(__AVX2__)
#define GQ8_TR 12u
#elif defined(__wasm_simd128__) && defined(__wasm_relaxed_simd__)
#define GQ8_TR 4u
#elif defined(__wasm_simd128__)
#define GQ8_TR 2u
#elif defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD)
#define GQ8_TR 2u
#else
#define GQ8_TR 6u
#endif
/* slab index of code (k, column j), kq = padded K; A53-class NEON and plain wasm SIMD128 keep each column's codes
   contiguous ([8][kq], kq a multiple of 8) so a vector of 8 activations meets 8 codes of one column (vmlal_s8 /
   i32x4.dot_i16x8_s, no broadcasts) */
#if GQ8_QUAD
#define GQ8_KALIGN 4u
#define GQ8_IDX(k, j, kq) ((size_t)((k) >> 2) * 32u + 4u * (j) + ((k) & 3u))
#elif defined(__aarch64__) || defined(__wasm_simd128__)
#define GQ8_KALIGN 8u
#define GQ8_IDX(k, j, kq) ((size_t)(j) * (kq) + (k))
#else
#define GQ8_KALIGN 4u
#define GQ8_IDX(k, j, kq) ((size_t)(k) * 8u + (j))
#endif
/* x86 stores activations as x + 128 (unsigned bytes for vpdpbusd / vpmaddubsw), so the tile broadcasts them straight
   from memory */
#if defined(__x86_64__) && defined(__AVX2__)
#define GQ8_XOR 0x80
#else
#define GQ8_XOR 0
#endif
/* wasm relaxed dot takes its second operand as 7-bit unsigned: the slab holds code + 8 there */
#if defined(__wasm_simd128__) && defined(__wasm_relaxed_simd__)
#define GQ8_WOFF 8
#else
#define GQ8_WOFF 0
#endif

static inline void gq8_quantize(const float *x, uint32_t K, uint32_t kq, float inv, int8_t *q, int32_t *sum) {
    const float magic = 12582912.0f;
    int32_t s = 0;
    uint32_t k0 = 0;
#ifdef GUST_F32_SIMD
    {   /* 16 at a time: integral values in int8 range, so the truncating conversions are exact */
        typedef float gv16f __attribute__((vector_size(64), aligned(4), may_alias));
        typedef int gv16i __attribute__((vector_size(64), aligned(4)));
        typedef signed char gv16b __attribute__((vector_size(16), aligned(1), may_alias));
        const gv16f iv = {inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv};
        const gv16f mg = {magic, magic, magic, magic, magic, magic, magic, magic, magic, magic, magic, magic, magic,
                          magic, magic, magic};
        gv16i sv = {0};
        for (; k0 + 16 <= K; k0 += 16) {
            const gv16i v = __builtin_convertvector((*(const gv16f *)(x + k0) * iv + mg) - mg, gv16i);
            sv = sv + v;
            *(gv16b *)(q + k0) = __builtin_convertvector(v ^ GQ8_XOR, gv16b);
        }
        for (uint32_t i = 0; i < 16; i++) s += sv[i];
    }
#endif
    for (uint32_t k = k0; k < K; k++) {
        const int32_t v = (int32_t)((x[k] * inv + magic) - magic);
        q[k] = (int8_t)(v ^ GQ8_XOR);
        s += v;
    }
    for (uint32_t k = K; k < kq; k++) q[k] = (int8_t)GQ8_XOR;
    *sum = s;
}

/* codes of output row n (k < K, zero up to kq) into column j of the slab; returns their sum */
static int32_t gq8_codes_row(const GfWeights *w, uint32_t n, uint32_t K, uint32_t kq, int8_t *dst, uint32_t j) {
    const uint32_t bits = w->bits, mask = (1u << bits) - 1u, sign = 1u << (bits - 1u);
    const uint64_t pos = (uint64_t)n * w->K * bits;
    const uint8_t *p = w->code + (pos >> 3);
    uint32_t acc = 0, nb = 0;
    int32_t sum = 0;
    if (bits == 4 && !(pos & 7u)) { /* byte-aligned 4-bit row: two codes per byte, low nibble first */
        for (uint32_t k = 0; k < kq; k++) {
            const int32_t v = k < K ? (int32_t)(((p[k >> 1] >> (4u * (k & 1u))) & 15u) ^ 8u) - 8 : 0;
            dst[GQ8_IDX(k, j, kq)] = (int8_t)(v + GQ8_WOFF);
            sum += v;
        }
        return sum;
    }
    if (pos & 7u) {
        acc = (uint32_t)*p++ >> (pos & 7u);
        nb = 8u - (uint32_t)(pos & 7u);
    }
    for (uint32_t k = 0; k < kq; k++) {
        int32_t v = 0;
        if (k < K) {
            while (nb < bits) {
                acc |= (uint32_t)*p++ << nb;
                nb += 8u;
            }
            v = (int32_t)((acc & mask) ^ sign) - (int32_t)sign;
            acc >>= bits;
            nb -= bits;
        }
        dst[GQ8_IDX(k, j, kq)] = (int8_t)(v + GQ8_WOFF);
        sum += v;
    }
    return sum;
}

/* All 8 columns of a slab (columns jn .. 7 zero) with their code sums; 4-bit byte-aligned rows (K even) are unpacked
   a 4-code group of all 8 columns at a time (16 bytes -> 32 codes), other rows by gq8_codes_row. */
static void gq8_codes_slab(const GfWeights *w, uint32_t j0, uint32_t jn, uint32_t K, uint32_t kq, int8_t *slab,
                           int32_t *wsum) {
#if GQ8_QUAD
    if (w->bits == 4 && !(K & 1u) && jn == 8u) {
        typedef unsigned char gv16u __attribute__((vector_size(16)));
        typedef unsigned char gv32u __attribute__((vector_size(32), aligned(1), may_alias));
        typedef short gv32s __attribute__((vector_size(64)));
        const uint8_t *row[8];
        for (uint32_t j = 0; j < 8u; j++) row[j] = w->code + (((uint64_t)(j0 + j) * w->K * 4u) >> 3);
        gv32s acc = {0};
        const uint32_t qfull = K / 4u; /* quads whose 4 codes all exist */
        for (uint32_t q = 0; q < qfull; q++) {
            gv16u v;
            for (uint32_t j = 0; j < 8u; j++) { v[2 * j] = row[j][2 * q]; v[2 * j + 1] = row[j][2 * q + 1]; }
            const gv16u lo = v & 15, hi = v >> 4;
            gv32u c = __builtin_shufflevector(lo, hi, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                                              25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            c = (c ^ 8) - 8; /* signed codes as bytes */
            acc = acc + __builtin_convertvector((__attribute__((vector_size(32))) signed char)c, gv32s);
            *(gv32u *)(slab + (size_t)q * 32u) = c + GQ8_WOFF;
        }
        for (uint32_t j = 0; j < 8u; j++) wsum[j] = acc[4 * j] + acc[4 * j + 1] + acc[4 * j + 2] + acc[4 * j + 3];
        for (uint32_t q = qfull; q < kq / 4u; q++) /* tail quad: K = 4 q + 2, two codes, rest zero */
            for (uint32_t j = 0; j < 8u; j++)
                for (uint32_t t = 0; t < 4u; t++) {
                    const uint32_t k = 4u * q + t;
                    const int32_t v = k < K ? (int32_t)(((row[j][k >> 1] >> (4u * (k & 1u))) & 15u) ^ 8u) - 8 : 0;
                    slab[GQ8_IDX(k, j, kq)] = (int8_t)(v + GQ8_WOFF);
                    wsum[j] += v;
                }
        return;
    }
#endif
    for (uint32_t jj = 0; jj < 8u; jj++) {
        if (jj < jn) {
            wsum[jj] = gq8_codes_row(w, j0 + jj, K, kq, slab, jj);
            continue;
        }
        wsum[jj] = 0;
        for (uint32_t k = 0; k < kq; k++) slab[GQ8_IDX(k, jj, kq)] = (int8_t)GQ8_WOFF;
    }
}

/* a[r][j] = sum over k < kq of x_r[k] * code[k][j], rows r < R, x_r = x + r * xk (int32, exact). wsum: per-column
   code sums (x86 unsigned trick), xsum: per-row activation sums (wasm 7-bit trick). */
static inline __attribute__((always_inline)) void gq8_tile(const int8_t *x, uint32_t xk, const int8_t *w, uint32_t kq,
                                                           const int32_t *wsum, const int32_t *xsum, int32_t *a,
                                                           const uint32_t R) {
#if defined(__x86_64__) && defined(__AVX2__)
    typedef int gv8i __attribute__((vector_size(32), aligned(4), may_alias));
    typedef uint32_t GqU32 __attribute__((may_alias, aligned(1)));
    gv8i acc[GQ8_TR];
    for (uint32_t r = 0; r < R; r++) acc[r] = (gv8i){0, 0, 0, 0, 0, 0, 0, 0};
#if !(defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
    typedef char gv32c __attribute__((vector_size(32), aligned(4), may_alias));
    typedef short gv16h __attribute__((vector_size(32), aligned(4), may_alias));
    const gv16h one16 = (gv16h){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#endif
    for (uint32_t q = 0; q < kq / 4u; q++) {
        const gv8i wv = *(const gv8i *)(w + (size_t)q * 32u);
        for (uint32_t r = 0; r < R; r++) {
            const int32_t xv = (int32_t)*(const GqU32 *)(x + (size_t)r * xk + 4u * q); /* stored as x + 128 */
            const gv8i b = (gv8i){xv, xv, xv, xv, xv, xv, xv, xv};
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            acc[r] = (gv8i)__builtin_ia32_vpdpbusd256((gv8i)acc[r], (gv8i)b, (gv8i)wv);
#else /* u8 x s8 pairs -> s16 (|sum| <= 2 * 255 * 8: no saturation), then pairs -> s32 */
            const gv16h t = (gv16h)__builtin_ia32_pmaddubsw256((gv32c)b, (gv32c)wv);
            acc[r] = acc[r] + (gv8i)__builtin_ia32_pmaddwd256(t, one16);
#endif
        }
    }
    const gv8i c = *(const gv8i *)wsum * 128;
    for (uint32_t r = 0; r < R; r++) *(gv8i *)(a + 8u * r) = acc[r] - c;
    (void)xsum;
#elif defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc[GQ8_TR][2];
    for (uint32_t r = 0; r < R; r++) acc[r][0] = acc[r][1] = vdupq_n_s32(0);
    for (uint32_t q = 0; q < kq / 4u; q++) {
        const int8x16_t w0 = vld1q_s8(w + (size_t)q * 32u), w1 = vld1q_s8(w + (size_t)q * 32u + 16u);
        for (uint32_t r = 0; r < R; r++) {
            int32_t xv;
            __builtin_memcpy(&xv, x + (size_t)r * xk + 4u * q, 4);
            const int8x16_t b = vreinterpretq_s8_s32(vdupq_n_s32(xv));
            acc[r][0] = vdotq_s32(acc[r][0], w0, b);
            acc[r][1] = vdotq_s32(acc[r][1], w1, b);
        }
    }
    for (uint32_t r = 0; r < R; r++) { vst1q_s32(a + 8u * r, acc[r][0]); vst1q_s32(a + 8u * r + 4u, acc[r][1]); }
    (void)wsum; (void)xsum;
#elif defined(__aarch64__)
    /* in-order NEON (Cortex-A53): 8 activations x 8 codes of one column per vmlal_s8 (int16 lanes, |code * q| <=
       1016, 32 k per block), then widening pairwise adds give the 8 column sums of the block in two int32 vectors */
    int32x4_t res[GQ8_TR][2];
    for (uint32_t r = 0; r < R; r++) res[r][0] = res[r][1] = vdupq_n_s32(0);
    for (uint32_t k0 = 0; k0 < kq; k0 += 32u) {
        const uint32_t k1 = k0 + 32u < kq ? k0 + 32u : kq;
        int16x8_t t[GQ8_TR][8];
        for (uint32_t r = 0; r < R; r++)
            for (uint32_t j = 0; j < 8u; j++) t[r][j] = vdupq_n_s16(0);
        for (uint32_t k = k0; k < k1; k += 8u) {
            int8x8_t xv[GQ8_TR];
            for (uint32_t r = 0; r < R; r++) xv[r] = vld1_s8(x + (size_t)r * xk + k);
            for (uint32_t j = 0; j < 8u; j++) {
                const int8x8_t wv = vld1_s8(w + (size_t)j * kq + k);
                for (uint32_t r = 0; r < R; r++) t[r][j] = vmlal_s8(t[r][j], xv[r], wv);
            }
        }
        for (uint32_t r = 0; r < R; r++) {
            const int32x4_t q01 = vpaddq_s32(vpaddlq_s16(t[r][0]), vpaddlq_s16(t[r][1]));
            const int32x4_t q23 = vpaddq_s32(vpaddlq_s16(t[r][2]), vpaddlq_s16(t[r][3]));
            const int32x4_t q45 = vpaddq_s32(vpaddlq_s16(t[r][4]), vpaddlq_s16(t[r][5]));
            const int32x4_t q67 = vpaddq_s32(vpaddlq_s16(t[r][6]), vpaddlq_s16(t[r][7]));
            res[r][0] = vaddq_s32(res[r][0], vpaddq_s32(q01, q23));
            res[r][1] = vaddq_s32(res[r][1], vpaddq_s32(q45, q67));
        }
    }
    for (uint32_t r = 0; r < R; r++) { vst1q_s32(a + 8u * r, res[r][0]); vst1q_s32(a + 8u * r + 4u, res[r][1]); }
    (void)wsum; (void)xsum;
#elif defined(__wasm_simd128__) && defined(__wasm_relaxed_simd__)
    typedef signed char gv16c __attribute__((vector_size(16), aligned(4), may_alias));
    gv4i acc[GQ8_TR][2];
    for (uint32_t r = 0; r < R; r++) acc[r][0] = acc[r][1] = (gv4i){0, 0, 0, 0};
    for (uint32_t q = 0; q < kq / 4u; q++) {
        const gv4i w0 = *(const gv4i *)(w + (size_t)q * 32u), w1 = *(const gv4i *)(w + (size_t)q * 32u + 16u);
        for (uint32_t r = 0; r < R; r++) {
            const int32_t xv = *(const GqI32 *)(x + (size_t)r * xk + 4u * q);
            const gv4i b = (gv4i){xv, xv, xv, xv};
            acc[r][0] = (gv4i)__builtin_wasm_relaxed_dot_i8x16_i7x16_add_s_i32x4((gv16c)b, (gv16c)w0, acc[r][0]);
            acc[r][1] = (gv4i)__builtin_wasm_relaxed_dot_i8x16_i7x16_add_s_i32x4((gv16c)b, (gv16c)w1, acc[r][1]);
        }
    }
    for (uint32_t r = 0; r < R; r++) {
        const int32_t c = 8 * xsum[r];
        const gv4i cv = (gv4i){c, c, c, c};
        *(gv4i *)(a + 8u * r) = acc[r][0] - cv;
        *(gv4i *)(a + 8u * r + 4u) = acc[r][1] - cv;
    }
    (void)wsum;
#elif defined(__wasm_simd128__)
    /* SIMD128 without relaxed dot: 8 activations and 8 codes of one column sign-extended to int16, i32x4.dot_i16x8_s
       (pmaddwd) into per-column partial sums, lanes added at the end */
    typedef signed char gv8c __attribute__((vector_size(8), aligned(1), may_alias));
    typedef short gv8h __attribute__((vector_size(16)));
    gv4i acc[GQ8_TR][8];
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t j = 0; j < 8u; j++) acc[r][j] = (gv4i){0, 0, 0, 0};
    for (uint32_t k = 0; k < kq; k += 8u) {
        gv8h xv[GQ8_TR];
        for (uint32_t r = 0; r < R; r++) xv[r] = __builtin_convertvector(*(const gv8c *)(x + (size_t)r * xk + k), gv8h);
        for (uint32_t j = 0; j < 8u; j++) {
            const gv8h wv = __builtin_convertvector(*(const gv8c *)(w + (size_t)j * kq + k), gv8h);
            for (uint32_t r = 0; r < R; r++) acc[r][j] += (gv4i)__builtin_wasm_dot_s_i32x4_i16x8(xv[r], wv);
        }
    }
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t j = 0; j < 8u; j++) a[8u * r + j] = acc[r][j][0] + acc[r][j][1] + acc[r][j][2] + acc[r][j][3];
    (void)wsum; (void)xsum;
#else
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t j = 0; j < 8u; j++) {
            int32_t s = 0;
            for (uint32_t k = 0; k < kq; k++) s += (int32_t)x[(size_t)r * xk + k] * w[GQ8_IDX(k, j, kq)];
            a[8u * r + j] = s;
        }
    (void)wsum; (void)xsum;
#endif
}
