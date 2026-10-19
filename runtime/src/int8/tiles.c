/* Vector INT8 building blocks for the convolution, attention and tensor operators: four-channel dots, register
   tiles (WebAssembly SIMD, AVX2) and AVX-512 VNNI tiles, exact vector requantization, and clamped copies. Every
   kernel returns the bits of the scalar definitions in kernels.c (dot, quant, clamp); the comments give the bound
   that makes each one exact. Part of the runtime/gust.c translation unit. */

/* 2^-t as a float64, t <= 1022 (exact power of two). */
static inline double pow2neg(uint32_t t) {
    union {
        uint64_t u;
        double d;
    } p = {(uint64_t)(1023 - t) << 52};
    return p.d;
}

/* Four output channels per input load. When the input tensor is proven
   non-negative (its producer clamps at zero), x fits unsigned 0..127 and the
   exact u8 x s8 pair products (|sum| <= 2*127*128 < 32768) map onto
   maddubs / relaxed i7 dot instructions with no saturation. Results are
   bit-identical to dot(). */
static inline void dot4(const int8_t *x, const int8_t *w, uint32_t n, uint32_t stride, int nonneg, int32_t r[4]) {
    const int8_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    uint32_t i = 0;
    r[0] = r[1] = r[2] = r[3] = 0;
#if defined(__wasm_relaxed_simd__)
    if (nonneg) {
        v128_t s0 = wasm_i32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
        for (; i + 16 <= n; i += 16) {
            v128_t v = wasm_v128_load(x + i);
            s0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(w0 + i), v, s0);
            s1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(w1 + i), v, s1);
            s2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(w2 + i), v, s2);
            s3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_load(w3 + i), v, s3);
        }
        v128_t t0 = wasm_i32x4_add(wasm_i32x4_shuffle(s0, s1, 0, 4, 1, 5), wasm_i32x4_shuffle(s0, s1, 2, 6, 3, 7));
        v128_t t1 = wasm_i32x4_add(wasm_i32x4_shuffle(s2, s3, 0, 4, 1, 5), wasm_i32x4_shuffle(s2, s3, 2, 6, 3, 7));
        v128_t t = wasm_i32x4_add(wasm_i32x4_shuffle(t0, t1, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t1, 2, 3, 6, 7));
        r[0] = wasm_i32x4_extract_lane(t, 0);
        r[1] = wasm_i32x4_extract_lane(t, 1);
        r[2] = wasm_i32x4_extract_lane(t, 2);
        r[3] = wasm_i32x4_extract_lane(t, 3);
    }
#elif defined(__AVX2__)
    if (nonneg) {
        const __m256i one = _mm256_set1_epi16(1);
        __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
        for (; i + 32 <= n; i += 32) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(x + i));
            s0 = _mm256_add_epi32(
                s0, _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *)(w0 + i))), one));
            s1 = _mm256_add_epi32(
                s1, _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *)(w1 + i))), one));
            s2 = _mm256_add_epi32(
                s2, _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *)(w2 + i))), one));
            s3 = _mm256_add_epi32(
                s3, _mm256_madd_epi16(_mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *)(w3 + i))), one));
        }
        __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));
        __m128i u = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
        if (i + 16 <= n) { /* 16-byte tail with the same exact u8 x s8 products */
            const __m128i o1 = _mm_set1_epi16(1);
            __m128i v = _mm_loadu_si128((const __m128i *)(x + i));
            __m128i q0 = _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w0 + i))), o1);
            __m128i q1 = _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w1 + i))), o1);
            __m128i q2 = _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w2 + i))), o1);
            __m128i q3 = _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w3 + i))), o1);
            u = _mm_add_epi32(u, _mm_hadd_epi32(_mm_hadd_epi32(q0, q1), _mm_hadd_epi32(q2, q3)));
            i += 16;
        }
        int32_t lanes[4];
        _mm_storeu_si128((__m128i *)lanes, u);
        r[0] = lanes[0];
        r[1] = lanes[1];
        r[2] = lanes[2];
        r[3] = lanes[3];
    }
#elif defined(__SSSE3__)
    if (nonneg) {
        const __m128i one = _mm_set1_epi16(1);
        __m128i s0 = _mm_setzero_si128(), s1 = s0, s2 = s0, s3 = s0;
        for (; i + 16 <= n; i += 16) {
            __m128i v = _mm_loadu_si128((const __m128i *)(x + i));
            s0 = _mm_add_epi32(s0,
                               _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w0 + i))), one));
            s1 = _mm_add_epi32(s1,
                               _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w1 + i))), one));
            s2 = _mm_add_epi32(s2,
                               _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w2 + i))), one));
            s3 = _mm_add_epi32(s3,
                               _mm_madd_epi16(_mm_maddubs_epi16(v, _mm_loadu_si128((const __m128i *)(w3 + i))), one));
        }
        __m128i t = _mm_hadd_epi32(_mm_hadd_epi32(s0, s1), _mm_hadd_epi32(s2, s3));
        int32_t lanes[4];
        _mm_storeu_si128((__m128i *)lanes, t);
        r[0] = lanes[0];
        r[1] = lanes[1];
        r[2] = lanes[2];
        r[3] = lanes[3];
    }
#endif
#if defined(__wasm_simd128__)
    if (i == 0 && n >= 16) {
        v128_t s0 = wasm_i32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
        for (; i + 16 <= n; i += 16) {
            v128_t v = wasm_v128_load(x + i), lo = wasm_i16x8_extend_low_i8x16(v), hi = wasm_i16x8_extend_high_i8x16(v),
                   q;
#define ROW_ACC(S, W)                                                                                                  \
    q = wasm_v128_load((W) + i);                                                                                       \
    S = wasm_i32x4_add(S, wasm_i32x4_add(wasm_i32x4_dot_i16x8(lo, wasm_i16x8_extend_low_i8x16(q)),                     \
                                         wasm_i32x4_dot_i16x8(hi, wasm_i16x8_extend_high_i8x16(q))));
            ROW_ACC(s0, w0) ROW_ACC(s1, w1) ROW_ACC(s2, w2) ROW_ACC(s3, w3)
#undef ROW_ACC
        }
        v128_t t0 = wasm_i32x4_add(wasm_i32x4_shuffle(s0, s1, 0, 4, 1, 5), wasm_i32x4_shuffle(s0, s1, 2, 6, 3, 7));
        v128_t t1 = wasm_i32x4_add(wasm_i32x4_shuffle(s2, s3, 0, 4, 1, 5), wasm_i32x4_shuffle(s2, s3, 2, 6, 3, 7));
        v128_t t = wasm_i32x4_add(wasm_i32x4_shuffle(t0, t1, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t1, 2, 3, 6, 7));
        r[0] = wasm_i32x4_extract_lane(t, 0);
        r[1] = wasm_i32x4_extract_lane(t, 1);
        r[2] = wasm_i32x4_extract_lane(t, 2);
        r[3] = wasm_i32x4_extract_lane(t, 3);
    }
#elif defined(__SSE2__)
    if (i == 0 && n >= 16) {
        __m128i zero = _mm_setzero_si128(), s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        for (; i + 16 <= n; i += 16) {
            __m128i v = _mm_loadu_si128((const __m128i *)(x + i)), vs = _mm_cmpgt_epi8(zero, v);
            __m128i lo = _mm_unpacklo_epi8(v, vs), hi = _mm_unpackhi_epi8(v, vs), q, qs;
#define ROW_ACC(S, W)                                                                                                  \
    q = _mm_loadu_si128((const __m128i *)((W) + i));                                                                   \
    qs = _mm_cmpgt_epi8(zero, q);                                                                                      \
    S = _mm_add_epi32(                                                                                                 \
        S, _mm_add_epi32(_mm_madd_epi16(lo, _mm_unpacklo_epi8(q, qs)), _mm_madd_epi16(hi, _mm_unpackhi_epi8(q, qs))));
            ROW_ACC(s0, w0) ROW_ACC(s1, w1) ROW_ACC(s2, w2) ROW_ACC(s3, w3)
#undef ROW_ACC
        }
        int32_t l0[4], l1[4], l2[4], l3[4];
        _mm_storeu_si128((__m128i *)l0, s0);
        _mm_storeu_si128((__m128i *)l1, s1);
        _mm_storeu_si128((__m128i *)l2, s2);
        _mm_storeu_si128((__m128i *)l3, s3);
        r[0] = l0[0] + l0[1] + l0[2] + l0[3];
        r[1] = l1[0] + l1[1] + l1[2] + l1[3];
        r[2] = l2[0] + l2[1] + l2[2] + l2[3];
        r[3] = l3[0] + l3[1] + l3[2] + l3[3];
    }
#endif
    (void)nonneg;
    for (; i < n; i++) {
        int32_t v = x[i];
        r[0] += v * w0[i];
        r[1] += v * w1[i];
        r[2] += v * w2[i];
        r[3] += v * w3[i];
    }
}

/* Two pixels x four output channels for non-negative inputs: every weight
   load feeds two pixels. Returns 0 when this build has no such kernel or n
   is not a multiple of the vector width (caller then uses dot4). */
static inline int dot4x2(const int8_t *x0, const int8_t *x1, const int8_t *w, uint32_t n, int32_t r0[4],
                         int32_t r1[4]) {
#if defined(__wasm_relaxed_simd__)
    if (n % 16) return 0;
    const int8_t *w0 = w, *w1 = w + n, *w2 = w + 2 * n, *w3 = w + 3 * n;
    v128_t a0 = wasm_i32x4_splat(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (uint32_t i = 0; i < n; i += 16) {
        v128_t u = wasm_v128_load(x0 + i), v = wasm_v128_load(x1 + i), q;
        q = wasm_v128_load(w0 + i);
        a0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, u, a0);
        b0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, v, b0);
        q = wasm_v128_load(w1 + i);
        a1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, u, a1);
        b1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, v, b1);
        q = wasm_v128_load(w2 + i);
        a2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, u, a2);
        b2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, v, b2);
        q = wasm_v128_load(w3 + i);
        a3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, u, a3);
        b3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q, v, b3);
    }
#define REDUCE4(S0, S1, S2, S3, R)                                                                                     \
    {                                                                                                                  \
        v128_t t0 = wasm_i32x4_add(wasm_i32x4_shuffle(S0, S1, 0, 4, 1, 5), wasm_i32x4_shuffle(S0, S1, 2, 6, 3, 7));    \
        v128_t t1 = wasm_i32x4_add(wasm_i32x4_shuffle(S2, S3, 0, 4, 1, 5), wasm_i32x4_shuffle(S2, S3, 2, 6, 3, 7));    \
        v128_t t = wasm_i32x4_add(wasm_i32x4_shuffle(t0, t1, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t1, 2, 3, 6, 7));     \
        wasm_v128_store(R, t);                                                                                         \
    }
    REDUCE4(a0, a1, a2, a3, r0);
    REDUCE4(b0, b1, b2, b3, r1);
#undef REDUCE4
    return 1;
#elif defined(__AVX2__)
    if (n % 32) return 0;
    const int8_t *w0 = w, *w1 = w + n, *w2 = w + 2 * n, *w3 = w + 3 * n;
    const __m256i one = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (uint32_t i = 0; i < n; i += 32) {
        __m256i u = _mm256_loadu_si256((const __m256i *)(x0 + i)), v = _mm256_loadu_si256((const __m256i *)(x1 + i)), q;
        q = _mm256_loadu_si256((const __m256i *)(w0 + i));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(u, q), one));
        b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(_mm256_maddubs_epi16(v, q), one));
        q = _mm256_loadu_si256((const __m256i *)(w1 + i));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(u, q), one));
        b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(_mm256_maddubs_epi16(v, q), one));
        q = _mm256_loadu_si256((const __m256i *)(w2 + i));
        a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(u, q), one));
        b2 = _mm256_add_epi32(b2, _mm256_madd_epi16(_mm256_maddubs_epi16(v, q), one));
        q = _mm256_loadu_si256((const __m256i *)(w3 + i));
        a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(u, q), one));
        b3 = _mm256_add_epi32(b3, _mm256_madd_epi16(_mm256_maddubs_epi16(v, q), one));
    }
#define REDUCE4(S0, S1, S2, S3, R)                                                                                     \
    {                                                                                                                  \
        __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(S0, S1), _mm256_hadd_epi32(S2, S3));                           \
        _mm_storeu_si128((__m128i *)(R), _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1)));    \
    }
    REDUCE4(a0, a1, a2, a3, r0);
    REDUCE4(b0, b1, b2, b3, r1);
#undef REDUCE4
    return 1;
#else
    (void)x0;
    (void)x1;
    (void)w;
    (void)n;
    (void)r0;
    (void)r1;
    return 0;
#endif
}

/* Register tiles for convolutions whose taps are contiguous (every 1x1, and
   unpadded kx1 / 1xk patches): TILE_P pixels x TILE_C output channels over one
   dot of length n (n%16==0). x[p] is pixel p's patch, channel j's weights
   start at w+j*row; t[j] lane p = exact dot(x[p],w_j). Non-negative inputs
   (0..127, u8) use u8 x s8 pair products (|pair| <= 2*127*128 < 2^15: no
   int16 saturation in maddubs, deterministic relaxed i8 x i7 dots) widened to
   int32 at once; other inputs are sign-extended to int16 for pmaddwd /
   i32x4.dot (exact for any input). Lanes hold <= terms*16384 < 2^31. */
#if defined(__wasm_simd128__) || (defined(__AVX2__) && !defined(GUST_INT8_VNNI))
#define INT8_TILE 1
#define TILE_P    4     /* pixels per register tile (lanes of the result) */
#define TILE_C    2     /* output channels per register tile */
#define TILE_L    16    /* tile dot lengths are multiples of TILE_L bytes */
#define TILE_G    16384 /* bytes of patches per pixel group (L1 resident) */
#endif
#ifdef INT8_TILE
/* quant() in f64 with per-lane f=m*2^-s (power-of-two scaling is exact) for
   SIMD sets without per-lane 64-bit arithmetic shifts. Let P=v*m (|P|<2^62).
   If |P|<2^53, q=v*f=P/2^s is exact and so is q+-0.5 (sign of q) unless
   |q|>=2^23-1; trunc of it is then the scalar half-away-from-zero rounding
   (s=0: q itself). Otherwise |q|>=2^23 after any rounding (monotonic, sign
   kept), far outside [lo,hi], so the clamp is unchanged. Clamping before
   trunc is equivalent since lo,hi are integers. */
static inline double qscale(int32_t m, uint32_t s) {
    union {
        uint64_t u;
        double d;
    } p = {(uint64_t)(1023 - s) << 52};
    return (double)m * p.d;
}
#endif
#if defined(__wasm_simd128__)
static inline v128_t wasm_sum4(v128_t a, v128_t b, v128_t c, v128_t d) { /* lane k = total of k-th argument */
    v128_t t0 = wasm_i32x4_add(wasm_i32x4_shuffle(a, b, 0, 4, 1, 5), wasm_i32x4_shuffle(a, b, 2, 6, 3, 7));
    v128_t t1 = wasm_i32x4_add(wasm_i32x4_shuffle(c, d, 0, 4, 1, 5), wasm_i32x4_shuffle(c, d, 2, 6, 3, 7));
    return wasm_i32x4_add(wasm_i32x4_shuffle(t0, t1, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t1, 2, 3, 6, 7));
}
static inline __attribute__((always_inline)) void tile_dot(const int8_t *const *x, const int8_t *w, uint32_t row,
                                                           uint32_t n, int nonneg, v128_t t[TILE_C]) {
    v128_t a[4][TILE_C]; /* TILE_P<=4; unused rows stay zero */
    for (uint32_t p = 0; p < 4; p++)
        for (uint32_t j = 0; j < TILE_C; j++) a[p][j] = wasm_i32x4_splat(0);
#if defined(__wasm_relaxed_simd__)
    if (nonneg) { /* widen with i32x4.dot against ones held in a register (extadd_pairwise reloads a constant in V8) */
        const v128_t one = wasm_i16x8_splat(1);
        for (uint32_t i = 0; i < n; i += 16) {
            v128_t v[TILE_P];
            for (uint32_t p = 0; p < TILE_P; p++) v[p] = wasm_v128_load(x[p] + i);
            for (uint32_t j = 0; j < TILE_C; j++) {
                v128_t q = wasm_v128_load(w + j * row + i);
                for (uint32_t p = 0; p < TILE_P; p++)
                    a[p][j] =
                        wasm_i32x4_add(a[p][j], wasm_i32x4_dot_i16x8(wasm_i16x8_relaxed_dot_i8x16_i7x16(q, v[p]), one));
            }
        }
    } else
#endif
        for (uint32_t i = 0; i < n; i += 8) { /* v128.load8x8_s: load and sign-extend in one instruction */
            v128_t q[TILE_C];
            for (uint32_t j = 0; j < TILE_C; j++) q[j] = wasm_i16x8_load8x8(w + j * row + i);
            for (uint32_t p = 0; p < TILE_P; p++) {
                v128_t v = wasm_i16x8_load8x8(x[p] + i);
                for (uint32_t j = 0; j < TILE_C; j++) a[p][j] = wasm_i32x4_add(a[p][j], wasm_i32x4_dot_i16x8(v, q[j]));
            }
        }
    (void)nonneg;
    for (uint32_t j = 0; j < TILE_C; j++) t[j] = wasm_sum4(a[0][j], a[1][j], a[2][j], a[3][j]);
}
/* f64x2 lanes q -> clamp(trunc(q+-0.5)), as int32 in the low word of t+1.5*2^52 */
static inline v128_t wasm_qround(v128_t q, v128_t lo, v128_t hi) {
    q = wasm_f64x2_trunc(
        wasm_f64x2_add(q, wasm_v128_or(wasm_v128_and(q, wasm_f64x2_splat(-0.0)), wasm_f64x2_splat(0.5))));
    return wasm_f64x2_add(wasm_f64x2_pmin(wasm_f64x2_pmax(q, lo), hi), wasm_f64x2_splat(6755399441055744.0));
}
/* quant(v) of four int32 lanes; f01/f23 hold lanes 0,1 / 2,3 scales, lo/hi are f64x2 splats. */
static inline v128_t wasm_quant(v128_t v, v128_t f01, v128_t f23, v128_t lo, v128_t hi) {
    v128_t q0 = wasm_qround(wasm_f64x2_mul(wasm_f64x2_convert_low_i32x4(v), f01), lo, hi);
    v128_t q1 =
        wasm_qround(wasm_f64x2_mul(wasm_f64x2_convert_low_i32x4(wasm_i32x4_shuffle(v, v, 2, 3, 2, 3)), f23), lo, hi);
    return wasm_i32x4_shuffle(q0, q1, 0, 2, 4, 6);
}
static inline v128_t wasm_quant16(const v128_t r[4], const v128_t b[4], const v128_t f[8], v128_t lo,
                                  v128_t hi) { /* 16 int8 = quant(r+b) */
    return wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(wasm_quant(wasm_i32x4_add(r[0], b[0]), f[0], f[1], lo, hi),
                                                           wasm_quant(wasm_i32x4_add(r[1], b[1]), f[2], f[3], lo, hi)),
                                   wasm_i16x8_narrow_i32x4(wasm_quant(wasm_i32x4_add(r[2], b[2]), f[4], f[5], lo, hi),
                                                           wasm_quant(wasm_i32x4_add(r[3], b[3]), f[6], f[7], lo, hi)));
}
typedef v128_t Lanes4;
typedef v128_t Scale4; /* tile driver interface */
#define T_LIMITS         const v128_t qlo = wasm_f64x2_splat(o->lo), qhi = wasm_f64x2_splat(o->hi)
#define T_BIAS(v)        wasm_i32x4_splat(v)
#define T_SCALE(m, s)    wasm_f64x2_splat(qscale(m, s))
#define T_QUANT(v, b, f) wasm_quant(wasm_i32x4_add(v, b), f, f, qlo, qhi)
#define T_STORE16(l, r)                                                                                                \
    wasm_v128_store(l,                                                                                                 \
                    wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(r[0], r[1]), wasm_i16x8_narrow_i32x4(r[2], r[3])))
#elif defined(__AVX2__) && !defined(GUST_INT8_VNNI)
static inline __m128i avx_sum4(__m256i a, __m256i b, __m256i c, __m256i d) { /* lane k = total of k-th argument */
    __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
    return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
}
#define AVX_LD(p)   _mm256_loadu_si256((const __m256i *)(p))
#define AVX_LD16(p) _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i *)(p))) /* upper half zero */
static inline __attribute__((always_inline)) void tile_dot(const int8_t *const *x, const int8_t *w, uint32_t row,
                                                           uint32_t n, int nonneg, __m128i t[TILE_C]) {
    __m256i a[4][TILE_C]; /* TILE_P<=4; unused rows stay zero */
    for (uint32_t p = 0; p < 4; p++)
        for (uint32_t j = 0; j < TILE_C; j++) a[p][j] = _mm256_setzero_si256();
    if (nonneg) {
        const __m256i one = _mm256_set1_epi16(1);
        uint32_t i = 0;
#define TILE_STEP(LD)                                                                                                  \
    {                                                                                                                  \
        __m256i v[TILE_P];                                                                                             \
        for (uint32_t p = 0; p < TILE_P; p++) v[p] = LD(x[p] + i);                                                     \
        for (uint32_t j = 0; j < TILE_C; j++) {                                                                        \
            __m256i q = LD(w + j * row + i);                                                                           \
            for (uint32_t p = 0; p < TILE_P; p++)                                                                      \
                a[p][j] = _mm256_add_epi32(a[p][j], _mm256_madd_epi16(_mm256_maddubs_epi16(v[p], q), one));            \
        }                                                                                                              \
    }
        for (; i + 32 <= n; i += 32) TILE_STEP(AVX_LD)
        if (i < n) TILE_STEP(AVX_LD16)
#undef TILE_STEP
    } else
        for (uint32_t i = 0; i < n; i += 16) {
            __m256i q[TILE_C];
            for (uint32_t j = 0; j < TILE_C; j++)
                q[j] = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(w + j * row + i)));
            for (uint32_t p = 0; p < TILE_P; p++) {
                __m256i v = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x[p] + i)));
                for (uint32_t j = 0; j < TILE_C; j++) a[p][j] = _mm256_add_epi32(a[p][j], _mm256_madd_epi16(v, q[j]));
            }
        }
    for (uint32_t j = 0; j < TILE_C; j++) t[j] = avx_sum4(a[0][j], a[1][j], a[2][j], a[3][j]);
}
/* quant(v) of four int32 lanes; f holds their scales, lo/hi are splats. */
static inline __m128i avx_quant(__m128i v, __m256d f, __m256d lo, __m256d hi) {
    __m256d q = _mm256_mul_pd(_mm256_cvtepi32_pd(v), f);
    q = _mm256_add_pd(q, _mm256_or_pd(_mm256_and_pd(q, _mm256_set1_pd(-0.0)), _mm256_set1_pd(0.5)));
    return _mm256_cvttpd_epi32(_mm256_min_pd(_mm256_max_pd(q, lo), hi));
}
static inline __m256d avx_qf(const int32_t *m, const uint32_t *s) { /* qscale() of four lanes */
    __m256i e = _mm256_slli_epi64(
        _mm256_cvtepu32_epi64(_mm_sub_epi32(_mm_set1_epi32(1023), _mm_loadu_si128((const __m128i *)s))), 52);
    return _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)m)), _mm256_castsi256_pd(e));
}
static inline __m128i avx_quant16(const __m128i r[4], const __m128i b[4], const __m256d f[4], __m256d lo,
                                  __m256d hi) { /* 16 int8 = quant(r+b) */
    return _mm_packs_epi16(_mm_packs_epi32(avx_quant(_mm_add_epi32(r[0], b[0]), f[0], lo, hi),
                                           avx_quant(_mm_add_epi32(r[1], b[1]), f[1], lo, hi)),
                           _mm_packs_epi32(avx_quant(_mm_add_epi32(r[2], b[2]), f[2], lo, hi),
                                           avx_quant(_mm_add_epi32(r[3], b[3]), f[3], lo, hi)));
}
typedef __m128i Lanes4;
typedef __m256d Scale4; /* tile driver interface */
#define T_LIMITS         const __m256d qlo = _mm256_set1_pd(o->lo), qhi = _mm256_set1_pd(o->hi)
#define T_BIAS(v)        _mm_set1_epi32(v)
#define T_SCALE(m, s)    _mm256_set1_pd(qscale(m, s))
#define T_QUANT(v, b, f) avx_quant(_mm_add_epi32(v, b), f, qlo, qhi)
#define T_STORE16(l, r)                                                                                                \
    _mm_storeu_si128((__m128i *)(l), _mm_packs_epi16(_mm_packs_epi32(r[0], r[1]), _mm_packs_epi32(r[2], r[3])))
#endif

#ifdef GUST_INT8_VNNI
#define VNNI_GROUP 32 /* pixels per patch group */
/* VNNI tiles: four pixels x four output channels via vpdpbusd (u8 x s8 into
   int32, no int16 stage, no saturation). Non-negative tensors (0..127) feed
   the u8 side directly. Signed inputs use x^0x80 = x+128 and subtract
   128*sum(w), itself summed with vpdpbusd(0x80..,w); the int32 lanes wrap
   modulo 2^32 and the true result fits int32 (gust_init bound), so both are
   exact. Channel tails use zero-masked loads (x and w) and never read past n.
   xp[p][k] is pixel p's tap k (NULL = padding); weights for channel j, tap k
   start at w+j*row+k*n. */
static inline __m512i vnni_sum4(__m512i a, __m512i b, __m512i c,
                                __m512i d) { /* 128-bit lane L: partial sums of a,b,c,d */
    __m512i t0 = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
    __m512i t1 = _mm512_add_epi32(_mm512_unpacklo_epi32(c, d), _mm512_unpackhi_epi32(c, d));
    return _mm512_add_epi32(_mm512_unpacklo_epi64(t0, t1), _mm512_unpackhi_epi64(t0, t1));
}
static inline __m512i vnni_join(__m512i u0, __m512i u1, __m512i u2,
                                __m512i u3) { /* lane 4*p+j = total of u_p's j partials */
    __m512i v01 = _mm512_add_epi32(_mm512_shuffle_i32x4(u0, u1, 0x44), _mm512_shuffle_i32x4(u0, u1, 0xEE));
    __m512i v23 = _mm512_add_epi32(_mm512_shuffle_i32x4(u2, u3, 0x44), _mm512_shuffle_i32x4(u2, u3, 0xEE));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(v01, v23, 0x88), _mm512_shuffle_i32x4(v01, v23, 0xDD));
}
#define VNNI_WEIGHTS                                                                                                   \
    __m512i q0 = LOAD(w0), q1 = LOAD(w1), q2 = LOAD(w2), q3 = LOAD(w3), v;                                             \
    if (sgn) {                                                                                                         \
        c0 = _mm512_dpbusd_epi32(c0, f, q0);                                                                           \
        c1 = _mm512_dpbusd_epi32(c1, f, q1);                                                                           \
        c2 = _mm512_dpbusd_epi32(c2, f, q2);                                                                           \
        c3 = _mm512_dpbusd_epi32(c3, f, q3);                                                                           \
    }
#define VNNI_DOT(A0, A1, A2, A3, X)                                                                                    \
    v = LOAD(X);                                                                                                       \
    if (sgn) v = _mm512_xor_si512(v, f);                                                                               \
    A0 = _mm512_dpbusd_epi32(A0, v, q0);                                                                               \
    A1 = _mm512_dpbusd_epi32(A1, v, q1);                                                                               \
    A2 = _mm512_dpbusd_epi32(A2, v, q2);                                                                               \
    A3 = _mm512_dpbusd_epi32(A3, v, q3);
#define VNNI_FULL(P) _mm512_loadu_si512((P) + i)
#define VNNI_TAIL(P) _mm512_maskz_loadu_epi8(t, (P) + i)
static inline __attribute__((always_inline)) __m512i vnni_tile(const int8_t *const (*xp)[64], uint32_t taps,
                                                               const int8_t *w, uint32_t row, uint32_t n, int sgn) {
    const __m512i f = _mm512_set1_epi8((char)0x80);
    __m512i a00 = _mm512_setzero_si512(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00, a13 = a00,
            a20 = a00, a21 = a00, a22 = a00, a23 = a00, a30 = a00, a31 = a00, a32 = a00, a33 = a00, c0 = a00, c1 = a00,
            c2 = a00, c3 = a00;
    for (uint32_t k = 0; k < taps; k++) {
        const int8_t *x0 = xp[0][k], *x1 = xp[1][k], *x2 = xp[2][k], *x3 = xp[3][k], *w0 = w + k * n, *w1 = w0 + row,
                     *w2 = w1 + row, *w3 = w2 + row;
        uint32_t i = 0;
#define LOAD VNNI_FULL
        for (; i + 64 <= n; i += 64) {
            VNNI_WEIGHTS VNNI_DOT(a00, a01, a02, a03, x0) VNNI_DOT(a10, a11, a12, a13, x1)
                VNNI_DOT(a20, a21, a22, a23, x2) VNNI_DOT(a30, a31, a32, a33, x3)
        }
#undef LOAD
#define LOAD VNNI_TAIL
        if (i < n) {
            const __mmask64 t = ~0ull >> (64 - (n - i));
            VNNI_WEIGHTS VNNI_DOT(a00, a01, a02, a03, x0) VNNI_DOT(a10, a11, a12, a13, x1)
                VNNI_DOT(a20, a21, a22, a23, x2) VNNI_DOT(a30, a31, a32, a33, x3)
        }
#undef LOAD
    }
    __m512i r = vnni_join(vnni_sum4(a00, a01, a02, a03), vnni_sum4(a10, a11, a12, a13), vnni_sum4(a20, a21, a22, a23),
                          vnni_sum4(a30, a31, a32, a33));
    if (sgn) {
        __m512i u = vnni_sum4(c0, c1, c2, c3);
        r = _mm512_sub_epi32(r, vnni_join(u, u, u, u));
    }
    return r;
}
/* Same lanes for tiles with padded taps or missing pixels (all-NULL rows give 0). */
static __attribute__((noinline)) __m512i vnni_edge(const int8_t *const (*xp)[64], uint32_t taps, const int8_t *w,
                                                   uint32_t row, uint32_t n, int sgn) {
    const __m512i f = _mm512_set1_epi8((char)0x80);
    __m512i u[4];
    for (int p = 0; p < 4; p++) {
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0;
        for (uint32_t k = 0; k < taps; k++)
            if (xp[p][k]) {
                const int8_t *x0 = xp[p][k], *w0 = w + k * n, *w1 = w0 + row, *w2 = w1 + row, *w3 = w2 + row;
                uint32_t i = 0;
#define LOAD VNNI_FULL
                for (; i + 64 <= n; i += 64) {
                    VNNI_WEIGHTS VNNI_DOT(a0, a1, a2, a3, x0)
                }
#undef LOAD
#define LOAD VNNI_TAIL
                if (i < n) {
                    const __mmask64 t = ~0ull >> (64 - (n - i));
                    VNNI_WEIGHTS VNNI_DOT(a0, a1, a2, a3, x0)
                }
#undef LOAD
            }
        u[p] = _mm512_sub_epi32(vnni_sum4(a0, a1, a2, a3), vnni_sum4(c0, c1, c2, c3));
    }
    return vnni_join(u[0], u[1], u[2], u[3]);
}
#undef VNNI_DOT
#undef VNNI_WEIGHTS
#undef VNNI_FULL
#undef VNNI_TAIL
/* Sixteen lanes of quant(): exact int64 products on even/odd lanes. Round half
   away from zero as one arithmetic shift: for v<0 and s>0,
   -((-v+2^(s-1))>>s) == (v+2^(s-1)-1)>>s; s=0 adds nothing. |v|<2^62. */
typedef struct {
    __m512i b, me, mo, se, so, re, ro, de, dd;
} Quant16; /* per-lane constants, even/odd lanes */
static inline Quant16 vnni_qprep(__m512i b, __m512i m, __m512i s) {
    const __m512i one = _mm512_set1_epi64(1);
    Quant16 q;
    q.b = b;
    q.me = m;
    q.mo = _mm512_srli_epi64(m, 32);
    q.se = _mm512_and_si512(s, _mm512_set1_epi64(0xFFFFFFFF));
    q.so = _mm512_srli_epi64(s, 32);
    __m512i pe = _mm512_sllv_epi64(one, q.se), po = _mm512_sllv_epi64(one, q.so);
    q.re = _mm512_srli_epi64(pe, 1);
    q.ro = _mm512_srli_epi64(po, 1);
    q.de = _mm512_sub_epi64(_mm512_srli_epi64(_mm512_sub_epi64(pe, one), 1), q.re);
    q.dd = _mm512_sub_epi64(_mm512_srli_epi64(_mm512_sub_epi64(po, one), 1), q.ro); /* -1 if s>0 else 0 */
    return q;
}
/* With lo>=0 a negative product ends at lo under either rounding (both give
   <=0), so the away-from-zero correction is skipped. */
static inline __attribute__((always_inline)) __m512i vnni_q64(__m512i v, __m512i m, __m512i s, __m512i r, __m512i d,
                                                              __m512i lo, __m512i hi, int relu) {
    __m512i a = _mm512_mul_epi32(v, m);
    a = _mm512_srav_epi64(
        _mm512_add_epi64(a, relu ? r : _mm512_add_epi64(r, _mm512_and_si512(_mm512_srai_epi64(a, 63), d))), s);
    return _mm512_min_epi64(_mm512_max_epi64(a, lo), hi);
}
static inline __attribute__((always_inline)) __m128i vnni_qapply_(__m512i v, const Quant16 *q, int32_t lo, int32_t hi,
                                                                  int relu) {
    const __m512i l = _mm512_set1_epi64(lo), h = _mm512_set1_epi64(hi);
    v = _mm512_add_epi32(v, q->b);
    __m512i e = vnni_q64(v, q->me, q->se, q->re, q->de, l, h, relu),
            o = vnni_q64(_mm512_srli_epi64(v, 32), q->mo, q->so, q->ro, q->dd, l, h, relu);
    return _mm512_cvtepi32_epi8(_mm512_mask_blend_epi32(0xAAAA, e, _mm512_slli_epi64(o, 32)));
}
static inline __m128i vnni_qapply(__m512i v, const Quant16 *q, int32_t lo, int32_t hi) { /* quant(v+bias) */
    return lo >= 0 ? vnni_qapply_(v, q, lo, hi, 1) : vnni_qapply_(v, q, lo, hi, 0);
}
/* quant(v) for lanes holding channels 0,8,1,9,..,7,15 (gemm_pack perm, constants permuted alike): the
   even/odd int64 halves are channels 0..7 / 8..15 and vpmovsqb narrows them with signed saturation, which is
   the clamp to the int8 range; a narrower [lo,hi] is applied to the bytes. relu as in vnni_qapply_. */
static inline __attribute__((always_inline)) __m128i vnni_qperm(__m512i v, const Quant16 *q, int32_t lo, int32_t hi,
                                                                int relu) { /* bias already in v */
    __m512i e = _mm512_mul_epi32(v, q->me), o = _mm512_mul_epi32(_mm512_srli_epi64(v, 32), q->mo);
    if (relu) {
        e = _mm512_add_epi64(e, q->re);
        o = _mm512_add_epi64(o, q->ro);
    } else {
        e = _mm512_add_epi64(e, _mm512_add_epi64(q->re, _mm512_and_si512(_mm512_srai_epi64(e, 63), q->de)));
        o = _mm512_add_epi64(o, _mm512_add_epi64(q->ro, _mm512_and_si512(_mm512_srai_epi64(o, 63), q->dd)));
    }
    const __m128i r = _mm_unpacklo_epi64(_mm512_cvtsepi64_epi8(_mm512_srav_epi64(e, q->se)),
                                         _mm512_cvtsepi64_epi8(_mm512_srav_epi64(o, q->so)));
    return lo > -128 || hi < 127 ? _mm_min_epi8(_mm_max_epi8(r, _mm_set1_epi8((char)lo)), _mm_set1_epi8((char)hi)) : r;
}
/* vnni_qperm for lanes in natural channel order: the narrowed even/odd halves are interleaved bytewise. */
static inline __attribute__((always_inline)) __m128i vnni_qsat(__m512i v, const Quant16 *q, int32_t lo, int32_t hi,
                                                               int relu) {
    v = _mm512_add_epi32(v, q->b);
    __m512i e = _mm512_mul_epi32(v, q->me), o = _mm512_mul_epi32(_mm512_srli_epi64(v, 32), q->mo);
    if (relu) {
        e = _mm512_add_epi64(e, q->re);
        o = _mm512_add_epi64(o, q->ro);
    } else {
        e = _mm512_add_epi64(e, _mm512_add_epi64(q->re, _mm512_and_si512(_mm512_srai_epi64(e, 63), q->de)));
        o = _mm512_add_epi64(o, _mm512_add_epi64(q->ro, _mm512_and_si512(_mm512_srai_epi64(o, 63), q->dd)));
    }
    const __m128i r = _mm_unpacklo_epi8(_mm512_cvtsepi64_epi8(_mm512_srav_epi64(e, q->se)),
                                        _mm512_cvtsepi64_epi8(_mm512_srav_epi64(o, q->so)));
    return lo > -128 || hi < 127 ? _mm_min_epi8(_mm_max_epi8(r, _mm_set1_epi8((char)lo)), _mm_set1_epi8((char)hi)) : r;
}
static inline __m128i vnni_quant(__m512i v, __m512i m, __m512i s, int32_t lo, int32_t hi) {
    Quant16 q = vnni_qprep(_mm512_setzero_si512(), m, s);
    return vnni_qapply(v, &q, lo, hi);
}
#endif
/* y[i]=clamp(x[i](+b[i])). Saturating to int8 first never changes a clamp to
   [lo,hi] inside int8, so the vector form is exact. No aliasing: gust_init
   rejects overlapping live tensors. */
#ifdef GUST_INT8_VNNI
/* rows of n bytes from stride sx to stride sy (SLICE / CONCAT whose clamp is the whole int8 range) */
static void copy_rows(int8_t *y, uint32_t sy, const int8_t *x, uint32_t sx, uint32_t n, uint32_t rows) {
    const __mmask64 t = ~0ull >> ((64 - (n & 63)) & 63);
    for (uint32_t r = 0; r < rows; r++, y += sy, x += sx) {
        uint32_t i = 0;
        for (; i + 64 <= n; i += 64) _mm512_storeu_si512(y + i, _mm512_loadu_si512(x + i));
        if (i < n) _mm512_mask_storeu_epi8(y + i, t, _mm512_maskz_loadu_epi8(t, x + i));
    }
}
#endif
#if defined(__aarch64__)
/* Requantization of four lanes, bit-identical to quant(): v * m in int64, |.| + 2^(s-1) shifted right by s, the
   sign restored, then clamped to [lo, hi]. |v * m| < 2^62, so abs never overflows; the saturating narrow to int32
   preserves every clamped result because lo and hi are int32. Returns the four int8 results as bytes 0..3. */
typedef struct {
    int32x4_t m, lo, hi;
    int64x2_t nsh0, nsh1, rnd0, rnd1; /* -shift and 2^(shift-1) (0 for shift 0) of lanes 0-1 and 2-3 */
} NgQuant;
static inline NgQuant ng_qprep(const int32_t *m, const uint32_t *s, uint32_t ms, const Op *o) {
    NgQuant q;
    int32_t mv[4];
    int64_t sv[4];
    for (int i = 0; i < 4; i++) {
        mv[i] = m[i * ms];
        sv[i] = (int64_t)s[i * ms];
    }
    q.m = vld1q_s32(mv);
    const int64x2_t s0 = vld1q_s64(sv), s1 = vld1q_s64(sv + 2), one = vdupq_n_s64(1);
    q.nsh0 = vnegq_s64(s0);
    q.nsh1 = vnegq_s64(s1);
    q.rnd0 = vshlq_s64(one, vsubq_s64(s0, one)); /* 1 << (s - 1); shift by -1 gives 0 for s = 0 */
    q.rnd1 = vshlq_s64(one, vsubq_s64(s1, one));
    q.lo = vdupq_n_s32(o->lo);
    q.hi = vdupq_n_s32(o->hi);
    return q;
}
static inline uint32_t ng_quant4(int32x4_t v, const NgQuant *q) {
    const int64x2_t p0 = vmull_s32(vget_low_s32(v), vget_low_s32(q->m)), p1 = vmull_high_s32(v, q->m);
    const uint64x2_t n0 = vcltzq_s64(p0), n1 = vcltzq_s64(p1);
    const int64x2_t a0 = vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(vaddq_s64(vabsq_s64(p0), q->rnd0)), q->nsh0)),
                    a1 = vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(vaddq_s64(vabsq_s64(p1), q->rnd1)), q->nsh1));
    const int64x2_t r0 = vbslq_s64(n0, vnegq_s64(a0), a0), r1 = vbslq_s64(n1, vnegq_s64(a1), a1);
    int32x4_t r = vcombine_s32(vqmovn_s64(r0), vqmovn_s64(r1));
    r = vminq_s32(vmaxq_s32(r, q->lo), q->hi);
    const int8x8_t b = vmovn_s16(vcombine_s16(vmovn_s32(r), vdup_n_s16(0)));
    return vget_lane_u32(vreinterpret_u32_s8(b), 0);
}

#endif
static void clampcopy(int8_t *y, const int8_t *x, const int8_t *b, uint32_t n, const Op *o) {
    uint32_t i = 0;
#ifdef GUST_INT8_VNNI
    const __m512i lo = _mm512_set1_epi8((char)o->lo), hi = _mm512_set1_epi8((char)o->hi);
    const __mmask64 t = ~0ull >> ((64 - (n & 63)) & 63);
    if (b) {
        for (; i + 64 <= n; i += 64)
            _mm512_storeu_si512(
                y + i,
                _mm512_min_epi8(
                    _mm512_max_epi8(_mm512_adds_epi8(_mm512_loadu_si512(x + i), _mm512_loadu_si512(b + i)), lo), hi));
        if (i < n)
            _mm512_mask_storeu_epi8(y + i, t,
                                    _mm512_min_epi8(_mm512_max_epi8(_mm512_adds_epi8(_mm512_maskz_loadu_epi8(t, x + i),
                                                                                     _mm512_maskz_loadu_epi8(t, b + i)),
                                                                    lo),
                                                    hi));
    } else {
        for (; i + 64 <= n; i += 64)
            _mm512_storeu_si512(y + i, _mm512_min_epi8(_mm512_max_epi8(_mm512_loadu_si512(x + i), lo), hi));
        if (i < n)
            _mm512_mask_storeu_epi8(y + i, t,
                                    _mm512_min_epi8(_mm512_max_epi8(_mm512_maskz_loadu_epi8(t, x + i), lo), hi));
    }
    return;
#elif defined(__AVX2__)
    const __m256i lo = _mm256_set1_epi8((char)o->lo), hi = _mm256_set1_epi8((char)o->hi);
    if (b)
        for (; i + 32 <= n; i += 32)
            _mm256_storeu_si256(
                (__m256i *)(y + i),
                _mm256_min_epi8(_mm256_max_epi8(_mm256_adds_epi8(_mm256_loadu_si256((const __m256i *)(x + i)),
                                                                 _mm256_loadu_si256((const __m256i *)(b + i))),
                                                lo),
                                hi));
    else
        for (; i + 32 <= n; i += 32)
            _mm256_storeu_si256((__m256i *)(y + i),
                                _mm256_min_epi8(_mm256_max_epi8(_mm256_loadu_si256((const __m256i *)(x + i)), lo), hi));
#elif defined(__aarch64__)
    const int8x16_t lo = vdupq_n_s8((int8_t)o->lo), hi = vdupq_n_s8((int8_t)o->hi);
    if (b)
        for (; i + 16 <= n; i += 16)
            vst1q_s8(y + i, vminq_s8(vmaxq_s8(vqaddq_s8(vld1q_s8(x + i), vld1q_s8(b + i)), lo), hi));
    else if (o->lo == -128 && o->hi == 127)
        for (; i + 16 <= n; i += 16) vst1q_s8(y + i, vld1q_s8(x + i));
    else
        for (; i + 16 <= n; i += 16) vst1q_s8(y + i, vminq_s8(vmaxq_s8(vld1q_s8(x + i), lo), hi));
#elif defined(__wasm_simd128__)
    const v128_t lo = wasm_i8x16_splat((int8_t)o->lo), hi = wasm_i8x16_splat((int8_t)o->hi);
    if (b)
        for (; i + 16 <= n; i += 16)
            wasm_v128_store(
                y + i, wasm_i8x16_min(
                           wasm_i8x16_max(wasm_i8x16_add_sat(wasm_v128_load(x + i), wasm_v128_load(b + i)), lo), hi));
    else
        for (; i + 16 <= n; i += 16)
            wasm_v128_store(y + i, wasm_i8x16_min(wasm_i8x16_max(wasm_v128_load(x + i), lo), hi));
#endif
    if (b)
        for (; i < n; i++) y[i] = clamp((int32_t)x[i] + b[i], o);
    else
        for (; i < n; i++) y[i] = clamp(x[i], o);
}

/* SSE2-only builds (no SSSE3 / AVX2 tiles): 1x1 convolutions in 4 pixel x 4 channel tiles. Each 16-channel chunk
   is sign-extended once per input row and once per weight row and reused across the tile (16 dots, 8 widenings).
   x holds four input rows and w four weight rows, both at stride n; out[r * 4 + c] receives row r, channel c. */
#if defined(__SSE2__) && !defined(__SSSE3__) && !defined(__AVX2__)
#define INT8_SSE2_TILE4 1
static inline void dot_tile4(const int8_t *x, const int8_t *w, uint32_t n, int32_t *out) {
    __m128i zero = _mm_setzero_si128(), acc[4][4];
    for (uint32_t r = 0; r < 4; r++)
        for (uint32_t c = 0; c < 4; c++) acc[r][c] = zero;
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i xl[4], xh[4];
        for (uint32_t r = 0; r < 4; r++) {
            __m128i v = _mm_loadu_si128((const __m128i *)(x + (size_t)r * n + i)), sign = _mm_cmpgt_epi8(zero, v);
            xl[r] = _mm_unpacklo_epi8(v, sign);
            xh[r] = _mm_unpackhi_epi8(v, sign);
        }
        for (uint32_t c = 0; c < 4; c++) {
            __m128i v = _mm_loadu_si128((const __m128i *)(w + (size_t)c * n + i)), sign = _mm_cmpgt_epi8(zero, v);
            __m128i wl = _mm_unpacklo_epi8(v, sign), wh = _mm_unpackhi_epi8(v, sign);
            for (uint32_t r = 0; r < 4; r++)
                acc[r][c] =
                    _mm_add_epi32(acc[r][c], _mm_add_epi32(_mm_madd_epi16(xl[r], wl), _mm_madd_epi16(xh[r], wh)));
        }
    }
    for (uint32_t r = 0; r < 4; r++) {
        __m128i t0 = _mm_add_epi32(_mm_unpacklo_epi32(acc[r][0], acc[r][1]), _mm_unpackhi_epi32(acc[r][0], acc[r][1]));
        __m128i t1 = _mm_add_epi32(_mm_unpacklo_epi32(acc[r][2], acc[r][3]), _mm_unpackhi_epi32(acc[r][2], acc[r][3]));
        _mm_storeu_si128((__m128i *)(out + r * 4),
                         _mm_add_epi32(_mm_unpacklo_epi64(t0, t1), _mm_unpackhi_epi64(t0, t1)));
    }
    for (; i < n; i++)
        for (uint32_t r = 0; r < 4; r++)
            for (uint32_t c = 0; c < 4; c++) out[r * 4 + c] += (int32_t)x[(size_t)r * n + i] * w[(size_t)c * n + i];
}

/* Four independent, exact signed requantizations (value + bias). Uniform
   shifts are common in per-channel PTQ; other tables use the scalar oracle. */
static inline void quant4(const int32_t *value, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o,
                          int8_t *out) {
    if (s[0] == s[1] && s[0] == s[2] && s[0] == s[3]) {
        uint32_t shift = s[0];
        int64_t rounding = shift ? ((int64_t)1 << (shift - 1)) : 0;
        __m128i v = _mm_loadu_si128((const __m128i *)value);
        if (bias) v = _mm_add_epi32(v, _mm_loadu_si128((const __m128i *)bias));
        __m128i sign = _mm_srai_epi32(v, 31), absolute = _mm_sub_epi32(_mm_xor_si128(v, sign), sign),
                mult = _mm_loadu_si128((const __m128i *)m);
        __m128i round = _mm_set1_epi64x(rounding), count = _mm_cvtsi32_si128((int)shift), limit = _mm_set1_epi64x(128),
                zero = _mm_setzero_si128();
        __m128i even = _mm_srl_epi64(_mm_add_epi64(_mm_mul_epu32(absolute, mult), round), count);
        __m128i odd = _mm_srl_epi64(
            _mm_add_epi64(_mm_mul_epu32(_mm_srli_si128(absolute, 4), _mm_srli_si128(mult, 4)), round), count);
        /* A magnitude >=128 is clipped before narrowing, including values
           with nonzero high 32 bits; no overflow or unsigned wraparound. */
        __m128i e = _mm_cmpeq_epi32(_mm_srli_epi64(even, 7), zero), d = _mm_cmpeq_epi32(_mm_srli_epi64(odd, 7), zero);
        e = _mm_and_si128(e, _mm_shuffle_epi32(e, _MM_SHUFFLE(2, 3, 0, 1)));
        d = _mm_and_si128(d, _mm_shuffle_epi32(d, _MM_SHUFFLE(2, 3, 0, 1)));
        even = _mm_or_si128(_mm_and_si128(e, even), _mm_andnot_si128(e, limit));
        odd = _mm_or_si128(_mm_and_si128(d, odd), _mm_andnot_si128(d, limit));
        __m128i magnitude = _mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(2, 0, 2, 0)),
                                               _mm_shuffle_epi32(odd, _MM_SHUFFLE(2, 0, 2, 0)));
        v = _mm_sub_epi32(_mm_xor_si128(magnitude, sign), sign);
        __m128i low = _mm_set1_epi32(o->lo), high = _mm_set1_epi32(o->hi), above = _mm_cmpgt_epi32(v, high);
        v = _mm_or_si128(_mm_and_si128(above, high), _mm_andnot_si128(above, v));
        __m128i below = _mm_cmpgt_epi32(low, v);
        v = _mm_or_si128(_mm_and_si128(below, low), _mm_andnot_si128(below, v));
        v = _mm_packs_epi32(v, zero);
        v = _mm_packs_epi16(v, zero);
        uint32_t packed = (uint32_t)_mm_cvtsi128_si32(v);
        out[0] = (int8_t)packed;
        out[1] = (int8_t)(packed >> 8);
        out[2] = (int8_t)(packed >> 16);
        out[3] = (int8_t)(packed >> 24);
        return;
    }
    for (uint32_t j = 0; j < 4; j++) out[j] = quant(value[j] + (bias ? bias[j] : 0), m[j], s[j], o);
}

#endif
