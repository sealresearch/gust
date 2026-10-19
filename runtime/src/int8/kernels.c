/* Scalar INT8 arithmetic shared by every operator: dot products, sparse index scan, clamp and requantization (the
   definitions every vector kernel in tiles.c and ops/ reproduces). Part of the runtime/gust.c translation unit. */

/* Exact signed INT8 dot products, including arbitrary channel tails. No
   packing or scratch allocation. Scalar remains the portability oracle. */
static inline int32_t dot(const int8_t *a, const int8_t *b, uint32_t n) {
    if (n == 3) return (int32_t)a[0] * b[0] + (int32_t)a[1] * b[1] + (int32_t)a[2] * b[2];
    uint32_t i = 0;
    int32_t result = 0;
#if defined(__wasm_simd128__)
    v128_t sum = wasm_i32x4_splat(0);
    for (; i + 16 <= n; i += 16) {
        v128_t x = wasm_v128_load(a + i), y = wasm_v128_load(b + i);
        sum = wasm_i32x4_add(sum, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(x), wasm_i16x8_extend_low_i8x16(y)));
        sum =
            wasm_i32x4_add(sum, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(x), wasm_i16x8_extend_high_i8x16(y)));
    }
    result = wasm_i32x4_extract_lane(sum, 0) + wasm_i32x4_extract_lane(sum, 1) + wasm_i32x4_extract_lane(sum, 2) +
             wasm_i32x4_extract_lane(sum, 3);
#elif defined(__SSE2__)
    __m128i sum = _mm_setzero_si128(), zero = _mm_setzero_si128();
    for (; i + 16 <= n; i += 16) {
        __m128i x = _mm_loadu_si128((const __m128i *)(a + i)), y = _mm_loadu_si128((const __m128i *)(b + i));
        __m128i xs = _mm_cmpgt_epi8(zero, x), ys = _mm_cmpgt_epi8(zero, y);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpacklo_epi8(x, xs), _mm_unpacklo_epi8(y, ys)));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_unpackhi_epi8(x, xs), _mm_unpackhi_epi8(y, ys)));
    }
    int32_t lanes[4];
    _mm_storeu_si128((__m128i *)lanes, sum);
    result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; i < n; i++) result += (int32_t)a[i] * b[i];
    return result;
}

/* Bounded sparse full-field classifiers. Skip zero blocks without changing
   signed products, bias, quantization, or the dense-input fallback. */
static int sparse_indices(const int8_t *x, uint32_t n, uint16_t *indices, uint32_t *used) {
    uint32_t i = 0, count = 0;
    for (; i + 16 <= n; i += 16) {
#if defined(__wasm_simd128__)
        if (!wasm_v128_any_true(wasm_v128_load(x + i))) continue;
#elif defined(__SSE2__)
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(x + i)), _mm_setzero_si128())) == 65535)
            continue;
#endif
        for (uint32_t j = i; j < i + 16; j++)
            if (x[j]) {
                if (count == 512) return 0;
                indices[count++] = (uint16_t)j;
            }
    }
    for (; i < n; i++)
        if (x[i]) {
            if (count == 512) return 0;
            indices[count++] = (uint16_t)i;
        }
    *used = count;
    return 1;
}

/* Saturate to the operator's activation range [lo, hi]. */
static int8_t clamp(int64_t v, const Op *o) {
    if (v < o->lo) v = o->lo;
    if (v > o->hi) v = o->hi;
    return (int8_t)v;
}
/* Fixed-point requantization: value * multiplier >> shift, round half away from zero. */
static int8_t quant(int32_t value, int32_t multiplier, uint32_t shift, const Op *o) {
    int64_t v = (int64_t)value * multiplier;
    if (shift) {
        int64_t n = v < 0 ? -v : v;
        n = (n + ((int64_t)1 << (shift - 1))) >> shift;
        v = v < 0 ? -n : n;
    }
    return clamp(v, o);
}
