/* Packed 3/4-bit convolution weights (capability bit 8, builds with GUST_PACKED_WEIGHTS). A convolution whose
   parameter1 is 3 or 4 stores its logical INT8 weights as signed two's-complement codes of that width, laid out
   consecutively and little-endian by bit (code i occupies bits [i*bits, (i+1)*bits) of the payload, low bits first)
   with no per-row padding. The accumulator, requantization and clamp are the byte-weight ones; only the weight
   fetch differs.

   Nothing is ever expanded wholesale: the operator (ops/conv.c op_conv_packed) decodes one channel block at a time
   inside the GEMM weight packing, or blocks of output channels (at most PACKED_BLOCK bytes) and a whole depthwise
   filter (at most PACKED_SCRATCH) into stack scratch for the byte kernels, so packed and byte graphs take the same
   fast paths. No heap, no float.
   Part of the runtime/gust.c translation unit. */
#ifdef GUST_PACKED_WEIGHTS

/* PACKED_DENSE_TERMS, PACKED_DEPTHWISE_WEIGHTS and PACKED_SCRATCH: see abi.h. */

/* Signed code `index` of a packed payload: low bits first, sign-extended from `bits` (3 or 4) to INT8. A code never
   spans more than two bytes, and the second byte is read only when the code reaches into it. */
static inline int8_t packed_code(const uint8_t *src, uint64_t index, uint32_t bits) {
    const uint64_t bit = index * bits;
    const uint8_t *p = src + (bit >> 3);
    const uint32_t shift = (uint32_t)(bit & 7);
    uint32_t v = (uint32_t)p[0] >> shift;
    if (shift + bits > 8) v |= (uint32_t)p[1] << (8 - shift);
    v &= (1u << bits) - 1u;
    const int32_t sign = (int32_t)(1u << (bits - 1));
    return (int8_t)((int32_t)(v ^ (uint32_t)sign) - sign);
}

/* Codes first .. first+count-1 into dst. Groups of eight codes start on a byte boundary (3 or 4 bytes), so the
   bulk loop touches whole bytes only; the unaligned head and the tail go through packed_code. */
static void packed_decode(const uint8_t *src, uint64_t first, uint32_t count, uint32_t bits, int8_t *dst) {
    uint32_t i = 0;
    for (; i < count && ((first + i) & 7); i++) dst[i] = packed_code(src, first + i, bits);
    if (bits == 4) {
        for (; i + 8 <= count; i += 8) {
            const uint8_t *p = src + (((first + i) * 4) >> 3);
            for (uint32_t j = 0; j < 4; j++) {
                dst[i + 2 * j] = (int8_t)((int32_t)((p[j] & 15u) ^ 8u) - 8);
                dst[i + 2 * j + 1] = (int8_t)((int32_t)((p[j] >> 4) ^ 8u) - 8);
            }
        }
    } else {
        for (; i + 8 <= count; i += 8) {
            const uint8_t *p = src + (((first + i) * 3) >> 3);
            const uint32_t group = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16;
            for (uint32_t j = 0; j < 8; j++) dst[i + j] = (int8_t)((int32_t)(((group >> (3 * j)) & 7u) ^ 4u) - 4);
        }
    }
    for (; i < count; i++) dst[i] = packed_code(src, first + i, bits);
}

/* Codes first .. first+count-1 into dst, vectorized for 4-bit codes (16 payload bytes -> 32 codes per step). */
static void packed_expand(const uint8_t *src, uint64_t first, uint32_t count, uint32_t bits, int8_t *dst) {
    uint32_t i = 0;
    if (bits == 4) {
        for (; i < count && ((first + i) & 1); i++) dst[i] = packed_code(src, first + i, 4);
#if defined(__wasm_simd128__) || defined(__SSE2__) || defined(__aarch64__)
        const uint8_t *p = src + ((first + i) >> 1);
#endif
#if defined(__aarch64__)
        const int8x16_t eight = vdupq_n_s8(8);
        for (; i + 32 <= count; i += 32, p += 16) { /* low nibble = even code, high nibble = odd code */
            const uint8x16_t v = vld1q_u8(p), l = vandq_u8(v, vdupq_n_u8(15)), h = vshrq_n_u8(v, 4);
            const int8x16_t a = vreinterpretq_s8_u8(vzip1q_u8(l, h)), b = vreinterpretq_s8_u8(vzip2q_u8(l, h));
            vst1q_s8(dst + i, vsubq_s8(veorq_s8(a, eight), eight));
            vst1q_s8(dst + i + 16, vsubq_s8(veorq_s8(b, eight), eight));
        }
#endif
#if defined(__wasm_simd128__)
        const v128_t low = wasm_i8x16_splat(15), eight = wasm_i8x16_splat(8);
        for (; i + 32 <= count; i += 32, p += 16) {
            const v128_t v = wasm_v128_load(p), l = wasm_v128_and(v, low), h = wasm_u8x16_shr(v, 4);
            const v128_t a = wasm_i8x16_shuffle(l, h, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
                         b = wasm_i8x16_shuffle(l, h, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            wasm_v128_store(dst + i, wasm_i8x16_sub(wasm_v128_xor(a, eight), eight));
            wasm_v128_store(dst + i + 16, wasm_i8x16_sub(wasm_v128_xor(b, eight), eight));
        }
#elif defined(__AVX2__)
        const __m256i low = _mm256_set1_epi8(15), eight = _mm256_set1_epi8(8);
        for (; i + 64 <= count; i += 64, p += 32) { /* in-lane unpacks: halves (0..15 | 16..31) of each half */
            const __m256i v = _mm256_loadu_si256((const __m256i *)p), l = _mm256_and_si256(v, low),
                          h = _mm256_and_si256(_mm256_srli_epi16(v, 4), low);
            const __m256i a = _mm256_sub_epi8(_mm256_xor_si256(_mm256_unpacklo_epi8(l, h), eight), eight),
                          b = _mm256_sub_epi8(_mm256_xor_si256(_mm256_unpackhi_epi8(l, h), eight), eight);
            _mm256_storeu_si256((__m256i *)(dst + i), _mm256_permute2x128_si256(a, b, 0x20));
            _mm256_storeu_si256((__m256i *)(dst + i + 32), _mm256_permute2x128_si256(a, b, 0x31));
        }
#endif
#if defined(__SSE2__)
        const __m128i low16 = _mm_set1_epi8(15), eight16 = _mm_set1_epi8(8);
        for (; i + 32 <= count; i += 32, p += 16) {
            const __m128i v = _mm_loadu_si128((const __m128i *)p), l = _mm_and_si128(v, low16),
                          h = _mm_and_si128(_mm_srli_epi16(v, 4), low16);
            _mm_storeu_si128((__m128i *)(dst + i),
                             _mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(l, h), eight16), eight16));
            _mm_storeu_si128((__m128i *)(dst + i + 16),
                             _mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(l, h), eight16), eight16));
        }
#endif
    }
    if (i < count) packed_decode(src, first + i, count - i, bits, dst + i);
}

#endif /* GUST_PACKED_WEIGHTS */
