/* Depthwise convolution kernels: VNNI (64 channels x 4 taps per vpdpbusd, row-interleaved [H,1,C] filters, pixel
   groups), AVX2 (row interleave, pixel groups) and WebAssembly SIMD pixel groups, and depthwise_pixel for any
   shape. Inputs proven non-negative take the u8 x s8 forms; signed inputs are exact through int16 pairs or the
   x ^ 0x80 correction. Part of the runtime/gust.c translation unit. */

#ifdef GUST_INT8_VNNI
static inline __attribute__((always_inline)) void
depthwise_vnni(const int8_t *const *patch, uint32_t taps, const int8_t *weights, uint32_t channels, const int32_t *bias,
               const int32_t *m, const uint32_t *s, const Op *o, int8_t *out, int nonneg) {
    /* 64 channels x 4 taps per step: byte-interleave taps k..k+3 so each dword
       holds one channel's four taps, and vpdpbusd sums them exactly into int32.
       Signed inputs: x^0x80 minus 128*sum(w), as in vnni_tile. Missing taps
       (padding or past the end) are zero in both x and w. */
    const __m512i f = _mm512_set1_epi8((char)0x80), zero = _mm512_setzero_si512();
    for (uint32_t ch = 0; ch < channels; ch += 64) {
        const __mmask64 t = channels - ch >= 64 ? ~0ull : ~0ull >> (64 - (channels - ch));
        __m512i r0 = zero, r1 = zero, r2 = zero, r3 = zero, c0 = zero, c1 = zero, c2 = zero, c3 = zero;
        for (uint32_t k = 0; k < taps; k += 4) {
            __m512i x[4], q[4];
            for (uint32_t j = 0; j < 4; j++) { /* branch-free: a missing tap loads nothing (mask 0 never faults) */
                const int8_t *p = k + j < taps ? patch[k + j] : NULL;
                const __mmask64 u = p ? t : 0;
                x[j] = _mm512_maskz_loadu_epi8(u, p ? p + ch : weights);
                q[j] = _mm512_maskz_loadu_epi8(u, weights + (p ? (k + j) * channels + ch : 0));
                if (!nonneg) x[j] = _mm512_xor_si512(x[j], f); /* masked-out bytes meet w=0 */
            }
            __m512i xl = _mm512_unpacklo_epi8(x[0], x[1]), xh = _mm512_unpackhi_epi8(x[0], x[1]),
                    yl = _mm512_unpacklo_epi8(x[2], x[3]), yh = _mm512_unpackhi_epi8(x[2], x[3]);
            __m512i wl = _mm512_unpacklo_epi8(q[0], q[1]), wh = _mm512_unpackhi_epi8(q[0], q[1]),
                    vl = _mm512_unpacklo_epi8(q[2], q[3]), vh = _mm512_unpackhi_epi8(q[2], q[3]);
            __m512i w0 = _mm512_unpacklo_epi16(wl, vl), w1 = _mm512_unpackhi_epi16(wl, vl),
                    w2 = _mm512_unpacklo_epi16(wh, vh), w3 = _mm512_unpackhi_epi16(wh, vh);
            r0 = _mm512_dpbusd_epi32(r0, _mm512_unpacklo_epi16(xl, yl), w0);
            r1 = _mm512_dpbusd_epi32(r1, _mm512_unpackhi_epi16(xl, yl),
                                     w1); /* 128-bit lane L: channels 16L+0..3, +4..7 */
            r2 = _mm512_dpbusd_epi32(r2, _mm512_unpacklo_epi16(xh, yh), w2);
            r3 = _mm512_dpbusd_epi32(r3, _mm512_unpackhi_epi16(xh, yh), w3); /* +8..11, +12..15 */
            if (!nonneg) {
                c0 = _mm512_dpbusd_epi32(c0, f, w0);
                c1 = _mm512_dpbusd_epi32(c1, f, w1);
                c2 = _mm512_dpbusd_epi32(c2, f, w2);
                c3 = _mm512_dpbusd_epi32(c3, f, w3);
            }
        }
        if (!nonneg) {
            r0 = _mm512_sub_epi32(r0, c0);
            r1 = _mm512_sub_epi32(r1, c1);
            r2 = _mm512_sub_epi32(r2, c2);
            r3 = _mm512_sub_epi32(r3, c3);
        }
        __m512i t0 = _mm512_shuffle_i32x4(r0, r1, 0x44), t1 = _mm512_shuffle_i32x4(r2, r3, 0x44),
                t2 = _mm512_shuffle_i32x4(r0, r1, 0xEE), t3 = _mm512_shuffle_i32x4(r2, r3, 0xEE);
        __m512i v[4] = {_mm512_shuffle_i32x4(t0, t1, 0x88), _mm512_shuffle_i32x4(t0, t1, 0xDD),
                        _mm512_shuffle_i32x4(t2, t3, 0x88), _mm512_shuffle_i32x4(t2, t3, 0xDD)};
        for (uint32_t j = 0; j < 4 && ch + 16 * j < channels; j++) {
            const uint32_t b = ch + 16 * j;
            const __mmask16 u = channels - b >= 16 ? 0xFFFF : (__mmask16)((1u << (channels - b)) - 1);
            _mm_mask_storeu_epi8(out + b, u,
                                 vnni_quant(_mm512_add_epi32(v[j], _mm512_maskz_loadu_epi32(u, bias + b)),
                                            _mm512_maskz_loadu_epi32(u, m + b), _mm512_maskz_loadu_epi32(u, s + b),
                                            o->lo, o->hi));
        }
    }
}
#ifdef GUST_INT8_VNNI
/* DEPTHWISE along the rows of a [H,1,C] tensor (kw=1, stride 1, dilation 1): the four-row interleave
   I(r)[c] = x[r..r+3][c] (padding rows x=0; signed inputs as x^0x80 with 128*sum(w) subtracted, as in
   vnni_tile) is built once per row and 16-channel block and shared by every output row t:
   y[t] = quant(sum_g vpdpbusd(I(t-py+4g), W[g]) + bias), W[g] = taps 4g..4g+3 interleaved (zero past kh). */
#define DWR_BUF 49152
static int depthwise_rows(const int8_t *x, uint32_t H, uint32_t T, uint32_t py, uint32_t taps, const int8_t *w,
                          uint32_t C, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o, int8_t *y,
                          int sgn) {
    const uint32_t G = (taps + 3) / 4, RB = DWR_BUF / 64;
    if (G > 16 || RB < 4 * G + 8) return 0;
    const uint32_t TB = RB - 4 * G + 4; /* output rows per chunk */
    __attribute__((aligned(64))) uint8_t I[DWR_BUF];
    __m512i W[16];
    const __m512i f = _mm512_set1_epi8((char)0x80), zero = _mm512_setzero_si512();
#define DWR_IL(R0, R1, R2, R3)                                                                                         \
    _mm512_inserti64x4(                                                                                                \
        _mm512_castsi256_si512(_mm256_inserti128_si256(                                                                \
            _mm256_castsi128_si256(_mm_unpacklo_epi16(_mm_unpacklo_epi8(R0, R1), _mm_unpacklo_epi8(R2, R3))),          \
            _mm_unpackhi_epi16(_mm_unpacklo_epi8(R0, R1), _mm_unpacklo_epi8(R2, R3)), 1)),                             \
        _mm256_inserti128_si256(                                                                                       \
            _mm256_castsi128_si256(_mm_unpacklo_epi16(_mm_unpackhi_epi8(R0, R1), _mm_unpackhi_epi8(R2, R3))),          \
            _mm_unpackhi_epi16(_mm_unpackhi_epi8(R0, R1), _mm_unpackhi_epi8(R2, R3)), 1),                              \
        1)
    for (uint32_t ch = 0; ch < C; ch += 16) {
        const uint32_t cn = C - ch < 16 ? C - ch : 16;
        const __mmask16 u = (__mmask16)(cn >= 16 ? 0xFFFF : (1u << cn) - 1);
        __m512i corr = zero;
        for (uint32_t g = 0; g < G; g++) {
            __m128i q[4];
            for (uint32_t j = 0; j < 4; j++)
                q[j] =
                    4 * g + j < taps ? _mm_maskz_loadu_epi8(u, w + (size_t)(4 * g + j) * C + ch) : _mm_setzero_si128();
            W[g] = DWR_IL(q[0], q[1], q[2], q[3]);
            if (sgn) corr = _mm512_dpbusd_epi32(corr, f, W[g]);
        }
        const Quant16 qc = vnni_qprep(_mm512_maskz_loadu_epi32(u, bias + ch), _mm512_maskz_loadu_epi32(u, m + ch),
                                      _mm512_maskz_loadu_epi32(u, s + ch));
        for (uint32_t t0 = 0; t0 < T; t0 += TB) {
            const uint32_t tn = T - t0 < TB ? T - t0 : TB, rn = tn + 4 * G - 4;
            const int64_t r0 = (int64_t)t0 - py; /* I rows r0..r0+rn-1 */
            for (uint32_t i = 0; i < rn; i++) {
                __m128i q[4];
                for (uint32_t j = 0; j < 4; j++) {
                    const int64_t r = r0 + i + j;
                    q[j] = r >= 0 && r < (int64_t)H ? _mm_maskz_loadu_epi8(u, x + (size_t)r * C + ch)
                                                    : _mm_setzero_si128();
                }
                __m512i v = DWR_IL(q[0], q[1], q[2], q[3]);
                if (sgn) v = _mm512_xor_si512(v, f);
                _mm512_store_si512(I + (size_t)i * 64, v);
            }
            for (uint32_t t = 0; t < tn; t += 8) {
                const uint32_t P = tn - t < 8 ? tn - t : 8;
                __m512i a[8];
                for (int p = 0; p < 8; p++) a[p] = zero;
                for (uint32_t g = 0; g < G; g++) {
                    const uint8_t *ig = I + (size_t)(t + 4 * g) * 64;
                    for (uint32_t p = 0; p < 8; p++)
                        a[p] = _mm512_dpbusd_epi32(a[p], _mm512_load_si512(ig + 64 * (p < P ? p : 0)), W[g]);
                }
                if (o->lo >= 0)
                    for (uint32_t p = 0; p < P; p++)
                        _mm_mask_storeu_epi8(y + (size_t)(t0 + t + p) * C + ch, u,
                                             vnni_qsat(_mm512_sub_epi32(a[p], corr), &qc, o->lo, o->hi, 1));
                else
                    for (uint32_t p = 0; p < P; p++)
                        _mm_mask_storeu_epi8(y + (size_t)(t0 + t + p) * C + ch, u,
                                             vnni_qsat(_mm512_sub_epi32(a[p], corr), &qc, o->lo, o->hi, 0));
            }
        }
    }
#undef DWR_IL
    return 1;
}
#endif
/* Depthwise for up to 16 taps over a group of pixels: the interleaved weights
   of each 64-channel chunk are built once and stay in registers while every
   pixel streams through. Signed inputs: a missing tap reads x=0x80 (offset 0),
   so the 128*sum(w) correction covers all taps uniformly and is hoisted too. */
static inline __attribute__((always_inline)) void depthwise_group(const int8_t *const (*xp)[64], const int *whole,
                                                                  uint32_t cnt, uint32_t taps, uint32_t groups,
                                                                  const int8_t *weights, uint32_t channels,
                                                                  const int32_t *bias, const int32_t *m,
                                                                  const uint32_t *s, const Op *o, int8_t *y, int sgn) {
    const __m512i f = _mm512_set1_epi8((char)0x80), zero = _mm512_setzero_si512();
    for (uint32_t ch = 0; ch < channels; ch += 64) {
        const __mmask64 t = channels - ch >= 64 ? ~0ull : ~0ull >> (64 - (channels - ch));
        __m512i W[16], c0 = zero, c1 = zero, c2 = zero, c3 = zero;
        for (uint32_t g = 0; g < 4; g++) {
            __m512i q[4];
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t k = 4 * g + j;
                q[j] = _mm512_maskz_loadu_epi8(k < taps ? t : 0, weights + (k < taps ? k * channels + ch : 0));
            }
            __m512i wl = _mm512_unpacklo_epi8(q[0], q[1]), wh = _mm512_unpackhi_epi8(q[0], q[1]),
                    vl = _mm512_unpacklo_epi8(q[2], q[3]), vh = _mm512_unpackhi_epi8(q[2], q[3]);
            W[4 * g] = _mm512_unpacklo_epi16(wl, vl);
            W[4 * g + 1] = _mm512_unpackhi_epi16(wl, vl);
            W[4 * g + 2] = _mm512_unpacklo_epi16(wh, vh);
            W[4 * g + 3] = _mm512_unpackhi_epi16(wh, vh);
            if (sgn) {
                c0 = _mm512_dpbusd_epi32(c0, f, W[4 * g]);
                c1 = _mm512_dpbusd_epi32(c1, f, W[4 * g + 1]);
                c2 = _mm512_dpbusd_epi32(c2, f, W[4 * g + 2]);
                c3 = _mm512_dpbusd_epi32(c3, f, W[4 * g + 3]);
            }
        }
        Quant16 qc[4];
        for (uint32_t j = 0; j < 4 && ch + 16 * j < channels; j++) {
            const uint32_t b = ch + 16 * j;
            const __mmask16 u = channels - b >= 16 ? 0xFFFF : (__mmask16)((1u << (channels - b)) - 1);
            qc[j] = vnni_qprep(_mm512_maskz_loadu_epi32(u, bias + b), _mm512_maskz_loadu_epi32(u, m + b),
                               _mm512_maskz_loadu_epi32(u, s + b));
        }
        for (uint32_t p = 0; p < cnt; p++) {
            __m512i r0 = zero, r1 = zero, r2 = zero, r3 = zero;
            for (uint32_t g = 0; g < 4; g++)
                if (g < groups) {
                    __m512i x[4];
                    if (whole[p])
                        for (
                            uint32_t j = 0; j < 4;
                            j++) { /* every tap present: plain loads, past-the-end taps read a valid row and meet w=0 */
                            const uint32_t k = 4 * g + j;
                            x[j] = _mm512_maskz_loadu_epi8(t, xp[p][k < taps ? k : 0] + ch);
                            if (sgn) x[j] = _mm512_xor_si512(x[j], f);
                        }
                    else
                        for (uint32_t j = 0; j < 4; j++) {
                            const uint32_t k = 4 * g + j;
                            const int8_t *a = k < taps ? xp[p][k] : NULL;
                            x[j] = _mm512_maskz_loadu_epi8(a ? t : 0, a ? a + ch : weights);
                            if (sgn) x[j] = _mm512_xor_si512(x[j], f);
                        }
                    __m512i xl = _mm512_unpacklo_epi8(x[0], x[1]), xh = _mm512_unpackhi_epi8(x[0], x[1]),
                            yl = _mm512_unpacklo_epi8(x[2], x[3]), yh = _mm512_unpackhi_epi8(x[2], x[3]);
                    r0 = _mm512_dpbusd_epi32(r0, _mm512_unpacklo_epi16(xl, yl), W[4 * g]);
                    r1 = _mm512_dpbusd_epi32(r1, _mm512_unpackhi_epi16(xl, yl),
                                             W[4 * g + 1]); /* lane L: channels 16L+0..3, +4..7 */
                    r2 = _mm512_dpbusd_epi32(r2, _mm512_unpacklo_epi16(xh, yh), W[4 * g + 2]);
                    r3 = _mm512_dpbusd_epi32(r3, _mm512_unpackhi_epi16(xh, yh), W[4 * g + 3]); /* +8..11, +12..15 */
                }
            if (sgn) {
                r0 = _mm512_sub_epi32(r0, c0);
                r1 = _mm512_sub_epi32(r1, c1);
                r2 = _mm512_sub_epi32(r2, c2);
                r3 = _mm512_sub_epi32(r3, c3);
            }
            __m512i t0 = _mm512_shuffle_i32x4(r0, r1, 0x44), t1 = _mm512_shuffle_i32x4(r2, r3, 0x44),
                    t2 = _mm512_shuffle_i32x4(r0, r1, 0xEE), t3 = _mm512_shuffle_i32x4(r2, r3, 0xEE);
            __m512i v[4] = {_mm512_shuffle_i32x4(t0, t1, 0x88), _mm512_shuffle_i32x4(t0, t1, 0xDD),
                            _mm512_shuffle_i32x4(t2, t3, 0x88), _mm512_shuffle_i32x4(t2, t3, 0xDD)};
            for (uint32_t j = 0; j < 4 && ch + 16 * j < channels; j++) {
                const uint32_t b = ch + 16 * j;
                const __mmask16 u = channels - b >= 16 ? 0xFFFF : (__mmask16)((1u << (channels - b)) - 1);
                _mm_mask_storeu_epi8(y + p * channels + b, u, vnni_qapply(v[j], &qc[j], o->lo, o->hi));
            }
        }
    }
}
#endif
#if defined(__wasm_simd128__)
#define LO8(a, b)  wasm_i8x16_shuffle(a, b, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23)
#define HI8(a, b)  wasm_i8x16_shuffle(a, b, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31)
#define LO16(a, b) wasm_i16x8_shuffle(a, b, 0, 8, 1, 9, 2, 10, 3, 11)
#define HI16(a, b) wasm_i16x8_shuffle(a, b, 4, 12, 5, 13, 6, 14, 7, 15)
/* Depthwise over a group of pixels, 16 channels at a time. Taps are
   interleaved so each int32 lane sums one channel's taps: four per lane via
   relaxed i8 x i7 dots for non-negative inputs, else two per lane as int16
   pairs (exact for any input). A chunk's interleaved weights are built once per
   group. xp[p][k] is NULL for padding and for k>=taps up to a multiple of 4. */
static void depthwise_wasm(const int8_t *const (*xp)[64], uint32_t cnt, uint32_t taps, const int8_t *w,
                           uint32_t channels, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o,
                           int8_t *y, int nonneg) {
    const v128_t zero = wasm_i32x4_splat(0), lo = wasm_f64x2_splat(o->lo), hi = wasm_f64x2_splat(o->hi);
    const uint32_t kt = (taps + 3) & ~3u;
    uint32_t ch = 0;
    for (; ch + 16 <= channels; ch += 16) {
        v128_t W[128], f[8], b[4], q[4], x[4], r[4];
        for (uint32_t j = 0; j < 4; j++) b[j] = wasm_v128_load(bias + ch + 4 * j);
        for (uint32_t j = 0; j < 8; j++)
            f[j] = wasm_f64x2_make(qscale(m[ch + 2 * j], s[ch + 2 * j]), qscale(m[ch + 2 * j + 1], s[ch + 2 * j + 1]));
#if defined(__wasm_relaxed_simd__)
        if (nonneg) {
            for (uint32_t k = 0; k < kt; k += 4) {
                for (uint32_t j = 0; j < 4; j++)
                    q[j] = k + j < taps ? wasm_v128_load(w + (k + j) * channels + ch) : zero;
                v128_t a = LO8(q[0], q[1]), c = HI8(q[0], q[1]), d = LO8(q[2], q[3]), e = HI8(q[2], q[3]);
                W[k] = LO16(a, d);
                W[k + 1] = HI16(a, d);
                W[k + 2] = LO16(c, e);
                W[k + 3] = HI16(c, e);
            }
            for (uint32_t p = 0; p < cnt; p++) {
                r[0] = r[1] = r[2] = r[3] = zero;
                for (uint32_t k = 0; k < kt; k += 4) {
                    for (uint32_t j = 0; j < 4; j++) x[j] = xp[p][k + j] ? wasm_v128_load(xp[p][k + j] + ch) : zero;
                    v128_t a = LO8(x[0], x[1]), c = HI8(x[0], x[1]), d = LO8(x[2], x[3]), e = HI8(x[2], x[3]);
                    r[0] = wasm_i32x4_add(
                        r[0], wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_relaxed_dot_i8x16_i7x16(W[k], LO16(a, d))));
                    r[1] = wasm_i32x4_add(r[1], wasm_i32x4_extadd_pairwise_i16x8(
                                                    wasm_i16x8_relaxed_dot_i8x16_i7x16(W[k + 1], HI16(a, d))));
                    r[2] = wasm_i32x4_add(r[2], wasm_i32x4_extadd_pairwise_i16x8(
                                                    wasm_i16x8_relaxed_dot_i8x16_i7x16(W[k + 2], LO16(c, e))));
                    r[3] = wasm_i32x4_add(r[3], wasm_i32x4_extadd_pairwise_i16x8(
                                                    wasm_i16x8_relaxed_dot_i8x16_i7x16(W[k + 3], HI16(c, e))));
                }
                wasm_v128_store(y + p * channels + ch, wasm_quant16(r, b, f, lo, hi));
            }
            continue;
        }
#endif
        for (uint32_t k = 0; k < kt; k += 2) {
            q[0] = k < taps ? wasm_v128_load(w + k * channels + ch) : zero;
            q[1] = k + 1 < taps ? wasm_v128_load(w + (k + 1) * channels + ch) : zero;
            v128_t a = LO8(q[0], q[1]), c = HI8(q[0], q[1]);
            W[2 * k] = wasm_i16x8_extend_low_i8x16(a);
            W[2 * k + 1] = wasm_i16x8_extend_high_i8x16(a);
            W[2 * k + 2] = wasm_i16x8_extend_low_i8x16(c);
            W[2 * k + 3] = wasm_i16x8_extend_high_i8x16(c);
        }
        for (uint32_t p = 0; p < cnt; p++) {
            r[0] = r[1] = r[2] = r[3] = zero;
            for (uint32_t k = 0; k < kt; k += 2) {
                x[0] = xp[p][k] ? wasm_v128_load(xp[p][k] + ch) : zero;
                x[1] = xp[p][k + 1] ? wasm_v128_load(xp[p][k + 1] + ch) : zero;
                v128_t a = LO8(x[0], x[1]), c = HI8(x[0], x[1]);
                r[0] = wasm_i32x4_add(r[0], wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a), W[2 * k]));
                r[1] = wasm_i32x4_add(r[1], wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a), W[2 * k + 1]));
                r[2] = wasm_i32x4_add(r[2], wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(c), W[2 * k + 2]));
                r[3] = wasm_i32x4_add(r[3], wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(c), W[2 * k + 3]));
            }
            wasm_v128_store(y + p * channels + ch, wasm_quant16(r, b, f, lo, hi));
        }
    }
    (void)nonneg;
    for (; ch < channels; ch++)
        for (uint32_t p = 0; p < cnt; p++) {
            int32_t acc = bias[ch];
            for (uint32_t k = 0; k < taps; k++)
                if (xp[p][k]) acc += (int32_t)xp[p][k][ch] * w[k * channels + ch];
            y[p * channels + ch] = quant(acc, m[ch], s[ch], o);
        }
}
#undef LO8
#undef HI8
#undef LO16
#undef HI16
#define DW_GROUP depthwise_wasm
#elif defined(INT8_TILE) && defined(__AVX2__)
/* depthwise_wasm for AVX2, 32 channels at a time; four taps per lane via
   maddubs (non-negative inputs, no saturation) or two as int16 pairs. In-lane
   unpacks leave 128-bit halves holding channels c and c+16; outputs are put
   back in order before quant. */
static void depthwise_avx2(const int8_t *const (*xp)[64], uint32_t cnt, uint32_t taps, const int8_t *w,
                           uint32_t channels, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o,
                           int8_t *y, int nonneg) {
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi16(1);
    const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi);
    const uint32_t kt = (taps + 3) & ~3u;
    uint32_t ch = 0;
    for (; ch + 32 <= channels; ch += 32) {
        __m256i W[128], q[4], x[4], r[4];
        __m128i b[8];
        __m256d f[8];
        for (uint32_t j = 0; j < 8; j++) {
            b[j] = _mm_loadu_si128((const __m128i *)(bias + ch + 4 * j));
            f[j] = avx_qf(m + ch + 4 * j, s + ch + 4 * j);
        }
        if (nonneg)
            for (uint32_t k = 0; k < kt; k += 4) {
                for (uint32_t j = 0; j < 4; j++) q[j] = k + j < taps ? AVX_LD(w + (k + j) * channels + ch) : zero;
                __m256i a = _mm256_unpacklo_epi8(q[0], q[1]), c = _mm256_unpackhi_epi8(q[0], q[1]),
                        d = _mm256_unpacklo_epi8(q[2], q[3]), e = _mm256_unpackhi_epi8(q[2], q[3]);
                W[k] = _mm256_unpacklo_epi16(a, d);
                W[k + 1] = _mm256_unpackhi_epi16(a, d);
                W[k + 2] = _mm256_unpacklo_epi16(c, e);
                W[k + 3] = _mm256_unpackhi_epi16(c, e);
            }
        else
            for (uint32_t k = 0; k < kt; k += 2) {
                q[0] = k < taps ? AVX_LD(w + k * channels + ch) : zero;
                q[1] = k + 1 < taps ? AVX_LD(w + (k + 1) * channels + ch) : zero;
                __m256i a = _mm256_unpacklo_epi8(q[0], q[1]), c = _mm256_unpackhi_epi8(q[0], q[1]);
                W[2 * k] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
                W[2 * k + 1] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(c));
                W[2 * k + 2] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
                W[2 * k + 3] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(c, 1));
            }
        for (uint32_t p = 0; p < cnt; p++) {
            __m256i u[4];
            __m128i v[8];
            r[0] = r[1] = r[2] = r[3] = zero;
            if (nonneg) {
                for (uint32_t k = 0; k < kt; k += 4) {
                    for (uint32_t j = 0; j < 4; j++) x[j] = xp[p][k + j] ? AVX_LD(xp[p][k + j] + ch) : zero;
                    __m256i a = _mm256_unpacklo_epi8(x[0], x[1]), c = _mm256_unpackhi_epi8(x[0], x[1]),
                            d = _mm256_unpacklo_epi8(x[2], x[3]), e = _mm256_unpackhi_epi8(x[2], x[3]);
                    r[0] = _mm256_add_epi32(
                        r[0], _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_unpacklo_epi16(a, d), W[k]), one));
                    r[1] = _mm256_add_epi32(
                        r[1], _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_unpackhi_epi16(a, d), W[k + 1]), one));
                    r[2] = _mm256_add_epi32(
                        r[2], _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_unpacklo_epi16(c, e), W[k + 2]), one));
                    r[3] = _mm256_add_epi32(
                        r[3], _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_unpackhi_epi16(c, e), W[k + 3]), one));
                } /* r halves: [0..3|16..19] [4..7|20..23] [8..11|24..27] [12..15|28..31] */
                u[0] = _mm256_permute2x128_si256(r[0], r[1], 0x20);
                u[1] = _mm256_permute2x128_si256(r[2], r[3], 0x20);
                u[2] = _mm256_permute2x128_si256(r[0], r[1], 0x31);
                u[3] = _mm256_permute2x128_si256(r[2], r[3], 0x31);
            } else {
                for (uint32_t k = 0; k < kt; k += 2) {
                    x[0] = xp[p][k] ? AVX_LD(xp[p][k] + ch) : zero;
                    x[1] = xp[p][k + 1] ? AVX_LD(xp[p][k + 1] + ch) : zero;
                    __m256i a = _mm256_unpacklo_epi8(x[0], x[1]), c = _mm256_unpackhi_epi8(x[0], x[1]);
                    r[0] = _mm256_add_epi32(
                        r[0], _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(a)), W[2 * k]));
                    r[1] = _mm256_add_epi32(
                        r[1], _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(c)), W[2 * k + 1]));
                    r[2] = _mm256_add_epi32(
                        r[2], _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1)), W[2 * k + 2]));
                    r[3] = _mm256_add_epi32(
                        r[3], _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(c, 1)), W[2 * k + 3]));
                } /* r: channels 0..7, 8..15, 16..23, 24..31 */
                u[0] = r[0];
                u[1] = r[1];
                u[2] = r[2];
                u[3] = r[3];
            }
            for (uint32_t j = 0; j < 4; j++) {
                v[2 * j] = _mm256_castsi256_si128(u[j]);
                v[2 * j + 1] = _mm256_extracti128_si256(u[j], 1);
            }
            _mm_storeu_si128((__m128i *)(y + p * channels + ch), avx_quant16(v, b, f, lo, hi));
            _mm_storeu_si128((__m128i *)(y + p * channels + ch + 16), avx_quant16(v + 4, b + 4, f + 4, lo, hi));
        }
    }
    for (; ch < channels; ch++)
        for (uint32_t p = 0; p < cnt; p++) {
            int32_t acc = bias[ch];
            for (uint32_t k = 0; k < taps; k++)
                if (xp[p][k]) acc += (int32_t)xp[p][k][ch] * w[k * channels + ch];
            y[p * channels + ch] = quant(acc, m[ch], s[ch], o);
        }
}
#define DW_GROUP depthwise_avx2
#endif
#if defined(INT8_TILE) && defined(__AVX2__) && !defined(GUST_INT8_VNNI)
static inline __m128i dwa_part(const int8_t *p, uint32_t n) {
    int8_t t[16] = {0};
    for (uint32_t i = 0; i < n; i++) t[i] = p[i];
    return _mm_loadu_si128((const __m128i *)t);
}
#endif
#if defined(INT8_TILE) && defined(__AVX2__) && !defined(GUST_INT8_VNNI)
/* DEPTHWISE along the rows of a [H,1,C] tensor (kw=1, stride 1, dilation 1), AVX2: the two-row interleave
   I(r)[c] = (x[r][c], x[r+1][c]) as int16 (padding rows 0) is built once per row and 16 channels and every output
   row t sums vpmaddwd(I(t-py+2g), W[g]) with W[g] = taps 2g, 2g+1 as int16 (zero past kh): exact for any int8
   operands (|pair| <= 2^15, lanes <= taps*2^14). In-lane unpacks hold channels 0..3|8..11 and 4..7|12..15;
   outputs are put back in order before quant. */
#define DWA_BUF 49152
static int depthwise_rows(const int8_t *x, uint32_t H, uint32_t T, uint32_t py, uint32_t taps, const int8_t *w,
                          uint32_t C, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o,
                          int8_t *y) {
    const uint32_t G = (taps + 1) / 2, RB = DWA_BUF / 64; /* I rows of 16 channels: 64 bytes */
    if (G > 32 || RB < 2 * G + 8) return 0;
    const uint32_t TB = RB - 2 * G + 2; /* output rows per chunk */
    __attribute__((aligned(32))) int16_t I[DWA_BUF / 2];
    __attribute__((aligned(32))) int16_t W[32][32];
    const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi);
    for (uint32_t ch = 0; ch < C; ch += 16) {
        const uint32_t cn = C - ch < 16 ? C - ch : 16;
        __m128i b[4];
        __m256d f[4];
        int32_t bv[16], mv[16];
        uint32_t sv[16];
        for (uint32_t l = 0; l < 16; l++) {
            bv[l] = l < cn ? bias[ch + l] : 0;
            mv[l] = l < cn ? m[ch + l] : 1;
            sv[l] = l < cn ? s[ch + l] : 0;
        }
        for (uint32_t j = 0; j < 4; j++) {
            b[j] = _mm_loadu_si128((const __m128i *)(bv + 4 * j));
            f[j] = avx_qf(mv + 4 * j, sv + 4 * j);
        }
#define DWA_ROW(P)                                                                                                     \
    ((P) ? _mm256_cvtepi8_epi16(cn >= 16 ? _mm_loadu_si128((const __m128i *)(P)) : dwa_part(P, cn))                    \
         : _mm256_setzero_si256())
        for (uint32_t g = 0; g < G; g++) {
            const __m256i r0 = DWA_ROW(w + (size_t)(2 * g) * C + ch),
                          r1 = 2 * g + 1 < taps ? DWA_ROW(w + (size_t)(2 * g + 1) * C + ch) : _mm256_setzero_si256();
            _mm256_store_si256((__m256i *)W[g], _mm256_unpacklo_epi16(r0, r1));
            _mm256_store_si256((__m256i *)(W[g] + 16), _mm256_unpackhi_epi16(r0, r1));
        }
        for (uint32_t t0 = 0; t0 < T; t0 += TB) {
            const uint32_t tn = T - t0 < TB ? T - t0 : TB, rn = tn + 2 * G - 2;
            const int64_t r0 = (int64_t)t0 - py; /* I rows r0..r0+rn-1 */
            for (uint32_t i = 0; i < rn; i++) {
                const int64_t ra = r0 + i, rb = ra + 1;
                const __m256i a = DWA_ROW(ra >= 0 && ra < (int64_t)H ? x + (size_t)ra * C + ch : NULL),
                              c = DWA_ROW(rb >= 0 && rb < (int64_t)H ? x + (size_t)rb * C + ch : NULL);
                _mm256_store_si256((__m256i *)(I + (size_t)i * 32), _mm256_unpacklo_epi16(a, c));
                _mm256_store_si256((__m256i *)(I + (size_t)i * 32 + 16), _mm256_unpackhi_epi16(a, c));
            }
            for (uint32_t t = 0; t < tn; t += 4) {
                const uint32_t P = tn - t < 4 ? tn - t : 4;
                __m256i al[4], ah[4];
                for (int p = 0; p < 4; p++) al[p] = ah[p] = _mm256_setzero_si256();
                for (uint32_t g = 0; g < G; g++) {
                    const __m256i wl = _mm256_load_si256((const __m256i *)W[g]),
                                  wh = _mm256_load_si256((const __m256i *)(W[g] + 16));
                    for (uint32_t p = 0; p < 4; p++) {
                        const int16_t *ip = I + (size_t)(t + (p < P ? p : 0) + 2 * g) * 32;
                        al[p] = _mm256_add_epi32(al[p], _mm256_madd_epi16(_mm256_load_si256((const __m256i *)ip), wl));
                        ah[p] = _mm256_add_epi32(ah[p],
                                                 _mm256_madd_epi16(_mm256_load_si256((const __m256i *)(ip + 16)), wh));
                    }
                }
                for (uint32_t p = 0; p < P; p++) { /* al: channels 0..3|8..11, ah: 4..7|12..15 */
                    const __m128i q0 = avx_quant(_mm_add_epi32(_mm256_castsi256_si128(al[p]), b[0]), f[0], lo, hi),
                                  q1 = avx_quant(_mm_add_epi32(_mm256_castsi256_si128(ah[p]), b[1]), f[1], lo, hi);
                    const __m128i q2 = avx_quant(_mm_add_epi32(_mm256_extracti128_si256(al[p], 1), b[2]), f[2], lo, hi),
                                  q3 = avx_quant(_mm_add_epi32(_mm256_extracti128_si256(ah[p], 1), b[3]), f[3], lo, hi);
                    const __m128i v = _mm_packs_epi16(_mm_packs_epi32(q0, q1), _mm_packs_epi32(q2, q3));
                    int8_t *d = y + (size_t)(t0 + t + p) * C + ch;
                    if (cn >= 16)
                        _mm_storeu_si128((__m128i *)d, v);
                    else {
                        int8_t l[16];
                        _mm_storeu_si128((__m128i *)l, v);
                        for (uint32_t k = 0; k < cn; k++) d[k] = l[k];
                    }
                }
            }
        }
#undef DWA_ROW
    }
    return 1;
}
#endif
static void depthwise_pixel(const int8_t *const *patch, uint32_t taps, const int8_t *weights, uint32_t channels,
                            const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o, int8_t *out,
                            int nonneg) {
    uint32_t ch = 0;
#ifdef GUST_INT8_VNNI
    if (nonneg)
        depthwise_vnni(patch, taps, weights, channels, bias, m, s, o, out, 1);
    else
        depthwise_vnni(patch, taps, weights, channels, bias, m, s, o, out, 0);
    return;
#endif
    (void)nonneg;
#if defined(__aarch64__)
    for (; ch + 16 <= channels; ch += 16) { /* |x * w| <= 2^14 fits int16; taps * 2^14 + bias fits int32 */
        int32x4_t a[4] = {vld1q_s32(bias + ch), vld1q_s32(bias + ch + 4), vld1q_s32(bias + ch + 8),
                          vld1q_s32(bias + ch + 12)};
        for (uint32_t k = 0; k < taps; k++)
            if (patch[k]) {
                const int8x16_t x = vld1q_s8(patch[k] + ch), w = vld1q_s8(weights + k * channels + ch);
                const int16x8_t l = vmull_s8(vget_low_s8(x), vget_low_s8(w)), h = vmull_high_s8(x, w);
                a[0] = vaddw_s16(a[0], vget_low_s16(l));
                a[1] = vaddw_high_s16(a[1], l);
                a[2] = vaddw_s16(a[2], vget_low_s16(h));
                a[3] = vaddw_high_s16(a[3], h);
            }
        for (uint32_t j = 0; j < 4; j++) {
            const NgQuant q = ng_qprep(m + ch + 4 * j, s + ch + 4 * j, 1, o);
            const uint32_t v = ng_quant4(a[j], &q);
            __builtin_memcpy(out + ch + 4 * j, &v, 4);
        }
    }
#endif
#if defined(__wasm_simd128__) || defined(__SSE2__)
    for (; ch + 16 <= channels; ch += 16) {
        int32_t accumulators[16];
#if defined(__wasm_simd128__)
        v128_t a0 = wasm_v128_load(bias + ch), a1 = wasm_v128_load(bias + ch + 4), a2 = wasm_v128_load(bias + ch + 8),
               a3 = wasm_v128_load(bias + ch + 12);
        for (uint32_t k = 0; k < taps; k++)
            if (patch[k]) {
                v128_t x = wasm_v128_load(patch[k] + ch), w = wasm_v128_load(weights + k * channels + ch);
                v128_t lo = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(x), wasm_i16x8_extend_low_i8x16(w));
                v128_t hi = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(x), wasm_i16x8_extend_high_i8x16(w));
                a0 = wasm_i32x4_add(a0, wasm_i32x4_extend_low_i16x8(lo));
                a1 = wasm_i32x4_add(a1, wasm_i32x4_extend_high_i16x8(lo));
                a2 = wasm_i32x4_add(a2, wasm_i32x4_extend_low_i16x8(hi));
                a3 = wasm_i32x4_add(a3, wasm_i32x4_extend_high_i16x8(hi));
            }
        wasm_v128_store(accumulators, a0);
        wasm_v128_store(accumulators + 4, a1);
        wasm_v128_store(accumulators + 8, a2);
        wasm_v128_store(accumulators + 12, a3);
#else
        __m128i zero = _mm_setzero_si128();
        __m128i a0 = _mm_loadu_si128((const __m128i *)(bias + ch)),
                a1 = _mm_loadu_si128((const __m128i *)(bias + ch + 4)),
                a2 = _mm_loadu_si128((const __m128i *)(bias + ch + 8)),
                a3 = _mm_loadu_si128((const __m128i *)(bias + ch + 12));
        for (uint32_t k = 0; k < taps; k++)
            if (patch[k]) {
                __m128i x = _mm_loadu_si128((const __m128i *)(patch[k] + ch)),
                        w = _mm_loadu_si128((const __m128i *)(weights + k * channels + ch));
                __m128i xs = _mm_cmpgt_epi8(zero, x), ws = _mm_cmpgt_epi8(zero, w);
                __m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(x, xs), _mm_unpacklo_epi8(w, ws)),
                        hi = _mm_mullo_epi16(_mm_unpackhi_epi8(x, xs), _mm_unpackhi_epi8(w, ws));
                __m128i ls = _mm_srai_epi16(lo, 15), hs = _mm_srai_epi16(hi, 15);
                a0 = _mm_add_epi32(a0, _mm_unpacklo_epi16(lo, ls));
                a1 = _mm_add_epi32(a1, _mm_unpackhi_epi16(lo, ls));
                a2 = _mm_add_epi32(a2, _mm_unpacklo_epi16(hi, hs));
                a3 = _mm_add_epi32(a3, _mm_unpackhi_epi16(hi, hs));
            }
        _mm_storeu_si128((__m128i *)accumulators, a0);
        _mm_storeu_si128((__m128i *)(accumulators + 4), a1);
        _mm_storeu_si128((__m128i *)(accumulators + 8), a2);
        _mm_storeu_si128((__m128i *)(accumulators + 12), a3);
#endif
        for (uint32_t j = 0; j < 16; j++) out[ch + j] = quant(accumulators[j], m[ch + j], s[ch + j], o);
    }
#endif
    for (; ch < channels; ch++) {
        int32_t acc = bias[ch];
        for (uint32_t k = 0; k < taps; k++)
            if (patch[k]) acc += (int32_t)patch[k][ch] * weights[k * channels + ch];
        out[ch] = quant(acc, m[ch], s[ch], o);
    }
}
