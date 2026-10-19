/* Dense convolution kernels and the convolution driver: conv_pixel (any patch), packed-weight GEMM tiles (AVX-512
   VNNI vpdpbusd; AVX2 vpmaddubsw with offset weights or non-negative inputs; AVX2 int16 tiles for any operands),
   then conv_rows, which routes every output pixel of a CONV / DEPTHWISE to the fastest exact kernel. Part of the
   runtime/gust.c translation unit. */

/* One output pixel of a general convolution from its patch (NULL = padding). */
static void conv_pixel(const int8_t *const *patch, uint32_t taps, uint32_t n, const int8_t *w, uint32_t channels,
                       const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o, int8_t *out, int nonneg) {
    if (n == 3 && taps <= 9) {
        int8_t rgb[27];
        for (uint32_t k = 0; k < taps; k++)
            for (uint32_t ch = 0; ch < 3; ch++) rgb[k * 3 + ch] = patch[k] ? patch[k][ch] : 0;
        for (uint32_t oc = 0; oc < channels; oc++)
            out[oc] = quant(bias[oc] + dot(rgb, w + oc * taps * 3, taps * 3), m[oc], s[oc], o);
        return;
    }
    uint32_t oc = 0;
    if (n >= 16)
        for (; oc + 4 <= channels; oc += 4) {
            int32_t acc[4] = {bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]};
            for (uint32_t k = 0; k < taps; k++)
                if (patch[k]) {
                    int32_t r[4];
                    dot4(patch[k], w + oc * taps * n + k * n, n, taps * n, nonneg, r);
                    acc[0] += r[0];
                    acc[1] += r[1];
                    acc[2] += r[2];
                    acc[3] += r[3];
                }
            for (uint32_t j = 0; j < 4; j++) out[oc + j] = quant(acc[j], m[oc + j], s[oc + j], o);
        }
    for (; oc < channels; oc++) {
        int32_t acc = bias[oc];
        for (uint32_t k = 0; k < taps; k++)
            if (patch[k]) acc += dot(patch[k], w + oc * taps * n + k * n, n);
        out[oc] = quant(acc, m[oc], s[oc], o);
    }
}
/* Packed-weight tiles for convolutions whose taps are contiguous (every 1x1, unpadded kx1) and for
   MATMUL: y_p[oc]=quant(dot(x_p,w_oc)+bias[oc]) for np pixels (x_p: K contiguous bytes at xs[p], output
   row ys[p]) and N channels (w_oc: K bytes at w+oc*wrow). One block of channels at a time is packed as
   Wp[g][channel][4] = w_oc[4g..4g+3] (zero past K and N) and every pixel streams through it, its four
   bytes x_p[4g..4g+3] broadcast to every channel lane: no horizontal sums, and the channels of an output
   pixel leave the tile as one vector. uniform: one multiplier/shift for all channels, no bias (MATMUL). */
#if defined(GUST_INT8_VNNI) || (defined(INT8_TILE) && defined(__AVX2__))
#define GEMM_CHUNK 512   /* pixels per call */
#define GEMM_MIN   16    /* fewer output pixels: packing every weight would cost more than it saves */
#define GEMM_BUF   40960 /* bytes of packed weights per channel block */
typedef int32_t __attribute__((may_alias, aligned(1))) Dword;
static inline int32_t ld32(const int8_t *p) {
    return *(const Dword *)(const void *)p;
}
static inline int32_t ld_tail(const int8_t *p, uint32_t n) { /* n<4 bytes, zero padded, nothing read past p+n */
    uint32_t v = 0;
    for (uint32_t i = 0; i < n; i++) v |= (uint32_t)(uint8_t)p[i] << (8 * i);
    return (int32_t)v;
}
#ifndef GUST_INT8_VNNI
/* Wp[g][cw][4] for channels oc0..oc0+nb-1 of a block of cw (a multiple of 8): every real byte becomes
   (w^flip)+off (flip 0x80: w+128 as u8; off: AVX2 small-weight offset), padding past K and nb stays 0.
   Eight weight rows x 32 bytes per step are transposed as 8x8 dwords in registers, one 32-byte store per group. */
static void gemm_pack(uint8_t *Wp, const int8_t *w, uint32_t wrow, uint32_t K, uint32_t oc0, uint32_t nb, uint32_t cw,
                      uint8_t flip, uint8_t off) {
    const uint32_t Gt = (K + 3) / 4;
    const size_t step = 4 * (size_t)cw;
    const __m256i fv = _mm256_set1_epi8((char)flip), ov = _mm256_set1_epi8((char)off),
                  idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    for (uint32_t c8 = 0; c8 < cw; c8 += 8)
        for (uint32_t k0 = 0; k0 < K; k0 += 32) {
            const uint32_t kn = K - k0 < 32 ? K - k0 : 32;
            const __m256i real = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)kn), idx);
            __m256i r[8], t[8], u[8];
            for (uint32_t i = 0; i < 8; i++) {
                const uint32_t j = c8 + i;
                if (j >= nb) {
                    r[i] = _mm256_setzero_si256();
                    continue;
                }
                const int8_t *row = w + (size_t)(oc0 + j) * wrow + k0;
                __m256i v;
                if (kn == 32)
                    v = _mm256_loadu_si256((const __m256i *)row);
                else {
                    int8_t tb[32] = {0};
                    for (uint32_t k = 0; k < kn; k++) tb[k] = row[k];
                    v = _mm256_loadu_si256((const __m256i *)tb);
                }
                r[i] = _mm256_and_si256(_mm256_add_epi8(_mm256_xor_si256(v, fv), ov), real);
            }
            for (uint32_t i = 0; i < 4; i++) {
                t[2 * i] = _mm256_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
                t[2 * i + 1] = _mm256_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
            }
            for (uint32_t b = 0; b < 2; b++) { /* u[4b+c], 128-bit half h: rows 4b..4b+3 of dword column 4h+c */
                u[4 * b] = _mm256_unpacklo_epi64(t[4 * b], t[4 * b + 2]);
                u[4 * b + 1] = _mm256_unpackhi_epi64(t[4 * b], t[4 * b + 2]);
                u[4 * b + 2] = _mm256_unpacklo_epi64(t[4 * b + 1], t[4 * b + 3]);
                u[4 * b + 3] = _mm256_unpackhi_epi64(t[4 * b + 1], t[4 * b + 3]);
            }
            for (uint32_t c = 0; c < 4; c++) {
                const uint32_t g = k0 / 4 + c;
                if (g < Gt)
                    _mm256_storeu_si256((__m256i *)(Wp + g * step + 4 * c8),
                                        _mm256_permute2x128_si256(u[c], u[4 + c], 0x20));
                if (g + 4 < Gt)
                    _mm256_storeu_si256((__m256i *)(Wp + (g + 4) * step + 4 * c8),
                                        _mm256_permute2x128_si256(u[c], u[4 + c], 0x31));
            }
        }
}
/* max |w| over the block's rows (|-128| = 128 as an unsigned byte). */
static uint32_t gemm_wmax(const int8_t *w, uint32_t wrow, uint32_t K, uint32_t oc0, uint32_t nb) {
    __m128i mx = _mm_setzero_si128();
    uint32_t m = 0;
    for (uint32_t j = 0; j < nb; j++) {
        const int8_t *r = w + (size_t)(oc0 + j) * wrow;
        uint32_t k = 0;
        for (; k + 16 <= K; k += 16) mx = _mm_max_epu8(mx, _mm_abs_epi8(_mm_loadu_si128((const __m128i *)(r + k))));
        for (; k < K; k++) {
            const uint32_t a = (uint32_t)(r[k] < 0 ? -(int32_t)r[k] : r[k]);
            m = a > m ? a : m;
        }
    }
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 8));
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 4));
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 2));
    mx = _mm_max_epu8(mx, _mm_srli_si128(mx, 1));
    const uint32_t v = (uint32_t)_mm_cvtsi128_si32(mx) & 255u;
    return v > m ? v : m;
}
#endif
#endif
#ifdef GUST_INT8_VNNI
/* Wp[g][cw][4] = w+128 (u8) for channels oc0..oc0+nb-1 of a block of cw (a multiple of 16), padding past K and
   nb 0; within each 16 channels lane 2i holds channel i and lane 2i+1 channel 8+i, so the even/odd 64-bit halves
   of an accumulator are channels 0..7 / 8..15 (vnni_qperm). 16 weight rows x 64 bytes (one cache line each) per
   step are transposed as 16x16 dwords in registers, one 64-byte store per group. */
static void gemm_pack512(uint8_t *Wp, const int8_t *w, uint32_t wrow, uint32_t K, uint32_t oc0, uint32_t nb,
                         uint32_t cw) {
    const uint32_t Gt = (K + 3) / 4;
    const size_t step = 4 * (size_t)cw;
    const __m512i f = _mm512_set1_epi8((char)0x80);
    static const uint8_t perm[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    for (uint32_t c16 = 0; c16 < cw; c16 += 16)
        for (uint32_t k0 = 0; k0 < K; k0 += 64) {
            const uint32_t kn = K - k0 < 64 ? K - k0 : 64;
            const __mmask64 real = kn >= 64 ? ~0ull : ~0ull >> (64 - kn);
            __m512i R[16], T[16], U[16];
            for (uint32_t L = 0; L < 16; L++) {
                const uint32_t j = c16 + perm[L];
                R[L] = j < nb
                           ? _mm512_maskz_mov_epi8(
                                 real,
                                 _mm512_xor_si512(_mm512_maskz_loadu_epi8(real, w + (size_t)(oc0 + j) * wrow + k0), f))
                           : _mm512_setzero_si512();
            }
            for (uint32_t i = 0; i < 8; i++) {
                T[2 * i] = _mm512_unpacklo_epi32(R[2 * i], R[2 * i + 1]);
                T[2 * i + 1] = _mm512_unpackhi_epi32(R[2 * i], R[2 * i + 1]);
            }
            for (uint32_t b = 0; b < 4; b++) { /* U[4b+c], 128-bit lane l: rows 4b..4b+3 of dword column 4l+c */
                U[4 * b] = _mm512_unpacklo_epi64(T[4 * b], T[4 * b + 2]);
                U[4 * b + 1] = _mm512_unpackhi_epi64(T[4 * b], T[4 * b + 2]);
                U[4 * b + 2] = _mm512_unpacklo_epi64(T[4 * b + 1], T[4 * b + 3]);
                U[4 * b + 3] = _mm512_unpackhi_epi64(T[4 * b + 1], T[4 * b + 3]);
            }
            for (uint32_t c = 0; c < 4; c++) {
                const __m512i X0 = _mm512_shuffle_i32x4(U[c], U[4 + c], 0x44),
                              X1 = _mm512_shuffle_i32x4(U[c], U[4 + c], 0xEE);
                const __m512i Y0 = _mm512_shuffle_i32x4(U[8 + c], U[12 + c], 0x44),
                              Y1 = _mm512_shuffle_i32x4(U[8 + c], U[12 + c], 0xEE);
                const __m512i O[4] = {_mm512_shuffle_i32x4(X0, Y0, 0x88), _mm512_shuffle_i32x4(X0, Y0, 0xDD),
                                      _mm512_shuffle_i32x4(X1, Y1, 0x88), _mm512_shuffle_i32x4(X1, Y1, 0xDD)};
                for (uint32_t l = 0; l < 4; l++) {
                    const uint32_t g = k0 / 4 + 4 * l + c;
                    if (g < Gt) _mm512_store_si512(Wp + g * step + 4 * c16, O[l]);
                }
            }
        }
}
/* vpdpbusd(Wp as u8 = w+128, x broadcast as s8) sums (w+128)*x per channel lane; minus 128*sum(x_p) that is
   the exact dot (int32 lanes wrap modulo 2^32 and the true result fits int32: gust_init bound). */
#define GV_P 8 /* pixels per tile; J (1 or 2) 16-channel vectors per pixel */
static inline
    __attribute__((always_inline)) void gv_block(const int8_t *const *xs, int8_t *const *ys, const int32_t *corr,
                                                 const int32_t *tail, uint32_t np, const uint8_t *Wp, uint32_t G,
                                                 uint32_t Kt, const Quant16 *q, uint32_t oc0, uint32_t nb, const int J,
                                                 const int relu, const int8_t *pf, size_t pfn, const Op *o) {
    const int PP = GV_P;
    const int32_t lo = o->lo, hi = o->hi;
    const size_t tiles = (np + PP - 1) / PP, per = (pfn + 64 * tiles - 1) / (64 * tiles);
    size_t pfo = 0;
    for (uint32_t p0 = 0; p0 < np; p0 += PP) {
        for (size_t i = 0; i < per && pfo < pfn; i++, pfo += 64)
            _mm_prefetch((const char *)(pf + pfo), _MM_HINT_T1); /* next block's weights into L2 */
        const uint32_t P = np - p0 < (uint32_t)PP ? np - p0 : (uint32_t)PP;
        const int8_t *x[GV_P];
        int32_t tl[GV_P];
        for (int p = 0; p < PP; p++) { /* short tiles repeat a pixel */
            const uint32_t i = p0 + ((uint32_t)p < P ? (uint32_t)p : 0);
            x[p] = xs[i];
            tl[p] = Kt ? tail[i] : 0;
        }
        __m512i a[GV_P][2]; /* start from bias-128*sum(x_p): the requantization adds nothing */
        for (int p = 0; p < PP; p++) {
            const __m512i c = _mm512_set1_epi32(corr[p0 + ((uint32_t)p < P ? (uint32_t)p : 0)]);
            for (int j = 0; j < J; j++) a[p][j] = _mm512_sub_epi32(q[j].b, c);
        }
        for (uint32_t g = 0; g < G; g++) {
            const uint8_t *wg = Wp + (size_t)g * 64 * J;
            __m512i w[2];
            for (int j = 0; j < J; j++) w[j] = _mm512_load_si512(wg + 64 * j);
            for (int p = 0; p < PP; p++) {
                const __m512i b = _mm512_set1_epi32(ld32(x[p] + 4 * g));
                for (int j = 0; j < J; j++) a[p][j] = _mm512_dpbusd_epi32(a[p][j], w[j], b);
            }
        }
        if (Kt) {
            const uint8_t *wg = Wp + (size_t)G * 64 * J;
            __m512i w[2];
            for (int j = 0; j < J; j++) w[j] = _mm512_load_si512(wg + 64 * j);
            for (int p = 0; p < PP; p++) {
                const __m512i b = _mm512_set1_epi32(tl[p]);
                for (int j = 0; j < J; j++) a[p][j] = _mm512_dpbusd_epi32(a[p][j], w[j], b);
            }
        }
        if (P == (uint32_t)PP && nb == 16u * J) { /* whole tile: constant indices keep the accumulators in registers */
            for (int p = 0; p < PP; p++)
                for (int j = 0; j < J; j++)
                    _mm_storeu_si128((__m128i *)(ys[p0 + p] + oc0 + 16 * j), vnni_qperm(a[p][j], &q[j], lo, hi, relu));
            continue;
        }
        __m512i t[GV_P][2];
        for (int p = 0; p < PP; p++)
            for (int j = 0; j < J; j++) t[p][j] = a[p][j];
        for (uint32_t p = 0; p < P; p++)
            for (int j = 0; j < J && 16u * j < nb; j++) {
                const uint32_t r = nb - 16u * j;
                const __mmask16 u = r >= 16 ? 0xFFFF : (__mmask16)((1u << r) - 1);
                _mm_mask_storeu_epi8(ys[p0 + p] + oc0 + 16 * j, u, vnni_qperm(t[p][j], &q[j], lo, hi, relu));
            }
    }
}
static inline int gemm_fits(uint32_t K) {
    return 64u * ((K + 3) / 4) <= GEMM_BUF;
}
/* Returns 0 (nothing written) when K is too long for the packing buffer. */
static int gemm(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, uint32_t wrow, uint32_t K,
                uint32_t N, const int32_t *bias, const int32_t *m, const uint32_t *s, int uniform, int small,
                const uint8_t *codes, const Op *o) {
    const uint32_t G = K / 4, Kt = K % 4, Gt = G + (Kt != 0);
    const int J = 128u * Gt <= GEMM_BUF ? 2 : 64u * Gt <= GEMM_BUF ? 1 : 0; /* 32 or 16 channels per packed block */
    (void)small;
    if (!J || !np) return J != 0;
    __attribute__((aligned(64))) uint8_t Wp[GEMM_BUF];
    int32_t corr[GEMM_CHUNK], tail[GEMM_CHUNK];
    __attribute__((aligned(64))) int8_t rows[GEMM_BUF]; /* the codes of one channel block, decoded (cw*K <= GEMM_BUF) */
    const __m512i one = _mm512_set1_epi8(1);
    for (uint32_t p = 0; p < np; p++) { /* 128*sum(x_p), modulo 2^32 */
        const int8_t *x = xs[p];
        __m512i acc = _mm512_setzero_si512(), a1 = acc;
        uint32_t k = 0;
        for (; k + 128 <= K; k += 128) {
            acc = _mm512_dpbusd_epi32(acc, one, _mm512_loadu_si512(x + k));
            a1 = _mm512_dpbusd_epi32(a1, one, _mm512_loadu_si512(x + k + 64));
        }
        for (; k + 64 <= K; k += 64) acc = _mm512_dpbusd_epi32(acc, one, _mm512_loadu_si512(x + k));
        if (k < K) acc = _mm512_dpbusd_epi32(acc, one, _mm512_maskz_loadu_epi8(~0ull >> (64 - (K - k)), x + k));
        acc = _mm512_add_epi32(acc, a1);
        corr[p] = (int32_t)(128u * (uint32_t)_mm512_reduce_add_epi32(acc));
        tail[p] = Kt ? ld_tail(x + 4 * G, Kt) : 0;
    }
    const uint32_t cw = 16u * J;
    for (uint32_t oc0 = 0; oc0 < N; oc0 += cw) {
        const uint32_t nb = N - oc0 < cw ? N - oc0 : cw;
        Quant16 q[2];
#ifdef GUST_PACKED_WEIGHTS
        if (codes) {
            packed_expand(codes, (uint64_t)oc0 * wrow, nb * wrow, o->p1, rows);
            gemm_pack512(Wp, rows, wrow, K, 0, nb, cw);
        } else
#endif
            gemm_pack512(Wp, w, wrow, K, oc0, nb, cw);
        for (int j = 0; j < J; j++) {
            const uint32_t b = oc0 + 16 * j, r = b < N ? N - b : 0;
            const __mmask16 u = r >= 16 ? 0xFFFF : (__mmask16)((1u << r) - 1);
            const __m512i pi = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
            q[j] = uniform ? vnni_qprep(_mm512_setzero_si512(), _mm512_set1_epi32(m[0]), _mm512_set1_epi32((int)s[0]))
                           : vnni_qprep(_mm512_permutexvar_epi32(pi, _mm512_maskz_loadu_epi32(u, bias + b)),
                                        _mm512_permutexvar_epi32(pi, _mm512_maskz_loadu_epi32(u, m + b)),
                                        _mm512_permutexvar_epi32(pi, _mm512_maskz_loadu_epi32(u, s + b)));
        }
        const int8_t *pf = w + (size_t)(oc0 + cw) * wrow;
        const size_t pfn = !codes && oc0 + cw < N ? (size_t)(N - oc0 - cw < cw ? N - oc0 - cw : cw) * wrow
                                                  : 0; /* rows are contiguous */
        if (o->lo >= 0) {
            if (J == 2)
                gv_block(xs, ys, corr, tail, np, Wp, G, Kt, q, oc0, nb, 2, 1, pf, pfn, o);
            else
                gv_block(xs, ys, corr, tail, np, Wp, G, Kt, q, oc0, nb, 1, 1, pf, pfn, o);
        } else {
            if (J == 2)
                gv_block(xs, ys, corr, tail, np, Wp, G, Kt, q, oc0, nb, 2, 0, pf, pfn, o);
            else
                gv_block(xs, ys, corr, tail, np, Wp, G, Kt, q, oc0, nb, 1, 0, pf, pfn, o);
        }
    }
    return 1;
}
#define GEMM 1
#elif defined(INT8_TILE) && defined(__AVX2__)
/* vpmaddubsw multiplies u8 x s8 and adds byte pairs into int16, which must not saturate. Two exact forms:
   small weights (|w| <= 63 everywhere, recorded by gust_init): the packed weights plus an offset
   (u8 = w+off, off = max|w| of the block: |pair| <= 2*2off*128 <= 32767) against x broadcast as s8, minus
   off*sum(x_p) per pixel; any input sign, no per-broadcast fix-up. Otherwise non-negative x (0..127)
   broadcast as u8 against the weights as s8 (|pair| <= 2*127*128). KD groups with KD*max|pair| <= 32767
   accumulate in int16 before vpmaddwd widens them: 4-bit weights take nine groups per widening. Lanes hold
   <= terms*255*128 < 2^31 (gust_init bound). */
#ifndef GA_P
#define GA_P 4 /* pixels per tile */
#define GA_J 2 /* 8-channel vectors per pixel */
#endif
static inline __attribute__((always_inline)) void
ga_block(const int8_t *const *xs, int8_t *const *ys, const int32_t *tail, const int32_t *sx, uint32_t np,
         const uint8_t *Wp, uint32_t G, uint32_t Kt, uint32_t KD, const int woff, uint32_t off, const __m128i *bc,
         const __m256d *f, uint32_t oc0, uint32_t nb, const int8_t *pf, size_t pfn, const Op *o) {
    const __m256i one = _mm256_set1_epi16(1);
    const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi);
    const size_t tiles = (np + GA_P - 1) / GA_P, per = (pfn + 64 * tiles - 1) / (64 * tiles);
    size_t pfo = 0;
#define GA_MUL(B, W) (woff ? _mm256_maddubs_epi16(W, B) : _mm256_maddubs_epi16(B, W))
    for (uint32_t p0 = 0; p0 < np; p0 += GA_P) {
        for (size_t i = 0; i < per && pfo < pfn; i++, pfo += 64)
            _mm_prefetch((const char *)(pf + pfo), _MM_HINT_T1); /* next block's weights into L2 */
        const uint32_t P = np - p0 < GA_P ? np - p0 : GA_P;
        const int8_t *x[GA_P];
        int32_t tl[GA_P];
        for (uint32_t p = 0; p < GA_P; p++) {
            const uint32_t i = p0 + (p < P ? p : 0);
            x[p] = xs[i];
            tl[p] = Kt ? tail[i] : 0;
        }
        __m256i a[GA_P][GA_J];
        for (int p = 0; p < GA_P; p++)
            for (int j = 0; j < GA_J; j++) a[p][j] = _mm256_setzero_si256();
        for (uint32_t g0 = 0; g0 < G; g0 += KD) {
            const uint32_t ge = G - g0 < KD ? G : g0 + KD;
            __m256i h[GA_P][GA_J];
            for (int p = 0; p < GA_P; p++)
                for (int j = 0; j < GA_J; j++) h[p][j] = _mm256_setzero_si256();
            const uint8_t *wg = Wp + (size_t)g0 * 32 * GA_J;
            const int8_t *xg[GA_P];
            for (int p = 0; p < GA_P; p++) xg[p] = x[p] + 4 * (size_t)g0;
            for (uint32_t g = g0; g < ge; g++, wg += 32 * GA_J) {
                __m256i wv[GA_J];
                for (int j = 0; j < GA_J; j++) wv[j] = _mm256_load_si256((const __m256i *)(wg + 32 * j));
                for (int p = 0; p < GA_P; p++) {
                    const __m256i b = _mm256_set1_epi32(ld32(xg[p]));
                    xg[p] += 4;
                    for (int j = 0; j < GA_J; j++) h[p][j] = _mm256_add_epi16(h[p][j], GA_MUL(b, wv[j]));
                }
            }
            for (int p = 0; p < GA_P; p++)
                for (int j = 0; j < GA_J; j++) a[p][j] = _mm256_add_epi32(a[p][j], _mm256_madd_epi16(h[p][j], one));
        }
        if (Kt) {
            const uint8_t *wg = Wp + (size_t)G * 32 * GA_J;
            for (int p = 0; p < GA_P; p++) {
                const __m256i b = _mm256_set1_epi32(tl[p]);
                for (int j = 0; j < GA_J; j++)
                    a[p][j] = _mm256_add_epi32(
                        a[p][j], _mm256_madd_epi16(GA_MUL(b, _mm256_load_si256((const __m256i *)(wg + 32 * j))), one));
            }
        }
        for (uint32_t p = 0; p < P; p++) {
            const __m128i c = _mm_set1_epi32(woff ? (int32_t)(off * (uint32_t)sx[p0 + p]) : 0);
            for (int j = 0; j < GA_J && 8u * j < nb; j++) {
                const __m128i r0 = avx_quant(
                    _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(a[p][j]), bc[2 * j]), c), f[2 * j], lo, hi);
                const __m128i r1 =
                    avx_quant(_mm_sub_epi32(_mm_add_epi32(_mm256_extracti128_si256(a[p][j], 1), bc[2 * j + 1]), c),
                              f[2 * j + 1], lo, hi);
                const __m128i v = _mm_packs_epi16(_mm_packs_epi32(r0, r1), r0);
                int8_t *d = ys[p0 + p] + oc0 + 8 * j;
                if (nb - 8u * j >= 8)
                    _mm_storel_epi64((__m128i *)d, v);
                else {
                    int8_t l[16];
                    _mm_storeu_si128((__m128i *)l, v);
                    for (uint32_t k = 0; k < nb - 8u * j; k++) d[k] = l[k];
                }
            }
        }
    }
#undef GA_MUL
}
static inline int gemm_fits(uint32_t K) {
    return 32u * GA_J * ((K + 3) / 4) <= GEMM_BUF;
}
/* small: every weight of the operator lies in [-63,63]; otherwise x must be non-negative. */
static int gemm(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, uint32_t wrow, uint32_t K,
                uint32_t N, const int32_t *bias, const int32_t *m, const uint32_t *s, int uniform, int small,
                const uint8_t *codes, const Op *o) {
    const uint32_t G = K / 4, Kt = K % 4, Gt = G + (Kt != 0), cw = 8u * GA_J;
    if ((size_t)cw * 4 * Gt > GEMM_BUF) return 0;
    if (!np) return 1;
    __attribute__((aligned(32))) uint8_t Wp[GEMM_BUF];
    int32_t tail[GEMM_CHUNK], sx[GEMM_CHUNK];
    __attribute__((aligned(32))) int8_t rows[GEMM_BUF]; /* the codes of one channel block, decoded (cw*K <= GEMM_BUF) */
    for (uint32_t p = 0; p < np; p++) {                 /* sum(x_p), exact in int32 */
        const int8_t *x = xs[p];
        uint32_t k = 0;
        int32_t sum = 0;
        tail[p] = Kt ? ld_tail(x + 4 * G, Kt) : 0;
        if (small) {
            __m256i acc = _mm256_setzero_si256();
            const __m256i one = _mm256_set1_epi16(1);
            for (; k + 16 <= K; k += 16)
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + k))), one));
            __m128i t = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            t = _mm_add_epi32(t, _mm_srli_si128(t, 8));
            t = _mm_add_epi32(t, _mm_srli_si128(t, 4));
            sum = _mm_cvtsi128_si32(t);
            for (; k < K; k++) sum += x[k];
        }
        sx[p] = sum;
    }
    for (uint32_t oc0 = 0; oc0 < N; oc0 += cw) {
        const uint32_t nb = N - oc0 < cw ? N - oc0 : cw;
        int32_t bv[8 * GA_J], mv[8 * GA_J];
        uint32_t sv[8 * GA_J];
        const int8_t *wb = w;
        uint32_t ob = oc0;
#ifdef GUST_PACKED_WEIGHTS
        if (codes) {
            packed_expand(codes, (uint64_t)oc0 * wrow, nb * wrow, o->p1, rows);
            wb = rows;
            ob = 0;
        }
#endif
        uint32_t wmax = gemm_wmax(wb, wrow, K, ob, nb);
        wmax = wmax < 1 ? 1 : wmax;
        const uint32_t off = small ? wmax : 0; /* u8 = w+off in [0,2*off] */
        gemm_pack(Wp, wb, wrow, K, ob, nb, cw, 0, (uint8_t)off);
        for (uint32_t j = 0; j < cw; j++) {
            bv[j] = j < nb ? (uniform ? 0 : bias[oc0 + j]) : 0;
            mv[j] = j < nb ? m[uniform ? 0 : oc0 + j] : 1;
            sv[j] = j < nb ? s[uniform ? 0 : oc0 + j] : 0;
        }
        __m128i bc[2 * GA_J];
        __m256d f[2 * GA_J];
        for (int j = 0; j < 2 * GA_J; j++) {
            bc[j] = _mm_loadu_si128((const __m128i *)(bv + 4 * j));
            f[j] = avx_qf(mv + 4 * j, sv + 4 * j);
        }
        uint32_t KD = 32767u / (off ? 4u * 128u * off : 2u * 127u * wmax);
        KD = KD < 1 ? 1 : KD;
        const int8_t *pf = w + (size_t)(oc0 + cw) * wrow;
        const size_t pfn = !codes && oc0 + cw < N ? (size_t)(N - oc0 - cw < cw ? N - oc0 - cw : cw) * wrow
                                                  : 0; /* rows are contiguous */
        if (off)
            ga_block(xs, ys, tail, sx, np, Wp, G, Kt, KD, 1, off, bc, f, oc0, nb, pf, pfn, o);
        else
            ga_block(xs, ys, tail, sx, np, Wp, G, Kt, KD, 0, 0, bc, f, oc0, nb, pf, pfn, o);
    }
    return 1;
}
#define GEMM 1
#endif
#ifdef GEMM
/* gemm() over threads (gust_parallel): tasks of GP_CHANNELS-multiple channel ranges, each the unchanged kernel on its
   range, so every output is computed by one task with the same code and results do not depend on the thread count.
   A range starting at channel a reads byte weights at w + a * wrow, and packed codes at bit a * wrow * bits: a
   multiple of 32 channels keeps that a whole byte for 3- and 4-bit codes. */
#define GP_CHANNELS 32u
#define GP_TASKS    16u
typedef struct {
    const int8_t *const *xs;
    int8_t *const *ys;
    const int8_t *w;
    const int32_t *bias, *m;
    const uint32_t *s;
    const uint8_t *codes;
    const Op *o;
    uint32_t np, wrow, K, N, step;
    int uniform, small;
} GpJob;
static void gp_task(void *opaque, uint32_t item, uint32_t thread) {
    (void)thread;
    const GpJob *j = opaque;
    const uint32_t a = item * j->step, n = j->N - a < j->step ? j->N - a : j->step;
    int8_t *ys[GEMM_CHUNK];
    for (uint32_t p = 0; p < j->np; p++) ys[p] = j->ys[p] + a;
    const size_t at = (size_t)a * j->wrow;
    gemm(j->xs, ys, j->np, j->w ? j->w + at : NULL, j->wrow, j->K, n, j->uniform ? NULL : j->bias + a,
         j->uniform ? j->m : j->m + a, j->uniform ? j->s : j->s + a, j->uniform, j->small,
         j->codes ? j->codes + at * j->o->p1 / 8u : NULL, j->o);
}
static void gemm_par(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, uint32_t wrow, uint32_t K,
                     uint32_t N, const int32_t *bias, const int32_t *m, const uint32_t *s, int uniform, int small,
                     const uint8_t *codes, const Op *o) {
#if defined(GUST_FLOAT32) && defined(GUST_THREADS)
    uint32_t step = (N + GP_TASKS - 1) / GP_TASKS;
    step = (step + GP_CHANNELS - 1) / GP_CHANNELS * GP_CHANNELS;
    if (np >= GEMM_MIN && N > step) {
        GpJob j = {xs, ys, w, bias, m, s, codes, o, np, wrow, K, N, step, uniform, small};
        gust_parallel(gp_task, &j, (N + step - 1) / step);
        return;
    }
#endif
    gemm(xs, ys, np, w, wrow, K, N, bias, m, s, uniform, small, codes, o);
}
#endif
#if defined(INT8_TILE) && defined(__AVX2__) && !defined(GUST_INT8_VNNI)
/* AVX2 int16 tiles for any int8 operands (MATMUL; convolutions with signed input and |w| > 63):
   y_p[oc]=quant(dot(x_p,w_oc)+bias[oc]) as in gemm(). A chunk of channels is packed as int16 pairs
   Wp[k/2][channel][2] (8 channels per vector) and each pixel's pair (x_2i, x_2i+1) is broadcast: vpmaddwd gives
   exact pair sums (|pair| <= 2^15; lanes <= K*2^14, the gust_init bound). P pixels x J blocks of 8 channels. */
#ifndef MA_P
#define MA_P 3 /* pixels per tile */
#define MA_J 3 /* 8-channel blocks per tile */
#endif
#define MA_BUF  49152
#define MA_KMAX 1024        /* longest dot */
#define MA_CMAX (64 * MA_J) /* channels per chunk at most (a multiple of 8*MA_J) */
static inline
    __attribute__((always_inline)) void ma_tile(const int8_t *Wp, uint32_t kp, const int16_t (*Xp)[MA_KMAX + 16],
                                                uint32_t CP, const int J, __m256i a[MA_P][MA_J]) {
    for (int p = 0; p < MA_P; p++)
        for (int j = 0; j < J; j++) a[p][j] = _mm256_setzero_si256();
    for (uint32_t cp = 0; cp < CP; cp++) {
        __m256i w[MA_J];
        for (int j = 0; j < J; j++) w[j] = _mm256_load_si256((const __m256i *)(Wp + ((size_t)cp * kp + 8 * j) * 4));
        for (int p = 0; p < MA_P; p++) {
            const __m256i q = _mm256_set1_epi32(ld32((const int8_t *)(Xp[p] + 2 * cp)));
            for (int j = 0; j < J; j++) a[p][j] = _mm256_add_epi32(a[p][j], _mm256_madd_epi16(w[j], q));
        }
    }
}
/* x row (K bytes) -> int16 (sign-extended), zero up to a multiple of 16. */
static inline void ma_widen(const int8_t *x, uint32_t K, int16_t *d) {
    uint32_t k = 0;
    for (; k + 16 <= K; k += 16)
        _mm256_storeu_si256((__m256i *)(d + k), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + k))));
    if (k < K) {
        int8_t t[16] = {0};
        for (uint32_t i = 0; k + i < K; i++) t[i] = x[k + i];
        _mm256_storeu_si256((__m256i *)(d + k), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)t)));
    }
}
static int gemm16(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, uint32_t wrow, uint32_t K,
                  uint32_t N, const int32_t *bias, const int32_t *m, const uint32_t *s, int uniform, const Op *o) {
    const uint32_t CP = (K + 1) / 2, KB = MA_BUF / (4 * CP) / (8 * MA_J) * (8 * MA_J),
                   KC = KB < MA_CMAX ? KB : MA_CMAX; /* channels per chunk */
    if (K > MA_KMAX || !KC) return 0;
    __attribute__((aligned(32))) int8_t Wp[MA_BUF];
    __attribute__((aligned(32))) int16_t Xp[MA_P][MA_KMAX + 16];
    int16_t row[MA_KMAX + 16];
    __m128i qb[MA_CMAX / 8][2];
    __m256d qf[MA_CMAX / 8][2]; /* per 8-channel block of a chunk */
    const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi);
    for (uint32_t c0 = 0; c0 < N; c0 += KC) {
        const uint32_t kn = N - c0 < KC ? N - c0 : KC, kp = (kn + 7) & ~7u; /* channels, padded to whole blocks */
        for (uint32_t ch = 0; ch < kp; ch++) {                              /* Wp[cp][ch] = (w_2cp, w_2cp+1) */
            if (ch < kn)
                ma_widen(w + (size_t)(c0 + ch) * wrow, K, row);
            else
                for (uint32_t k = 0; k < 2 * CP; k++) row[k] = 0;
            for (uint32_t cp = 0; cp < CP; cp++)
                *(Dword *)(void *)(Wp + ((size_t)cp * kp + ch) * 4) = ld32((const int8_t *)(row + 2 * cp));
        }
        for (uint32_t ch = 0; ch < kp; ch += 8) { /* quant constants of each 8-channel block */
            int32_t bv[8], mv[8];
            uint32_t sv[8];
            for (uint32_t l = 0; l < 8; l++) {
                const uint32_t oc = c0 + ch + l;
                const int in = ch + l < kn;
                bv[l] = in && !uniform ? bias[oc] : 0;
                mv[l] = in ? m[uniform ? 0 : oc] : 1;
                sv[l] = in ? s[uniform ? 0 : oc] : 0;
            }
            qb[ch / 8][0] = _mm_loadu_si128((const __m128i *)bv);
            qb[ch / 8][1] = _mm_loadu_si128((const __m128i *)(bv + 4));
            qf[ch / 8][0] = avx_qf(mv, sv);
            qf[ch / 8][1] = avx_qf(mv + 4, sv + 4);
        }
        for (uint32_t p0 = 0; p0 < np; p0 += MA_P) {
            const uint32_t P = np - p0 < MA_P ? np - p0 : MA_P;
            for (uint32_t p = 0; p < MA_P; p++) ma_widen(xs[p0 + (p < P ? p : 0)], K, Xp[p]);
            for (uint32_t j0 = 0; j0 < kp / 8; j0 += MA_J) {
                const uint32_t J = kp / 8 - j0 < MA_J ? kp / 8 - j0 : MA_J;
                __m256i a[MA_P][MA_J];
#define MA_T(JJ) ma_tile(Wp + j0 * 32, kp, (const int16_t (*)[MA_KMAX + 16]) Xp, CP, JJ, a);
                switch (J) {
                case 1: MA_T(1) break;
                case 2: MA_T(2) break;
                default: MA_T(MA_J)
                }
#undef MA_T
                for (uint32_t j = 0; j < J; j++) {
                    const uint32_t k0 = 8 * (j0 + j), oc = c0 + k0;
                    if (k0 >= kn) break;
                    const __m128i b0 = qb[j0 + j][0], b1 = qb[j0 + j][1];
                    const __m256d f0 = qf[j0 + j][0], f1 = qf[j0 + j][1];
                    for (uint32_t p = 0; p < P; p++) {
                        const __m128i r0 = avx_quant(_mm_add_epi32(_mm256_castsi256_si128(a[p][j]), b0), f0, lo, hi),
                                      r1 = avx_quant(_mm_add_epi32(_mm256_extracti128_si256(a[p][j], 1), b1), f1, lo,
                                                     hi);
                        const __m128i v = _mm_packs_epi16(_mm_packs_epi32(r0, r1), r0);
                        int8_t *d = ys[p0 + p] + oc;
                        if (kn - k0 >= 8)
                            _mm_storel_epi64((__m128i *)d, v);
                        else {
                            int8_t l[16];
                            _mm_storeu_si128((__m128i *)l, v);
                            for (uint32_t i2 = 0; i2 < kn - k0; i2++) d[i2] = l[i2];
                        }
                    }
                }
            }
        }
    }
    return 1;
}
#endif

/* The taps of output pixel p (NULL = padding); returns 1 when all are real, 2 when they are also contiguous in
   memory (one dot of taps * channels bytes), else 0. */
static int conv_patch(const Op *o, const Tensor *a, const Tensor *z, const int8_t *x, uint32_t p,
                      const int8_t **patch) {
    const uint32_t hy = p / z->w, wx = p % z->w, taps = o->kh * o->kw;
    int full = 1;
    for (uint32_t ky = 0; ky < o->kh; ky++)
        for (uint32_t kx = 0; kx < o->kw; kx++) {
            const int64_t iy = (int64_t)hy * o->sy + (int64_t)ky * o->dy - o->py,
                          ix = (int64_t)wx * o->sx + (int64_t)kx * o->dx - o->px;
            const int8_t *t =
                (iy < 0 || ix < 0 || iy >= a->h || ix >= a->w) ? NULL : x + ((uint32_t)iy * a->w + (uint32_t)ix) * a->c;
            patch[ky * o->kw + kx] = t;
            full &= t != NULL;
        }
    if (!full) return 0;
    for (uint32_t k = 1; k < taps; k++)
        if ((uintptr_t)patch[k] - (uintptr_t)patch[0] != (uintptr_t)k * a->c) return 1;
    return 2;
}

/* GUST_CONV / GUST_DEPTHWISE over the first `rows` output rows for N output channels: weights w (N rows of
   kh * kw * C bytes, tap-major; depthwise: kh * kw rows of N), bias / m / s of those channels, output channel 0 of
   pixel p at y + p * ystride. A depthwise call always covers every channel (N = ystride = C). small: every weight
   lies in [-63, 63]. Paths, fastest first: packed GEMM tiles for pixels whose taps are real and contiguous (the
   rest through conv_pixel), VNNI or AVX2 tiles, row-interleaved and grouped depthwise, register tiles, 1x1 dots,
   the general patch walk. All produce quant(bias + sum(input * weight)) exactly. */
static void conv_rows(const Context *c, const Op *o, uint32_t rows, const int8_t *w, const int32_t *bias,
                      const int32_t *m, const uint32_t *s, uint32_t N, int8_t *y, uint32_t ystride, int small) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    const uint32_t npv = rows * z->w, taps = o->kh * o->kw, n = a->c, row = taps * n;
    const int nonneg = (int)((c->nonneg[o->a >> 5] >> (o->a & 31)) & 1u);
    (void)small;
    (void)row;
#ifdef GEMM
#ifdef GUST_INT8_VNNI
    const int gemm_ok = 1;
#else
    const int gemm_ok = nonneg || small;
#endif
    if (o->code == GUST_CONV && npv >= GEMM_MIN && taps <= 64 && gemm_fits(row) && gemm_ok) {
        const int8_t *xs[GEMM_CHUNK];
        int8_t *ys[GEMM_CHUNK];
        for (uint32_t pb = 0; pb < npv; pb += GEMM_CHUNK) {
            uint32_t nf = 0;
            for (uint32_t p = pb; p < pb + GEMM_CHUNK && p < npv; p++) {
                const int8_t *patch[64];
                if (conv_patch(o, a, z, x, p, patch) == 2) {
                    xs[nf] = patch[0];
                    ys[nf++] = y + (size_t)p * ystride;
                } else
                    conv_pixel(patch, taps, n, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
            }
            gemm_par(xs, ys, nf, w, row, row, N, bias, m, s, 0, small, NULL, o);
        }
        return;
    }
#ifndef GUST_INT8_VNNI
    if (o->code == GUST_CONV && npv >= GEMM_MIN && taps <= 64 && row <= MA_KMAX) { /* signed input, |w| > 63 */
        const int8_t *xs[GEMM_CHUNK];
        int8_t *ys[GEMM_CHUNK];
        for (uint32_t pb = 0; pb < npv; pb += GEMM_CHUNK) {
            uint32_t nf = 0;
            for (uint32_t p = pb; p < pb + GEMM_CHUNK && p < npv; p++) {
                const int8_t *patch[64];
                if (conv_patch(o, a, z, x, p, patch) == 2) {
                    xs[nf] = patch[0];
                    ys[nf++] = y + (size_t)p * ystride;
                } else
                    conv_pixel(patch, taps, n, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
            }
            gemm16(xs, ys, nf, w, row, row, N, bias, m, s, 0, o);
        }
        return;
    }
#endif
#endif
#ifdef GEMM128
    if (o->code == GUST_CONV && taps <= 64) {
        const int8_t *xs[NG_CHUNK];
        int8_t *ys[NG_CHUNK];
        for (uint32_t pb = 0; pb < npv; pb += NG_CHUNK) {
            uint32_t nf = 0;
            for (uint32_t p = pb; p < pb + NG_CHUNK && p < npv; p++) {
                const int8_t *patch[64];
                if (conv_patch(o, a, z, x, p, patch) == 2) {
                    xs[nf] = patch[0];
                    ys[nf++] = y + (size_t)p * ystride;
                } else
                    conv_pixel(patch, taps, n, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
            }
            g128_gemm(xs, ys, nf, w, row, row, N, bias, m, s, o);
        }
        return;
    }
#endif
#ifdef GUST_INT8_VNNI
    if (o->code == GUST_DEPTHWISE && a->w == 1 && z->w == 1 && o->kw == 1 && o->sy == 1 && o->dy == 1 &&
        depthwise_rows(x, a->h, rows, o->py, o->kh, w, N, bias, m, s, o, y, !nonneg))
        return;
    if ((o->code == GUST_CONV && taps <= 64) || (o->code == GUST_DEPTHWISE && taps <= 16)) {
        /* Tiles of four pixels x four channels. Pixels go in groups of up to 32 whose patches (~16 KB of input)
           stay in L1 while every 4-channel weight block sweeps the group. Padded taps are NULL. */
        const int sgn = !nonneg;
        uint32_t g = o->code == GUST_DEPTHWISE ? VNNI_GROUP : (16384 / row) & ~3u;
        g = g < 4 ? 4 : g > VNNI_GROUP ? VNNI_GROUP : g;
        const int8_t *xp[VNNI_GROUP][64];
        int full[VNNI_GROUP / 4];
        for (uint32_t pb = 0; pb < npv; pb += g) {
            const uint32_t tiles = (npv - pb < g ? npv - pb + 3 : g) / 4;
            for (uint32_t p = 0; p < 4 * tiles; p++) {
                const uint32_t q = pb + p, hy = q / z->w, wx = q % z->w;
                if (p % 4 == 0) full[p / 4] = 3;
                for (uint32_t ky = 0; ky < o->kh; ky++)
                    for (uint32_t kx = 0; kx < o->kw; kx++) {
                        const int64_t iy = (int64_t)hy * o->sy + (int64_t)ky * o->dy - o->py,
                                      ix = (int64_t)wx * o->sx + (int64_t)kx * o->dx - o->px;
                        const int8_t *t = (q >= npv || iy < 0 || ix < 0 || iy >= a->h || ix >= a->w)
                                              ? NULL
                                              : x + ((uint32_t)iy * a->w + (uint32_t)ix) * a->c;
                        xp[p][ky * o->kw + kx] = t;
                        if (!t) full[p / 4] = 0;
                    }
                for (uint32_t k = 1; k < taps && full[p / 4]; k++) /* 3: taps contiguous */
                    if ((uintptr_t)xp[p][k] - (uintptr_t)xp[p][0] != (uintptr_t)k * n) full[p / 4] = 1;
            }
            if (o->code == GUST_DEPTHWISE) {
                const uint32_t cnt = npv - pb < g ? npv - pb : g;
                int whole[VNNI_GROUP];
                for (uint32_t p = 0; p < cnt; p++) {
                    whole[p] = 1;
                    for (uint32_t k = 0; k < taps; k++) whole[p] &= xp[p][k] != NULL;
                }
#define DEPTHWISE_GROUPS(G)                                                                                            \
    if (sgn)                                                                                                           \
        depthwise_group(xp, whole, cnt, taps, G, w, N, bias, m, s, o, y + pb * N, 1);                                  \
    else                                                                                                               \
        depthwise_group(xp, whole, cnt, taps, G, w, N, bias, m, s, o, y + pb * N, 0);
                switch ((taps + 3) / 4) {
                case 1: DEPTHWISE_GROUPS(1) break;
                case 2: DEPTHWISE_GROUPS(2) break;
                case 3: DEPTHWISE_GROUPS(3) break;
                default: DEPTHWISE_GROUPS(4)
                }
#undef DEPTHWISE_GROUPS
                continue;
            }
            uint32_t oc = 0;
            for (; oc + 4 <= N; oc += 4) {
                const Quant16 q4 = vnni_qprep(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(bias + oc))),
                                              _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(m + oc))),
                                              _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(s + oc))));
                for (uint32_t ti = 0; ti < tiles; ti++) {
                    const int8_t *const(*xt)[64] = xp + 4 * ti;
                    const uint32_t p0 = pb + 4 * ti;
                    /* contiguous taps (and their weights, w + j * row + k * n) form one dot of length row */
                    const uint32_t tk = full[ti] == 3 ? 1 : taps, tn = full[ti] == 3 ? row : n;
                    __m512i t = !full[ti] ? vnni_edge(xt, taps, w + oc * row, row, n, sgn)
                                : sgn     ? vnni_tile(xt, tk, w + oc * row, row, tn, 1)
                                          : vnni_tile(xt, tk, w + oc * row, row, tn, 0);
                    __m128i v = vnni_qapply(t, &q4, o->lo, o->hi);
                    int8_t *yo = y + (size_t)p0 * ystride + oc;
                    _mm_storeu_si32(yo, v);
                    if (p0 + 1 < npv) _mm_storeu_si32(yo + ystride, _mm_srli_si128(v, 4));
                    if (p0 + 2 < npv) _mm_storeu_si32(yo + 2 * ystride, _mm_srli_si128(v, 8));
                    if (p0 + 3 < npv) _mm_storeu_si32(yo + 3 * ystride, _mm_srli_si128(v, 12));
                }
            }
            for (; oc < N; oc++)
                for (uint32_t p = 0; p < 4 * tiles && pb + p < npv; p++) {
                    int32_t acc = bias[oc];
                    for (uint32_t k = 0; k < taps; k++)
                        if (xp[p][k]) acc += dot(xp[p][k], w + oc * row + k * n, n);
                    y[(size_t)(pb + p) * ystride + oc] = quant(acc, m[oc], s[oc], o);
                }
        }
        return;
    }
#endif
#if defined(INT8_TILE) && defined(__AVX2__) && !defined(GUST_INT8_VNNI)
    if (o->code == GUST_DEPTHWISE && a->w == 1 && z->w == 1 && o->kw == 1 && o->sy == 1 && o->dy == 1 &&
        depthwise_rows(x, a->h, rows, o->py, o->kh, w, N, bias, m, s, o, y))
        return;
#endif
#ifdef DW_GROUP
    if (o->code == GUST_DEPTHWISE && taps <= 64) { /* groups of 32 pixels share each chunk's interleaved weights */
        const int8_t *xp[32][64];
        for (uint32_t pb = 0; pb < npv; pb += 32) {
            const uint32_t cnt = npv - pb < 32 ? npv - pb : 32;
            for (uint32_t p = 0; p < cnt; p++) {
                for (uint32_t k = taps; k < ((taps + 3) & ~3u); k++) xp[p][k] = NULL;
                conv_patch(o, a, z, x, pb + p, xp[p]);
            }
            DW_GROUP((const int8_t *const(*)[64])xp, cnt, taps, w, N, bias, m, s, o, y + pb * N, nonneg);
        }
        return;
    }
#endif
#ifdef INT8_TILE
    if (o->code == GUST_CONV && n % TILE_L == 0 && taps <= 64) {
        /* Groups of pixels (~16 KB of patches) stay in L1 while each TILE_C-channel weight block sweeps them in
           TILE_P x TILE_C register tiles. Pixels whose taps are padded or not contiguous take conv_pixel. */
        const uint32_t ct = N - N % TILE_C;
        uint32_t g = TILE_G / row / TILE_P * TILE_P;
        g = g < TILE_P ? TILE_P : g > 32 ? 32 : g;
        const int8_t *xs[32];
        uint32_t ps[32];
        for (uint32_t pb = 0; pb < npv; pb += g) {
            uint32_t nf = 0;
            for (uint32_t p = pb; p < pb + g && p < npv; p++) {
                const int8_t *patch[64];
                if (conv_patch(o, a, z, x, p, patch) == 2) {
                    xs[nf] = patch[0];
                    ps[nf++] = p;
                } else
                    conv_pixel(patch, taps, n, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
            }
            T_LIMITS;
            for (uint32_t oc = 0; oc < ct; oc += TILE_C) {
                Lanes4 b[TILE_C];
                Scale4 f[TILE_C];
                for (uint32_t j = 0; j < TILE_C; j++) {
                    b[j] = T_BIAS(bias[oc + j]);
                    f[j] = T_SCALE(m[oc + j], s[oc + j]);
                }
                for (uint32_t i = 0; i < nf; i += TILE_P) {
                    const uint32_t P = nf - i < TILE_P ? nf - i : TILE_P;
                    Lanes4 t[TILE_C], r[4];
                    int8_t l[16];
                    const int8_t *xt[TILE_P];
                    for (uint32_t p = 0; p < TILE_P; p++) xt[p] = xs[i + (p < P ? p : 0)]; /* short tiles repeat */
                    tile_dot(xt, w + oc * row, row, row, nonneg, t);
                    for (uint32_t j = 0; j < 4; j++) r[j] = j < TILE_C ? T_QUANT(t[j], b[j], f[j]) : r[0];
                    T_STORE16(l, r); /* l[4j+p] */
                    for (uint32_t p = 0; p < P; p++)
                        for (uint32_t j = 0; j < TILE_C; j++) y[(size_t)ps[i + p] * ystride + oc + j] = l[4 * j + p];
                }
            }
            for (uint32_t oc = ct; oc < N; oc++)
                for (uint32_t f = 0; f < nf; f++)
                    y[(size_t)ps[f] * ystride + oc] = quant(bias[oc] + dot(xs[f], w + oc * row, row), m[oc], s[oc], o);
        }
        return;
    }
#endif
    if (o->code == GUST_CONV && o->kh == 1 && o->kw == 1 && o->sy == 1 && o->sx == 1 && o->py == 0 && o->px == 0) {
        uint32_t p = 0;
#ifdef INT8_SSE2_TILE4
        if (n >= 16 && N >= 4) { /* blocks of 16 pixels stay cached while each 4-channel weight tile sweeps them */
            const uint32_t quads = npv & ~3u, tiled = N & ~3u;
            for (uint32_t block = 0; block < quads; block += 16) {
                const uint32_t end = quads - block < 16 ? quads : block + 16;
                for (uint32_t oc = 0; oc < tiled; oc += 4)
                    for (uint32_t q = block; q < end; q += 4) {
                        int32_t sums[16];
                        dot_tile4(x + (size_t)q * n, w + (size_t)oc * n, n, sums);
                        for (uint32_t r = 0; r < 4; r++)
                            quant4(sums + r * 4, bias + oc, m + oc, s + oc, o, y + (size_t)(q + r) * ystride + oc);
                    }
                for (uint32_t oc = tiled; oc < N; oc++)
                    for (uint32_t q = block; q < end; q++)
                        y[(size_t)q * ystride + oc] =
                            quant(bias[oc] + dot(x + (size_t)q * n, w + oc * n, n), m[oc], s[oc], o);
            }
            p = quads;
        }
#endif
        if (nonneg)
            for (; p + 2 <= npv; p += 2) {
                const int8_t *x0 = x + p * n, *x1 = x0 + n;
                int8_t *y0 = y + (size_t)p * ystride, *y1 = y0 + ystride;
                uint32_t oc = 0;
                for (; oc + 4 <= N; oc += 4) {
                    int32_t r0[4], r1[4];
                    if (!dot4x2(x0, x1, w + oc * n, n, r0, r1)) {
                        dot4(x0, w + oc * n, n, n, 1, r0);
                        dot4(x1, w + oc * n, n, n, 1, r1);
                    }
                    for (uint32_t j = 0; j < 4; j++) {
                        y0[oc + j] = quant(bias[oc + j] + r0[j], m[oc + j], s[oc + j], o);
                        y1[oc + j] = quant(bias[oc + j] + r1[j], m[oc + j], s[oc + j], o);
                    }
                }
                for (; oc < N; oc++) {
                    y0[oc] = quant(bias[oc] + dot(x0, w + oc * n, n), m[oc], s[oc], o);
                    y1[oc] = quant(bias[oc] + dot(x1, w + oc * n, n), m[oc], s[oc], o);
                }
            }
        for (; p < npv; p++) {
            const int8_t *xr = x + p * n;
            int8_t *yr = y + (size_t)p * ystride;
            uint32_t oc = 0;
            for (; oc + 4 <= N; oc += 4) {
                int32_t r[4];
                dot4(xr, w + oc * n, n, n, nonneg, r);
                for (uint32_t j = 0; j < 4; j++) yr[oc + j] = quant(bias[oc + j] + r[j], m[oc + j], s[oc + j], o);
            }
            for (; oc < N; oc++) yr[oc] = quant(bias[oc] + dot(xr, w + oc * n, n), m[oc], s[oc], o);
        }
        return;
    }
    const int8_t *patch[4096];
    for (uint32_t p = 0; p < npv; p++) {
        conv_patch(o, a, z, x, p, patch);
        if (o->code == GUST_DEPTHWISE)
            depthwise_pixel(patch, taps, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
        else
            conv_pixel(patch, taps, n, w, N, bias, m, s, o, y + (size_t)p * ystride, nonneg);
    }
}
