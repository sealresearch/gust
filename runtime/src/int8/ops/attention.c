/* GUST_MATMUL and GUST_ATTEND: exact integer self-attention for sequence
   models laid out as [T,1,C] tensors. Compiled only with GUST_ATTENTION.
   Part of the runtime/gust.c translation unit; see docs/FORMAT.md. */
#ifdef GUST_ATTENTION

/* Keys visible to query row t of a masked MATMUL: min(Tb, floor(t * Tb / Ta) + p0 + 1). */
static uint32_t live_keys(const Op *o, uint32_t t, uint32_t queries, uint32_t keys) {
    uint64_t k = (uint64_t)t * keys / queries + o->p0 + 1;
    return !o->p1 || k >= keys ? keys : (uint32_t)k;
}

/* GUST_MATMUL: y[t][s] = quant(dot(x_t, b_s)) with one multiplier/shift and no bias; keys s >= live_keys(t) are
   -128 instead (gust_init keeps lo >= -127 then, so -128 only means masked). b plays the 1x1 convolution's
   weights; blocks of 4 queries stop at their last live key. Keys s >= kr (the real rows of b under gust_run_valid)
   are zero, so their live scores are quant(0) = clamp(0) without a dot product. */
static void matmul(const Context *c, const Op *o, const int8_t *x, const int8_t *b, int8_t *y, uint32_t rows,
                   uint32_t kr) { /* query rows < rows */
    const uint32_t Ta = c->t[o->a].h, Tb = c->t[o->b].h, C = c->t[o->a].c;
    const int8_t z0 = clamp(0, o);
#define LIVE(t) live_keys(o, t, Ta, Tb)
#define REAL(t) (LIVE(t) < kr ? LIVE(t) : kr)
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift);
    const int nonneg = (int)((c->nonneg[o->a >> 5] >> (o->a & 31)) & 1u);
#ifdef GUST_INT8_VNNI
    if (rows >= GEMM_MIN &&
        gemm_fits(C)) { /* keys are the channels: each query chunk runs to its last live key, then masks */
        const int8_t *xs[GEMM_CHUNK];
        int8_t *ys[GEMM_CHUNK];
        for (uint32_t t0 = 0; t0 < rows; t0 += GEMM_CHUNK) {
            const uint32_t cnt = rows - t0 < GEMM_CHUNK ? rows - t0 : GEMM_CHUNK;
            for (uint32_t i = 0; i < cnt; i++) {
                xs[i] = x + (size_t)(t0 + i) * C;
                ys[i] = y + (size_t)(t0 + i) * Tb;
            }
            gemm_par(xs, ys, cnt, b, C, C, REAL(t0 + cnt - 1), NULL, &mul, &sh, 1, 0, NULL, o);
            for (uint32_t i = 0; i < cnt; i++) {
                for (uint32_t j = kr; j < LIVE(t0 + i); j++) ys[i][j] = z0;
                for (uint32_t j = LIVE(t0 + i); j < Tb; j++) ys[i][j] = -128;
            }
        }
        return;
    }
#elif defined(INT8_TILE) && defined(__AVX2__)
    if (rows >= GEMM_MIN && C <= MA_KMAX) { /* keys are the channels, as for VNNI */
        const int8_t *xs[GEMM_CHUNK];
        int8_t *ys[GEMM_CHUNK];
        for (uint32_t t0 = 0; t0 < rows; t0 += GEMM_CHUNK) {
            const uint32_t cnt = rows - t0 < GEMM_CHUNK ? rows - t0 : GEMM_CHUNK;
            for (uint32_t i = 0; i < cnt; i++) {
                xs[i] = x + (size_t)(t0 + i) * C;
                ys[i] = y + (size_t)(t0 + i) * Tb;
            }
            gemm16(xs, ys, cnt, b, C, C, REAL(t0 + cnt - 1), NULL, &mul, &sh, 1, o);
            for (uint32_t i = 0; i < cnt; i++) {
                for (uint32_t j = kr; j < LIVE(t0 + i); j++) ys[i][j] = z0;
                for (uint32_t j = LIVE(t0 + i); j < Tb; j++) ys[i][j] = -128;
            }
        }
        return;
    }
#elif defined(GEMM128)
    if (!o->p1) { /* unmasked: every key is live */
        const int8_t *xs[NG_CHUNK];
        int8_t *ys[NG_CHUNK];
        (void)nonneg;
        for (uint32_t t0 = 0; t0 < rows; t0 += NG_CHUNK) {
            const uint32_t cnt = rows - t0 < NG_CHUNK ? rows - t0 : NG_CHUNK;
            for (uint32_t i = 0; i < cnt; i++) {
                xs[i] = x + (size_t)(t0 + i) * C;
                ys[i] = y + (size_t)(t0 + i) * Tb;
            }
            g128_scores(xs, ys, cnt, b, C, kr, mul, sh, o);
            for (uint32_t i = 0; i < cnt; i++)
                for (uint32_t j = kr; j < Tb; j++) ys[i][j] = z0;
        }
        return;
    }
#endif
    for (uint32_t t0 = 0; t0 < rows; t0 += 4) {
        const uint32_t P = rows - t0 < 4 ? rows - t0 : 4, se = REAL(t0 + P - 1);
        uint32_t k = 0;
#ifdef GUST_INT8_VNNI
        { /* 4 queries x 4 keys per vnni tile (signed queries: x^0x80 correction) */
            const int8_t *xp[4][64];
            int8_t l[16];
            for (uint32_t p = 0; p < 4; p++) xp[p][0] = p < P ? x + (t0 + p) * C : NULL;
            const Quant16 q = vnni_qprep(_mm512_setzero_si512(), _mm512_set1_epi32(mul), _mm512_set1_epi32((int)sh));
            const int8_t *const(*xt)[64] = (const int8_t *const(*)[64])xp;
            for (; k + 4 <= se; k += 4) {
                __m512i r = P < 4    ? vnni_edge(xt, 1, b + k * C, C, C, !nonneg)
                            : nonneg ? vnni_tile(xt, 1, b + k * C, C, C, 0)
                                     : vnni_tile(xt, 1, b + k * C, C, C, 1);
                _mm_storeu_si128((__m128i *)l, vnni_qapply(r, &q, o->lo, o->hi)); /* l[4p+j] */
                for (uint32_t p = 0; p < P; p++)
                    for (uint32_t j = 0; j < 4; j++) y[(t0 + p) * Tb + k + j] = l[4 * p + j];
            }
        }
#elif defined(INT8_TILE)
        if (C % TILE_L == 0) { /* TILE_P queries x TILE_C keys per register tile */
            T_LIMITS;
            const Lanes4 zb = T_BIAS(0);
            const Scale4 f = T_SCALE(mul, sh);
            const int8_t *xt[TILE_P];
            for (uint32_t p = 0; p < TILE_P; p++) xt[p] = x + (t0 + (p < P ? p : 0)) * C;
            for (; k + TILE_C <= se; k += TILE_C) {
                Lanes4 tt[TILE_C], r[4];
                int8_t l[16];
                tile_dot(xt, b + k * C, C, C, nonneg, tt);
                for (uint32_t j = 0; j < 4; j++) r[j] = j < TILE_C ? T_QUANT(tt[j], zb, f) : r[0];
                T_STORE16(l, r); /* l[4j+p] */
                for (uint32_t p = 0; p < P; p++)
                    for (uint32_t j = 0; j < TILE_C; j++) y[(t0 + p) * Tb + k + j] = l[4 * j + p];
            }
        }
#endif
        for (uint32_t p = 0; p < P; p++) {
            const int8_t *xr = x + (t0 + p) * C;
            int8_t *yr = y + (t0 + p) * Tb;
            uint32_t j = k;
            for (; j + 4 <= se; j += 4) {
                int32_t r[4];
                dot4(xr, b + j * C, C, C, nonneg, r);
                for (uint32_t q = 0; q < 4; q++) yr[j + q] = quant(r[q], mul, sh, o);
            }
            for (; j < se; j++) yr[j] = quant(dot(xr, b + j * C, C), mul, sh, o);
            for (j = se; j < LIVE(t0 + p); j++) yr[j] = z0;
            for (j = LIVE(t0 + p); j < Tb; j++) yr[j] = -128;
        }
    }
#undef LIVE
#undef REAL
}
/* acc[j]=sum_k w[k]*b[k*C+j] for j<n<=64, k<cnt<=256: |w*b| <= 2^15*2^7, so
   int32 is exact (<= 2^30). Vector paths take keys in pairs: -w fits int16
   (w <= 2^15), so the int16 dot of (-w_k,-w_k+1) with (b_k,b_k+1) is exactly
   -(w_k*b_k+w_k+1*b_k+1) (pmaddwd / vpdpwssd / i32x4.dot); an odd last key
   pairs with itself at weight 0 and pairs of zero weights are skipped. Lanes
   span channels. */
static void attend_acc(const int8_t *b, uint32_t C, uint32_t n, const uint32_t *w, uint32_t cnt, int32_t acc[64]) {
    uint32_t j = 0;
#if defined(GUST_INT8_VNNI) || defined(__wasm_simd128__) || defined(__SSE2__)
    int32_t pw[128];
    const int8_t *r0[128], *r1[128];
    uint32_t np = 0;
    for (uint32_t k = 0; k < cnt; k += 2) {
        const uint32_t w1 = k + 1 < cnt ? w[k + 1] : 0;
        if (!(w[k] | w1)) continue;
        pw[np] = (int32_t)((uint32_t)(uint16_t)(0u - w[k]) | (uint32_t)(uint16_t)(0u - w1) << 16);
        r0[np] = b + k * C;
        r1[np] = b + (k + 1 < cnt ? k + 1 : k) * C;
        np++;
    }
#endif
#ifdef GUST_INT8_VNNI
    const __mmask64 t = ~0ull >> (64 - n);
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (uint32_t p = 0; p < np; p++) {
        const __m512i x = _mm512_maskz_loadu_epi8(t, r0[p]), y = _mm512_maskz_loadu_epi8(t, r1[p]),
                      q = _mm512_set1_epi32(pw[p]);
        const __m512i x0 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(x)),
                      y0 = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(y));
        const __m512i x1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x, 1)),
                      y1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(y, 1));
        a0 = _mm512_dpwssd_epi32(a0, _mm512_unpacklo_epi16(x0, y0), q);
        a1 = _mm512_dpwssd_epi32(a1, _mm512_unpackhi_epi16(x0, y0), q);
        a2 = _mm512_dpwssd_epi32(a2, _mm512_unpacklo_epi16(x1, y1), q);
        a3 = _mm512_dpwssd_epi32(a3, _mm512_unpackhi_epi16(x1, y1), q);
    } /* 128-bit lane L of a0/a1: channels 8L+0..3 / 8L+4..7 (a2,a3: +32); negate back in channel order */
    const __m512i lo = _mm512_set_epi32(23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0),
                  hi = _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8),
                  z = _mm512_setzero_si512();
    _mm512_storeu_si512(acc, _mm512_sub_epi32(z, _mm512_permutex2var_epi32(a0, lo, a1)));
    _mm512_storeu_si512(acc + 16, _mm512_sub_epi32(z, _mm512_permutex2var_epi32(a0, hi, a1)));
    _mm512_storeu_si512(acc + 32, _mm512_sub_epi32(z, _mm512_permutex2var_epi32(a2, lo, a3)));
    _mm512_storeu_si512(acc + 48, _mm512_sub_epi32(z, _mm512_permutex2var_epi32(a2, hi, a3)));
    return;
#elif defined(__AVX2__)
    const __m256i z = _mm256_setzero_si256();
#define PAIR_ACC(A, B, O)                                                                                              \
    {                                                                                                                  \
        const __m256i x = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(r0[p] + j + (O)))),                   \
                      y = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(r1[p] + j + (O))));                   \
        A = _mm256_add_epi32(A, _mm256_madd_epi16(_mm256_unpacklo_epi16(x, y), q));                                    \
        B = _mm256_add_epi32(B, _mm256_madd_epi16(_mm256_unpackhi_epi16(x, y), q));                                    \
    }
#define PAIR_OUT(A, B, O)                                                                                              \
    _mm256_storeu_si256((__m256i *)(acc + j + (O)), _mm256_sub_epi32(z, _mm256_permute2x128_si256(A, B, 0x20)));       \
    _mm256_storeu_si256(                                                                                               \
        (__m256i *)(acc + j + (O) + 8),                                                                                \
        _mm256_sub_epi32(z, _mm256_permute2x128_si256(A, B, 0x31))); /* halves: channels 0..3|8..11, 4..7|12..15 */
    for (; j + 32 <= n; j += 32) {
        __m256i a0 = z, a1 = z, a2 = z, a3 = z;
        for (uint32_t p = 0; p < np; p++) {
            const __m256i q = _mm256_set1_epi32(pw[p]);
            PAIR_ACC(a0, a1, 0) PAIR_ACC(a2, a3, 16)
        }
        PAIR_OUT(a0, a1, 0) PAIR_OUT(a2, a3, 16)
    }
    for (; j + 16 <= n; j += 16) {
        __m256i a0 = z, a1 = z;
        for (uint32_t p = 0; p < np; p++) {
            const __m256i q = _mm256_set1_epi32(pw[p]);
            PAIR_ACC(a0, a1, 0)
        }
        PAIR_OUT(a0, a1, 0)
    }
#undef PAIR_ACC
#undef PAIR_OUT
#elif defined(__wasm_simd128__)
    const v128_t z = wasm_i32x4_splat(0);
    for (; j + 16 <= n; j += 16) {
        v128_t a0 = z, a1 = z, a2 = z, a3 = z;
        for (uint32_t p = 0; p < np; p++) {
            const v128_t x = wasm_v128_load(r0[p] + j), y = wasm_v128_load(r1[p] + j), q = wasm_i32x4_splat(pw[p]);
            const v128_t l = wasm_i8x16_shuffle(x, y, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
                         h = wasm_i8x16_shuffle(x, y, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            a0 = wasm_i32x4_add(a0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(l), q));
            a1 = wasm_i32x4_add(a1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(l), q));
            a2 = wasm_i32x4_add(a2, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(h), q));
            a3 = wasm_i32x4_add(a3, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(h), q));
        }
        wasm_v128_store(acc + j, wasm_i32x4_sub(z, a0));
        wasm_v128_store(acc + j + 4, wasm_i32x4_sub(z, a1));
        wasm_v128_store(acc + j + 8, wasm_i32x4_sub(z, a2));
        wasm_v128_store(acc + j + 12, wasm_i32x4_sub(z, a3));
    }
#elif defined(__SSE2__)
    const __m128i z = _mm_setzero_si128();
    for (; j + 16 <= n; j += 16) {
        __m128i a0 = z, a1 = z, a2 = z, a3 = z;
        for (uint32_t p = 0; p < np; p++) {
            const __m128i x = _mm_loadu_si128((const __m128i *)(r0[p] + j)),
                          y = _mm_loadu_si128((const __m128i *)(r1[p] + j)), q = _mm_set1_epi32(pw[p]);
            const __m128i l = _mm_unpacklo_epi8(x, y), h = _mm_unpackhi_epi8(x, y), ls = _mm_cmpgt_epi8(z, l),
                          hs = _mm_cmpgt_epi8(z, h);
            a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_unpacklo_epi8(l, ls), q));
            a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_unpackhi_epi8(l, ls), q));
            a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_unpacklo_epi8(h, hs), q));
            a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_unpackhi_epi8(h, hs), q));
        }
        _mm_storeu_si128((__m128i *)(acc + j), _mm_sub_epi32(z, a0));
        _mm_storeu_si128((__m128i *)(acc + j + 4), _mm_sub_epi32(z, a1));
        _mm_storeu_si128((__m128i *)(acc + j + 8), _mm_sub_epi32(z, a2));
        _mm_storeu_si128((__m128i *)(acc + j + 12), _mm_sub_epi32(z, a3));
    }
#endif
    if (j < n) { /* remaining channels key by key (rows stream contiguously; zero weights skipped) */
        for (uint32_t i = j; i < n; i++) acc[i] = 0;
        for (uint32_t k = 0; k < cnt; k++) {
            if (!w[k]) continue;
            const int32_t wk = (int32_t)w[k];
            const int8_t *r = b + (size_t)k * C;
            for (uint32_t i = j; i < n; i++) acc[i] += wk * r[i];
        }
    }
}
/* y=clamp(round(round(num*2^16/den)*mul/2^(sh+16))), both roundings half away
   from zero; den=0 (every key masked) gives clamp(0). |num| <= 128*den and
   den <= 2^31: |num|*2^16 < 2^63, quotient <= 2^23, quotient*mul < 2^54. The
   quotient floor(m/den), m=|num|*2^16+floor(den/2) < 2^55, starts from the f64
   estimate m*(1/den) (within 1, as den >= 2^15) and is corrected exactly. */
static int8_t attend_out(int64_t num, uint64_t den, double inv, int32_t mul, uint32_t sh, const Op *o) {
    if (!den) return clamp(0, o);
    const uint64_t n = (uint64_t)(num < 0 ? -num : num), m = (n << 16) + (den >> 1);
    uint64_t q = (uint64_t)((double)m * inv);
    while (q * den > m) q--;
    while (m - q * den >= den) q++;
    const uint64_t r = (q * (uint32_t)mul + ((uint64_t)1 << (sh + 15))) >> (sh + 16);
    return clamp(num < 0 ? -(int64_t)r : (int64_t)r, o);
}
/* GUST_ATTEND row t: top=max live a[t][s] (p1: a==-128 is dead), w_s=lut[top-a]
   (dead: 0), den=sum w_s, num_c=sum w_s*b[s][c] exactly (int32 per 256 keys,
   then int64), y[t][c]=attend_out(num_c,den). No probability is quantized.
   Dead keys are -128, the minimum, so the plain row max is the live max
   whenever any key is live. */
static void attend(const Context *c, const Op *o, const int8_t *x, const int8_t *b, int8_t *y, uint32_t rows,
                   uint32_t keys) { /* rows < rows; keys >= keys weigh 0 */
    const uint32_t S = c->t[o->a].c, C = c->t[o->b].c, *lut = (const uint32_t *)(c->blob + o->weights);
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift);
    for (uint32_t t = 0; t < rows; t++) {
        const int8_t *row = x + t * S;
        int8_t *yr = y + t * C;
        int32_t top = -128;
        uint64_t den = 0;
        double inv = 0;
        for (uint32_t k = 0; k < keys; k++) top = row[k] > top ? row[k] : top;
        const int dead = o->p1 && top == -128;
        for (uint32_t c0 = 0; c0 < C; c0 += 64) {
            const uint32_t n = C - c0 < 64 ? C - c0 : 64;
            int64_t num[64];
            int32_t acc[64];
            uint32_t w[256];
            for (uint32_t j = 0; j < n; j++) num[j] = 0;
            for (uint32_t s0 = 0; !dead && s0 < keys; s0 += 256) {
                const uint32_t cnt = keys - s0 < 256 ? keys - s0 : 256;
                for (uint32_t k = 0; k < cnt; k++) {
                    const int8_t v = row[s0 + k];
                    w[k] = o->p1 && v == -128 ? 0 : lut[top - v];
                }
                if (!c0)
                    for (uint32_t k = 0; k < cnt; k++) den += w[k];
                attend_acc(b + s0 * C + c0, C, n, w, cnt, acc);
                for (uint32_t j = 0; j < n; j++) num[j] += acc[j];
            }
            if (!c0 && den) inv = 1.0 / (double)den;
            for (uint32_t j = 0; j < n; j++) yr[c0 + j] = attend_out(num[j], den, inv, mul, sh, o);
        }
    }
}
/* Vector ATTEND, same integers as attend(): per query row, top and the weights w_s=lut[top-a_s] (0 for dead
   and past-the-end keys; every lut entry <= 2^15 fits uint16), den=sum w, num_c=sum w_s*v[s][c] exactly, then
   attend_out(num_c,den) evaluated in f64 with every intermediate an exact integer (att_out below). Taken for
   keys <= 4096 (and V within the packing buffer); larger cases use attend(). */
#if defined(GUST_INT8_VNNI) || (defined(INT8_TILE) && defined(__AVX2__))
#define ATT_BUF  49152 /* bytes of packed V */
#define ATT_KEYS 4096
/* attend_out in f64 for den in [1, 2^27] (keys <= 4096): n=|num| <= 128*den, m=n*2^16+floor(den/2) < 2^51 and
   q=floor(m/den) <= 2^23+1. The estimate e=m*(1/den) has relative error <= 2^-52, |e-m/den| <= 2^-28 < 1/den: a
   quotient k-eps below an integer has eps >= 1/den, so floor(e) is never above q, and at most one below (an exact
   quotient); one exact test m-floor(e)*den >= den (products < 2^53) fixes it. Then
   r=floor((q*mul+2^(sh+15))/2^(sh+16)) = floor((q*mh+floor((q*ml+2^(sh+15))/2^16))/2^sh) with mul=mh*2^16+ml:
   products < 2^40, sums < 2^46 (fused or not, every value is exact), and power-of-two scaling and floor are
   exact. y=clamp(sign(num)*r). */
#endif
#ifdef GUST_INT8_VNNI
typedef struct {
    __m512d den, inv, half, mh, ml, rnd, s16, ssh, lo, hi;
} AttK;
static inline __m256i att_out8(__m512d num, const AttK *k) { /* ml, rnd hold ml*2^-16, 2^(sh-1) */
    const __m512d n = _mm512_abs_pd(num), m = _mm512_fmadd_pd(n, _mm512_set1_pd(65536.0), k->half);
    __m512d q = _mm512_roundscale_pd(_mm512_mul_pd(m, k->inv), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    q = _mm512_mask_add_pd(q, _mm512_cmp_pd_mask(_mm512_fnmadd_pd(q, k->den, m), k->den, _CMP_GE_OQ), q,
                           _mm512_set1_pd(1.0));
    const __m512d b1 =
        _mm512_roundscale_pd(_mm512_fmadd_pd(q, k->ml, k->rnd), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m512d r = _mm512_roundscale_pd(_mm512_mul_pd(_mm512_fmadd_pd(q, k->mh, b1), k->ssh),
                                     _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    r = _mm512_mask_sub_pd(r, _mm512_cmp_pd_mask(num, _mm512_setzero_pd(), _CMP_LT_OQ), _mm512_setzero_pd(), r);
    return _mm512_cvttpd_epi32(_mm512_min_pd(_mm512_max_pd(r, k->lo), k->hi));
}
static inline int hmax_epi8(__m512i v) {
    const __m256i a = _mm256_max_epi8(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
    __m128i b = _mm_max_epi8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    b = _mm_max_epi8(b, _mm_srli_si128(b, 8));
    b = _mm_max_epi8(b, _mm_srli_si128(b, 4));
    b = _mm_max_epi8(b, _mm_srli_si128(b, 2));
    b = _mm_max_epi8(b, _mm_srli_si128(b, 1));
    return (int8_t)_mm_cvtsi128_si32(b);
}
/* Weights of one query row as low/high byte planes wl/wh (zero past keys, written up to a multiple of 32);
   returns den, 0 for a dead row. lut[d] for d = top-a in [0,255]: four 64-entry vpermi2w lookups selected
   by bits 6 and 7 of d. */
static inline uint32_t att_weights(const int8_t *row, uint32_t keys, uint32_t p1, const __m512i *T, uint8_t *wl,
                                   uint8_t *wh) {
    __m512i mx = _mm512_set1_epi8(-128);
    uint32_t k = 0;
    for (; k + 64 <= keys; k += 64) mx = _mm512_max_epi8(mx, _mm512_loadu_si512(row + k));
    if (k < keys)
        mx = _mm512_max_epi8(mx, _mm512_mask_loadu_epi8(_mm512_set1_epi8(-128), ~0ull >> (64 - (keys - k)), row + k));
    const int top = hmax_epi8(mx);
    if (p1 && top == -128) return 0;
    const __m512i vt = _mm512_set1_epi16((short)top), b6 = _mm512_set1_epi16(64), b7 = _mm512_set1_epi16(128),
                  dead = _mm512_set1_epi16(-128);
    __m512i ds = _mm512_setzero_si512();
    for (k = 0; k < keys; k += 32) {
        const __mmask32 kv = keys - k >= 32 ? ~0u : (1u << (keys - k)) - 1;
        const __m512i a = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(kv, row + k)), d = _mm512_sub_epi16(vt, a);
        const __mmask32 live = p1 ? kv & ~_mm512_cmpeq_epi16_mask(a, dead) : kv, s6 = _mm512_test_epi16_mask(d, b6);
        const __m512i r0 = _mm512_mask_blend_epi16(s6, _mm512_permutex2var_epi16(T[0], d, T[1]),
                                                   _mm512_permutex2var_epi16(T[2], d, T[3]));
        const __m512i r1 = _mm512_mask_blend_epi16(s6, _mm512_permutex2var_epi16(T[4], d, T[5]),
                                                   _mm512_permutex2var_epi16(T[6], d, T[7]));
        const __m512i w = _mm512_maskz_mov_epi16(live, _mm512_mask_blend_epi16(_mm512_test_epi16_mask(d, b7), r0, r1));
        const __m256i l = _mm512_cvtepi16_epi8(w), h = _mm512_cvtepi16_epi8(_mm512_srli_epi16(w, 8));
        _mm256_storeu_si256((__m256i *)(wl + k), l);
        _mm256_storeu_si256((__m256i *)(wh + k), h);
        ds = _mm512_add_epi64(ds, _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_sad_epu8(l, _mm256_setzero_si256())),
                                                     _mm256_sad_epu8(h, _mm256_setzero_si256()), 1));
    }
    const __m256i lo4 = _mm512_castsi512_si256(ds),
                  hi4 = _mm512_extracti64x4_epi64(ds, 1); /* den = sum lo + 256 sum hi */
    return (uint32_t)(_mm256_extract_epi64(lo4, 0) + _mm256_extract_epi64(lo4, 1) + _mm256_extract_epi64(lo4, 2) +
                      _mm256_extract_epi64(lo4, 3) +
                      256 * (_mm256_extract_epi64(hi4, 0) + _mm256_extract_epi64(hi4, 1) +
                             _mm256_extract_epi64(hi4, 2) + _mm256_extract_epi64(hi4, 3)));
}
/* Two queries x JN 16-channel blocks: vpdpbusd(weight bytes broadcast as u8, Vp as s8) per byte plane;
   lanes <= 255*128*keys < 2^31. */
static inline __attribute__((always_inline)) void att_block(const int8_t *Vp, uint32_t G, const uint8_t *const *wl,
                                                            const uint8_t *const *wh, const int JN, const AttK *k,
                                                            uint32_t nq, int8_t *const *yq, uint32_t c0, uint32_t C) {
    __m512i al[2][5], ah[2][5];
    for (int q = 0; q < 2; q++)
        for (int j = 0; j < JN; j++) al[q][j] = ah[q][j] = _mm512_setzero_si512();
    for (uint32_t g = 0; g < G; g++) {
        const __m512i l0 = _mm512_set1_epi32(ld32((const int8_t *)wl[0] + 4 * g)),
                      h0 = _mm512_set1_epi32(ld32((const int8_t *)wh[0] + 4 * g));
        const __m512i l1 = _mm512_set1_epi32(ld32((const int8_t *)wl[1] + 4 * g)),
                      h1 = _mm512_set1_epi32(ld32((const int8_t *)wh[1] + 4 * g));
        const int8_t *vg = Vp + (size_t)g * 64 * JN;
        for (int j = 0; j < JN; j++) {
            const __m512i v = _mm512_load_si512(vg + 64 * j);
            al[0][j] = _mm512_dpbusd_epi32(al[0][j], l0, v);
            ah[0][j] = _mm512_dpbusd_epi32(ah[0][j], h0, v);
            al[1][j] = _mm512_dpbusd_epi32(al[1][j], l1, v);
            ah[1][j] = _mm512_dpbusd_epi32(ah[1][j], h1, v);
        }
    }
    const __m512d s8 = _mm512_set1_pd(256.0);
    for (uint32_t q = 0; q < nq; q++)
        for (int j = 0; j < JN; j++) {
            const uint32_t c = c0 + 16 * j;
            if (c >= C) break;
            const __m512d n0 = _mm512_add_pd(_mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(ah[q][j])), s8),
                                             _mm512_cvtepi32_pd(_mm512_castsi512_si256(al[q][j])));
            const __m512d n1 =
                _mm512_add_pd(_mm512_mul_pd(_mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(ah[q][j], 1)), s8),
                              _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(al[q][j], 1)));
            const __m512i r = _mm512_inserti64x4(_mm512_castsi256_si512(att_out8(n0, k + q)), att_out8(n1, k + q), 1);
            const __mmask16 u = C - c >= 16 ? 0xFFFF : (__mmask16)((1u << (C - c)) - 1);
            _mm_mask_storeu_epi8(yq[q] + c, u, _mm512_cvtepi32_epi8(r));
        }
}
static int attend_fast(const Context *c, const Op *o, const int8_t *x, const int8_t *b, int8_t *y, uint32_t rows,
                       uint32_t keys) {
    const uint32_t S = c->t[o->a].c, C = c->t[o->b].c, *lut = (const uint32_t *)(c->blob + o->weights),
                   G = (keys + 3) / 4, cb = (C + 15) / 16;
    if (!keys || keys > ATT_KEYS || 64u * G > ATT_BUF) return 0;
    uint32_t nb = ATT_BUF / (64u * G);
    nb = nb > 5 ? 5 : nb;
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift);
    __attribute__((aligned(64))) int8_t Vp[ATT_BUF];
    __attribute__((aligned(64))) uint8_t W[4][ATT_KEYS + 64];
    __m512i T[8];
    for (int i = 0; i < 8; i++)
        T[i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_loadu_si512(lut + 32 * i))),
                                  _mm512_cvtepi32_epi16(_mm512_loadu_si512(lut + 32 * i + 16)), 1);
    AttK k[2];
    for (int q = 0; q < 2; q++) {
        k[q].mh = _mm512_set1_pd((double)(mul >> 16));
        k[q].ml = _mm512_set1_pd((double)(mul & 0xFFFF) / 65536.0);
        k[q].rnd = _mm512_set1_pd((double)(1ull << (sh + 15)) / 65536.0);
        k[q].s16 = _mm512_set1_pd(1.0 / 65536.0);
        k[q].ssh = _mm512_set1_pd(pow2neg(sh));
        k[q].lo = _mm512_set1_pd(o->lo);
        k[q].hi = _mm512_set1_pd(o->hi);
    }
    for (uint32_t j0 = 0; j0 < cb; j0 += nb) {
        const uint32_t jn = cb - j0 < nb ? cb - j0 : nb;
        for (uint32_t g = 0; g < G; g++)
            for (uint32_t j = 0; j < jn; j++) { /* Vp[g][j]: channels 16(j0+j)+0..15 x keys 4g+0..3 */
                const uint32_t ch = 16 * (j0 + j), cn = C - ch < 16 ? C - ch : 16;
                const __mmask16 u = (__mmask16)(cn >= 16 ? 0xFFFF : (1u << cn) - 1);
                __m128i r[4];
                for (uint32_t i = 0; i < 4; i++)
                    r[i] = 4 * g + i < keys ? _mm_maskz_loadu_epi8(u, b + (size_t)(4 * g + i) * C + ch)
                                            : _mm_setzero_si128();
                const __m128i a0 = _mm_unpacklo_epi8(r[0], r[1]), a1 = _mm_unpackhi_epi8(r[0], r[1]),
                              a2 = _mm_unpacklo_epi8(r[2], r[3]), a3 = _mm_unpackhi_epi8(r[2], r[3]);
                __m128i *d = (__m128i *)(Vp + ((size_t)g * jn + j) * 64);
                _mm_store_si128(d, _mm_unpacklo_epi16(a0, a2));
                _mm_store_si128(d + 1, _mm_unpackhi_epi16(a0, a2));
                _mm_store_si128(d + 2, _mm_unpacklo_epi16(a1, a3));
                _mm_store_si128(d + 3, _mm_unpackhi_epi16(a1, a3));
            }
        for (uint32_t t0 = 0; t0 < rows; t0 += 2) {
            const uint32_t nq = rows - t0 < 2 ? 1 : 2;
            const uint8_t *wl[2], *wh[2];
            int8_t *yq[2];
            uint32_t live = 0;
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t t = t0 + (q < nq ? q : 0);
                uint32_t den;
                wl[q] = W[2 * q];
                wh[q] = W[2 * q + 1];
                yq[q] = y + (size_t)t * C;
                den = q < nq ? att_weights(x + (size_t)t * S, keys, o->p1, T, W[2 * q], W[2 * q + 1]) : 1;
                if (q < nq && !den) {
                    for (uint32_t ch = 16 * j0; ch < 16 * (j0 + jn) && ch < C; ch++) yq[q][ch] = clamp(0, o);
                    wl[q] = wl[0];
                    wh[q] = wh[0];
                    continue;
                }
                if (q < nq)
                    live |= 1u << q;
                else {
                    wl[q] = wl[0];
                    wh[q] = wh[0];
                }
                k[q].den = _mm512_set1_pd((double)den);
                k[q].inv = _mm512_set1_pd(1.0 / (double)den);
                k[q].half = _mm512_set1_pd((double)(den >> 1));
            }
            if (!live) continue;
            if (live == 2) {
                wl[0] = wl[1];
                wh[0] = wh[1];
                yq[0] = yq[1];
                k[0] = k[1];
            }
            const uint32_t n = live == 3 ? 2 : 1;
#define ATT_JN(JN) att_block(Vp, G, wl, wh, JN, k, n, yq, 16 * j0, C);
            switch (jn) {
            case 1: ATT_JN(1) break;
            case 2: ATT_JN(2) break;
            case 3: ATT_JN(3) break;
            case 4: ATT_JN(4) break;
            default: ATT_JN(5)
            }
#undef ATT_JN
        }
    }
    return 1;
}
#define ATTEND_FAST 1
#elif defined(INT8_TILE) && defined(__AVX2__)
typedef struct {
    __m256d den, inv, half, mh, ml, rnd, s16, ssh, lo, hi;
} AttK;
static inline __m128i att_out4(__m256d num, const AttK *k) {
    const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd(),
                  n = _mm256_andnot_pd(_mm256_set1_pd(-0.0), num);
    const __m256d m = _mm256_add_pd(_mm256_mul_pd(n, _mm256_set1_pd(65536.0)), k->half);
    __m256d q = _mm256_floor_pd(_mm256_mul_pd(m, k->inv));
    q = _mm256_add_pd(
        q, _mm256_and_pd(_mm256_cmp_pd(_mm256_sub_pd(m, _mm256_mul_pd(q, k->den)), k->den, _CMP_GE_OQ), one));
    const __m256d b1 = _mm256_floor_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(q, k->ml), k->rnd), k->s16));
    __m256d r = _mm256_floor_pd(_mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(q, k->mh), b1), k->ssh));
    r = _mm256_blendv_pd(r, _mm256_sub_pd(zero, r), _mm256_cmp_pd(num, zero, _CMP_LT_OQ));
    return _mm256_cvttpd_epi32(_mm256_min_pd(_mm256_max_pd(r, k->lo), k->hi));
}
/* -w as int16 (fits: w <= 2^15) for keys 0..keys-1, zero up to a multiple of 16; returns den (0: dead row). */
static inline uint32_t att_weights(const int8_t *row, uint32_t keys, uint32_t p1, const uint32_t *lut, int16_t *wn) {
    __m256i mx = _mm256_set1_epi8(-128);
    uint32_t k = 0;
    int top = -128;
    for (; k + 32 <= keys; k += 32) mx = _mm256_max_epi8(mx, _mm256_loadu_si256((const __m256i *)(row + k)));
    {
        __m128i v = _mm_max_epi8(_mm256_castsi256_si128(mx), _mm256_extracti128_si256(mx, 1));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 8));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 4));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 2));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 1));
        top = (int8_t)_mm_cvtsi128_si32(v);
    }
    for (; k < keys; k++) top = row[k] > top ? row[k] : top;
    if (p1 && top == -128) return 0;
    const __m256i vt = _mm256_set1_epi32(top), dead = _mm256_set1_epi32(-128),
                  idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), zero = _mm256_setzero_si256();
    __m256i ds = zero;
    for (k = 0; k < keys; k += 16) {
        __m256i n[2];
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t k0 = k + 8 * h;
            int8_t tmp[8];
            const int8_t *src = row + k0;
            if (k0 + 8 > keys) {
                for (uint32_t i = 0; i < 8; i++) tmp[i] = k0 + i < keys ? row[k0 + i] : 0;
                src = tmp;
            }
            const __m256i a = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)src));
            __m256i live = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(keys - k0)), idx); /* k0+i < keys */
            if (p1) live = _mm256_andnot_si256(_mm256_cmpeq_epi32(a, dead), live);
            const __m256i w = _mm256_and_si256(
                live, _mm256_mask_i32gather_epi32(zero, (const int *)lut, _mm256_sub_epi32(vt, a), live, 4));
            ds = _mm256_add_epi32(ds, w);
            n[h] = _mm256_sub_epi32(zero, w);
        }
        _mm256_storeu_si256((__m256i *)(wn + k), _mm256_permute4x64_epi64(_mm256_packs_epi32(n[0], n[1]), 0xD8));
    }
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(ds), _mm256_extracti128_si256(ds, 1));
    return (uint32_t)(_mm_cvtsi128_si32(s) + _mm_extract_epi32(s, 1) + _mm_extract_epi32(s, 2) +
                      _mm_extract_epi32(s, 3));
}
/* Two queries x JB 8-channel blocks over key pairs p0..p1-1 (<= 128 pairs: |lane| <= 128*2^23 = 2^30):
   vpmaddwd(V pair as int16, (-w_s,-w_s+1) broadcast) = -(w_s*v_s+w_s+1*v_s+1) exactly. */
static inline __attribute__((always_inline)) void att_block(const int8_t *Vp, uint32_t Cp, uint32_t p0, uint32_t p1,
                                                            const int16_t *const *wn, const int JB, uint32_t c0,
                                                            double (*num)[ATT_KEYS / 16]) {
    __m256i a[2][6];
    for (int q = 0; q < 2; q++)
        for (int j = 0; j < JB; j++) a[q][j] = _mm256_setzero_si256();
    for (uint32_t p = p0; p < p1; p++) {
        const __m256i w0 = _mm256_set1_epi32(ld32((const int8_t *)(wn[0] + 2 * p))),
                      w1 = _mm256_set1_epi32(ld32((const int8_t *)(wn[1] + 2 * p)));
        const int8_t *vp = Vp + ((size_t)p * Cp + c0) * 2;
        for (int j = 0; j < JB; j++) {
            const __m256i v = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(vp + 16 * j)));
            a[0][j] = _mm256_add_epi32(a[0][j], _mm256_madd_epi16(v, w0));
            a[1][j] = _mm256_add_epi32(a[1][j], _mm256_madd_epi16(v, w1));
        }
    }
    for (int q = 0; q < 2; q++)
        for (int j = 0; j < JB; j++) {
            double *d = num[q] + c0 + 8 * j;
            _mm256_storeu_pd(d, _mm256_sub_pd(_mm256_loadu_pd(d), _mm256_cvtepi32_pd(_mm256_castsi256_si128(a[q][j]))));
            _mm256_storeu_pd(
                d + 4, _mm256_sub_pd(_mm256_loadu_pd(d + 4), _mm256_cvtepi32_pd(_mm256_extracti128_si256(a[q][j], 1))));
        }
}
static int attend_fast(const Context *c, const Op *o, const int8_t *x, const int8_t *b, int8_t *y, uint32_t rows,
                       uint32_t keys) {
    const uint32_t S = c->t[o->a].c, C = c->t[o->b].c, *lut = (const uint32_t *)(c->blob + o->weights),
                   P = (keys + 1) / 2, Cp = (C + 15) & ~15u;
    if (!keys || keys > ATT_KEYS || C > ATT_KEYS / 16 || (size_t)P * Cp * 2 > ATT_BUF) return 0;
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift);
    __attribute__((aligned(32))) int8_t Vp[ATT_BUF];
    __attribute__((aligned(32))) int16_t W[2][ATT_KEYS + 32];
    double num[2][ATT_KEYS / 16];
    for (uint32_t p = 0; p < P; p++)
        for (uint32_t ch = 0; ch < Cp; ch += 16) { /* Vp[p][c][2] = (v[2p][c], v[2p+1][c]) */
            __m128i r0, r1;
            if (ch + 16 <= C) {
                r0 = _mm_loadu_si128((const __m128i *)(b + (size_t)2 * p * C + ch));
                r1 = 2 * p + 1 < keys ? _mm_loadu_si128((const __m128i *)(b + (size_t)(2 * p + 1) * C + ch))
                                      : _mm_setzero_si128();
            } else {
                int8_t t0[16], t1[16];
                for (uint32_t i = 0; i < 16; i++) {
                    t0[i] = ch + i < C ? b[(size_t)2 * p * C + ch + i] : 0;
                    t1[i] = ch + i < C && 2 * p + 1 < keys ? b[(size_t)(2 * p + 1) * C + ch + i] : 0;
                }
                r0 = _mm_loadu_si128((const __m128i *)t0);
                r1 = _mm_loadu_si128((const __m128i *)t1);
            }
            _mm_storeu_si128((__m128i *)(Vp + ((size_t)p * Cp + ch) * 2), _mm_unpacklo_epi8(r0, r1));
            _mm_storeu_si128((__m128i *)(Vp + ((size_t)p * Cp + ch) * 2 + 16), _mm_unpackhi_epi8(r0, r1));
        }
    AttK k[2];
    for (int q = 0; q < 2; q++) {
        k[q].mh = _mm256_set1_pd((double)(mul >> 16));
        k[q].ml = _mm256_set1_pd((double)(mul & 0xFFFF));
        k[q].rnd = _mm256_set1_pd((double)(1ull << (sh + 15)));
        k[q].s16 = _mm256_set1_pd(1.0 / 65536.0);
        k[q].ssh = _mm256_set1_pd(pow2neg(sh));
        k[q].lo = _mm256_set1_pd(o->lo);
        k[q].hi = _mm256_set1_pd(o->hi);
    }
    for (uint32_t t0 = 0; t0 < rows; t0 += 2) {
        const uint32_t nq = rows - t0 < 2 ? 1 : 2;
        const int16_t *wn[2];
        int8_t *yq[2];
        uint32_t live = 0;
        for (uint32_t q = 0; q < 2; q++) {
            const uint32_t t = t0 + (q < nq ? q : 0);
            wn[q] = W[q];
            yq[q] = y + (size_t)t * C;
            const uint32_t den = q < nq ? att_weights(x + (size_t)t * S, keys, o->p1, lut, W[q]) : 0;
            if (q < nq && !den) {
                for (uint32_t ch = 0; ch < C; ch++) yq[q][ch] = clamp(0, o);
                continue;
            }
            if (q < nq) {
                live |= 1u << q;
                k[q].den = _mm256_set1_pd((double)den);
                k[q].inv = _mm256_set1_pd(1.0 / (double)den);
                k[q].half = _mm256_set1_pd((double)(den >> 1));
            }
        }
        if (!live) continue;
        if (live == 2) {
            wn[0] = wn[1];
            yq[0] = yq[1];
            k[0] = k[1];
        }
        if (live != 3) wn[1] = wn[0];
        const uint32_t n = live == 3 ? 2 : 1;
        for (uint32_t q = 0; q < 2; q++)
            for (uint32_t ch = 0; ch < Cp; ch++) num[q][ch] = 0;
        for (uint32_t pb = 0; pb < P; pb += 128) {
            const uint32_t pe = P - pb < 128 ? P : pb + 128;
            for (uint32_t c0 = 0; c0 < Cp; c0 += 48) {
                const uint32_t jb = (Cp - c0) / 8 < 6 ? (Cp - c0) / 8 : 6;
#define ATT_JB(JB) att_block(Vp, Cp, pb, pe, wn, JB, c0, num);
                switch (jb) {
                case 2: ATT_JB(2) break;
                case 4: ATT_JB(4) break;
                default: ATT_JB(6)
                }
#undef ATT_JB
            }
        }
        for (uint32_t q = 0; q < n; q++)
            for (uint32_t ch = 0; ch < C; ch += 8) {
                const __m128i r0 = att_out4(_mm256_loadu_pd(num[q] + ch), k + q),
                              r1 = att_out4(_mm256_loadu_pd(num[q] + ch + 4), k + q);
                const __m128i v = _mm_packs_epi16(_mm_packs_epi32(r0, r1), r0);
                if (C - ch >= 8)
                    _mm_storel_epi64((__m128i *)(yq[q] + ch), v);
                else {
                    int8_t l[16];
                    _mm_storeu_si128((__m128i *)l, v);
                    for (uint32_t i = 0; i < C - ch; i++) yq[q][ch + i] = l[i];
                }
            }
    }
    return 1;
}
#define ATTEND_FAST 1
#endif

/* GUST_MATMUL for query rows t < rows (gust_run_valid shortens them). */
static void op_matmul(const Context *c, const Op *o, uint32_t rows, uint32_t keys) {
    matmul(c, o, c->arena + c->t[o->a].offset, c->arena + c->t[o->b].offset, c->arena + c->t[o->out].offset, rows,
           keys);
}

#if defined(GEMM128) && !defined(NG_X86)
/* 128-bit ATTEND (NEON, WebAssembly SIMD128), the integers of attend(): tiles of 4 query rows share each 16-channel
   value load. Weights are stored negated as int16 (w <= 2^15, so -w fits); each key adds -w_s * v[s][c] into int32
   lanes (NEON smlal by scalar, WebAssembly extmul + add); per 256 keys |sum| <= 256 * 2^15 * 128 = 2^30 is exact,
   then int64 per channel. Tasks of NA_ROWS query rows. */
#if defined(__aarch64__)
#define NA_WIDEN_LO(v)        vmovl_s8(vget_low_s8(v))
#define NA_WIDEN_HI(v)        vmovl_high_s8(v)
#define NA_MAC_LO(a, v16, w)  vmlal_n_s16(a, vget_low_s16(v16), w)
#define NA_MAC_HI(a, v16, w)  vmlal_high_n_s16(a, v16, w)
#define NA_ST32(p, a)         vst1q_s32(p, a)
#else
#define NA_WIDEN_LO(v)        wasm_i16x8_extend_low_i8x16(v)
#define NA_WIDEN_HI(v)        wasm_i16x8_extend_high_i8x16(v)
#define NA_MAC_LO(a, v16, w)  wasm_i32x4_add(a, wasm_i32x4_extmul_low_i16x8(v16, wasm_i16x8_splat(w)))
#define NA_MAC_HI(a, v16, w)  wasm_i32x4_add(a, wasm_i32x4_extmul_high_i16x8(v16, wasm_i16x8_splat(w)))
#define NA_ST32(p, a)         wasm_v128_store(p, a)
#endif
#define NA_KEYS 2048
#define NA_ROWS 8
typedef struct {
    const Context *c;
    const Op *o;
    const int8_t *x, *b;
    int8_t *y;
    uint32_t rows, keys;
} NaJob;
static void na_task(void *opaque, uint32_t item, uint32_t thread) {
    (void)thread;
    const NaJob *j = opaque;
    const Context *c = j->c;
    const Op *o = j->o;
    const uint32_t S = c->t[o->a].c, C = c->t[o->b].c, *lut = (const uint32_t *)(c->blob + o->weights);
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift), keys = j->keys;
    const uint32_t t1 = (item + 1) * NA_ROWS < j->rows ? (item + 1) * NA_ROWS : j->rows;
    int16_t nw[4][NA_KEYS] __attribute__((aligned(16)));
    for (uint32_t t0 = item * NA_ROWS; t0 < t1; t0 += 4) {
        const uint32_t R = t1 - t0 < 4 ? t1 - t0 : 4;
        uint64_t den[4] = {0, 0, 0, 0};
        double inv[4] = {0, 0, 0, 0};
        int dead[4] = {1, 1, 1, 1};
        for (uint32_t r = 0; r < 4; r++) {
            if (r >= R) {
                for (uint32_t k = 0; k < keys; k++) nw[r][k] = 0;
                continue;
            }
            const int8_t *row = j->x + (size_t)(t0 + r) * S;
            int32_t top = -128;
            uint32_t k = 0;
#if defined(__aarch64__)
            int8x16_t mx = vdupq_n_s8(-128);
            for (; k + 16 <= keys; k += 16) mx = vmaxq_s8(mx, vld1q_s8(row + k));
            top = vmaxvq_s8(mx);
#endif
            for (; k < keys; k++) top = row[k] > top ? row[k] : top;
            dead[r] = o->p1 && top == -128;
            for (k = 0; k < keys; k++) {
                const int8_t v = row[k];
                const uint32_t w = dead[r] || (o->p1 && v == -128) ? 0u : lut[top - v];
                den[r] += w;
                nw[r][k] = (int16_t)(0 - (int32_t)w);
            }
            if (dead[r]) den[r] = 0;
            if (den[r]) inv[r] = 1.0 / (double)den[r];
        }
        for (uint32_t c0 = 0; c0 < C; c0 += 16) {
            const uint32_t n = C - c0 < 16 ? C - c0 : 16;
            int64_t num[4][16];
            for (uint32_t r = 0; r < 4; r++)
                for (uint32_t i = 0; i < 16; i++) num[r][i] = 0;
            for (uint32_t s0 = 0; s0 < keys; s0 += 256) {
                const uint32_t se = keys - s0 < 256 ? keys : s0 + 256;
                if (n == 16) {
                    G32 a[4][4];
                    for (uint32_t r = 0; r < 4; r++)
                        for (uint32_t q = 0; q < 4; q++) a[r][q] = G_Z32;
                    for (uint32_t k = s0; k < se; k++) {
                        const G8 v = G_LD(j->b + (size_t)k * C + c0);
                        const G16 lo = NA_WIDEN_LO(v), hi = NA_WIDEN_HI(v);
                        for (uint32_t r = 0; r < 4; r++) {
                            const int16_t w = nw[r][k];
                            a[r][0] = NA_MAC_LO(a[r][0], lo, w);
                            a[r][1] = NA_MAC_HI(a[r][1], lo, w);
                            a[r][2] = NA_MAC_LO(a[r][2], hi, w);
                            a[r][3] = NA_MAC_HI(a[r][3], hi, w);
                        }
                    }
                    for (uint32_t r = 0; r < 4; r++)
                        for (uint32_t q = 0; q < 4; q++) {
                            int32_t l4[4];
                            NA_ST32(l4, a[r][q]);
                            for (uint32_t l = 0; l < 4; l++) num[r][4 * q + l] -= l4[l];
                        }
                } else
                    for (uint32_t r = 0; r < 4; r++)
                        for (uint32_t i = 0; i < n; i++) {
                            int32_t acc = 0;
                            for (uint32_t k = s0; k < se; k++) acc += (int32_t)nw[r][k] * j->b[(size_t)k * C + c0 + i];
                            num[r][i] -= acc;
                        }
            }
            for (uint32_t r = 0; r < R; r++)
                for (uint32_t i = 0; i < n; i++)
                    j->y[(size_t)(t0 + r) * C + c0 + i] = attend_out(num[r][i], den[r], inv[r], mul, sh, o);
        }
    }
}
static int g128_attend(const Context *c, const Op *o, const int8_t *x, const int8_t *b, int8_t *y, uint32_t rows,
                       uint32_t keys) {
    if (keys > NA_KEYS) return 0;
    NaJob j = {c, o, x, b, y, rows, keys};
    const uint32_t tasks = (rows + NA_ROWS - 1) / NA_ROWS;
#if defined(GUST_FLOAT32) && defined(GUST_THREADS)
    if (tasks > 1) {
        gust_parallel(na_task, &j, tasks);
        return 1;
    }
#endif
    for (uint32_t i = 0; i < tasks; i++) na_task(&j, i, 0);
    return 1;
}
#endif
/* GUST_ATTEND for query rows t < rows over the first `keys` keys: the vector form when the shape fits its buffers. */
/* Query rows are independent: threaded builds without the 128-bit tiles run blocks of AT_ROWS rows as tasks (each
   the same code on its rows, so results do not depend on the thread count). */
#define AT_ROWS 32u
typedef struct {
    const Context *c;
    const Op *o;
    const int8_t *x, *v;
    int8_t *y;
    uint32_t rows, keys;
} AtJob;
static void at_task(void *opaque, uint32_t item, uint32_t thread) {
    (void)thread;
    const AtJob *j = opaque;
    const uint32_t r0 = item * AT_ROWS, n = j->rows - r0 < AT_ROWS ? j->rows - r0 : AT_ROWS;
    const uint32_t S = j->c->t[j->o->a].c, C = j->c->t[j->o->b].c;
    const int8_t *x = j->x + (size_t)r0 * S;
    int8_t *y = j->y + (size_t)r0 * C;
#ifdef ATTEND_FAST
    if (attend_fast(j->c, j->o, x, j->v, y, n, j->keys)) return;
#endif
    attend(j->c, j->o, x, j->v, y, n, j->keys);
}
static void op_attend(const Context *c, const Op *o, uint32_t rows, uint32_t keys) {
    const int8_t *x = c->arena + c->t[o->a].offset, *v = c->arena + c->t[o->b].offset;
    int8_t *y = c->arena + c->t[o->out].offset;
#if defined(GEMM128) && !defined(NG_X86)
    if (g128_attend(c, o, x, v, y, rows, keys)) return;
#endif
    AtJob j = {c, o, x, v, y, rows, keys};
    const uint32_t tasks = (rows + AT_ROWS - 1) / AT_ROWS;
#if defined(GUST_FLOAT32) && defined(GUST_THREADS)
    if (tasks > 1) {
        gust_parallel(at_task, &j, tasks);
        return;
    }
#endif
    for (uint32_t i = 0; i < tasks; i++) at_task(&j, i, 0);
}

#endif
