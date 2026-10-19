/* GUST_NORM, GUST_LUT and GUST_MUL for Conformer-style sequence models laid out as [T,1,C] tensors (any [H,W,C]
   works; statistics are per position). Compiled only with GUST_CONFORMER. Part of the runtime/gust.c translation
   unit; see docs/FORMAT.md. Every kernel is exact integer arithmetic (float64 steps only where every value is an
   exactly representable integer or power-of-two multiple) with no allocation. */
#ifdef GUST_CONFORMER

#if defined(__AVX2__) && !defined(GUST_INT8_VNNI)
typedef int32_t __attribute__((may_alias, aligned(1))) Dword32;
static inline int32_t ld32x(const int8_t *p) {
    return *(const Dword32 *)(const void *)p;
}
#endif
/* GUST_LUT: y=t[x+128] over n bytes. AVX-512BW: entries in pairs as 128 words, u=x+128 looks up word u/2
   (two 64-word vpermi2w selected by bit 6 of u/2) and keeps its low or high byte. AVX2: 16 vpshufb lookups by
   the low nibble, one per 16-entry slice, then a blend tree on the four high bits. */
static void lut_apply(const int8_t *t, const int8_t *x, int8_t *y, uint64_t n) {
    uint64_t i = 0;
#ifdef GUST_INT8_VNNI
    __m512i T[4];
    for (int k = 0; k < 4; k++) T[k] = _mm512_loadu_si512(t + 64 * k);
    const __m512i flip = _mm512_set1_epi8((char)0x80), b6 = _mm512_set1_epi16(64), one = _mm512_set1_epi16(1),
                  lowb = _mm512_set1_epi16(255);
    for (; i < n;
         i += 64) { /* word lane = bytes (u0,u1): word u0/2 = lane>>1, word u1/2 = lane>>9 (index bits 0..6 used) */
        const __mmask64 m = n - i >= 64 ? ~0ull : ~0ull >> (64 - (n - i));
        const __m512i u = _mm512_xor_si512(_mm512_maskz_loadu_epi8(m, x + i), flip), he = _mm512_srli_epi16(u, 1),
                      ho = _mm512_srli_epi16(u, 9);
        const __m512i re =
            _mm512_mask_blend_epi16(_mm512_test_epi16_mask(he, b6), _mm512_permutex2var_epi16(T[0], he, T[1]),
                                    _mm512_permutex2var_epi16(T[2], he, T[3]));
        const __m512i ro =
            _mm512_mask_blend_epi16(_mm512_test_epi16_mask(ho, b6), _mm512_permutex2var_epi16(T[0], ho, T[1]),
                                    _mm512_permutex2var_epi16(T[2], ho, T[3]));
        const __m512i ve = _mm512_and_si512(_mm512_srlv_epi16(re, _mm512_slli_epi16(_mm512_and_si512(u, one), 3)),
                                            lowb); /* byte u0&1 of the word */
        const __m512i vo = _mm512_slli_epi16(
            _mm512_srlv_epi16(ro, _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(u, 8), one), 3)), 8);
        _mm512_mask_storeu_epi8(y + i, m, _mm512_or_si512(ve, vo));
    }
#elif defined(__AVX2__)
    __m256i T[16];
    for (int k = 0; k < 16; k++) T[k] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(t + 16 * k)));
    const __m256i flip = _mm256_set1_epi8((char)0x80), low = _mm256_set1_epi8(15);
    for (; i + 32 <= n; i += 32) {
        const __m256i u = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(x + i)), flip),
                      l = _mm256_and_si256(u, low);
        __m256i r[8];
        for (int k = 0; k < 8; k++)
            r[k] =
                _mm256_blendv_epi8(_mm256_shuffle_epi8(T[k], l), _mm256_shuffle_epi8(T[k + 8], l), u); /* bit 7 of u */
        const __m256i m2 = _mm256_slli_epi16(u, 1), m1 = _mm256_slli_epi16(u, 2), m0 = _mm256_slli_epi16(u, 3);
        for (int k = 0; k < 4; k++) r[k] = _mm256_blendv_epi8(r[k], r[k + 4], m2);
        for (int k = 0; k < 2; k++) r[k] = _mm256_blendv_epi8(r[k], r[k + 2], m1);
        _mm256_storeu_si256((__m256i *)(y + i), _mm256_blendv_epi8(r[0], r[1], m0));
    }
#elif defined(__aarch64__)
    int8x16x4_t T[4];
    for (int k = 0; k < 4; k++) T[k] = vld1q_s8_x4(t + 64 * k);
    const uint8x16_t flip = vdupq_n_u8(0x80), d64 = vdupq_n_u8(64);
    for (; i + 16 <= n; i += 16) { /* u = x + 128; tbx leaves lanes whose index is past its 64 entries */
        const uint8x16_t u0 = veorq_u8(vreinterpretq_u8_s8(vld1q_s8(x + i)), flip), u1 = vsubq_u8(u0, d64),
                         u2 = vsubq_u8(u1, d64), u3 = vsubq_u8(u2, d64);
        int8x16_t r = vqtbl4q_s8(T[0], u0);
        r = vqtbx4q_s8(r, T[1], u1);
        r = vqtbx4q_s8(r, T[2], u2);
        vst1q_s8(y + i, vqtbx4q_s8(r, T[3], u3));
    }
#endif
    for (; i < n; i++) y[i] = t[x[i] + 128];
}
#if !defined(__SSE2__) && !defined(__aarch64__)
/* floor(sqrt(v)) by the bitwise digit method: exact, identical on every build. */
static uint64_t isqrt64(uint64_t v) {
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    for (; bit; bit >>= 2) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else
            r >>= 1;
    }
    return r;
}
#endif
/* GUST_NORM over positions rows of C channels. Integer definition, per position:
   s=sum x, q=sum x^2, D=C*q-s^2+E, den=max(1,isqrt(D)), inv=round(2^46/den) (one division
   per position); per channel n=C*x-s, p=n*inv, zq=sign(p)*((|p|+2^30)>>31) (the value
   (x-mean)/sqrt(var+eps) in Q15), y=quant(zq*g+bias) computed without int32 wrap.
   Bounds (C<=4096): |s|<=2^19, q<=2^26, D<2^39+2^32. Since n^2<=(C-1)*D and den>sqrt(D)/2,
   |n|/den<2*sqrt(C-1)<128, so |p|<=|n|*(2^46/den+1/2)<2^53 and |zq|<=2^22+1.
   The per-channel step runs in f64, exactly, so scalar and vector code (fused multiply-adds
   or not) agree bit for bit: n and inv are exact, so are p=n*inv (<2^53) and q=p*2^-31;
   q+-0.5 (|q|<2^22, granularity 2^-31) is exact and its truncation is zq. a=zq*g+bias is
   an exact integer (|a|<2^29+2^31 by the gust_init bias bound; no int32 wrap). v=a*m*2^-t
   is exact when |a*m|<2^53 and so is v+-0.5 (the qscale argument above); otherwise
   |v|>=2^23 and the clamp to [lo,hi] (integers, applied before truncation) is unchanged. */
#ifdef __SSE2__
/* floor(sqrt(v)) for v < 2^53: the correctly rounded f64 square root is within one of it; exact correction. */
static inline uint64_t isqrt_sse(uint64_t v) {
    uint64_t r = (uint64_t)_mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_set_sd((double)v)));
    while (r * r > v) r--;
    while ((r + 1) * (r + 1) <= v) r++;
    return r;
}
#define ISQRT isqrt_sse
#elif defined(__aarch64__)
/* floor(sqrt(v)) for v < 2^53: the correctly rounded f64 square root is within one of it; exact correction. */
static inline uint64_t isqrt_neon(uint64_t v) {
    uint64_t r = (uint64_t)vgetq_lane_f64(vsqrtq_f64(vdupq_n_f64((double)v)), 0);
    while (r * r > v) r--;
    while ((r + 1) * (r + 1) <= v) r++;
    return r;
}
#define ISQRT isqrt_neon
#else
#define ISQRT isqrt64
#endif
#if defined(__aarch64__)
/* trunc(v + copysign(0.5, v)): round half away from zero of an f64 whose halves are exact (see above). */
static inline float64x2_t norm_rhalf(float64x2_t v) {
    const uint64x2_t sign = vandq_u64(vreinterpretq_u64_f64(v), vdupq_n_u64(0x8000000000000000ull));
    return vrndq_f64(vaddq_f64(v, vreinterpretq_f64_u64(vorrq_u64(sign, vreinterpretq_u64_f64(vdupq_n_f64(0.5))))));
}
#endif
static void norm(const Op *o, const int8_t *x, int8_t *y, uint32_t positions, uint32_t C, const int8_t *g,
                 const int32_t *bias, const int32_t *m, const uint32_t *s) {
    /* Rows go in blocks of 32 (statistics first); per-channel constants are converted to f64
       once per block and chunk of 256 channels. */
    const double lo = o->lo, hi = o->hi;
    const int32_t cc = (int32_t)C;
    for (uint32_t p0 = 0; p0 < positions; p0 += 32) {
        const uint32_t P = positions - p0 < 32 ? positions - p0 : 32;
        double inv[32];
        int32_t sums[32];
        for (uint32_t p = 0; p < P; p++) {
            const int8_t *xr = x + (uint64_t)(p0 + p) * C;
            int32_t sum = 0, sq = 0;
            uint32_t k = 0;
#if defined(GUST_INT8_VNNI) || defined(__AVX2__)
            { /* exact int32 sums: |sum| <= 2^19, q <= 2^26 */
                const __m256i one = _mm256_set1_epi16(1);
                __m256i s0 = _mm256_setzero_si256(), q0 = s0;
                for (; k + 16 <= C; k += 16) {
                    const __m256i v = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xr + k)));
                    s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(v, one));
                    q0 = _mm256_add_epi32(q0, _mm256_madd_epi16(v, v));
                }
                __m128i a = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1)),
                        b = _mm_add_epi32(_mm256_castsi256_si128(q0), _mm256_extracti128_si256(q0, 1));
                a = _mm_add_epi32(a, _mm_srli_si128(a, 8));
                a = _mm_add_epi32(a, _mm_srli_si128(a, 4));
                b = _mm_add_epi32(b, _mm_srli_si128(b, 8));
                b = _mm_add_epi32(b, _mm_srli_si128(b, 4));
                sum = _mm_cvtsi128_si32(a);
                sq = _mm_cvtsi128_si32(b);
            }
#elif defined(__aarch64__)
            { /* exact int32 sums: |sum| <= 2^19, q <= 2^26 */
                int32x4_t s0 = vdupq_n_s32(0), q0 = s0;
                for (; k + 16 <= C; k += 16) {
                    const int8x16_t v = vld1q_s8(xr + k);
                    s0 = vpadalq_s16(s0, vpaddlq_s8(v));
                    q0 = vpadalq_s16(q0, vmull_s8(vget_low_s8(v), vget_low_s8(v)));
                    q0 = vpadalq_s16(q0, vmull_high_s8(v, v));
                }
                sum = vaddvq_s32(s0);
                sq = vaddvq_s32(q0);
            }
#endif
            for (; k < C; k++) {
                const int32_t v = xr[k];
                sum += v;
                sq += v * v;
            }
            uint64_t den = ISQRT((uint64_t)((int64_t)C * sq - (int64_t)sum * sum) + o->p0);
            if (!den) den = 1;
            inv[p] = (double)((((uint64_t)1 << 46) + den / 2) / den) * pow2neg(31);
            sums[p] = sum; /* inv*2^-31 */
        }
        for (uint32_t c0 = 0; c0 < C; c0 += 256) {
            const uint32_t n = C - c0 < 256 ? C - c0 : 256;
            double gd[256], bd[256], fd[256];
            for (uint32_t k = 0; k < n; k++) {
                gd[k] = g[c0 + k];
                bd[k] = bias[c0 + k];
                fd[k] = (double)m[c0 + k] * pow2neg(s[c0 + k]);
            }
            for (uint32_t p = 0; p < P; p++) {
                const int8_t *xr = x + (uint64_t)(p0 + p) * C + c0;
                int8_t *yr = y + (uint64_t)(p0 + p) * C + c0;
                const double fi = inv[p];
                const int32_t sum = sums[p];
                uint32_t k = 0;
#if defined(__wasm_simd128__)
                { /* the same f64 steps, two channels per f64x2; trunc(q+copysign(0.5,q)) rounds half away */
                    const v128_t vc = wasm_i32x4_splat(cc), vs = wasm_i32x4_splat(sum), vf = wasm_f64x2_splat(fi),
                                 vlo = wasm_f64x2_splat(lo), vhi = wasm_f64x2_splat(hi);
                    const v128_t sg = wasm_f64x2_splat(-0.0), half = wasm_f64x2_splat(0.5);
                    for (; k + 4 <= n; k += 4) {
                        const v128_t nv = wasm_i32x4_sub(
                            wasm_i32x4_mul(
                                wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(wasm_v128_load32_zero(xr + k))),
                                vc),
                            vs);
                        v128_t r[2];
                        for (uint32_t j = 0; j < 2; j++) {
                            const v128_t q = wasm_f64x2_mul(
                                wasm_f64x2_convert_low_i32x4(j ? wasm_i32x4_shuffle(nv, nv, 2, 3, 2, 3) : nv), vf);
                            const v128_t zq =
                                wasm_f64x2_trunc(wasm_f64x2_add(q, wasm_v128_or(wasm_v128_and(q, sg), half)));
                            r[j] = wasm_qround(
                                wasm_f64x2_mul(wasm_f64x2_add(wasm_f64x2_mul(zq, wasm_v128_load(gd + k + 2 * j)),
                                                              wasm_v128_load(bd + k + 2 * j)),
                                               wasm_v128_load(fd + k + 2 * j)),
                                vlo, vhi);
                        }
                        const v128_t z = wasm_i32x4_shuffle(r[0], r[1], 0, 2, 4, 6);
                        wasm_v128_store32_lane(
                            yr + k,
                            wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(z, z), wasm_i16x8_narrow_i32x4(z, z)), 0);
                    }
                }
#elif defined(GUST_INT8_VNNI)
                { /* the same f64 steps, eight channels per vector; trunc(q+copysign(0.5,q)) rounds half away */
                    /* int64 <-> f64 by the 1.5*2^52 bias (exact for |i| < 2^51; zmm conversions are slow on some
                       cores) */
                    const __m512i vc = _mm512_set1_epi64(cc), vs = _mm512_set1_epi64(sum),
                                  sg = _mm512_set1_epi64((long long)0x8000000000000000ull),
                                  mg = _mm512_set1_epi64(0x4338000000000000ll);
                    const __m512d vf = _mm512_set1_pd(fi), vlo = _mm512_set1_pd(lo), vhi = _mm512_set1_pd(hi),
                                  half = _mm512_set1_pd(0.5), mgd = _mm512_set1_pd(6755399441055744.0);
#define NORM_HALF(q)                                                                                                   \
    _mm512_castsi512_pd(_mm512_or_si512(_mm512_and_si512(_mm512_castpd_si512(q), sg), _mm512_castpd_si512(half)))
                    /* q+-0.5 and v+-0.5 as single fused steps: both sums are representable (the f64 argument above), so
                       the one rounding is exact; signs come from n and a (fi, fd > 0). */
                    for (; k + 8 <= n; k += 8) {
                        const __m512i nv = _mm512_sub_epi64(
                            _mm512_mul_epi32(_mm512_cvtepi8_epi64(_mm_loadl_epi64((const __m128i *)(xr + k))), vc), vs);
                        const __m512d nd = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_add_epi64(nv, mg)), mgd);
                        const __m512d zq =
                            _mm512_roundscale_pd(_mm512_fmadd_pd(nd, vf, NORM_HALF(_mm512_castsi512_pd(nv))),
                                                 _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                        const __m512d a = _mm512_fmadd_pd(zq, _mm512_loadu_pd(gd + k), _mm512_loadu_pd(bd + k));
                        const __m512d r =
                            _mm512_min_pd(_mm512_max_pd(_mm512_roundscale_pd(
                                                            _mm512_fmadd_pd(a, _mm512_loadu_pd(fd + k), NORM_HALF(a)),
                                                            _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC),
                                                        vlo),
                                          vhi);
                        _mm_storel_epi64((__m128i *)(yr + k), _mm512_cvtepi64_epi8(_mm512_castpd_si512(_mm512_add_pd(
                                                                  r, mgd)))); /* low byte of the biased integer */
                    }
#undef NORM_HALF
                }
#elif defined(__AVX2__)
                { /* the same f64 steps, four channels per vector */
                    const __m128i vc = _mm_set1_epi32(cc), vs = _mm_set1_epi32(sum);
                    const __m256d vf = _mm256_set1_pd(fi), vlo = _mm256_set1_pd(lo), vhi = _mm256_set1_pd(hi),
                                  sg = _mm256_set1_pd(-0.0), half = _mm256_set1_pd(0.5);
                    for (; k + 8 <= n; k += 8) {
                        __m128i z[2];
                        for (uint32_t j = 0; j < 2; j++) {
                            const __m128i nv = _mm_sub_epi32(
                                _mm_mullo_epi32(_mm_cvtepi8_epi32(_mm_cvtsi32_si128(ld32x(xr + k + 4 * j))), vc), vs);
                            const __m256d q = _mm256_mul_pd(_mm256_cvtepi32_pd(nv), vf);
                            const __m256d zq =
                                _mm256_round_pd(_mm256_add_pd(q, _mm256_or_pd(_mm256_and_pd(q, sg), half)),
                                                _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                            const __m256d v =
                                _mm256_mul_pd(_mm256_add_pd(_mm256_mul_pd(zq, _mm256_loadu_pd(gd + k + 4 * j)),
                                                            _mm256_loadu_pd(bd + k + 4 * j)),
                                              _mm256_loadu_pd(fd + k + 4 * j));
                            z[j] = _mm256_cvttpd_epi32(_mm256_min_pd(
                                _mm256_max_pd(_mm256_add_pd(v, _mm256_or_pd(_mm256_and_pd(v, sg), half)), vlo), vhi));
                        }
                        _mm_storel_epi64((__m128i *)(yr + k), _mm_packs_epi16(_mm_packs_epi32(z[0], z[1]), z[0]));
                    }
                }
#elif defined(__aarch64__)
                { /* the same f64 steps, two channels per f64x2; trunc(q + copysign(0.5, q)) rounds half away */
                    const int32x4_t vc = vdupq_n_s32(cc), vs = vdupq_n_s32(sum);
                    const float64x2_t vf = vdupq_n_f64(fi), vlo = vdupq_n_f64(lo), vhi = vdupq_n_f64(hi);
                    for (; k + 8 <= n; k += 8) {
                        const int16x8_t x16 = vmovl_s8(vld1_s8(xr + k));
                        const int32x4_t nv[2] = {vsubq_s32(vmulq_s32(vmovl_s16(vget_low_s16(x16)), vc), vs),
                                                 vsubq_s32(vmulq_s32(vmovl_high_s16(x16), vc), vs)};
                        int32x4_t z[2];
                        for (uint32_t j = 0; j < 2; j++) {
                            float64x2_t r[2];
                            for (uint32_t h = 0; h < 2; h++) {
                                const int64x2_t n64 = h ? vmovl_high_s32(nv[j]) : vmovl_s32(vget_low_s32(nv[j]));
                                const uint32_t kk = k + 4 * j + 2 * h;
                                const float64x2_t zq = norm_rhalf(vmulq_f64(vcvtq_f64_s64(n64), vf));
                                const float64x2_t v = vmulq_f64(
                                    vaddq_f64(vmulq_f64(zq, vld1q_f64(gd + kk)), vld1q_f64(bd + kk)), vld1q_f64(fd + kk));
                                r[h] = vminq_f64(vmaxq_f64(norm_rhalf(v), vlo), vhi);
                            }
                            z[j] = vcombine_s32(vmovn_s64(vcvtq_s64_f64(r[0])), vmovn_s64(vcvtq_s64_f64(r[1])));
                        }
                        vst1_s8(yr + k, vmovn_s16(vcombine_s16(vmovn_s32(z[0]), vmovn_s32(z[1]))));
                    }
                }
#endif
                for (; k < n; k++) {
                    const double q = (double)(cc * xr[k] - sum) * fi, zq = (double)(int32_t)(q < 0 ? q - 0.5 : q + 0.5);
                    const double v = (zq * gd[k] + bd[k]) * fd[k], r = v < 0 ? v - 0.5 : v + 0.5;
                    yr[k] = (int8_t)(int32_t)(r < lo ? lo : r > hi ? hi : r);
                }
            }
        }
    }
}

/* GUST_NORM over the real rows. */
/* Positions are independent, so threaded builds split them into tasks of NORM_TASK_ROWS (same code per position:
   results do not depend on the thread count). */
#define NORM_TASK_ROWS 8u
typedef struct {
    const Context *c;
    const Op *o;
    uint32_t positions;
} NormJob;
static void norm_task(void *opaque, uint32_t item, uint32_t thread) {
    (void)thread;
    const NormJob *j = opaque;
    const Context *c = j->c;
    const Op *o = j->o;
    const uint32_t C = c->t[o->out].c, p0 = item * NORM_TASK_ROWS,
                   n = j->positions - p0 < NORM_TASK_ROWS ? j->positions - p0 : NORM_TASK_ROWS;
    norm(o, c->arena + c->t[o->a].offset + (size_t)p0 * C, c->arena + c->t[o->out].offset + (size_t)p0 * C, n, C,
         (const int8_t *)(c->blob + o->weights), (const int32_t *)(c->blob + o->bias),
         (const int32_t *)(c->blob + o->mult), (const uint32_t *)(c->blob + o->shift));
}
static void op_norm(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *z = c->t + o->out;
    NormJob j = {c, o, rows * z->w};
    const uint32_t tasks = (j.positions + NORM_TASK_ROWS - 1) / NORM_TASK_ROWS;
#if defined(GUST_FLOAT32) && defined(GUST_THREADS)
    if (tasks > 1) {
        gust_parallel(norm_task, &j, tasks);
        return;
    }
#endif
    for (uint32_t i = 0; i < tasks; i++) norm_task(&j, i, 0);
}

/* y = table[x + 128] over the real rows; gust_init checked that every entry lies inside [lo, hi]. */
static void op_lut(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *z = c->t + o->out;
    lut_apply((const int8_t *)(c->blob + o->weights), c->arena + c->t[o->a].offset, c->arena + z->offset,
              (uint64_t)rows * z->w * z->c);
}

/* y = quant(a * b) over the real rows with one multiplier/shift and no bias; |a * b| <= 2^14, so the vector paths
   use the exact requantization helpers. */
static void op_mul(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *z = c->t + o->out;
    const int8_t *x = c->arena + c->t[o->a].offset, *b = c->arena + c->t[o->b].offset;
    int8_t *y = c->arena + z->offset;
    const int32_t mul = *(const int32_t *)(c->blob + o->mult);
    const uint32_t sh = *(const uint32_t *)(c->blob + o->shift);
    const uint64_t n = (uint64_t)rows * z->w * z->c;
    uint64_t i = 0;
#ifdef GUST_INT8_VNNI
    const Quant16 q = vnni_qprep(_mm512_setzero_si512(), _mm512_set1_epi32(mul), _mm512_set1_epi32((int)sh));
#define MUL_Q(R)                                                                                                       \
    for (; i + 32 <= n; i += 32) {                                                                                     \
        const __m512i pr = _mm512_mullo_epi16(_mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(x + i))),      \
                                              _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b + i))));     \
        _mm_storeu_si128((__m128i *)(y + i),                                                                           \
                         vnni_qsat(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(pr)), &q, o->lo, o->hi, R));           \
        _mm_storeu_si128((__m128i *)(y + i + 16),                                                                      \
                         vnni_qsat(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(pr, 1)), &q, o->lo, o->hi, R));     \
    }
    if (o->lo >= 0) {
        MUL_Q(1)
    } else {
        MUL_Q(0)
    }
#undef MUL_Q
#elif defined(INT8_TILE) && defined(__AVX2__)
    {
        const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi), f1 = _mm256_set1_pd(qscale(mul, sh));
        const __m256d f[4] = {f1, f1, f1, f1};
        const __m128i zb[4] = {_mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128()};
        for (; i + 16 <= n; i += 16) {
            __m128i r[4];
            for (uint32_t j = 0; j < 4; j++)
                r[j] = _mm_mullo_epi32(_mm_cvtepi8_epi32(_mm_loadu_si32(x + i + 4 * j)),
                                       _mm_cvtepi8_epi32(_mm_loadu_si32(b + i + 4 * j)));
            _mm_storeu_si128((__m128i *)(y + i), avx_quant16(r, zb, f, lo, hi));
        }
    }
#elif defined(__aarch64__)
    {
        const NgQuant q = ng_qprep(&mul, &sh, 0, o);
        for (; i + 16 <= n; i += 16) {
            const int8x16_t u = vld1q_s8(x + i), v = vld1q_s8(b + i);
            const int16x8_t l = vmull_s8(vget_low_s8(u), vget_low_s8(v)), h = vmull_high_s8(u, v);
            const uint32_t r[4] = {ng_quant4(vmovl_s16(vget_low_s16(l)), &q), ng_quant4(vmovl_high_s16(l), &q),
                                   ng_quant4(vmovl_s16(vget_low_s16(h)), &q), ng_quant4(vmovl_high_s16(h), &q)};
            __builtin_memcpy(y + i, r, 16);
        }
    }
#elif defined(__wasm_simd128__)
    {
        const v128_t lo = wasm_f64x2_splat(o->lo), hi = wasm_f64x2_splat(o->hi), f1 = wasm_f64x2_splat(qscale(mul, sh)),
                     z0 = wasm_i32x4_splat(0);
        const v128_t f[8] = {f1, f1, f1, f1, f1, f1, f1, f1}, zb[4] = {z0, z0, z0, z0};
        for (; i + 16 <= n; i += 16) {
            const v128_t u = wasm_v128_load(x + i), v = wasm_v128_load(b + i);
            const v128_t l = wasm_i16x8_mul(wasm_i16x8_extend_low_i8x16(u), wasm_i16x8_extend_low_i8x16(v)),
                         h = wasm_i16x8_mul(wasm_i16x8_extend_high_i8x16(u), wasm_i16x8_extend_high_i8x16(v));
            const v128_t r[4] = {wasm_i32x4_extend_low_i16x8(l), wasm_i32x4_extend_high_i16x8(l),
                                 wasm_i32x4_extend_low_i16x8(h), wasm_i32x4_extend_high_i16x8(h)};
            wasm_v128_store(y + i, wasm_quant16(r, zb, f, lo, hi));
        }
    }
#endif
    for (; i < n; i++) y[i] = quant((int32_t)x[i] * b[i], mul, sh, o);
}

#endif
