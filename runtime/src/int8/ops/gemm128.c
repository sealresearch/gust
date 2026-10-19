/* INT8 GEMM on 128-bit vectors (AArch64 NEON, WebAssembly SIMD128) for convolutions whose taps are contiguous (every
   1x1) and for MATMUL scores: y_p[oc] = quant(bias[oc] + dot(x_p, w_oc)) for np pixels and N channels, exactly as
   kernels.c computes it. Part of the runtime/gust.c translation unit.

   Tiles of 4 pixels x 4 channels. Per 16 bytes of K, two int8 x int8 products enter each int16 lane of the tile's
   accumulator (NEON smlal / smlal2; WebAssembly extmul + add, which V8 lowers to smull + add on ARM); every `run`
   steps the int16 lanes are pairwise-added into int32 (sadalp / extadd_pairwise). With |x| <= 128 and |w| <= wmax a
   lane holds 2 * run * 128 * wmax <= 32767, so nothing saturates: run = 15 for 4-bit codes (wmax 8), 2 for
   |w| <= 63, 1 for |w| <= 127; a weight of -128 widens every product (run 0). The int32 sum of every lane is
   exact, so the order of additions does not matter, and g_quant4 reproduces quant().

   Shape: on in-order cores (Cortex-A53) a 128-bit load costs two cycles and does not overlap the multiplies, so a
   tile maximizes multiplies per loaded byte within 32 vector registers: 16 accumulators and 8 operands (32
   multiplies per 8 loads). The low-half products of all 16 pairs precede the high-half ones, so dependent
   accumulations are 16 instructions apart. */
#if defined(__x86_64__) && defined(__SSE2__) && !defined(__AVX2__)
#define NG_X86 1 /* the SSE2 base variant of x86-64 builds (AVX2 and VNNI variants keep ops/gemm.c) */
#endif
#if defined(__aarch64__) || defined(__wasm_simd128__) || defined(NG_X86)
#if defined(__aarch64__)
typedef int8x16_t G8;
typedef int16x8_t G16;
typedef int32x4_t G32;
typedef NgQuant GQuant;
#define G_LD(p)            vld1q_s8(p)
#define G_LD32(p)          vld1q_s32(p)
#define G_Z16              vdupq_n_s16(0)
#define G_Z32              vdupq_n_s32(0)
#define G_ADD32(a, b)      vaddq_s32(a, b)
#define G_MLAL_LO(h, x, w) vmlal_s8(h, vget_low_s8(x), vget_low_s8(w))
#define G_MLAL_HI(h, x, w) vmlal_high_s8(h, x, w)
#define G_MULL_LO(x, w)    vmull_s8(vget_low_s8(x), vget_low_s8(w))
#define G_MULL_HI(x, w)    vmull_high_s8(x, w)
#define G_PADAL(a, h)      vpadalq_s16(a, h)
#define G_HADD4(a0, a1, a2, a3) vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)) /* lane j: total of aj */
#define g_qprep            ng_qprep
#define g_quant4           ng_quant4
/* Dot-product instructions (FEAT_DotProd: Cortex-A55, A75 and later, Apple, Neoverse): detected once at run time, so
   one binary serves every AArch64 core. GUST_ARM=base forces the plain NEON tiles (tests). Results are identical. */
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <stdlib.h>
#include <string.h>
static int ng_has_dot(void) {
    static int v = -1;
    if (v >= 0) return v;
    int d = 0;
#if defined(__ARM_FEATURE_DOTPROD)
    d = 1;
#elif defined(__linux__) || defined(__ANDROID__)
    d = (getauxval(AT_HWCAP) & (1ul << 20)) != 0; /* HWCAP_ASIMDDP */
#elif defined(__APPLE__)
    int on = 0;
    size_t len = sizeof on;
    d = !sysctlbyname("hw.optional.arm.FEAT_DotProd", &on, &len, NULL, 0) && on;
#endif
    const char *f = getenv("GUST_ARM");
    if (f && !strcmp(f, "base")) d = 0;
    v = d;
    return v;
}
/* The 4 x 4 tile with sdot: each instruction adds four exact int8 products into each int32 lane, so there is no int16
   stage and no bound on K beyond int32 (K * 2^14). */
__attribute__((target("dotprod"))) static void ng_tile_dot(const int8_t *const *x, const int8_t *w, size_t wrow,
                                                           uint32_t K, int32x4_t c[4]) {
    const uint32_t K16 = K & ~15u;
    const int8_t *w0 = w, *w1 = w + wrow, *w2 = w + 2 * wrow, *w3 = w + 3 * wrow;
    int32x4_t a[4][4];
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 4; j++) a[p][j] = vdupq_n_s32(0);
    for (uint32_t k = 0; k < K16; k += 16) {
        const int8x16_t wv[4] = {vld1q_s8(w0 + k), vld1q_s8(w1 + k), vld1q_s8(w2 + k), vld1q_s8(w3 + k)};
        const int8x16_t xv[4] = {vld1q_s8(x[0] + k), vld1q_s8(x[1] + k), vld1q_s8(x[2] + k), vld1q_s8(x[3] + k)};
        for (int p = 0; p < 4; p++)
            for (int j = 0; j < 4; j++) a[p][j] = vdotq_s32(a[p][j], xv[p], wv[j]);
    }
    for (int p = 0; p < 4; p++) {
        c[p] = vpaddq_s32(vpaddq_s32(a[p][0], a[p][1]), vpaddq_s32(a[p][2], a[p][3]));
        if (K16 < K) {
            int32_t t[4] = {0, 0, 0, 0};
            for (uint32_t i = K16; i < K; i++) {
                const int32_t xi = x[p][i];
                t[0] += xi * w0[i];
                t[1] += xi * w1[i];
                t[2] += xi * w2[i];
                t[3] += xi * w3[i];
            }
            c[p] = vaddq_s32(c[p], vld1q_s32(t));
        }
    }
}
/* max |w| of n weights (|-128| = 128). */
static uint32_t g_wmax(const int8_t *w, size_t n) {
    uint8x16_t m = vdupq_n_u8(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) m = vmaxq_u8(m, vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(w + i))));
    uint32_t r = vmaxvq_u8(m);
    for (; i < n; i++) {
        const uint32_t a = w[i] < 0 ? (uint32_t)(-(int32_t)w[i]) : (uint32_t)w[i];
        if (a > r) r = a;
    }
    return r;
}
#elif defined(NG_X86)
typedef __m128i G8;
typedef __m128i G16;
typedef __m128i G32;
#define G_LD(p)       _mm_loadu_si128((const __m128i *)(const void *)(p))
#define G_LD32(p)     _mm_loadu_si128((const __m128i *)(const void *)(p))
#define G_Z32         _mm_setzero_si128()
#define G_ADD32(a, b) _mm_add_epi32(a, b)
static inline __m128i g_hadd4(__m128i a0, __m128i a1, __m128i a2, __m128i a3) { /* lane j: total of aj */
    const __m128i t0 = _mm_add_epi32(_mm_unpacklo_epi32(a0, a1), _mm_unpackhi_epi32(a0, a1)),
                  t1 = _mm_add_epi32(_mm_unpacklo_epi32(a2, a3), _mm_unpackhi_epi32(a2, a3));
    return _mm_add_epi32(_mm_unpacklo_epi64(t0, t1), _mm_unpackhi_epi64(t0, t1));
}
#define G_HADD4(a0, a1, a2, a3) g_hadd4(a0, a1, a2, a3)
/* quant() of four lanes in f64, the contract of tiles.c qscale: f = m * 2^-s, q = v * f, clamp(q +- 0.5) truncated
   (clamping before truncation is equivalent because lo and hi are integers). */
typedef struct {
    __m128d f01, f23, lo, hi;
} GQuant;
static inline double g_qscale(int32_t m, uint32_t s) {
    union {
        uint64_t u;
        double d;
    } p = {(uint64_t)(1023 - s) << 52};
    return (double)m * p.d;
}
static inline GQuant g_qprep(const int32_t *m, const uint32_t *s, uint32_t ms, const Op *o) {
    GQuant q;
    q.f01 = _mm_set_pd(g_qscale(m[ms], s[ms]), g_qscale(m[0], s[0]));
    q.f23 = _mm_set_pd(g_qscale(m[3 * ms], s[3 * ms]), g_qscale(m[2 * ms], s[2 * ms]));
    q.lo = _mm_set1_pd(o->lo);
    q.hi = _mm_set1_pd(o->hi);
    return q;
}
static inline __m128i g_qround(__m128d q, const GQuant *g) {
    const __m128d half = _mm_or_pd(_mm_and_pd(q, _mm_set1_pd(-0.0)), _mm_set1_pd(0.5));
    return _mm_cvttpd_epi32(_mm_min_pd(_mm_max_pd(_mm_add_pd(q, half), g->lo), g->hi));
}
static inline uint32_t g_quant4(__m128i v, const GQuant *g) {
    const __m128i r0 = g_qround(_mm_mul_pd(_mm_cvtepi32_pd(v), g->f01), g),
                  r1 = g_qround(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v, 8)), g->f23), g);
    const __m128i r = _mm_unpacklo_epi64(r0, r1), h = _mm_packs_epi32(r, r);
    return (uint32_t)_mm_cvtsi128_si32(_mm_packs_epi16(h, h));
}
static uint32_t g_wmax(const int8_t *w, size_t n) {
    uint32_t r = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t a = w[i] < 0 ? (uint32_t)(-(int32_t)w[i]) : (uint32_t)w[i];
        if (a > r) r = a;
    }
    return r;
}
#include <cpuid.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h> /* SSSE3 intrinsics, used only by target("ssse3") functions */
static int ng_has_ssse3(void) {
    static int v = -1;
    if (v < 0) {
        unsigned a, b, c, d;
        const char *f = getenv("GUST_X86"); /* tests: sse2 disables the SSSE3 tiles */
        v = __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 9)) != 0 && !(f && !strcmp(f, "sse2"));
    }
    return v;
}
#else
typedef v128_t G8;
typedef v128_t G16;
typedef v128_t G32;
#define G_LD(p)            wasm_v128_load(p)
#define G_LD32(p)          wasm_v128_load(p)
#define G_Z16              wasm_i16x8_splat(0)
#define G_Z32              wasm_i32x4_splat(0)
#define G_ADD32(a, b)      wasm_i32x4_add(a, b)
#define G_MLAL_LO(h, x, w) wasm_i16x8_add(h, wasm_i16x8_extmul_low_i8x16(x, w))
#define G_MLAL_HI(h, x, w) wasm_i16x8_add(h, wasm_i16x8_extmul_high_i8x16(x, w))
#define G_MULL_LO(x, w)    wasm_i16x8_extmul_low_i8x16(x, w)
#define G_MULL_HI(x, w)    wasm_i16x8_extmul_high_i8x16(x, w)
#define G_PADAL(a, h)      wasm_i32x4_add(a, wasm_i32x4_extadd_pairwise_i16x8(h))
static inline v128_t g_hadd4(v128_t a0, v128_t a1, v128_t a2, v128_t a3) { /* lane j: total of aj */
    const v128_t t0 = wasm_i32x4_add(wasm_i32x4_shuffle(a0, a1, 0, 4, 1, 5), wasm_i32x4_shuffle(a0, a1, 2, 6, 3, 7)),
                 t1 = wasm_i32x4_add(wasm_i32x4_shuffle(a2, a3, 0, 4, 1, 5), wasm_i32x4_shuffle(a2, a3, 2, 6, 3, 7));
    return wasm_i32x4_add(wasm_i32x4_shuffle(t0, t1, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t1, 2, 3, 6, 7));
}
#define G_HADD4(a0, a1, a2, a3) g_hadd4(a0, a1, a2, a3)
/* The exact f64 requantization of tiles.c (wasm_quant) for four channels. */
typedef struct {
    v128_t f01, f23, lo, hi;
} GQuant;
static inline GQuant g_qprep(const int32_t *m, const uint32_t *s, uint32_t ms, const Op *o) {
    GQuant q;
    q.f01 = wasm_f64x2_make(qscale(m[0], s[0]), qscale(m[ms], s[ms]));
    q.f23 = wasm_f64x2_make(qscale(m[2 * ms], s[2 * ms]), qscale(m[3 * ms], s[3 * ms]));
    q.lo = wasm_f64x2_splat(o->lo);
    q.hi = wasm_f64x2_splat(o->hi);
    return q;
}
static inline uint32_t g_quant4(v128_t v, const GQuant *q) {
    const v128_t r = wasm_quant(v, q->f01, q->f23, q->lo, q->hi), h = wasm_i16x8_narrow_i32x4(r, r);
    return wasm_u32x4_extract_lane(wasm_i8x16_narrow_i16x8(h, h), 0);
}
static uint32_t g_wmax(const int8_t *w, size_t n) {
    uint32_t r = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t a = w[i] < 0 ? (uint32_t)(-(int32_t)w[i]) : (uint32_t)w[i];
        if (a > r) r = a;
    }
    return r;
}
#endif
#define NG_P 4
#define NG_J 4
#define NG_CHUNK 512
#define NG_TASKS 16       /* channel tasks per operator, for balance over up to 4-8 threads */
#define NG_PARALLEL_MIN 2 /* fewer pixels: one thread */
#define NG_XBYTES 8192u /* bytes of input pixels per L1 block */
#define NG_BLOCK 16384u   /* bytes of decoded weights per channel block (L1 resident with one pixel tile) */

/* Steps of 16 bytes between int16 -> int32 widenings for weights of magnitude <= wmax (0: widen every product). */
static inline uint32_t ng_run(uint32_t wmax) {
    const uint32_t r = 32767u / (2u * 128u * (wmax ? wmax : 1u));
    return r > 64u ? 64u : r;
}

#if defined(NG_X86)
/* SSE2: operands sign-extended to int16 and multiplied in pairs by pmaddwd into int32 lanes (exact: |pair| <=
   2^15). Two pixels x two channels per pass (16 registers), four passes per 4 x 4 tile. */
static inline __m128i x86_lo16(__m128i v) { return _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8); }
static inline __m128i x86_hi16(__m128i v) { return _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 8); }
static void ng_tile(const int8_t *const *x, const int8_t *w, size_t wrow, uint32_t K, uint32_t run, __m128i c[4]) {
    (void)run;
    const uint32_t K16 = K & ~15u;
    __m128i a[4][4];
    for (int pp = 0; pp < 4; pp += 2)
        for (int jj = 0; jj < 4; jj += 2) {
            const int8_t *x0 = x[pp], *x1 = x[pp + 1], *u0 = w + (size_t)jj * wrow, *u1 = u0 + wrow;
            __m128i s00 = _mm_setzero_si128(), s01 = s00, s10 = s00, s11 = s00;
            for (uint32_t k = 0; k < K16; k += 16) {
                const __m128i xa = G_LD(x0 + k), xb = G_LD(x1 + k), wa = G_LD(u0 + k), wb = G_LD(u1 + k);
                const __m128i xal = x86_lo16(xa), xah = x86_hi16(xa), xbl = x86_lo16(xb), xbh = x86_hi16(xb);
                const __m128i wal = x86_lo16(wa), wah = x86_hi16(wa), wbl = x86_lo16(wb), wbh = x86_hi16(wb);
                s00 = _mm_add_epi32(s00, _mm_add_epi32(_mm_madd_epi16(xal, wal), _mm_madd_epi16(xah, wah)));
                s01 = _mm_add_epi32(s01, _mm_add_epi32(_mm_madd_epi16(xal, wbl), _mm_madd_epi16(xah, wbh)));
                s10 = _mm_add_epi32(s10, _mm_add_epi32(_mm_madd_epi16(xbl, wal), _mm_madd_epi16(xbh, wah)));
                s11 = _mm_add_epi32(s11, _mm_add_epi32(_mm_madd_epi16(xbl, wbl), _mm_madd_epi16(xbh, wbh)));
            }
            a[pp][jj] = s00;
            a[pp][jj + 1] = s01;
            a[pp + 1][jj] = s10;
            a[pp + 1][jj + 1] = s11;
        }
    const int8_t *w0 = w, *w1 = w + wrow, *w2 = w + 2 * wrow, *w3 = w + 3 * wrow;
    for (int p = 0; p < 4; p++) {
        c[p] = G_HADD4(a[p][0], a[p][1], a[p][2], a[p][3]);
        if (K16 < K) {
            int32_t t[4] = {0, 0, 0, 0};
            for (uint32_t i = K16; i < K; i++) {
                const int32_t xi = x[p][i];
                t[0] += xi * w0[i];
                t[1] += xi * w1[i];
                t[2] += xi * w2[i];
                t[3] += xi * w3[i];
            }
            c[p] = G_ADD32(c[p], G_LD32(t));
        }
    }
}
/* SSSE3 (selected at run time): weights stored offset into 0..127 meet pmaddubsw as unsigned bytes against signed
   activations; pairs are at most 2 * 127 * 128 < 2^15, so nothing saturates, and pmaddwd by ones widens them
   exactly. The caller removes off * sum(x_p). Four pixels x two channels per pass, two passes per tile. */
__attribute__((target("ssse3"))) static void ng_tile_off(const int8_t *const *x, const int8_t *w, size_t wrow,
                                                         uint32_t K, __m128i c[4]) {
    const uint32_t K16 = K & ~15u;
    const __m128i one = _mm_set1_epi16(1);
    __m128i a[4][4];
    for (int jj = 0; jj < 4; jj += 2) {
        const int8_t *u0 = w + (size_t)jj * wrow, *u1 = u0 + wrow;
        __m128i s[4][2];
        for (int p = 0; p < 4; p++) s[p][0] = s[p][1] = _mm_setzero_si128();
        for (uint32_t k = 0; k < K16; k += 16) {
            const __m128i wa = G_LD(u0 + k), wb = G_LD(u1 + k);
            for (int p = 0; p < 4; p++) {
                const __m128i xv = G_LD(x[p] + k);
                s[p][0] = _mm_add_epi32(s[p][0], _mm_madd_epi16(_mm_maddubs_epi16(wa, xv), one));
                s[p][1] = _mm_add_epi32(s[p][1], _mm_madd_epi16(_mm_maddubs_epi16(wb, xv), one));
            }
        }
        for (int p = 0; p < 4; p++) {
            a[p][jj] = s[p][0];
            a[p][jj + 1] = s[p][1];
        }
    }
    const int8_t *w0 = w, *w1 = w + wrow, *w2 = w + 2 * wrow, *w3 = w + 3 * wrow;
    for (int p = 0; p < 4; p++) {
        c[p] = G_HADD4(a[p][0], a[p][1], a[p][2], a[p][3]);
        if (K16 < K) {
            int32_t t[4] = {0, 0, 0, 0};
            for (uint32_t i = K16; i < K; i++) {
                const int32_t xi = x[p][i];
                t[0] += xi * w0[i];
                t[1] += xi * w1[i];
                t[2] += xi * w2[i];
                t[3] += xi * w3[i];
            }
            c[p] = G_ADD32(c[p], G_LD32(t));
        }
    }
}
#else
/* Sums of a 4 x 4 tile over K: c[p] holds channels j = 0..3 of pixel p. */
static inline __attribute__((always_inline)) void ng_tile(const int8_t *const *x, const int8_t *w, size_t wrow,
                                                          uint32_t K, uint32_t run, G32 c[4]) {
    const uint32_t K16 = K & ~15u;
    const int8_t *w0 = w, *w1 = w + wrow, *w2 = w + 2 * wrow, *w3 = w + 3 * wrow;
    G32 a[4][4];
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 4; j++) a[p][j] = G_Z32;
    uint32_t k = 0;
    if (run == 0) {
        for (; k < K16; k += 16) {
            const G8 wv[4] = {G_LD(w0 + k), G_LD(w1 + k), G_LD(w2 + k), G_LD(w3 + k)};
            for (int p = 0; p < 4; p++) {
                const G8 xv = G_LD(x[p] + k);
                for (int j = 0; j < 4; j++) {
                    a[p][j] = G_PADAL(a[p][j], G_MULL_LO(xv, wv[j]));
                    a[p][j] = G_PADAL(a[p][j], G_MULL_HI(xv, wv[j]));
                }
            }
        }
    } else
        while (k < K16) {
            const uint32_t end = K16 - k > 16u * run ? k + 16u * run : K16;
            G16 h[4][4];
            for (int p = 0; p < 4; p++)
                for (int j = 0; j < 4; j++) h[p][j] = G_Z16;
            for (; k < end; k += 16) {
                const G8 wv[4] = {G_LD(w0 + k), G_LD(w1 + k), G_LD(w2 + k), G_LD(w3 + k)};
                const G8 xv[4] = {G_LD(x[0] + k), G_LD(x[1] + k), G_LD(x[2] + k), G_LD(x[3] + k)};
                for (int p = 0; p < 4; p++)
                    for (int j = 0; j < 4; j++) h[p][j] = G_MLAL_LO(h[p][j], xv[p], wv[j]);
                for (int p = 0; p < 4; p++)
                    for (int j = 0; j < 4; j++) h[p][j] = G_MLAL_HI(h[p][j], xv[p], wv[j]);
            }
            for (int p = 0; p < 4; p++)
                for (int j = 0; j < 4; j++) a[p][j] = G_PADAL(a[p][j], h[p][j]);
        }
    for (int p = 0; p < 4; p++) {
        c[p] = G_HADD4(a[p][0], a[p][1], a[p][2], a[p][3]);
        if (K16 < K) {
            int32_t t[4] = {0, 0, 0, 0};
            for (uint32_t i = K16; i < K; i++) {
                const int32_t xi = x[p][i];
                t[0] += xi * w0[i];
                t[1] += xi * w1[i];
                t[2] += xi * w2[i];
                t[3] += xi * w3[i];
            }
            c[p] = G_ADD32(c[p], G_LD32(t));
        }
    }
}
#endif

#if defined(__wasm_simd128__)
/* Whether relaxed-SIMD builds run the GEMM with the relaxed dot product (default) or the plain SIMD128 tiles; results
   are identical, and plain SIMD128 builds ignore it. V8 emulates the relaxed dot product on ARM cores without
   dot-product instructions (Cortex-A53), where the plain tiles are about 1.8x faster; elsewhere the relaxed form is
   faster. A WebAssembly module has no clock to measure this, so the host may time a run each way and choose. */
static int ng_relaxed = 1;
GUST_API void gust_prefer_relaxed_dot(uint32_t on) { ng_relaxed = on != 0; }
#endif
#if defined(__wasm_relaxed_simd__)
/* Relaxed-SIMD form (WebAssembly builds for Chrome and Firefox): weights stored offset by `off` into 0..127 (i7), so
   i32x4.relaxed_dot_i8x16_i7x16_add adds four exact products per lane: pairs of i8 x i7 are at most 2 * 128 * 127
   in magnitude, below the i16 saturation of its x86 lowering, and lanes stay far inside int32. The caller removes
   off * sum(x_p) afterwards. V8 maps it to vpdpbusd / pmaddubsw on x86 and sdot on ARM with dot products. */
static inline __attribute__((always_inline)) void ng_tile_relaxed(const int8_t *const *x, const int8_t *w, size_t wrow,
                                                                  uint32_t K, v128_t c[4]) {
    const uint32_t K16 = K & ~15u;
    const int8_t *w0 = w, *w1 = w + wrow, *w2 = w + 2 * wrow, *w3 = w + 3 * wrow;
    v128_t a[4][4];
    for (int p = 0; p < 4; p++)
        for (int j = 0; j < 4; j++) a[p][j] = wasm_i32x4_splat(0);
    for (uint32_t k = 0; k < K16; k += 16) {
        const v128_t wv[4] = {G_LD(w0 + k), G_LD(w1 + k), G_LD(w2 + k), G_LD(w3 + k)};
        const v128_t xv[4] = {G_LD(x[0] + k), G_LD(x[1] + k), G_LD(x[2] + k), G_LD(x[3] + k)};
        for (int p = 0; p < 4; p++)
            for (int j = 0; j < 4; j++) a[p][j] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(xv[p], wv[j], a[p][j]);
    }
    for (int p = 0; p < 4; p++) {
        c[p] = G_HADD4(a[p][0], a[p][1], a[p][2], a[p][3]);
        if (K16 < K) {
            int32_t t[4] = {0, 0, 0, 0};
            for (uint32_t i = K16; i < K; i++) {
                const int32_t xi = x[p][i];
                t[0] += xi * w0[i];
                t[1] += xi * w1[i];
                t[2] += xi * w2[i];
                t[3] += xi * w3[i];
            }
            c[p] = G_ADD32(c[p], G_LD32(t));
        }
    }
}
#define ng_tile_off ng_tile_relaxed
#endif
#if defined(__wasm_relaxed_simd__) || defined(NG_X86)
/* Offset-weight dot products (relaxed-SIMD WebAssembly; SSSE3 at run time on the x86 base variant). */
#define NG_OFF 1
static inline int ng_off_on(void) {
#if defined(NG_X86)
    return ng_has_ssse3();
#else
    return ng_relaxed;
#endif
}
/* Adds off to n weight bytes (the offset operand form; every result lies in 0..127). */
static void ng_offset(int8_t *w, uint32_t n, int8_t off) {
    uint32_t i = 0;
#if defined(NG_X86)
    const __m128i v = _mm_set1_epi8(off);
    for (; i + 16 <= n; i += 16)
        _mm_storeu_si128((__m128i *)(void *)(w + i), _mm_add_epi8(_mm_loadu_si128((const __m128i *)(const void *)(w + i)), v));
#else
    const v128_t v = wasm_i8x16_splat(off);
    for (; i + 16 <= n; i += 16) wasm_v128_store(w + i, wasm_i8x16_add(wasm_v128_load(w + i), v));
#endif
    for (; i < n; i++) w[i] = (int8_t)(w[i] + off);
}
#endif

/* Channels oc0 .. oc0 + nb - 1 of the GEMM for every pixel: weights w (nb rows of wrow bytes, K used). bias NULL:
   none; ms 0: one multiplier and shift for every channel (MATMUL), 1: per channel. off > 0 (relaxed-SIMD builds):
   weights are stored as w + off, and off * sum(x_p) is removed from every sum. */
static void ng_block(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, size_t wrow, uint32_t K,
                     uint32_t oc0, uint32_t nb, const int32_t *bias, const int32_t *m, const uint32_t *s,
                     uint32_t run, uint32_t ms, uint32_t off, const Op *o) {
    const uint32_t nt = nb & ~3u;
#if !defined(NG_OFF)
    (void)off;
#endif
#if defined(__aarch64__)
    const int use_dot = ng_has_dot();
#endif
    /* Blocks of pixels (about NG_XBYTES of input) stay in L1 while every 4-channel group of the block's
       weights sweeps them. */
    uint32_t pb = NG_XBYTES / (K > 16 ? K : 16) & ~3u;
    if (pb < 4) pb = 4;
    for (uint32_t q0 = 0; q0 < np; q0 += pb) {
        const uint32_t qe = np - q0 < pb ? np : q0 + pb;
#if defined(NG_OFF)
        int32_t sx[NG_XBYTES / 16 + 4]; /* off * sum(x_p) of the block's pixels (pb <= NG_XBYTES / 16 when off) */
        if (off)
            for (uint32_t p = q0; p < qe; p++) {
                int32_t t = 0;
                for (uint32_t k = 0; k < K; k++) t += xs[p][k];
                sx[p - q0] = (int32_t)off * t;
            }
#endif
        for (uint32_t j0 = 0; j0 < nt; j0 += 4) {
            const uint32_t oc = oc0 + j0;
            const GQuant q = g_qprep(m + oc * ms, s + oc * ms, ms, o);
            const G32 b = bias ? G_LD32(bias + oc) : G_Z32;
            for (uint32_t p0 = q0; p0 < qe; p0 += 4) {
                const uint32_t P = qe - p0 < 4 ? qe - p0 : 4;
                const int8_t *x[4];
                for (uint32_t p = 0; p < 4; p++) x[p] = xs[p0 + (p < P ? p : 0)]; /* short tiles repeat a pixel */
                G32 c[4];
#if defined(NG_OFF)
                if (off) {
                    ng_tile_off(x, w + (size_t)j0 * wrow, wrow, K, c);
                    for (uint32_t p = 0; p < P; p++) {
                        int32_t l[4];
                        __builtin_memcpy(l, &c[p], 16);
                        for (int i = 0; i < 4; i++) l[i] -= sx[p0 + p - q0];
                        __builtin_memcpy(&c[p], l, 16);
                    }
                } else
#endif
#if defined(__aarch64__)
                    if (use_dot)
                        ng_tile_dot(x, w + (size_t)j0 * wrow, wrow, K, c);
                    else
#endif
                        ng_tile(x, w + (size_t)j0 * wrow, wrow, K, run, c);
                for (uint32_t p = 0; p < P; p++) {
                    const uint32_t v = g_quant4(G_ADD32(c[p], b), &q);
                    __builtin_memcpy(ys[p0 + p] + oc, &v, 4);
                }
            }
        }
    }
    for (uint32_t j0 = nt; j0 < nb; j0++) {
        const uint32_t oc = oc0 + j0;
        for (uint32_t p = 0; p < np; p++) {
            int32_t d = dot(xs[p], w + (size_t)j0 * wrow, K);
            if (off) /* offset weights: remove off * sum(x) */
                for (uint32_t k = 0; k < K; k++) d -= (int32_t)off * xs[p][k];
            ys[p][oc] = quant((bias ? bias[oc] : 0) + d, m[oc * ms], s[oc * ms], o);
        }
    }
}

/* The GEMM of one operator, split into tasks of nb output channels (gust_parallel; every output is computed by one
   task with the same code, so results do not depend on the thread count). packed: codes of `bits` bits (row = K
   codes per channel, consecutive; |w| <= 2^(bits-1)), each task decoding its block into an L1 buffer; otherwise
   byte weights, N rows of wrow bytes. */
typedef struct {
    const int8_t *const *xs;
    int8_t *const *ys;
    const int8_t *w;
    const uint8_t *packed;
    const int32_t *bias, *m;
    const uint32_t *s;
    const Op *o;
    size_t wrow;
    uint32_t np, K, N, nb, bits, ms;
} NgJob;
static void ng_task(void *opaque, uint32_t item, uint32_t thread) {
    (void)thread;
    const NgJob *j = opaque;
    const uint32_t oc = item * j->nb, cnt = j->N - oc < j->nb ? j->N - oc : j->nb;
#ifdef GUST_PACKED_WEIGHTS
    if (j->packed) {
        int8_t buf[NG_BLOCK] __attribute__((aligned(16)));
        packed_expand(j->packed, (uint64_t)oc * j->K, cnt * j->K, j->bits, buf);
        uint32_t off = 0;
#if defined(NG_OFF)
        if (ng_off_on()) {
            off = 1u << (j->bits - 1); /* codes -2^(b-1) .. 2^(b-1)-1 become 0 .. 2^b - 1 */
            ng_offset(buf, cnt * j->K, (int8_t)off);
        }
#endif
        ng_block(j->xs, j->ys, j->np, buf, j->K, j->K, oc, cnt, j->bias, j->m, j->s, ng_run(1u << (j->bits - 1)), 1,
                 off, j->o);
        return;
    }
#endif
    const int8_t *wb = j->w + (size_t)oc * j->wrow;
    uint32_t wmax = 0;
    for (uint32_t r = 0; r < cnt; r++) {
        const uint32_t v = g_wmax(wb + (size_t)r * j->wrow, j->K);
        if (v > wmax) wmax = v;
    }
#if defined(NG_OFF)
    if (ng_off_on() && wmax <= 63 && (size_t)cnt * j->K <= NG_BLOCK) { /* |w| <= 63: w + 64 fits i7 */
        int8_t buf[NG_BLOCK] __attribute__((aligned(16)));
        for (uint32_t r = 0; r < cnt; r++) __builtin_memcpy(buf + (size_t)r * j->K, wb + (size_t)r * j->wrow, j->K);
        ng_offset(buf, cnt * j->K, 64);
        ng_block(j->xs, j->ys, j->np, buf, j->K, j->K, oc, cnt, j->bias, j->m, j->s, 0, j->ms, 64, j->o);
        return;
    }
#endif
    ng_block(j->xs, j->ys, j->np, wb, j->wrow, j->K, oc, cnt, j->bias, j->m, j->s, wmax >= 128 ? 0 : ng_run(wmax), j->ms,
             0, j->o);
}
static void ng_run_job(NgJob *j) {
    /* Blocks fit NG_BLOCK and give at least NG_TASKS tasks for balance across threads. */
    uint32_t nb = NG_BLOCK / (j->K ? j->K : 1) / NG_J * NG_J, even = (j->N + NG_TASKS - 1) / NG_TASKS;
    even = (even + NG_J - 1) / NG_J * NG_J;
    if (even < nb) nb = even;
    if (nb < NG_J) nb = NG_J;
    j->nb = nb;
    const uint32_t tasks = (j->N + nb - 1) / nb;
#if defined(GUST_FLOAT32) && defined(GUST_THREADS)
    if (j->np >= NG_PARALLEL_MIN) {
        gust_parallel(ng_task, j, tasks);
        return;
    }
#endif
    for (uint32_t i = 0; i < tasks; i++) ng_task(j, i, 0);
}
static void g128_gemm(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *w, size_t wrow,
                      uint32_t K, uint32_t N, const int32_t *bias, const int32_t *m, const uint32_t *s, const Op *o) {
    NgJob j = {xs, ys, w, NULL, bias, m, s, o, wrow, np, K, N, 0, 0, 1};
    ng_run_job(&j);
}
/* MATMUL scores: queries x keys (rows of K bytes), one multiplier and shift, no bias. */
static void g128_scores(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const int8_t *keys, uint32_t K,
                        uint32_t N, int32_t mul, uint32_t sh, const Op *o) {
    NgJob j = {xs, ys, keys, NULL, NULL, &mul, &sh, o, K, np, K, N, 0, 0, 0};
    ng_run_job(&j);
}
#ifdef GUST_PACKED_WEIGHTS
static void g128_gemm_packed(const int8_t *const *xs, int8_t *const *ys, uint32_t np, const uint8_t *packed,
                             uint32_t bits, uint32_t K, uint32_t N, const int32_t *bias, const int32_t *m,
                             const uint32_t *s, const Op *o) {
    NgJob j = {xs, ys, NULL, packed, bias, m, s, o, K, np, K, N, 0, bits, 1};
    ng_run_job(&j);
}
#endif
#define GEMM128 1
#endif
