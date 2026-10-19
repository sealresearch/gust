/* float32 graph kernels: shared definitions: vector types, profiling hooks, weight decoding. Part of the src/gust.c translation unit. */
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
/* float32-graph-v1 kernels: packed-code restoration, convolution tiles and the attention inner loops.

   Every kernel fixes its reduction order, and the SIMD and scalar variants spell out the same order, so all builds
   return identical bits:
     - a convolution output sums over k = 0 .. K-1 in increasing order; lanes are output channels, never the
       reduction axis, so tiling, blocking and SIMD width cannot change a result;
     - the attention dot product keeps two partial sums, one for the even and one for the odd head dimensions
       (a trailing odd dimension joins the even sum), added as even + odd;
     - the attention value sum runs over the keys in increasing order for each output dimension.
   Multiplies and adds are always separate rounded operations (contraction is off). */
#pragma clang fp contract(off)

/* GUST_F32_PROFILE (measurement builds only): milliseconds per operator code (slot = code) and per int16-convolution
   phase (slots 1 scales/quantize, 2 code panels, 3 tiles + epilogue); slot 4 measures zeroing output tails,
   outside the operator totals. Read/reset each slot with gust_profile(slot). */
#ifdef GUST_F32_PROFILE
#if defined(__wasm__)
__attribute__((import_module("env"), import_name("gust_now"))) double gust_now(void);
#else
double gust_now(void); /* native measurement builds: milliseconds, provided by the benchmark */
#endif
static double gust_prof[64];
#if defined(__wasm__)
__attribute__((export_name("gust_profile")))
#endif
double gust_profile(uint32_t slot) {
    const double v = slot < 64u ? gust_prof[slot] : 0.0;
    if (slot < 64u) gust_prof[slot] = 0.0;
    return v;
}
#define GP_START(v) const double v = gust_now()
#define GP_ADD(slot, v) (gust_prof[slot] += gust_now() - (v))
#else
#define GP_START(v) (void)0
#define GP_ADD(slot, v) (void)0
#endif

typedef float GfF __attribute__((may_alias));
typedef double GfD __attribute__((may_alias));
#ifdef GUST_F32_SIMD
typedef float gv4f __attribute__((vector_size(16), aligned(4), may_alias));
typedef float gv2f __attribute__((vector_size(8), aligned(4), may_alias));
typedef double gv2d __attribute__((vector_size(16), aligned(8), may_alias));
#endif

/* Packed weights of one F_CONV operator: row-major [N][K] codes of `bits` bits as a little-endian bit stream, and
   either one float32 scale per row or one float16 scale per 64 consecutive weights of a row. */
typedef struct {
    const uint8_t *code;
    const uint8_t *scale;
    uint64_t K;
    uint32_t bits, grouped, ngroups;
} GfWeights;

/* Restores weights k0 .. k0+kl-1 of output row n as float32(code) * float32(scale) into dst[0], dst[stride], ...
   A sequential bit reader that touches only the bytes the codes occupy. */
static inline __attribute__((always_inline)) void gf_decode_row(const GfWeights *w, const uint32_t bits, uint32_t n,
                                                                uint32_t k0, uint32_t kl, float *dst, uint32_t stride) {
#ifdef GUST_FLOAT32_TEMPORAL
    if (bits == 32) { /* float32 weights (float32-temporal-v1): float32(value) * float32(scale), no codes */
        const GfF *f = (const GfF *)(w->code) + (size_t)n * w->K + k0;
        const float s = *(const GfF *)(w->scale + 4ull * n);
        for (uint32_t i = 0; i < kl; i++) dst[(size_t)i * stride] = f[i] * s;
        return;
    }
#endif
    const uint64_t pos = ((uint64_t)n * w->K + k0) * bits;
    if (bits == 4 && !(pos & 7u) && !w->grouped) { /* byte-aligned 4-bit row: two codes per byte, low nibble first */
        const uint8_t *b = w->code + (pos >> 3);
        const float s = *(const GfF *)(w->scale + 4ull * n);
        uint32_t i = 0;
        for (; i + 2 <= kl; i += 2, b++) {
            dst[(size_t)i * stride] = (float)((int32_t)((*b & 15u) ^ 8u) - 8) * s;
            dst[(size_t)(i + 1) * stride] = (float)((int32_t)((*b >> 4) ^ 8u) - 8) * s;
        }
        if (i < kl) dst[(size_t)i * stride] = (float)((int32_t)((*b & 15u) ^ 8u) - 8) * s;
        return;
    }
    if ((bits == 4 || bits == 8) && !(pos & 7u) && w->grouped) {
        /* Byte-aligned grouped 4- and 8-bit rows (float16 scale per 64 codes): whole bytes instead of the bit reader,
           the same float32(code) * float32(scale) as the general path below. */
        const uint8_t *base = w->code + (pos >> 3), *gs = w->scale + 2ull * n * w->ngroups;
        uint32_t i = 0;
        while (i < kl) {
            const uint32_t kk = k0 + i, g = kk >> 6, room = 64u - (kk & 63u);
            const uint32_t seg = kl - i < room ? kl - i : room;
            const float s = gf_f16((uint16_t)(gs[2u * g] | (uint32_t)gs[2u * g + 1u] << 8));
            if (bits == 4)
                for (uint32_t j = i; j < i + seg; j++) {
                    const uint32_t v = (j & 1u) ? (uint32_t)base[j >> 1] >> 4 : base[j >> 1] & 15u;
                    dst[(size_t)j * stride] = (float)((int32_t)(v ^ 8u) - 8) * s;
                }
            else
                for (uint32_t j = i; j < i + seg; j++) dst[(size_t)j * stride] = (float)(int8_t)base[j] * s;
            i += seg;
        }
        return;
    }
    const uint8_t *p = w->code + (pos >> 3);
    const uint32_t sh = (uint32_t)(pos & 7u), mask = (1u << bits) - 1u, sign = 1u << (bits - 1u);
    uint32_t acc = 0, nb = 0;
    if (sh) {
        acc = (uint32_t)*p++ >> sh;
        nb = 8u - sh;
    }
    float row = 0.0f;
    const uint8_t *gs = w->scale;
    if (w->grouped)
        gs += 2ull * n * w->ngroups;
    else
        row = *(const GfF *)(w->scale + 4ull * n);
    uint32_t i = 0;
    while (i < kl) {
        uint32_t seg = kl - i;
        float s = row;
        if (w->grouped) {
            const uint32_t kk = k0 + i, g = kk >> 6, room = 64u - (kk & 63u);
            s = gf_f16((uint16_t)(gs[2u * g] | (uint32_t)gs[2u * g + 1u] << 8));
            if (seg > room) seg = room;
        }
        for (uint32_t j = 0; j < seg; j++) {
            while (nb < bits) {
                acc |= (uint32_t)*p++ << nb;
                nb += 8u;
            }
            const uint32_t v = acc & mask;
            acc >>= bits;
            nb -= bits;
            dst[(size_t)(i + j) * stride] = (float)((int32_t)(v ^ sign) - (int32_t)sign) * s;
        }
        i += seg;
    }
}

/* Restores a [kl][4] panel for outputs n0 .. n0+nc-1 (unused columns are zero). */
static void gf_decode_panel(const GfWeights *w, uint32_t n0, uint32_t nc, uint32_t k0, uint32_t kl, float *dst) {
    for (uint32_t j = 0; j < 4; j++) {
        if (j >= nc) {
            for (uint32_t k = 0; k < kl; k++) dst[(size_t)k * 4 + j] = 0.0f;
            continue;
        }
        switch (w->bits) {
        case 2: gf_decode_row(w, 2, n0 + j, k0, kl, dst + j, 4); break;
        case 3: gf_decode_row(w, 3, n0 + j, k0, kl, dst + j, 4); break;
        case 4: gf_decode_row(w, 4, n0 + j, k0, kl, dst + j, 4); break;
        case 5: gf_decode_row(w, 5, n0 + j, k0, kl, dst + j, 4); break;
        case 6: gf_decode_row(w, 6, n0 + j, k0, kl, dst + j, 4); break;
        case 7: gf_decode_row(w, 7, n0 + j, k0, kl, dst + j, 4); break;
#ifdef GUST_FLOAT32_TEMPORAL
        case 32: gf_decode_row(w, 32, n0 + j, k0, kl, dst + j, 4); break;
#endif
        default: gf_decode_row(w, 8, n0 + j, k0, kl, dst + j, 4); break;
        }
    }
}
