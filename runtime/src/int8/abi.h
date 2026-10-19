/* Binary layout of a model artifact and the private execution context.
   Part of the runtime/gust.c translation unit; not a public header. */
#include "../../include/gust.h"
#include <limits.h>
#if defined(GUST_CONFORMER) && !defined(GUST_ATTENTION)
#error "GUST_CONFORMER needs GUST_ATTENTION (valid-length execution)"
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif
/* AVX-512 VNNI kernels (int8/tiles.c, ops/gemm.c, ops/depthwise.c): the x86-64 vnni variant (dispatch_x86.c). */
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__)
#define GUST_INT8_VNNI 1
#endif

#ifdef GUST_PACKED_WEIGHTS
/* Limits of packed 3/4-bit convolutions (the loader refuses larger ones; byte INT8 keeps its wider limits). A
   dense operator decodes four output rows at a time, a depthwise one its whole filter, into PACKED_SCRATCH bytes. */
#define PACKED_DENSE_TERMS       4096u  /* kh * kw * input channels of a dense convolution */
#define PACKED_DEPTHWISE_WEIGHTS 32768u /* kh * kw * channels of a depthwise convolution */
#define PACKED_SCRATCH           32768u
/* A packed dense convolution runs the byte kernels on blocks of output channels whose codes are decoded into this
   many stack bytes (at least 16 rows of PACKED_DENSE_TERMS); a depthwise one decodes its filter into PACKED_SCRATCH. */
#define PACKED_BLOCK 65536u
#endif

/* Fixed-size little-endian records; see docs/FORMAT.md. */
typedef struct {
    uint8_t magic[8];
    uint32_t version, total, crc, nt, no, to, oo, data, arena, ni, nout, flags, r0, r1;
} Header;
typedef struct {
    uint32_t h, w, c, offset, bytes;
    float scale;
    uint32_t flags, reserved;
} Tensor;
typedef struct {
    uint32_t code, a, b, out, weights, bias, mult, shift, kh, kw, sy, sx, py, px, dy, dx, p0, p1;
    int32_t lo, hi;
} Op;
#ifdef GUST_FLOAT32
#define GUST_F32_ACT_SLOTS (1u << 21) /* int16 values: rows * (K rounded up to even) of one int16-activation conv */
#endif
typedef struct {
    const Header *header;
    const Tensor *t;
    const Op *ops;
    const uint8_t *blob;
    int8_t *arena;
    uint32_t ready;
    /* Set by gust_init when every operator supports valid-length execution (gust_run_valid). */
    uint32_t valid_ok;
    /* Filled by gust_init. nonneg: bit per tensor, every value is >= 0 (its producer clamps at zero), so the kernels
       may treat it as unsigned 0..127. small: bit per operator, a convolution whose weights all lie in [-63, 63]
       (packed 3/4-bit codes always do), for the AVX2 offset-weight tiles. */
    uint32_t nonneg[128];
    uint32_t small[128];
    /* Filled by gust_init: per-operator offset into packed_weights (UINT32_MAX
       when unused). Pointwise weights are widened to int16 for the madd path;
       uniform requantize operators store an exact 256-entry lookup table. */
    uint32_t packed_offsets[4096];
    int16_t packed_weights[32768];
#ifdef GUST_FLOAT32
#ifdef GUST_SHARED_SCRATCH
    /* GUST_SHARED_SCRATCH: the two scratch areas below live once per process (format.c) and every context points at
       them, 5 MiB less per loaded graph; runs of different contexts must then be serialized too. */
    int16_t *f32_act;
    int16_t (*f32_slab)[32768];
#else
    /* float32 F_CONV p0 bit 4 (int16 activations): every input row of the operator quantized once. */
    int16_t f32_act[GUST_F32_ACT_SLOTS] __attribute__((aligned(16)));
    /* 64 KiB per thread of the float32 work sharing (threads.c): an int16 code slab or a float32 weight panel */
    int16_t f32_slab[16][32768] __attribute__((aligned(16)));
#endif
#endif
} Context;
_Static_assert(sizeof(Header) == 64, "header ABI");
_Static_assert(sizeof(Tensor) == 32, "tensor ABI");
_Static_assert(sizeof(Op) == 80, "operator ABI");
