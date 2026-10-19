/* float32-graph-v1: shared constants and forward declarations (builds with GUST_FLOAT32 only).
   Part of the runtime/gust.c translation unit; see runtime/src/float32/README.md and docs/FORMAT.md. */
#ifndef GUST_FLOAT32_ABI_H
#define GUST_FLOAT32_ABI_H
#if defined(GUST_FLOAT32) && !defined(GUST_ATTENTION)
#error "GUST_FLOAT32 builds also define the ASR features (GUST_ATTENTION, GUST_CONFORMER, GUST_PACKED_WEIGHTS)"
#endif

/* Header.flags value of a float32 graph; Context.ready value of a loaded float32 graph (legacy contexts use 1). */
#define GUST_F32_FLAGS 1u
#define GUST_F32_READY 2u
/* Tensor.flags bit 4: float32 tensor (required on every tensor of a float32 graph). */
#define GUST_F32_TENSOR 4u
/* Capability bits 9 (float32 graphs) and 11 (per-input rows). A decoder build (GUST_FLOAT32_DECODER,
   src/float32/decoder.h) reports bit 10 as well through the same macro, which gust_capabilities() ORs in, and bit 12
   (float32-t5-v1, T5.md): RMS norm, ReLU, additive attention bias, grouped gather scales and grouped partial sums,
   the operator modes of an encoder-decoder T5. */
#ifdef GUST_FLOAT32_DECODER
#define GUST_F32_DECODER_CAPABILITY 1024u
#define GUST_F32_T5_CAPABILITY      4096u
#define GUST_FLOAT32_T5 /* float32-t5-v1 is part of every decoder build */
/* Bit 13 (float32-temporal-v1, TEMPORAL.md): depthwise temporal convolution, FiLM conditioning, row mean and
   int16 activations with 5- to 8-bit weights, for convolution networks over time. */
#define GUST_F32_TEMPORAL_CAPABILITY 8192u
#define GUST_FLOAT32_TEMPORAL /* float32-temporal-v1 is part of every decoder build */
#define GUST_F32_CAPABILITY                                                                                            \
    (512u | 2048u | GUST_F32_DECODER_CAPABILITY | GUST_F32_T5_CAPABILITY | GUST_F32_TEMPORAL_CAPABILITY)
#else
#define GUST_F32_CAPABILITY (512u | 2048u)
#endif

/* Largest tensor dimension, and the reduction block of the convolution kernels. The block never changes a result
   (partial sums are carried exactly), only how many weights are restored at once: 4 output columns of at most
   GUST_F32_KBLOCK weights each fit the 64 KiB Context scratch. Test builds shrink it. */
#define GUST_F32_MAX_DIM (1u << 24)
#ifndef GUST_F32_KBLOCK
#define GUST_F32_KBLOCK 4096u
#endif
#if GUST_F32_KBLOCK < 1u || GUST_F32_KBLOCK > 4096u
#error "GUST_F32_KBLOCK must be in 1..4096"
#endif
/* Longest key sequence of F_MHA: its score row lives in the 64 KiB Context scratch. */
#define GUST_F32_MAX_KEYS 16384u
/* F_CONV p0 bit 4 (int16 activations): most output rows (one float32 scale each in the scratch) and the widest
   reduction (a GQ_TR-row tile and an 8-column panel of int16 must fit beside the scales). */
#define GQ_MAX_ROWS 2048u
#define GQ_MAX_K 2048u
/* Widest attention head. */
#define GUST_F32_MAX_HEAD 64u

#if defined(__wasm_simd128__) || defined(__SSE2__) || defined(__aarch64__)
#define GUST_F32_SIMD 1
#endif

/* Operators 32 to 37; 17 to 31 and everything above 37 stay unsupported. */
enum { GUST_F_CONV = 32, GUST_F_ADD = 33, GUST_F_NORM = 34, GUST_F_UNARY = 35, GUST_F_ROPE = 36, GUST_F_MHA = 37 };

static int float_init(Context *c, const void *blob, uint32_t bytes, void *arena, uint32_t capacity);
static int float_run(Context *c, gust_stop_callback stop, void *opaque, uint32_t valid,
                     gust_trace_callback observer, void *trace_opaque);
static int float_run_inputs(Context *c, gust_stop_callback stop, void *opaque, uint32_t valid,
                            const uint32_t *input_rows, gust_trace_callback observer, void *trace_opaque);
#endif
