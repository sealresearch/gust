#ifndef GUST_H
#define GUST_H
#include <stdint.h>
#include <stddef.h>
#ifdef _WIN32
#define GUST_API __declspec(dllexport)
#else
#define GUST_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
enum {
    GUST_OK = 0,
    GUST_FORMAT = 1,
    GUST_CHECKSUM = 2,
    GUST_SHAPE = 3,
    GUST_MEMORY = 4,
    GUST_OPERATOR = 5,
    GUST_QUANTIZATION = 6,
    GUST_GRAPH = 7,
    GUST_BUDGET = 8
};
enum {
    GUST_CONV = 1,
    GUST_DEPTHWISE = 2,
    GUST_ADD = 3,
    GUST_POOL = 4,
    GUST_RESIZE = 5,
    GUST_CONCAT = 6,
    GUST_REQUANTIZE = 7,
    GUST_SLICE = 8,
    GUST_GATHER = 9,
    GUST_TOPK_POOL = 10,
    GUST_WEIGHTED_POOL = 11,
    /* Attention (capability bit 5, builds with GUST_ATTENTION): exact integer scores and softmax-weighted values. */
    GUST_MATMUL = 12,
    GUST_ATTEND = 13,
    /* Conformer (capability bit 7, builds with GUST_CONFORMER): per-position channel normalization, a 256-entry
       table activation and an elementwise product. */
    GUST_NORM = 14,
    GUST_LUT = 15,
    GUST_MUL = 16
};
/* Blob and arena remain caller-owned and must outlive the context. Blob must
   stay immutable. Context/blob require 8/4-byte alignment respectively. */
/* The instruction set this build runs ("x86-64 AVX2", "AArch64 NEON dotprod", "wasm32 SIMD", ...); x86-64 dispatch
   builds pick their variant from CPUID at the first call. */
GUST_API const char *gust_variant(void);
GUST_API uint32_t gust_context_bytes(void);
/* bit0 SIMD; bit1 gather-u16-le-v1; bit2 gather-u16-le-i8-v1; bit3 topk-pool-v1; bit4 weighted-pool-v1;
   bit5 attention-int-v1 (opcodes 12 and 13); bit6 valid-frames-v1 (gust_run_valid with a nonzero length);
   bit7 norm-lut-mul-v1 (opcodes 14, 15 and 16); bit8 packed-conv-weights-v1 (convolution parameter1 = 3 or 4).
   Bits 5 and 6 are set only by builds compiled with GUST_ATTENTION and bit 7 only by builds compiled with
   GUST_CONFORMER (which needs GUST_ATTENTION); native builds and gust-asr*.wasm define both. The small specialist
   WebAssembly modules omit them and reject opcodes 12 to 16 as GUST_OPERATOR. Bit 8 is set only by builds compiled
   with GUST_PACKED_WEIGHTS (native builds and gust-asr*.wasm); builds without it reject a packed convolution as
   GUST_OPERATOR in gust_init.
   bit9 float32-graph-v1 (artifact Header.flags 1, operators 32 to 37: packed-weight float32 activation graphs, see
   docs/FORMAT.md and runtime/src/float32/README.md), set only by builds compiled with GUST_FLOAT32; every other build
   rejects such an artifact as GUST_FORMAT without writing the arena. For a float32 graph gust_tensor returns float32
   storage (16-byte aligned) and gust_run_valid counts the real rows of tensor 0. Bit10 adds float32 decoder
   operators (GUST_FLOAT32_DECODER); bit11 adds gust_run_valid_inputs (GUST_FLOAT32).
   Decoder builds also report bit12 float32-t5-v1 (RMS norm, ReLU, attention bias and grouped weight modes) and
   bit13 float32-temporal-v1 (temporal depthwise convolution, FiLM, row mean and wider packed codes). */
GUST_API uint32_t gust_capabilities(void);
GUST_API int gust_init(void *context, const void *blob, uint32_t blob_bytes, void *arena, uint32_t arena_bytes);
GUST_API int gust_run(void *context);
/* gust_run over a padded input: only the first valid_frames rows (height) of tensor 0 are real. 0 or tensor 0's
   height runs every row, bit-exact with gust_run on every build. A larger value returns GUST_SHAPE; any other value
   needs capability bit 6 and an artifact of operators 1-8 and 12-16, otherwise GUST_OPERATOR (nothing is written).
   Tensor t keeps ceil(valid_frames * rows_t / rows_0) real rows; input rows past that are zeroed in the caller's
   buffer, every operator computes and writes only its real rows (the rest are zero), POOL averages the real rows
   and ATTEND ignores keys past them, so a padded window equals exact-length inference; NORM, LUT and MUL compute
   real rows only, so padded rows stay zero whatever their beta or table(0). See docs/FORMAT.md. */
GUST_API int gust_run_valid(void *context, uint32_t valid_frames);
/* Capability bit14 (16384) valid-outputs-v1, INT8 graphs: gust_run_valid that clears only what is observable. Every
   real row of every tensor and every output tensor (flag 2) is bit-identical to gust_run_valid; other tensors hold
   unspecified bytes past their real rows (a padded graph skips clearing megabytes of intermediate rows). Rows a
   convolution or resize reads past its input's real rows are still zeroed first. Float32 graphs: GUST_OPERATOR. */
GUST_API int gust_run_valid_outputs(void *context, uint32_t valid_frames);
#if defined(__wasm__)
/* WebAssembly builds: 1 (default) runs INT8 matrix products of relaxed-SIMD builds with the relaxed dot product, 0 with
   the plain SIMD128 tiles. Results are identical; only speed differs (the plain tiles win on ARM cores without dot
   products, such as the Cortex-A53, where V8 emulates the relaxed instruction). Plain SIMD128 builds ignore it. */
GUST_API void gust_prefer_relaxed_dot(uint32_t on);
#endif
/* Capability bit11 (2048), float32 stateless graphs: one count per input tensor in tensor-index order, each
   1..height. The counts are consumed before arena writes and must not overlap Context storage. Output row
   counts propagate from operands; produced tails are zero. Context/arena/scratch capacities do not grow.
   NULL/unready Context: GUST_GRAPH; unsupported INT8 or dynamic position/cache graph: GUST_OPERATOR;
   null array, wrong count, out-of-range rows or inconsistent joins: GUST_SHAPE; Context overlap: GUST_MEMORY.
   Every refusal leaves the arena untouched and the Context usable by subsequent calls. */
GUST_API int gust_run_valid_inputs(void *context, const uint32_t *input_rows, uint32_t input_count);
/* Stops before an operator when the callback returns nonzero (GUST_BUDGET).
   Partial output must never be consumed; the next full run recomputes the
   graph from its inputs. */
typedef int (*gust_stop_callback)(void *opaque);
GUST_API int gust_run_checked(void *context, gust_stop_callback stop, void *opaque);
/* Per-call, model-independent diagnostics. No allocation or Context/arena growth. A non-NULL observer receives
   RUN_BEGIN, ordered OP_BEGIN/OP_END pairs, then RUN_END with the returned status (including preflight errors).
   Operator events identify the artifact's opcode/output tensor, full shape and active output rows. Other fields
   are zero on run events; operator_index/output_tensor are UINT32_MAX there. Status is meaningful on END events.
   Events are borrowed for the callback only. The observer may inspect active output rows on OP_END, but must not
   mutate/re-enter the context, arena or row counts. Intermediate storage can alias and be reused by later ops;
   padding need not be materialized at OP_END. No timestamps or tensor copies: the observer chooses what to keep.
   NULL observer is equivalent to the corresponding ordinary API. Concurrent use follows the ordinary runtime
   rules (including serialized contexts in shared-scratch builds); observer state is never process-global. */
enum { GUST_TRACE_RUN_BEGIN = 1, GUST_TRACE_OP_BEGIN = 2, GUST_TRACE_OP_END = 3, GUST_TRACE_RUN_END = 4 };
typedef struct {
    uint32_t kind, operator_index, opcode, output_tensor;
    uint32_t rows, height, width, channels;
    int32_t status;
} gust_trace_event;
typedef void (*gust_trace_callback)(void *opaque, const gust_trace_event *event);
GUST_API int gust_run_traced(void *context, uint32_t valid_rows, gust_trace_callback observer, void *opaque);
GUST_API int gust_run_valid_inputs_traced(void *context, const uint32_t *input_rows, uint32_t input_count,
                                        gust_trace_callback observer, void *opaque);
GUST_API void *gust_tensor(void *context, uint32_t index);
GUST_API uint32_t gust_tensor_bytes(void *context, uint32_t index);
/* bit10 float32-decoder-v1 (operators 38 to 45 and state tensors, tensor flag 8; see docs/FORMAT.md and
   runtime/src/float32/DECODER.md), set only by builds compiled with GUST_FLOAT32_DECODER, which also define
   gust_reset; every other build neither reports the bit nor exports the symbol, and refuses such an artifact
   (GUST_OPERATOR or GUST_FORMAT) without writing the arena.
   gust_reset clears every state tensor (key/value caches and the position counter) of a loaded float32 graph and
   nothing else: inputs, outputs and the context stay usable, no allocation. NULL or a context that is not loaded:
   GUST_GRAPH; a loaded INT8 graph: GUST_OPERATOR (nothing is written in either case); a float32 graph without state:
   GUST_OK. The caller must not run the context concurrently. */
GUST_API int gust_reset(void *context);
/* Host work on the runtime's worker pool (float32 builds): runs fn(opaque, item, thread) once for every item in
   0..items-1, on the calling thread and the pool's workers (gust_threads_start, or WebAssembly workers parked in
   gust_worker), and returns when all have finished. The thread number (< 16) names the runner, for per-thread
   scratch. Items must be independent; one call (or graph run) at a time per process. Without threads the items
   run in order on the caller. */
typedef void (*gust_task)(void *opaque, uint32_t item, uint32_t thread);
GUST_API void gust_parallel(gust_task fn, void *opaque, uint32_t items);
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
/* Native builds with GUST_THREADS + GUST_THREADS_POSIX: start worker threads (process-wide pool, once) so float32
   operators split their work over n threads; results are identical for any n. Returns the thread count in use. */
GUST_API uint32_t gust_threads_start(uint32_t n);
#endif
#ifdef __cplusplus
}
#endif
#endif
