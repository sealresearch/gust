/* Gust runtime: one C11 translation unit, no allocation and no libc on the hot path, identical results on every
   build (native x86 / ARM, WebAssembly SIMD and scalar). Compile this file alone:

     include/gust.h       public API
     src/int8/            INT8 graphs: artifact loader, executor, scalar and vector kernels, operators 1-16 (ops/)
     src/float32/         float32 graphs (capability bit 9): operators 32-37 and 46, int16 / int8 activation
                          convolutions, attention, threads; decoder state (bit 10, DECODER.md). See README.md there.
*/
#include "rename.h"
#include "int8/abi.h"
#ifdef GUST_FLOAT32
#include "float32/abi.h"
#ifdef GUST_FLOAT32_DECODER
#include "float32/decoder.h"
#endif
#endif
#include "trace.h"
#include "int8/kernels.c"
#include "int8/tiles.c"
#include "int8/format.c"
#include "int8/ops/packed.c"
#include "int8/ops/depthwise.c"
#include "int8/ops/gemm128.c"
#include "int8/ops/gemm.c"
#include "int8/ops/conv.c"
#include "int8/ops/gather.c"
#include "int8/ops/tensor.c"
#include "int8/ops/attention.c"
#include "int8/ops/norm.c"
#include "int8/executor.c"
#ifdef GUST_FLOAT32
#include "float32/math.c"
#include "float32/kernels/common.c"
#include "float32/kernels/conv32.c"
#include "float32/kernels/attention.c"
#include "float32/kernels/fast.c"
#include "float32/kernels/conv16.c"
#include "float32/kernels/conv16_neon.c"
#include "float32/kernels/conv8.c"
#include "float32/threads.c"
#include "float32/ops.c"
#include "float32/executor.c"
#include "float32/format.c"
#ifdef GUST_FLOAT32_DECODER
#include "float32/decoder.c"
#include "float32/temporal.c"
#endif
#endif

#ifndef GUST_VARIANT
GUST_API const char *gust_variant(void) {
#if defined(__wasm_simd128__) && defined(__wasm_relaxed_simd__)
    return "wasm32 SIMD + relaxed SIMD";
#elif defined(__wasm_simd128__)
    return "wasm32 SIMD";
#elif defined(__wasm__)
    return "wasm32";
#elif defined(__aarch64__)
    return ng_has_dot() ? "AArch64 NEON dotprod" : "AArch64 NEON"; /* detected at run time (int8/ops/gemm128.c) */
#elif defined(__AVX512VNNI__)
    return "x86-64 AVX-512 VNNI";
#elif defined(__AVX2__)
    return "x86-64 AVX2";
#elif defined(__x86_64__)
    return "x86-64 SSE2";
#else
    return "scalar";
#endif
}
#endif
