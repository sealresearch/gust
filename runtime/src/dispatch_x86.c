/* One x86-64 binary for every CPU: gust.c is compiled three times (base = SSE2, avx2, vnni = AVX-512 VNNI) with
   GUST_VARIANT renaming its entry points (rename.h); the variant is chosen once from CPUID. All variants return the
   same bits, so the choice only changes speed. Build with -DGUST_DISPATCH; threads use the chosen variant's pool. */
#include "gust.h"

#define DECL(v)                                                                                                       \
    uint32_t gust_context_bytes_##v(void);                                                                             \
    uint32_t gust_capabilities_##v(void);                                                                              \
    int gust_init_##v(void *, const void *, uint32_t, void *, uint32_t);                                               \
    int gust_run_##v(void *);                                                                                          \
    int gust_run_valid_##v(void *, uint32_t);                                                                          \
    int gust_run_valid_outputs_##v(void *, uint32_t);                                                                  \
    int gust_run_valid_inputs_##v(void *, const uint32_t *, uint32_t);                                                 \
    int gust_run_traced_##v(void *, uint32_t, gust_trace_callback, void *);                                          \
    int gust_run_valid_inputs_traced_##v(void *, const uint32_t *, uint32_t, gust_trace_callback, void *);             \
    int gust_run_checked_##v(void *, gust_stop_callback, void *);                                                      \
    void *gust_tensor_##v(void *, uint32_t);                                                                           \
    uint32_t gust_tensor_bytes_##v(void *, uint32_t);                                                                  \
    int gust_reset_##v(void *);                                                                                        \
    uint32_t gust_threads_start_##v(uint32_t);                                                                         \
    void gust_parallel_##v(gust_task, void *, uint32_t);
DECL(base)
DECL(avx2)
DECL(vnni)

#include <cpuid.h>
#include <stdlib.h>
#include <string.h>
static uint64_t xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
static int variant(void) { /* 0 base, 1 avx2, 2 vnni; the OS must also save the wider registers (XCR0) */
    static int v = -1;
    if (v >= 0) return v;
    unsigned a, b, c, d;
    v = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27)) || !(c & (1u << 28))) return v; /* OSXSAVE, AVX */
    const uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 6u) != 6u) return v;
    if (__get_cpuid_max(0, 0) < 7) return v;
    __cpuid_count(7, 0, a, b, c, d);
    if (b & (1u << 5)) v = 1;                                                           /* AVX2 */
    if (v && (xcr0 & 0xe0u) == 0xe0u && (b & (1u << 16)) && (b & (1u << 30)) && (b & (1u << 31)) && (c & (1u << 11)))
        v = 2; /* F, BW, VL, VNNI */
    const char *f = getenv("GUST_X86"); /* tests: force a lower variant (base, avx2; sse2 = base without SSSE3) */
    if (f && (!strcmp(f, "base") || !strcmp(f, "sse2"))) v = 0;
    else if (f && !strcmp(f, "avx2") && v > 1) v = 1;
    return v;
}
#define PICK(name, ...) \
    (variant() == 2 ? name##_vnni(__VA_ARGS__) : variant() == 1 ? name##_avx2(__VA_ARGS__) : name##_base(__VA_ARGS__))

GUST_API const char *gust_variant(void) { static const char *n[3] = {"x86-64 SSE2", "x86-64 AVX2", "x86-64 AVX-512 VNNI"}; return n[variant()]; }
GUST_API uint32_t gust_context_bytes(void) { return variant() == 2 ? gust_context_bytes_vnni() : variant() == 1 ? gust_context_bytes_avx2() : gust_context_bytes_base(); }
GUST_API uint32_t gust_capabilities(void) { return variant() == 2 ? gust_capabilities_vnni() : variant() == 1 ? gust_capabilities_avx2() : gust_capabilities_base(); }
GUST_API int gust_init(void *c, const void *b, uint32_t n, void *a, uint32_t an) { return PICK(gust_init, c, b, n, a, an); }
GUST_API int gust_run(void *c) { return PICK(gust_run, c); }
GUST_API int gust_run_valid(void *c, uint32_t v) { return PICK(gust_run_valid, c, v); }
GUST_API int gust_run_valid_outputs(void *c, uint32_t v) { return PICK(gust_run_valid_outputs, c, v); }
GUST_API int gust_run_valid_inputs(void *c, const uint32_t *v, uint32_t n) { return PICK(gust_run_valid_inputs, c, v, n); }
GUST_API int gust_run_traced(void *c, uint32_t v, gust_trace_callback t, void *o) { return PICK(gust_run_traced, c, v, t, o); }
GUST_API int gust_run_valid_inputs_traced(void *c, const uint32_t *v, uint32_t n, gust_trace_callback t, void *o) { return PICK(gust_run_valid_inputs_traced, c, v, n, t, o); }
GUST_API int gust_run_checked(void *c, gust_stop_callback s, void *o) { return PICK(gust_run_checked, c, s, o); }
GUST_API void *gust_tensor(void *c, uint32_t i) { return PICK(gust_tensor, c, i); }
GUST_API uint32_t gust_tensor_bytes(void *c, uint32_t i) { return PICK(gust_tensor_bytes, c, i); }
GUST_API int gust_reset(void *c) { return PICK(gust_reset, c); }
GUST_API void gust_parallel(gust_task f, void *o, uint32_t n) {
    if (variant() == 2) gust_parallel_vnni(f, o, n);
    else if (variant() == 1) gust_parallel_avx2(f, o, n);
    else gust_parallel_base(f, o, n);
}
#if defined(GUST_THREADS)
GUST_API uint32_t gust_threads_start(uint32_t n) { return PICK(gust_threads_start, n); }
#endif
