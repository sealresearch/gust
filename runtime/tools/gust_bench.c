/* Native Gust benchmark for one float32 graph (e.g. relay.gust): runs it at several valid lengths, prints the median
   time and an FNV-1a hash of the output bytes (equal hashes on two machines = bit-identical results).
     gust_bench relay.gust [lengths=35,350,650] [reps=7] [threads=1]
   Token-input graphs get deterministic pseudo-random ids; other inputs get a fixed pattern. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gust.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#elif !defined(__wasm__)
#include <sys/resource.h>
#endif

typedef struct {
    uint8_t magic[8];
    uint32_t version, total, crc, nt, no, to, oo, data, arena, ni, nout, flags, r0, r1;
} Hdr;
typedef struct {
    uint32_t h, w, c, offset, bytes;
    float scale;
    uint32_t flags, reserved;
} Tns;

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
#ifdef GUST_F32_PROFILE
double gust_now(void) { return now() * 1e3; }
double gust_profile(uint32_t slot);
#endif
static void *amalloc(size_t bytes) { /* 64-byte aligned, never freed (aligned_alloc is missing on Windows) */
    uint8_t *p = malloc(bytes + 64);
    return p ? (void *)(((uintptr_t)p + 63u) & ~(uintptr_t)63u) : NULL;
}
/* GUST_BENCH_OUTPUTS=1: INT8 graphs run with gust_run_valid_outputs (identical outputs; intermediate tails stay
   uncleared, so the arena hash differs). */
static int outputs_only;
static int run(void *ctx, uint32_t len) { return outputs_only ? gust_run_valid_outputs(ctx, len) : gust_run_valid(ctx, len); }
static int cmp(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s graph.gust [lengths] [reps=7] [threads=1]\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = amalloc((size_t)((n + 63) / 64 * 64));
    if (fread(blob, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    Hdr h;
    memcpy(&h, blob, sizeof h);
    const Tns *t = (const Tns *)(blob + h.to);
    void *ctx = amalloc((gust_context_bytes() + 63u) / 64u * 64u);
    const uint32_t arena_bytes = (h.arena + 63u) / 64u * 64u;
    void *arena = amalloc(arena_bytes);
    memset(arena, 0xa5, arena_bytes); /* deterministic untouched bytes for the complete-arena parity hash */
    int s = gust_init(ctx, blob, (uint32_t)n, arena, arena_bytes);
    if (s) { fprintf(stderr, "gust_init: %d\n", s); return 1; }
    int out = -1;
    for (uint32_t i = 0; i < h.nt; i++)
        if (t[i].flags & 2u) out = (int)i;
    outputs_only = getenv("GUST_BENCH_OUTPUTS") && getenv("GUST_BENCH_OUTPUTS")[0] == '1';
#if defined(__wasm__)
    if (getenv("GUST_BENCH_RELAXED") && getenv("GUST_BENCH_RELAXED")[0] == '0') gust_prefer_relaxed_dot(0);
#endif
    const char *ls = argc > 2 ? argv[2] : "35,350,650";
    const int reps = argc > 3 ? atoi(argv[3]) : 7;
    if (reps < 1 || reps > 64) { fprintf(stderr, "reps must be 1..64\n"); return 2; }
    uint32_t threads = 1;
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
    threads = gust_threads_start(argc > 4 ? (uint32_t)atoi(argv[4]) : 1u);
#endif
    printf("%s: %u tensors, %u ops, arena %.1f MB, tensor0 [%u x %u], output tensor %d, %u thread(s)\n", argv[1], h.nt,
           h.no, h.arena / 1e6, t[0].h, t[0].c, out, threads);
    char buf[256];
    strncpy(buf, ls, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        uint32_t len = (uint32_t)atoi(tok);
        if (len > t[0].h) len = t[0].h;
        for (uint32_t i = 0; i < h.nt; i++) {
            if (!(t[i].flags & 1u)) continue;
            float *x = gust_tensor(ctx, i);
            const uint32_t cnt = gust_tensor_bytes(ctx, i) / 4u;
            for (uint32_t k = 0; k < cnt; k++)
                x[k] = (i == 0 && t[i].c == 1) ? (float)(5u + (k * 7919u) % 15000u) : (float)((k * 37u) % 101u) * 0.01f;
            if (t[i].h == 1 && t[i].c == 1) x[0] = 0.0f; /* scalar inputs (counts): zero */
        }
        double ms[64];
        const int r = reps < 64 ? reps : 64;
        if ((s = run(ctx, len))) { fprintf(stderr, "run: %d\n", s); return 1; }
        for (int k = 0; k < r; k++) {
            const double t0 = now();
            if ((s = run(ctx, len))) { fprintf(stderr, "run: %d\n", s); return 1; }
            ms[k] = (now() - t0) * 1e3;
        }
        qsort(ms, (size_t)r, sizeof ms[0], cmp);
        uint64_t hash = 1469598103934665603ull;
        if (out >= 0) {
            const uint8_t *o = gust_tensor(ctx, (uint32_t)out);
            for (uint32_t k = 0, b = gust_tensor_bytes(ctx, (uint32_t)out); k < b; k++) hash = (hash ^ o[k]) * 1099511628211ull;
        }
        uint64_t arena_hash = 1469598103934665603ull;
        for (uint32_t k = 0; k < arena_bytes; k++) arena_hash = (arena_hash ^ ((const uint8_t *)arena)[k]) * 1099511628211ull;
        printf("len %4u: median %8.2f ms  (min %.2f)  hash %016llx arena %016llx\n", len, ms[r / 2], ms[0],
               (unsigned long long)hash, (unsigned long long)arena_hash);
#ifdef GUST_F32_PROFILE
        {
            static const char *names[48] = {[1] = "conv16 quantize", [2] = "conv16 code panels", [3] = "conv16 tiles",
                [4] = "zero tails", [32] = "conv (all)", [33] = "add", [34] = "norm", [35] = "unary (gelu)", [36] = "rope",
                [37] = "attention", [41] = "att pack", [42] = "att scores", [43] = "att softmax", [44] = "att PV", [38] = "mul", [40] = "gather", [46] = "mha cache"};
            const double runs = (double)(r + 1);
            for (int k = 0; k < 48; k++) {
                const double v = gust_profile((uint32_t)k);
                if (v > 0.0) printf("    %-20s %8.2f ms/run\n", names[k] ? names[k] : "?", v / runs);
            }
        }
#endif
    }
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pm;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm))
        printf("peak RSS %llu KiB\n", (unsigned long long)(pm.PeakWorkingSetSize / 1024u));
#elif !defined(__wasm__)
    struct rusage usage;
    if (!getrusage(RUSAGE_SELF, &usage)) {
#ifdef __APPLE__
        printf("peak RSS %ld KiB\n", usage.ru_maxrss / 1024l);
#else
        printf("peak RSS %ld KiB\n", usage.ru_maxrss);
#endif
    }
#endif
    return 0;
}
