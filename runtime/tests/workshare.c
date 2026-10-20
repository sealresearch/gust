/* Regression for changing task counts while workers finish the previous drain, including low-word wrap.
   Build with the runtime feature flags and GUST_THREADS + the platform backend. */
#include "../src/gust.c"
#include <stdio.h>

static uint32_t visits[64], invalid;
static void count(Context *c, const Op *o, uint32_t begin, uint32_t end) {
    (void)c;
    (void)o;
    if (begin >= end || begin / GT_CHUNK >= 64u) {
        __atomic_fetch_add(&invalid, 1u, __ATOMIC_SEQ_CST);
        return;
    }
    __atomic_fetch_add(visits + begin / GT_CHUNK, 1u, __ATOMIC_SEQ_CST);
}

/* A worker may resume while next has been invalidated but new arguments are not yet published. */
#ifndef __wasm__
static int stalled_worker(void) {
    const uint64_t pending[] = {65537ull, 4294967297ull};
    for (uint32_t test = 0; test < 2u; test++) {
        gt_pool.gen = pending[test] - 1u;
        gt_pool.next = pending[test] << 16;
        gt_pool.n = gt_pool.done = 2u;
        gt_pool.args.kind = GT_ROWS;
        gt_pool.args.p[0] = gt_pool.args.p[1] = NULL;
        gt_pool.args.p[2] = (const void *)count;
        gt_pool.args.u[0] = 2u * GT_CHUNK;
        gt_drain(1u, 1u);
        if (visits[0] || visits[1] || invalid || gt_pool.done != 2u || gt_pool.next != (pending[test] << 16)) {
            fputs("stalled worker claimed an unpublished round\n", stderr);
            return 1;
        }
    }
    gt_pool.next = 0;
    gt_pool.n = gt_pool.done = 0;
    return 0;
}
int main(void) {
    if (stalled_worker()) return 1;
    __atomic_store_n(&gt_pool.gen, 0xfffffff0ull, __ATOMIC_SEQ_CST);
    if (gust_threads_start(8) < 2u) { fputs("need at least two threads\n", stderr); return 2; }
#else
GT_EXPORT("test_run") int test_run(void) {
    __atomic_store_n(&gt_pool.gen, 0xfffffff0ull, __ATOMIC_SEQ_CST);
    gust_set_threads(8);
#endif
    for (uint32_t round = 0; round < 100000u; round++) {
        const uint32_t n = round & 1u ? 63u : 2u;
        for (uint32_t i = 0; i < 64; i++) __atomic_store_n(visits + i, 0u, __ATOMIC_SEQ_CST);
        gt_rows(count, NULL, NULL, n * GT_CHUNK);
        for (uint32_t i = 0; i < 64; i++)
            if (__atomic_load_n(visits + i, __ATOMIC_SEQ_CST) != (i < n) || __atomic_load_n(&invalid, __ATOMIC_SEQ_CST)) {
                fprintf(stderr, "workshare mismatch round=%u tasks=%u index=%u\n", round, n, i);
                return 1;
            }
    }
    puts("workshare: 100000 changing rounds, every task exactly once");
    return 0;
}
#ifdef __wasm__
GT_EXPORT("test_pool_bytes") uint32_t test_pool_bytes(void) { return sizeof gt_pool; }
GT_EXPORT("test_generation") uint64_t test_generation(void) { return __atomic_load_n(&gt_pool.gen, __ATOMIC_SEQ_CST); }
#endif
