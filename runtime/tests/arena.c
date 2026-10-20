/* Compare optimized execution with eager clearing (a checked run) at varied lengths, byte by byte over the WHOLE
   arena. One graph allocation, plus a 4 KiB comparison buffer: suitable for the Pi's memory budget. Build with
   the runtime feature/ISA flags, then: arena-test graph.gust [threads=1]. Reference bytes go to a temporary file. */
#include "../src/gust.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *aligned_mem(size_t n) {
    void *p = malloc(n + 63u);
    return p ? (void *)(((uintptr_t)p + 63u) & ~(uintptr_t)63u) : NULL;
}

static int never_stop(void *p) { (void)p; return 0; }

static void inputs(Context *c, uint32_t rows, unsigned seed) {
    memset(c->arena, (int)seed, c->header->arena);
#ifdef GUST_FLOAT32_DECODER
    gust_reset(c);
#endif
    for (uint32_t t = 0; t < c->header->nt; t++) {
        if (!(c->t[t].flags & 1u)) continue;
        float *x = gf_tensor(c, t);
        for (uint32_t k = 0; k < c->t[t].bytes / 4u; k++)
            x[k] = c->t[t].c == 1 ? (float)(k % 4u) : (float)((k * 37u) % 101u) * 0.01f;
        if (c->t[t].h == 1 && c->t[t].c == 1) x[0] = 0.0f;
    }
    /* Invalid input rows may be NaN; they must never enter valid computations or be modified by execution. */
    const Tensor *input = c->t;
    memset(gf_tensor(c, 0) + (size_t)rows * input->c, 0xff, (size_t)(input->h - rows) * input->c * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s graph.gust [threads=1]\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    rewind(f);
    if (bytes < (long)sizeof(Header)) return 1;
    uint8_t *blob = aligned_mem((size_t)bytes);
    if (!blob || fread(blob, 1, (size_t)bytes, f) != (size_t)bytes) return 1;
    fclose(f);
    const Header *h = (const Header *)blob;
    Context *c = aligned_mem(gust_context_bytes());
    void *arena = aligned_mem(h->arena);
    if (!c || !arena || gust_init(c, blob, (uint32_t)bytes, arena, h->arena)) return 1;
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
    gust_threads_start(argc > 2 ? (uint32_t)atoi(argv[2]) : 1u);
#endif
    const uint32_t lengths[] = {1, 2, 3, 5, 7, 11, 35, 350, 600, c->t[0].h};
    unsigned cases = 0;
    for (uint32_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
        const uint32_t len = lengths[i];
        if (len > c->t[0].h) continue;
        const unsigned seed = i & 1u ? 0xffu : 0xa5u + i; /* Also start produced padding with NaNs. */
        inputs(c, len, seed);
        const int eager = float_run(c, never_stop, NULL, len, NULL, NULL);
        if (eager) { /* Some convolution graphs cannot accept very short sequences. */
            if (gust_run_valid(c, len) != eager) return 1;
            continue;
        }
        FILE *ref = tmpfile();
        if (!ref || fwrite(arena, 1, h->arena, ref) != h->arena) return 1;
        rewind(ref);
        inputs(c, len, seed);
        if (gust_run_valid(c, len)) return 1;
        uint8_t buf[4096];
        for (uint32_t p = 0; p < h->arena;) {
            const size_t n = h->arena - p < sizeof buf ? h->arena - p : sizeof buf;
            if (fread(buf, 1, n, ref) != n || memcmp(buf, (const uint8_t *)arena + p, n)) {
                fprintf(stderr, "%s: arena mismatch len=%u offset=%u\n", argv[1], len, p);
                for (size_t k = 0; k < n; k++)
                    if (buf[k] != ((const uint8_t *)arena)[p + k]) {
                        fprintf(stderr, "first differing byte %u: eager=%02x optimized=%02x\n",
                                p + (uint32_t)k, buf[k], ((const uint8_t *)arena)[p + k]);
                        break;
                    }
                return 1;
            }
            p += (uint32_t)n;
        }
        fclose(ref);
        cases++;
    }
    printf("%s: %u lengths, whole arena byte-identical\n", argv[1], cases);
    return cases ? 0 : 1;
}
