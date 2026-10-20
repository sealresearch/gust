/* Per-input row API: valid head outputs match dense execution; padding and refused calls stay untouched. */
#include "../src/gust.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t counts[4096], saved[4096];
static void *mem(size_t n) {
    void *p = malloc(n + 63u);
    return p ? (void *)(((uintptr_t)p + 63u) & ~(uintptr_t)63u) : NULL;
}
static uint64_t hash(const Context *c) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < c->header->arena; i++) h = (h ^ (uint8_t)c->arena[i]) * 1099511628211ull;
    return h;
}
static int compare_outputs(const Context *c, FILE *ref) {
    uint8_t buffer[4096];
    rewind(ref);
    for (uint32_t t = 0; t < c->header->nt; t++) {
        if (!(c->t[t].flags & 2u)) continue;
        for (uint32_t b = 0; b < saved[t];) {
            const uint32_t n = saved[t] - b < sizeof buffer ? saved[t] - b : sizeof buffer;
            if (fread(buffer, 1, n, ref) != n || memcmp(buffer, (uint8_t *)gf_tensor(c, t) + b, n)) {
                fprintf(stderr, "input-rows mismatch tensor=%u offset=%u\n", t, b);
                return 1;
            }
            b += n;
        }
    }
    return 0;
}
static void prepare(Context *c, int nan_padding) {
    memset(c->arena, 0xa5, c->header->arena);
    uint32_t input = 0;
    for (uint32_t i = 0; i < c->header->nt; i++) {
        const Tensor *t = c->t + i;
        if (!(t->flags & 1u)) continue;
        float *x = gf_tensor(c, i);
        for (uint32_t r = 0; r < t->h; r++)
            for (uint32_t j = 0; j < t->c; j++) {
                const size_t k = (size_t)r * t->c + j;
                if (r >= counts[input]) {
                    if (nan_padding) memset(x + k, 0xff, sizeof(float));
                    else x[k] = 0.0f;
                } else x[k] = t->c == 1 ? (float)(r % 4u) : (float)((k * 37u + i * 11u) % 101u) * 0.01f;
            }
        input++;
    }
}
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    rewind(f);
    uint8_t *blob = mem((size_t)bytes);
    if (!blob || fread(blob, 1, (size_t)bytes, f) != (size_t)bytes) return 1;
    fclose(f);
    const Header *h = (const Header *)blob;
    Context *c = mem(gust_context_bytes());
    void *arena = mem(h->arena);
    if (!c || !arena || gust_init(c, blob, (uint32_t)bytes, arena, h->arena)) return 1;
    if (!(gust_capabilities() & 2048u) || gust_run_valid_inputs(NULL, counts, 0) != GUST_GRAPH) return 1;
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
    gust_threads_start(argc > 2 ? (uint32_t)atoi(argv[2]) : 1u);
#endif
    if (argc > 3 && !strcmp(argv[3], "--unsupported")) {
        uint32_t n = 0;
        for (uint32_t t = 0; t < h->nt; t++) if (c->t[t].flags & 1u) counts[n++] = c->t[t].h;
        prepare(c, 0);
        const uint64_t before = hash(c);
        if (gust_run_valid_inputs(c, counts, n) != GUST_OPERATOR || hash(c) != before) return 1;
        if (gust_run_valid(c, c->t[0].h < 35u ? c->t[0].h : 35u)) return 1;
        puts("dynamic graph refused without writes; legacy execution remains usable");
        return 0;
    }
    const uint32_t request[] = {1, 5, 17, 191}, instance[] = {1, 3, 7, 31}, option[] = {1, 5, 63, 64};
    for (uint32_t test = 0; test < 4; test++) {
        uint32_t input = 0;
        for (uint32_t t = 0; t < h->nt; t++) {
            if (!(c->t[t].flags & 1u)) continue;
            uint32_t n = input == 0 ? request[test] : c->t[t].h == 64 ? option[test] : instance[test];
            counts[input++] = n < c->t[t].h ? n : c->t[t].h;
        }
        prepare(c, 1);
        const uint64_t before = hash(c);
        if (gust_run_valid_inputs(c, NULL, input) != GUST_SHAPE ||
            gust_run_valid_inputs(c, counts, input + 1u) != GUST_SHAPE ||
            gust_run_valid_inputs(c, (const uint32_t *)c, input) != GUST_MEMORY) return 1;
        const uint32_t first = counts[0];
        counts[0] = 0;
        if (gust_run_valid_inputs(c, counts, input) != GUST_SHAPE) return 1;
        counts[0] = c->t[0].h + 1u;
        if (gust_run_valid_inputs(c, counts, input) != GUST_SHAPE || hash(c) != before) return 1;
        counts[0] = first;
        const uint32_t last = counts[input - 1u];
        counts[input - 1u] = 0;
        if (gust_run_valid_inputs(c, counts, input) != GUST_SHAPE || hash(c) != before) return 1;
        counts[input - 1u] = last;
        if (gust_run_valid_inputs(c, counts, input)) return 1;
        FILE *ref = tmpfile();
        if (!ref) return 1;
        for (uint32_t t = 0; t < h->nt; t++) {
            if (!(c->t[t].flags & 2u)) continue;
            saved[t] = c->packed_offsets[t] * c->t[t].c * 4u;
            const uint8_t *out = (const uint8_t *)gf_tensor(c, t);
            if (fwrite(out, 1, saved[t], ref) != saved[t]) return 1;
            for (uint32_t b = saved[t]; b < c->t[t].bytes; b++) if (out[b]) return 1;
        }
        input = 0;
        for (uint32_t t = 0; t < h->nt; t++) {
            if (!(c->t[t].flags & 1u)) continue;
            const uint8_t *x = (const uint8_t *)gf_tensor(c, t);
            for (uint32_t b = counts[input] * c->t[t].c * 4u; b < c->t[t].bytes; b++) if (x[b] != 0xff) return 1;
            input++;
        }
        prepare(c, 0);
        if (gust_run_valid(c, counts[0]) || compare_outputs(c, ref)) return 1;
        if (test == 0) {
            /* Arena-resident counts must be consumed before output writes, without reading input padding. */
            prepare(c, 0);
            const Tensor *t = c->t;
            const size_t used = (size_t)counts[0] * t->c * sizeof(float);
            if (t->bytes - used < input * sizeof(uint32_t)) return 1;
            uint32_t *arena_counts = (uint32_t *)((uint8_t *)gf_tensor(c, 0) + used);
            memcpy(arena_counts, counts, input * sizeof(uint32_t));
            if (gust_run_valid_inputs(c, arena_counts, input) || compare_outputs(c, ref)) return 1;
        }
        fclose(ref);
    }
    printf("%s: four multi-input cases exact, NaN padding and refusal guards pass\n", argv[1]);
    return 0;
}
