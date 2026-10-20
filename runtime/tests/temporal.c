/* float32-temporal-v1 (src/float32/TEMPORAL.md) on small graphs built in memory: F_DWCONV (dilations, valid rows,
   zero tails), F_FILM (both gain forms), F_MEAN, F_CONV int16 activations with 8-bit codes (bounded levels, the
   worst-case sum that must not overflow) and float32 weights, each against its contract bit for bit, per-input rows,
   and the refusals that keep each mode unambiguous. Build with the runtime's feature flags (build.sh test-temporal):
     zig cc -std=c11 -O2 -ffp-contract=off $FEAT -Iruntime/include runtime/tests/temporal.c -o out/temporal.exe */
#include "../src/gust.c"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t blob[1u << 22] __attribute__((aligned(64)));
static uint8_t arena[1u << 22] __attribute__((aligned(64)));
static uint8_t *context;
static uint32_t nt, no, data, end, arena_end;

static Header *head(void) { return (Header *)(void *)blob; }
static Tensor *tens(uint32_t i) { return (Tensor *)(void *)(blob + 64 + 32 * i); }
static Op *op(uint32_t i) { return (Op *)(void *)(blob + 64 + 32 * nt + 80 * i); }

static void begin(uint32_t tensors, uint32_t ops) {
    memset(blob, 0, sizeof blob);
    nt = tensors;
    no = ops;
    data = (64 + 32 * nt + 80 * no + 15u) & ~15u;
    end = data;
    arena_end = 0;
}
/* flags: 1 input, 2 output (float32 is added) */
static uint32_t tensor(uint32_t i, uint32_t h, uint32_t c, uint32_t flags) {
    Tensor *t = tens(i);
    t->h = h;
    t->w = 1;
    t->c = c;
    t->offset = arena_end;
    t->bytes = 4 * h * c;
    t->scale = 1.0f;
    t->flags = flags | 4u;
    arena_end = (arena_end + t->bytes + 15u) & ~15u;
    return i;
}
static uint32_t put(const void *p, uint32_t n) {
    const uint32_t at = end;
    memcpy(blob + at, p, n);
    end = (end + n + 15u) & ~15u;
    return at;
}
static int load(void) {
    Header *h = head();
    memcpy(h->magic, (const uint8_t[8]){'G', 'U', 'S', 'T', 0, 0, 1, 0}, 8);
    h->version = 1;
    h->total = end;
    h->nt = nt;
    h->no = no;
    h->to = 64;
    h->oo = 64 + 32 * nt;
    h->data = data;
    h->arena = arena_end;
    h->flags = 1;
    h->ni = h->nout = 0;
    for (uint32_t i = 0; i < nt; i++) {
        h->ni += (tens(i)->flags & 1u) != 0;
        h->nout += (tens(i)->flags & 2u) != 0;
    }
    h->crc = gf_crc(blob, end);
    return gust_init(context, blob, end, arena, sizeof arena);
}
static float *at(uint32_t t) { return gf_tensor((const Context *)(void *)context, t); }

static uint32_t seed = 11;
static float rnd(void) { /* uniform in [-1, 1) */
    seed = seed * 1103515245u + 12345u;
    return (float)(seed >> 8) / 8388608.0f - 1.0f;
}
static int failures;
static void check(int ok, const char *what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}
static int same_bits(const float *a, const float *b, size_t n) { return !memcmp(a, b, n * sizeof(float)); }
static int zero_from(const float *a, size_t start, size_t n) {
    for (size_t i = start; i < n; i++)
        if (a[i] != 0.0f) return 0;
    return 1;
}

#pragma clang fp contract(off)
static void dwconv(void) {
    enum { T = 23, C = 20, KH = 7 };
    static float x[T * C], w[KH * C], b[C], ref[T * C];
    for (uint32_t i = 0; i < T * C; i++) x[i] = rnd() * 2.0f;
    for (uint32_t i = 0; i < KH * C; i++) w[i] = rnd();
    for (uint32_t i = 0; i < C; i++) b[i] = rnd();
    const uint32_t dilations[3] = {1, 2, 4};
    for (uint32_t d = 0; d < 3; d++)
        for (uint32_t valid = T; valid >= T - 6; valid -= 6) {
            begin(2, 1);
            tensor(0, T, C, 1);
            tensor(1, T, C, 2);
            *op(0) = (Op){.code = GUST_F_DWCONV, .a = 0, .out = 1, .weights = put(w, sizeof w), .bias = put(b, sizeof b),
                          .kh = KH, .dy = dilations[d]};
            if (load() != GUST_OK) {
                check(0, "F_DWCONV loads");
                return;
            }
            memcpy(at(0), x, sizeof x);
            const int r = gust_run_valid(context, valid);
            for (uint32_t t = 0; t < valid; t++)
                for (uint32_t j = 0; j < C; j++) {
                    float y = b[j];
                    for (uint32_t k = 0; k < KH; k++) {
                        const int s = (int)t + ((int)k - 3) * (int)dilations[d];
                        if (s >= 0 && s < (int)valid) y = y + x[s * C + j] * w[k * C + j];
                    }
                    ref[t * C + j] = y;
                }
            char what[96];
            snprintf(what, sizeof what, "F_DWCONV dilation %u, %u of %u rows: contract bits, zero tail", dilations[d], valid, T);
            check(r == GUST_OK && same_bits(at(1), ref, (size_t)valid * C) && zero_from(at(1), (size_t)valid * C, T * C), what);
        }
}

static void film_mean(void) {
    enum { T = 9, C = 24 };
    static float x[T * C], u[2 * C], ref[T * C];
    for (uint32_t i = 0; i < T * C; i++) x[i] = rnd() * 3.0f;
    for (uint32_t i = 0; i < 2 * C; i++) u[i] = rnd() * 2.0f;
    for (uint32_t mode = 0; mode < 2; mode++) {
        begin(3, 1);
        tensor(0, T, C, 1);
        tensor(1, 1, 2 * C, 1);
        tensor(2, T, C, 2);
        *op(0) = (Op){.code = GUST_F_FILM, .a = 0, .b = 1, .out = 2, .p0 = mode};
        if (load() != GUST_OK) {
            check(0, "F_FILM loads");
            return;
        }
        memcpy(at(0), x, sizeof x);
        memcpy(at(1), u, sizeof u);
        const uint32_t rows[2] = {T - 2, 1};
        const int r = gust_run_valid_inputs(context, rows, 2);
        for (uint32_t t = 0; t < T - 2; t++)
            for (uint32_t j = 0; j < C; j++) {
                const float g = mode ? 1.0f + 0.5f * gf_tanhf(u[j]) : 1.0f + u[j];
                ref[t * C + j] = x[t * C + j] * g + u[C + j];
            }
        check(r == GUST_OK && same_bits(at(2), ref, (T - 2) * C) && zero_from(at(2), (T - 2) * C, T * C),
              mode ? "F_FILM tanh gain (per-input rows): contract bits" : "F_FILM plain gain (per-input rows): contract bits");
    }
    begin(2, 1);
    tensor(0, T, C, 1);
    tensor(1, 1, C, 2);
    *op(0) = (Op){.code = GUST_F_MEAN, .a = 0, .out = 1};
    check(load() == GUST_OK, "F_MEAN loads");
    memcpy(at(0), x, sizeof x);
    const int r = gust_run_valid(context, 5);
    for (uint32_t j = 0; j < C; j++) {
        double s = 0.0;
        for (uint32_t t = 0; t < 5; t++) s = s + (double)x[t * C + j];
        ref[j] = (float)(s / 5.0);
    }
    check(r == GUST_OK && same_bits(at(1), ref, C), "F_MEAN over 5 valid rows: contract bits");
}

/* int16 activations, 8-bit codes: levels L = min(32767, floor((2^31 - 1) / (128 K))), q = round-half-even(x L / m). */
static void conv16_8bit(uint32_t T, uint32_t K, uint32_t N, int extreme, const char *what) {
    static float x[64 * 2048], ref[64 * 2048], scale[2048], bias[2048];
    static int8_t codes[2048 * 2048];
    for (uint32_t i = 0; i < T * K; i++) x[i] = extreme ? (i & 1 ? -4.0f : 4.0f) : rnd() * 3.0f;
    for (uint32_t i = 0; i < N * K; i++) codes[i] = extreme ? (int8_t)(i & 1 ? 127 : -128) : (int8_t)(rnd() * 127.99f);
    for (uint32_t i = 0; i < N; i++) scale[i] = 0.001f + (float)i * 1e-5f, bias[i] = rnd();
    begin(2, 1);
    tensor(0, T, K, 1);
    tensor(1, T, N, 2);
    *op(0) = (Op){.code = GUST_F_CONV, .a = 0, .out = 1, .weights = put(codes, N * K), .mult = put(scale, 4 * N),
                  .bias = put(bias, 4 * N), .kh = 1, .kw = 1, .sy = 1, .sx = 1, .dy = 1, .dx = 1, .p0 = 4, .p1 = 8};
    if (load() != GUST_OK) {
        check(0, what);
        return;
    }
    memcpy(at(0), x, (size_t)T * K * 4);
    const int r = gust_run(context);
    const uint32_t bound = (uint32_t)(2147483647u / (128u * K)), L = bound < 32767u ? bound : 32767u;
    int fits = 1;
    for (uint32_t t = 0; t < T; t++) {
        float m = 0.0f;
        for (uint32_t k = 0; k < K; k++) m = fabsf(x[t * K + k]) > m ? fabsf(x[t * K + k]) : m;
        const float s = m / (float)L, inv = (float)L / m;
        for (uint32_t n = 0; n < N; n++) {
            int64_t acc = 0;
            for (uint32_t k = 0; k < K; k++)
                acc += (int64_t)codes[n * K + k] * (int64_t)nearbyintf(x[t * K + k] * inv);
            fits &= acc >= INT32_MIN && acc <= INT32_MAX;
            ref[t * N + n] = (float)(int32_t)acc * (s * scale[n]) + bias[n];
        }
    }
    check(r == GUST_OK && fits && same_bits(at(1), ref, (size_t)T * N), what);
}

static void float_weights(void) {
    enum { T = 13, K = 40, N = 21 };
    static float x[T * K], w[N * K], s[N], b[N], ref[T * N];
    for (uint32_t i = 0; i < T * K; i++) x[i] = rnd();
    for (uint32_t i = 0; i < N * K; i++) w[i] = rnd() * 0.3f;
    for (uint32_t i = 0; i < N; i++) s[i] = 1.0f, b[i] = rnd();
    begin(2, 1);
    tensor(0, T, K, 1);
    tensor(1, T, N, 2);
    *op(0) = (Op){.code = GUST_F_CONV, .a = 0, .out = 1, .weights = put(w, sizeof w), .mult = put(s, sizeof s),
                  .bias = put(b, sizeof b), .kh = 1, .kw = 1, .sy = 1, .sx = 1, .dy = 1, .dx = 1, .p1 = 32};
    check(load() == GUST_OK, "F_CONV float32 weights (p1 32) load");
    memcpy(at(0), x, sizeof x);
    const int r = gust_run(context);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t n = 0; n < N; n++) {
            float y = b[n];
            for (uint32_t k = 0; k < K; k++) y = y + x[t * K + k] * (w[n * K + k] * s[n]);
            ref[t * N + n] = y;
        }
    check(r == GUST_OK && same_bits(at(1), ref, T * N), "F_CONV float32 weights: sequential float32 sums bit for bit");
}

static void refusals(void) {
    static float w[64 * 8], b[8], s[8];
    for (uint32_t i = 0; i < 8; i++) s[i] = 1.0f;
    const uint32_t bad_kh[3] = {6, 33, 0};
    for (uint32_t i = 0; i < 3; i++) {
        begin(2, 1);
        tensor(0, 8, 8, 1);
        tensor(1, 8, 8, 2);
        *op(0) = (Op){.code = GUST_F_DWCONV, .a = 0, .out = 1, .weights = put(w, 4 * 8 * 33), .kh = bad_kh[i], .dy = 1};
        check(load() != GUST_OK, i == 0 ? "F_DWCONV with an even kernel is refused"
                                 : i == 1 ? "F_DWCONV with 33 taps is refused" : "F_DWCONV with no taps is refused");
    }
    begin(2, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 8, 8, 2);
    *op(0) = (Op){.code = GUST_F_DWCONV, .a = 0, .out = 1, .weights = put(w, 4 * 8 * 3), .kh = 3};
    check(load() != GUST_OK, "F_DWCONV with dilation 0 is refused");
    begin(3, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 1, 8, 1);
    tensor(2, 8, 8, 2);
    *op(0) = (Op){.code = GUST_F_FILM, .a = 0, .b = 1, .out = 2};
    check(load() != GUST_OK, "F_FILM with a [1, C] modulation (not [1, 2C]) is refused");
    begin(3, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 1, 16, 1);
    tensor(2, 8, 8, 2);
    *op(0) = (Op){.code = GUST_F_FILM, .a = 0, .b = 1, .out = 2, .p0 = 2};
    check(load() != GUST_OK, "F_FILM p0 2 is refused");
    begin(2, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 2, 8, 2);
    *op(0) = (Op){.code = GUST_F_MEAN, .a = 0, .out = 1};
    check(load() != GUST_OK, "F_MEAN into two rows is refused");
    begin(2, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 8, 8, 2);
    *op(0) = (Op){.code = GUST_F_CONV, .a = 0, .out = 1, .weights = put(w, 4 * 64), .mult = put(s, sizeof s), .kh = 1,
                  .kw = 1, .sy = 1, .sx = 1, .dy = 1, .dx = 1, .p0 = 4, .p1 = 32};
    check(load() != GUST_OK, "float32 weights with int16 activations are refused");
    op(0)->p0 = 12;
    op(0)->p1 = 8;
    check(load() != GUST_OK, "int8 activations with 8-bit codes are refused");
    op(0)->p0 = 1;
    op(0)->p1 = 32;
    check(load() != GUST_OK, "float32 weights with group scales are refused");
    begin(2, 1);
    tensor(0, 8, 8, 1);
    tensor(1, 8, 8, 2);
    *op(0) = (Op){.code = 50, .a = 0, .out = 1};
    check(load() == GUST_OPERATOR, "operator 50 is refused");
    (void)b;
}

int main(void) {
    uint8_t *raw = calloc(1, gust_context_bytes() + 64u); /* the Context is 64-byte aligned */
    context = raw + (64u - (uintptr_t)raw % 64u);
    printf("float32-temporal-v1 (capability %s)\n", gust_capabilities() & 8192u ? "bit 13 reported" : "MISSING");
    if (!(gust_capabilities() & 8192u)) return 1;
    dwconv();
    film_mean();
    conv16_8bit(7, 256, 24, 0, "int16 activations x 8-bit codes (K 256): contract bits");
    conv16_8bit(5, 1280, 16, 0, "int16 activations x 8-bit codes (K 1280, bounded levels): contract bits");
    conv16_8bit(3, 2048, 8, 1, "int16 activations x 8-bit codes, worst-case sums (K 2048): no overflow, contract bits");
    float_weights();
    refusals();
    printf(failures ? "float32-temporal-v1: %d FAILURES\n" : "float32-temporal-v1: all checks passed\n", failures);
    return failures != 0;
}
