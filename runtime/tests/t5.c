/* float32-t5-v1 operator modes (src/float32/T5.md) on small graphs built in memory: RMS norm and ReLU, F_MHA with an
   additive bias (all and valid query rows), F_CONV grouped partial sums in the exact contract order (one row and
   several rows give the same bits), F_GATHER with float16 group scales, and the refusals that keep each mode
   unambiguous. F_MHA_PREFIX with a bias runs in cached decoder step graphs. Build with the runtime's
   feature flags; no artifacts needed:
     zig cc -std=c11 -O2 -ffp-contract=off $FEAT -Iruntime/include runtime/tests/t5.c -o out/t5.exe */
#include "../src/gust.c"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t blob[1u << 20] __attribute__((aligned(64)));
static uint8_t arena[1u << 20] __attribute__((aligned(64)));
static uint8_t context[1u << 18] __attribute__((aligned(64)));
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
    for (uint32_t i = 0; i < nt; i++) {
        h->ni += (tens(i)->flags & 1u) != 0;
        h->nout += (tens(i)->flags & 2u) != 0;
    }
    h->crc = gf_crc(blob, end);
    return gust_init(context, blob, end, arena, sizeof arena);
}
static float *at(uint32_t t) { return gf_tensor((const Context *)(void *)context, t); }

static uint32_t seed = 7;
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

static void rms_relu(void) {
    enum { T = 5, C = 64 };
    float gamma[C], eps = 1e-6f, x[T * C], ref[T * C];
    for (uint32_t i = 0; i < C; i++) gamma[i] = rnd();
    for (uint32_t i = 0; i < T * C; i++) x[i] = rnd() * 3.0f;
    begin(3, 2);
    tensor(0, T, C, 1);
    tensor(1, T, C, 0);
    tensor(2, T, C, 2);
    *op(0) = (Op){.code = GUST_F_NORM, .a = 0, .out = 1, .weights = put(gamma, sizeof gamma), .mult = put(&eps, 4), .p0 = 4};
    *op(1) = (Op){.code = GUST_F_UNARY, .a = 1, .out = 2, .p0 = 3};
    check(load() == GUST_OK, "RMS norm + ReLU graph loads");
    memcpy(at(0), x, sizeof x);
    check(gust_run(context) == GUST_OK, "RMS norm + ReLU runs");
    for (uint32_t t = 0; t < T; t++) {
        double sq = 0.0;
        for (uint32_t i = 0; i < C; i++) sq = sq + (double)x[t * C + i] * (double)x[t * C + i];
        const double inv = 1.0 / sqrt(sq / (double)C + (double)eps);
        for (uint32_t i = 0; i < C; i++) {
            const float y = (float)(((double)x[t * C + i] * inv) * (double)gamma[i]);
            ref[t * C + i] = y > 0.0f ? y : 0.0f;
        }
    }
    check(same_bits(at(2), ref, T * C), "RMS norm + ReLU equal the float64 contract bit for bit");
}

static void mha_bias(void) {
    enum { T = 6, C = 64, H = 4, D = C / H };
    float q[T * C], k[T * C], v[T * C], b[H * T * T], scale = 0.5f;
    for (uint32_t i = 0; i < T * C; i++) q[i] = rnd(), k[i] = rnd(), v[i] = rnd();
    for (uint32_t i = 0; i < H * T * T; i++) b[i] = rnd() * 2.0f;
    begin(5, 1);
    tensor(0, T, C, 1);
    tensor(1, T, C, 1);
    tensor(2, T, C, 1);
    tensor(3, H * T, T, 1);
    tensor(4, T, C, 2);
    *op(0) = (Op){.code = GUST_F_MHA, .a = 0, .b = 1, .out = 4, .weights = 2, .bias = 3, .mult = put(&scale, 4),
                  .kh = H, .p0 = 4};
    check(load() == GUST_OK, "F_MHA with a bias loads");
    for (uint32_t valid = T; valid >= 4; valid -= 2) {
        memcpy(at(0), q, sizeof q), memcpy(at(1), k, sizeof k), memcpy(at(2), v, sizeof v), memcpy(at(3), b, sizeof b);
        check(gust_run_valid(context, valid) == GUST_OK, valid == T ? "F_MHA bias runs (all rows)" : "F_MHA bias runs (4 valid queries)");
        double err = 0.0;
        for (uint32_t i = 0; i < valid; i++)
            for (uint32_t h = 0; h < H; h++) {
                double s[T], m = -1e300, sum = 0.0;
                for (uint32_t j = 0; j < T; j++) {
                    double d = 0.0;
                    for (uint32_t e = 0; e < D; e++) d += (double)q[i * C + h * D + e] * (double)k[j * C + h * D + e];
                    s[j] = d * scale + b[(h * T + i) * T + j];
                    if (s[j] > m) m = s[j];
                }
                for (uint32_t j = 0; j < T; j++) sum += (s[j] = exp(s[j] - m));
                for (uint32_t e = 0; e < D; e++) {
                    double y = 0.0;
                    for (uint32_t j = 0; j < T; j++) y += s[j] * v[j * C + h * D + e];
                    const double diff = fabs(y / sum - (double)at(4)[i * C + h * D + e]);
                    if (diff > err) err = diff;
                }
            }
        check(err < 2e-5, "F_MHA bias matches the float64 reference (max error < 2e-5)");
    }
}

static void grouped(uint32_t rows, const uint8_t *codes, const uint16_t *scales, const float *bias, const float *x,
                    float *out) {
    enum { N = 10, K = 128 };
    begin(2, 1);
    tensor(0, rows, K, 1);
    tensor(1, rows, N, 2);
    *op(0) = (Op){.code = GUST_F_CONV, .a = 0, .out = 1, .weights = put(codes, N * K / 2), .bias = put(bias, 4 * N),
                  .mult = put(scales, 2 * N * (K / 64)), .kh = 1, .kw = 1, .sy = 1, .sx = 1, .dy = 1, .dx = 1,
                  .p0 = 17, .p1 = 4};
    if (load() != GUST_OK) {
        check(0, "grouped partial sums graph loads");
        return;
    }
    memcpy(at(0), x, 4u * rows * K);
    check(gust_run(context) == GUST_OK, "grouped partial sums run");
    memcpy(out, at(1), 4u * rows * N);
}
static void grouped_sums(void) {
    enum { N = 10, K = 128, T = 5, G = K / 64 };
    static uint8_t codes[N * K / 2];
    uint16_t scales[N * G];
    float bias[N], x[T * K], one[T * N], many[T * N], ref[T * N];
    int8_t q[N * K];
    for (uint32_t i = 0; i < N * K; i++) q[i] = (int8_t)((seed = seed * 1103515245u + 12345u) >> 28) - 8;
    for (uint32_t i = 0; i < N * K; i += 2) codes[i / 2] = (uint8_t)((q[i] & 15) | ((q[i + 1] & 15) << 4));
    for (uint32_t i = 0; i < N * G; i++) scales[i] = (uint16_t)(0x2000u + i * 37u); /* positive finite float16 */
    for (uint32_t i = 0; i < N; i++) bias[i] = rnd();
    for (uint32_t i = 0; i < T * K; i++) x[i] = rnd();
    grouped(T, codes, scales, bias, x, many);
    for (uint32_t t = 0; t < T; t++) grouped(1, codes, scales, bias, x + t * K, one + t * N);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t n = 0; n < N; n++) {
            float y = bias[n];
            for (uint32_t g = 0; g < G; g++) {
                float p = 0.0f;
                for (uint32_t k = 0; k < 64; k++) p = p + (float)q[n * K + g * 64 + k] * x[t * K + g * 64 + k];
                y = y + gf_f16(scales[n * G + g]) * p;
            }
            ref[t * N + n] = y;
        }
    check(same_bits(one, many, T * N), "grouped partial sums: one row and several rows give the same bits");
    check(same_bits(many, ref, T * N), "grouped partial sums follow the contract order bit for bit");
}

static void gather_group64(void) {
    enum { R = 10, C = 128, G = C / 64 };
    static uint8_t codes[R * C / 2];
    uint16_t scales[R * G];
    int8_t q[R * C];
    for (uint32_t i = 0; i < R * C; i++) q[i] = (int8_t)((seed = seed * 1103515245u + 12345u) >> 28) - 8;
    for (uint32_t i = 0; i < R * C; i += 2) codes[i / 2] = (uint8_t)((q[i] & 15) | ((q[i + 1] & 15) << 4));
    for (uint32_t i = 0; i < R * G; i++) scales[i] = (uint16_t)(0x2400u + i * 53u);
    begin(2, 1);
    tensor(0, 1, 1, 1);
    tensor(1, 1, C, 2);
    *op(0) = (Op){.code = GUST_F_GATHER, .a = 0, .out = 1, .weights = put(codes, sizeof codes),
                  .mult = put(scales, sizeof scales), .kh = 1, .p0 = R, .p1 = 4};
    check(load() == GUST_OK, "F_GATHER with group scales loads");
    int ok = 1;
    for (uint32_t token = 0; token < R; token++) {
        at(0)[0] = (float)token;
        ok &= gust_run(context) == GUST_OK;
        for (uint32_t i = 0; i < C; i++) ok &= at(1)[i] == (float)q[token * C + i] * gf_f16(scales[token * G + i / 64]);
    }
    check(ok, "F_GATHER group scales: every row is float32(code) * float32(scale) exactly");
}

static void refusals(void) {
    enum { T = 2, C = 64 };
    float gamma[C], eps = 1e-6f, scale = 1.0f;
    for (uint32_t i = 0; i < C; i++) gamma[i] = 1.0f;
    begin(2, 1);
    tensor(0, T, C, 1);
    tensor(1, T, C, 2);
    *op(0) = (Op){.code = GUST_F_NORM, .a = 0, .out = 1, .weights = put(gamma, sizeof gamma), .bias = put(gamma, sizeof gamma),
                  .mult = put(&eps, 4), .p0 = 4};
    check(load() != GUST_OK, "RMS norm with a beta is refused");
    op(0)->bias = 0, op(0)->p0 = 3;
    check(load() != GUST_OK, "F_NORM p0 3 is refused");
    *op(0) = (Op){.code = GUST_F_UNARY, .a = 0, .out = 1, .p0 = 4};
    check(load() != GUST_OK, "F_UNARY p0 4 is refused");
    begin(4, 1);
    tensor(0, T, C, 1);
    tensor(1, T, C, 1);
    tensor(2, T, C, 1);
    tensor(3, T, C, 2);
    *op(0) = (Op){.code = GUST_F_MHA, .a = 0, .b = 1, .out = 3, .weights = 2, .bias = 1, .mult = put(&scale, 4), .kh = 4};
    check(load() != GUST_OK, "an F_MHA bias without p0 bit 2 is refused");
    op(0)->p0 = 5;
    check(load() != GUST_OK, "an F_MHA bias with banding is refused");
    begin(2, 1);
    tensor(0, 1, 64, 1);
    tensor(1, 1, 4, 2);
    static uint8_t codes[4 * 64 / 2];
    uint16_t scales[4] = {0x3c00, 0x3c00, 0x3c00, 0x3c00};
    *op(0) = (Op){.code = GUST_F_CONV, .a = 0, .out = 1, .weights = put(codes, sizeof codes), .mult = put(scales, sizeof scales),
                  .kh = 1, .kw = 1, .sy = 1, .sx = 1, .dy = 1, .dx = 1, .p0 = 16, .p1 = 4};
    check(load() != GUST_OK, "grouped partial sums without group scales (p0 16) are refused");
    op(0)->p0 = 19;
    check(load() != GUST_OK, "grouped partial sums with float64 sums (p0 19) are refused");
}

int main(void) {
    printf("float32-t5-v1 (capability %s)\n", gust_capabilities() & 4096u ? "bit 12 reported" : "MISSING");
    if (!(gust_capabilities() & 4096u)) return 1;
    rms_relu();
    mha_bias();
    grouped_sums();
    gather_group64();
    refusals();
    printf(failures ? "float32-t5-v1: %d FAILURES\n" : "float32-t5-v1: all checks passed\n", failures);
    return failures != 0;
}
