/* float32-graph-v1 operators 32 to 37. gust_init has validated every index, offset, scale and shape; the executor
   (float32/executor.c) has computed the valid row count of every tensor, so each operator reads only valid input
   rows and computes exactly `rows` output rows (the executor zeroes the rest of the output). No allocation; scratch
   is the 64 KiB Context.packed_weights storage (restored weights or one attention score row) and a few small
   stack arrays. Contraction is off, no host math. */
#pragma clang fp contract(off)

static inline float *gf_tensor(const Context *c, uint32_t t) {
    return (float *)(void *)(c->arena + c->t[t].offset);
}
static inline float *gf_scratch(const Context *c) {
    return (float *)(void *)c->packed_weights;
}

/* One 8-column slab of an int16-activation convolution over every row (kept small so the tile accumulators stay
   in registers): out[t][j] for t < rows, j < jn. */
static __attribute__((noinline)) void gq_slab(const int16_t *xq, uint32_t kk, uint32_t rows, const int16_t *slab,
                                              uint32_t kp, const float *scale, const GfF *ws, const GfF *bias,
                                              float *out, uint32_t N, uint32_t jn) {
    int32_t acc[GQ_TR * 8u] __attribute__((aligned(16)));
    for (uint32_t t = 0; t < rows;) {
        const uint32_t left = rows - t;
        const uint32_t R = left >= GQ_TR ? GQ_TR : left >= GQ_TR / 2u ? GQ_TR / 2u : 1u;
        for (uint32_t i = 0; i < GQ_TR * 8u; i++) acc[i] = 0;
        if (R == GQ_TR)
            gq_tile(xq + (size_t)t * kk, kk, slab, kp, acc, GQ_TR);
        else if (R == GQ_TR / 2u)
            gq_tile(xq + (size_t)t * kk, kk, slab, kp, acc, GQ_TR / 2u);
        else
            gq_tile(xq + (size_t)t * kk, kk, slab, kp, acc, 1u);
        for (uint32_t r = 0; r < R; r++) gq_epilogue(acc + 8u * r, scale[t + r], ws, bias, out + (size_t)(t + r) * N, jn);
        t += R;
    }
}

/* F_CONV p0 bit 4: int16 activations (kernels.c, gq_*). Every input row is quantized once into Context.f32_act
   ([rows][kk], kk = K rounded up to even) with its scale in the 64 KiB scratch (float[rows]); then each 8-column
   slab of output channels restores its codes ([kk / 2][8][2] in the thread's Context.f32_slab) and runs over every
   row. Tasks (threads.c): row chunks of 32 (GT_QUANT), slabs (GT_SLAB). */
enum { GT_QUANT = 1, GT_SLAB = 2, GT_HEAD = 3, GT_ROWS = 4, GT_PACK = 5, GT_QCHUNK = 6, GT_COLS = 7, GT_QUANT8 = 8,
       GT_SLAB8 = 9, GT_ZERO = 10, GT_HOST = 11 };

/* Host work on the operators' pool (gust_parallel, gust.h): one task per item; the thread number names the runner. */
typedef struct {
    gust_task fn;
    void *opaque;
} GtHost;
GUST_API void gust_parallel(gust_task fn, void *opaque, uint32_t items) {
    const GtHost host = {fn, opaque};
    GtArgs args;
    args.p[0] = &host;
    gt_parallel(GT_HOST, &args, items);
}

/* Disjoint 64 KiB ranges of an output tail; use the existing worker pool and no scratch. */
static void gf_zero_task(const GtArgs *a, uint32_t i) {
    float *y = (float *)(void *)a->p[0];
    const uint32_t n = a->u[0], start = i * 16384u, end = n - start < 16384u ? n : start + 16384u;
    for (uint32_t k = start; k < end; k++) y[k] = 0.0f;
}

static void gq_quant_task(const GtArgs *a, uint32_t i) {
    const float *x = (const float *)a->p[0];
    float *scale = (float *)a->p[1];
    int16_t *xq = (int16_t *)a->p[2];
    const uint32_t rows = a->u[0], K = a->u[1], kk = a->u[2], xrow = a->u[3];
    const float levels = (float)a->u[4]; /* 32767, or fewer for 5- to 8-bit codes (gq_levels) */
    const uint32_t t1 = 32u * i + 32u < rows ? 32u * i + 32u : rows;
    for (uint32_t t = 32u * i; t < t1; t++) {
        const float *xr = x + (size_t)t * xrow;
        const float m = gq_absmax(xr, K);
        scale[t] = m > 0.0f ? m / levels : 0.0f;
        gq_quantize(xr, K, kk, m > 0.0f ? levels / m : 0.0f, xq + (size_t)t * kk);
    }
}

#if defined(__aarch64__)
static __attribute__((noinline)) void gq4_slab(const int16_t *xq, uint32_t kk, uint32_t rows, const uint8_t *packed,
                                              uint32_t kp, const float *scale, const GfF *ws, const GfF *bias,
                                              float *out, uint32_t N, uint32_t jn) {
    int32_t acc[GQ_TR * 8u] __attribute__((aligned(32)));
    for (uint32_t t = 0; t < rows;) {
        const uint32_t left = rows - t;
        const uint32_t R = left >= GQ_TR ? GQ_TR : left >= GQ_TR / 2u ? GQ_TR / 2u : 1u;
        for (uint32_t j = 0; j < GQ_TR * 8u; j++) acc[j] = 0;
        if (R == GQ_TR) gq4_tile(xq + (size_t)t * kk, kk, packed, kp, acc, GQ_TR);
        else if (R == GQ_TR / 2u) gq4_tile(xq + (size_t)t * kk, kk, packed, kp, acc, GQ_TR / 2u);
        else gq4_tile(xq + (size_t)t * kk, kk, packed, kp, acc, 1u);
        for (uint32_t r = 0; r < R; r++)
            gq_epilogue(acc + 8u * r, scale[t + r], ws, bias, out + (size_t)(t + r) * N, jn);
        t += R;
    }
}
#endif

static void gq_slab_task(const GtArgs *a, uint32_t i, uint32_t tid) {
    Context *c = (Context *)a->p[0];
    const GfWeights *w = (const GfWeights *)a->p[1];
    const int16_t *xq = (const int16_t *)a->p[2];
    const float *scale = (const float *)a->p[3];
    const GfF *wscale = (const GfF *)a->p[4], *bias = (const GfF *)a->p[5];
    float *out = (float *)a->p[6];
    const uint32_t rows = a->u[0], K = a->u[1], kk = a->u[2], N = a->u[3], kp = kk / 2u, j0 = 8u * i;
    const uint32_t jn = N - j0 < 8u ? N - j0 : 8u;
    int16_t *slab = c->f32_slab[tid];
#if defined(__aarch64__)
    if (w->bits == 4u && !(K & 1u) && rows <= 64u) {
        uint8_t *packed = (uint8_t *)slab;
        gq4_codes_slab(w, j0, jn, K, packed);
        gq4_slab(xq, kk, rows, packed, kp, scale, wscale + j0, bias ? bias + j0 : (const GfF *)0,
                 out + j0, N, jn);
        return;
    }
#endif
    for (uint32_t jj = 0; jj < 8u; jj++) {
        if (jj < jn) {
            gq_codes_row(w, j0 + jj, K, slab, jj);
            if (K & 1u) slab[(size_t)(kp - 1u) * 16u + 2u * jj + 1u] = 0;
        } else {
            for (uint32_t p = 0; p < kp; p++) slab[(size_t)p * 16u + 2u * jj] = slab[(size_t)p * 16u + 2u * jj + 1u] = 0;
        }
    }
    gq_slab(xq, kk, rows, slab, kp, scale, wscale + j0, bias ? bias + j0 : (const GfF *)0, out + j0, N, jn);
}

/* F_CONV p0 = 4 | 8: int8 activations (kernels.c, gq8_*). Rows quantized once into Context.f32_act as int8
   ([rows][kq], kq = K rounded up to 4), per-row scales and activation sums in the 64 KiB scratch (float[2048],
   int32[2048]); each 8-column slab restores its codes into the thread's Context.f32_slab and runs over every row. */
static void gq8_quant_task(const GtArgs *a, uint32_t i) {
    const float *x = (const float *)a->p[0];
    float *scale = (float *)a->p[1];
    int8_t *xq = (int8_t *)a->p[2];
    int32_t *xsum = (int32_t *)a->p[3];
    const uint32_t rows = a->u[0], K = a->u[1], kq = a->u[2], xrow = a->u[3];
    const uint32_t t1 = 32u * i + 32u < rows ? 32u * i + 32u : rows;
    for (uint32_t t = 32u * i; t < t1; t++) {
        const float *xr = x + (size_t)t * xrow;
        const float m = gq_absmax(xr, K);
        scale[t] = m > 0.0f ? m / 127.0f : 0.0f;
        gq8_quantize(xr, K, kq, m > 0.0f ? 127.0f / m : 0.0f, xq + (size_t)t * kq, xsum + t);
    }
}

static __attribute__((noinline)) void gq8_slab(const int8_t *xq, uint32_t kq, uint32_t rows, const int8_t *slab,
                                               const int32_t *wsum, const int32_t *xsum, const float *scale,
                                               const GfF *ws, const GfF *bias, float *out, uint32_t N, uint32_t jn) {
    int32_t acc[GQ8_TR * 8u] __attribute__((aligned(32)));
    for (uint32_t t = 0; t < rows;) {
        const uint32_t R = rows - t >= GQ8_TR ? GQ8_TR : 1u;
        if (R == GQ8_TR)
            gq8_tile(xq + (size_t)t * kq, kq, slab, kq, wsum, xsum + t, acc, GQ8_TR);
        else
            gq8_tile(xq + (size_t)t * kq, kq, slab, kq, wsum, xsum + t, acc, 1u);
        for (uint32_t r = 0; r < R; r++) gq_epilogue(acc + 8u * r, scale[t + r], ws, bias, out + (size_t)(t + r) * N, jn);
        t += R;
    }
}

static void gq8_slab_task(const GtArgs *a, uint32_t i, uint32_t tid) {
    Context *c = (Context *)a->p[0];
    const GfWeights *w = (const GfWeights *)a->p[1];
    const int8_t *xq = (const int8_t *)a->p[2];
    const float *scale = (const float *)a->p[3];
    const GfF *wscale = (const GfF *)a->p[4], *bias = (const GfF *)a->p[5];
    float *out = (float *)a->p[6];
    const int32_t *xsum = (const int32_t *)a->p[7];
    const uint32_t rows = a->u[0], K = a->u[1], kq = a->u[2], N = a->u[3], j0 = 8u * i;
    const uint32_t jn = N - j0 < 8u ? N - j0 : 8u;
    int8_t *slab = (int8_t *)c->f32_slab[tid];
    int32_t wsum[8] __attribute__((aligned(32)));
    gq8_codes_slab(w, j0, jn, K, kq, slab, wsum);
    gq8_slab(xq, kq, rows, slab, wsum, xsum, scale, wscale + j0, bias ? bias + j0 : (const GfF *)0, out + j0, N, jn);
}

static __attribute__((noinline)) void gq8_op_conv(Context *c, const Op *o, const GfWeights *w, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const uint32_t N = z->c, K = (uint32_t)w->K, kq = (K + GQ8_KALIGN - 1u) & ~(GQ8_KALIGN - 1u);
    float *scale = (float *)(void *)c->packed_weights;
    int32_t *xsum = (int32_t *)(void *)(c->packed_weights + 2u * GQ_MAX_ROWS);
    GtArgs args;
    args.p[0] = gf_tensor(c, o->a);
    args.p[1] = scale;
    args.p[2] = c->f32_act;
    args.p[3] = xsum;
    args.u[0] = rows;
    args.u[1] = K;
    args.u[2] = kq;
    args.u[3] = o->sy * a->c;
    GP_START(gp1);
    gt_parallel(GT_QUANT8, &args, (rows + 31u) / 32u);
    GP_ADD(1, gp1);
    args.p[0] = c;
    args.p[1] = w;
    args.p[2] = c->f32_act;
    args.p[3] = scale;
    args.p[4] = w->scale;
    args.p[5] = o->bias ? (const void *)(c->blob + o->bias) : (const void *)0;
    args.p[6] = gf_tensor(c, o->out);
    args.p[7] = xsum;
    args.u[3] = N;
    GP_START(gp3);
    gt_parallel(GT_SLAB8, &args, (N + 7u) / 8u);
    GP_ADD(3, gp3);
}

/* Largest activation code of an int16-activation convolution: 32767 for codes of at most 4 bits (|code| <= 8,
   K <= 2048), otherwise the largest L with L * 2^(bits - 1) * K <= 2^31 - 1, so no int32 partial sum overflows
   (float32-temporal-v1). */
static uint32_t gq_levels(uint32_t bits, uint32_t K) {
    if (bits <= 4u) return 32767u;
    const uint32_t bound = (uint32_t)(2147483647u / ((1u << (bits - 1u)) * K));
    return bound < 32767u ? bound : 32767u;
}

static __attribute__((noinline)) void gq_op_conv(Context *c, const Op *o, const GfWeights *w, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const uint32_t N = z->c, K = (uint32_t)w->K, kk = (K + 1u) & ~1u;
    GtArgs args;
    args.u[4] = gq_levels(w->bits, K);
    args.p[0] = gf_tensor(c, o->a);
    args.p[1] = c->packed_weights; /* float scale[rows] */
    args.p[2] = c->f32_act;
    args.u[0] = rows;
    args.u[1] = K;
    args.u[2] = kk;
    args.u[3] = o->sy * a->c;
    GP_START(gp1);
    gt_parallel(GT_QUANT, &args, (rows + 31u) / 32u);
    GP_ADD(1, gp1);
    args.p[0] = c;
    args.p[1] = w;
    args.p[2] = c->f32_act;
    args.p[3] = c->packed_weights;
    args.p[4] = w->scale;
    args.p[5] = o->bias ? (const void *)(c->blob + o->bias) : (const void *)0;
    args.p[6] = gf_tensor(c, o->out);
    args.u[3] = N;
    GP_START(gp3);
    gt_parallel(GT_SLAB, &args, (N + 7u) / 8u);
    GP_ADD(3, gp3);
}

/* float32-mode convolution of output columns n0 .. n0 + GF_TW - 1 over every row, weights restored into `panel`
   (GF_TW x min(K, GF_KB) floats). One GT_COLS task per column block. */
static void gf_conv32_cols(Context *c, const Op *o, const GfWeights *w, uint32_t rows, uint32_t n0, float *panel) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const uint32_t N = z->c, K = (uint32_t)w->K, kb = K < GF_KB ? K : GF_KB, nc = N - n0 < GF_TW ? N - n0 : GF_TW;
    const int chunk = K > GF_KB;
    const size_t xrow = (size_t)(o->sy * a->c);
    const float *x = gf_tensor(c, o->a);
    float *out = gf_tensor(c, o->out);
    const GfF *bias = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    float acc[GF_TR * GF_TW] __attribute__((aligned(32)));
    if (!chunk) gf_decode_panel_w(w, n0, nc, 0, K, panel);
    for (uint32_t t = 0; t < rows;) {
        const uint32_t R = rows - t >= GF_TR ? GF_TR : 1u;
        const float *xt = x + (size_t)t * xrow;
        for (uint32_t r = 0; r < R; r++)
            for (uint32_t j = 0; j < GF_TW; j++) acc[r * GF_TW + j] = (j < nc && bias) ? bias[n0 + j] : 0.0f;
        for (uint32_t k0 = 0; k0 < K; k0 += kb) {
            const uint32_t kl = K - k0 < kb ? K - k0 : kb;
            if (chunk) gf_decode_panel_w(w, n0, nc, k0, kl, panel);
            if (R == GF_TR)
                gf_tile32w(xt + k0, xrow, panel, kl, acc, GF_TR);
            else
                gf_tile32w(xt + k0, xrow, panel, kl, acc, 1u);
        }
        for (uint32_t r = 0; r < R; r++)
            for (uint32_t j = 0; j < nc; j++) out[(size_t)(t + r) * N + n0 + j] = acc[r * GF_TW + j];
        t += R;
    }
}

#ifdef GUST_FLOAT32_T5
/* F_CONV p0 = 17 (float32-t5-v1): grouped partial sums over 4-bit codes with float16 scales per 64 weights (kh 1,
   K a multiple of 64, float32 sums). For output n of row t:
     y = bias (or 0); for each group g in increasing order: p = 0; for k = 0 .. 63: p = p + float32(code) * x[k];
     y = y + scale_g * p.
   The integer codes are multiplied by the inputs and each group is scaled once (the factored order of the dedicated
   T5 runtime); every code is read once per row instead of restoring code * scale first. Two code paths (one row:
   codes read straight from the payload; several rows: one group of 4 outputs restored to the scratch and reused)
   perform the same operations in the same order, so the row count never changes a bit; lanes are output channels
   only, so SIMD and scalar builds agree. */
static void gf_op_conv_grouped_sums(Context *c, const Op *o, uint32_t rows) {
    static const float q4[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f};
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const uint32_t N = z->c, K = a->c, G = K / 64u;
    const uint8_t *code = c->blob + o->weights, *sc = c->blob + o->mult;
    const float *x = gf_tensor(c, o->a);
    float *out = gf_tensor(c, o->out);
    const GfF *bias = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    float *q = gf_scratch(c); /* [64][4]: restored codes of 4 outputs, k-major (several-row path) */
    for (uint32_t n0 = 0; n0 < N; n0 += 4) {
        const uint32_t nc = N - n0 < 4 ? N - n0 : 4;
        for (uint32_t t = 0; t < rows; t++)
            for (uint32_t j = 0; j < nc; j++) out[(size_t)t * N + n0 + j] = bias ? bias[n0 + j] : 0.0f;
        for (uint32_t g = 0; g < G; g++) {
            float s[4] = {0.f, 0.f, 0.f, 0.f};
            const uint8_t *row[4];
            for (uint32_t j = 0; j < 4; j++) {
                const uint32_t n = n0 + (j < nc ? j : 0);
                const size_t at = 2u * ((size_t)n * G + g);
                row[j] = code + (((size_t)n * K + (size_t)g * 64u) >> 1);
                if (j < nc) s[j] = gf_f16((uint16_t)(sc[at] | (uint32_t)sc[at + 1u] << 8));
            }
            if (rows == 1) {
                const float *xg = x + (size_t)g * 64u;
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
                for (uint32_t k = 0; k < 64u; k += 2) {
                    const uint32_t b0 = row[0][k >> 1], b1 = row[1][k >> 1], b2 = row[2][k >> 1], b3 = row[3][k >> 1];
                    const float xa = xg[k], xb = xg[k + 1];
                    p0 = p0 + q4[b0 & 15u] * xa;
                    p1 = p1 + q4[b1 & 15u] * xa;
                    p2 = p2 + q4[b2 & 15u] * xa;
                    p3 = p3 + q4[b3 & 15u] * xa;
                    p0 = p0 + q4[b0 >> 4] * xb;
                    p1 = p1 + q4[b1 >> 4] * xb;
                    p2 = p2 + q4[b2 >> 4] * xb;
                    p3 = p3 + q4[b3 >> 4] * xb;
                }
                const float p[4] = {p0, p1, p2, p3};
                for (uint32_t j = 0; j < nc; j++) out[n0 + j] = out[n0 + j] + s[j] * p[j];
                continue;
            }
            for (uint32_t k = 0; k < 64u; k += 2)
                for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t b = row[j][k >> 1];
                    q[k * 4u + j] = q4[b & 15u];
                    q[(k + 1u) * 4u + j] = q4[b >> 4];
                }
            for (uint32_t t = 0; t < rows; t++) {
                const float *xg = x + (size_t)t * K + (size_t)g * 64u;
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
                for (uint32_t k = 0; k < 64u; k++) {
                    const float xv = xg[k];
                    p0 = p0 + q[k * 4u] * xv;
                    p1 = p1 + q[k * 4u + 1u] * xv;
                    p2 = p2 + q[k * 4u + 2u] * xv;
                    p3 = p3 + q[k * 4u + 3u] * xv;
                }
                const float p[4] = {p0, p1, p2, p3};
                float *ot = out + (size_t)t * N + n0;
                for (uint32_t j = 0; j < nc; j++) ot[j] = ot[j] + s[j] * p[j];
            }
        }
    }
}
#endif

/* F_CONV. Output row t is the sum over the K = kh * C reduction window starting at input row t * sy.
   p0 bit 1: float64 products and sums of float32(code * scale), one float32 cast per output; otherwise sequential
   float32 products and sums. The bias seeds the sum. K beyond GUST_F32_KBLOCK is processed in blocks whose weights
   are restored per row tile; partial sums are carried exactly, so no block size changes a result. */
static void gf_op_conv(Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    GfWeights w;
    w.code = c->blob + o->weights;
    w.scale = c->blob + o->mult;
    w.bits = o->p1;
    w.grouped = o->p0 & 1u;
    w.K = (uint64_t)o->kh * a->c;
    w.ngroups = (uint32_t)((w.K + 63u) / 64u);
    const uint32_t N = z->c, K = (uint32_t)w.K, accurate = (o->p0 >> 1) & 1u;
    const size_t xrow = (size_t)(o->sy * a->c); /* wraps only when it is unused (rows == 1) */
#ifdef GUST_FLOAT32_T5
    if (o->p0 & 16u) {
        gf_op_conv_grouped_sums(c, o, rows);
        return;
    }
#endif
    if (o->p0 & 8u) {
        gq8_op_conv(c, o, &w, rows);
        return;
    }
    if (o->p0 & 4u) {
        gq_op_conv(c, o, &w, rows);
        return;
    }
    const float *x = gf_tensor(c, o->a);
    float *out = gf_tensor(c, o->out);
    float *panel = gf_scratch(c);
    const GfF *bias = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    if (!accurate) { /* float32 mode: same sums, wide panels (kernels.c), column blocks shared out (threads.c) */
        const uint32_t blocks = (N + GF_TW - 1u) / GF_TW;
        if (gt_threads() > 1u && blocks > 1u) {
            GtArgs args;
            args.p[0] = c;
            args.p[1] = o;
            args.p[2] = &w;
            args.u[0] = rows;
            gt_parallel(GT_COLS, &args, blocks);
        } else {
            for (uint32_t n0 = 0; n0 < N; n0 += GF_TW) gf_conv32_cols(c, o, &w, rows, n0, panel);
        }
        return;
    }
    const uint32_t block = K < GUST_F32_KBLOCK ? K : GUST_F32_KBLOCK;
    const int chunked = K > GUST_F32_KBLOCK;
    for (uint32_t n0 = 0; n0 < N; n0 += 4) {
        const uint32_t nc = N - n0 < 4 ? N - n0 : 4;
        if (!chunked) gf_decode_panel(&w, n0, nc, 0, K, panel);
        for (uint32_t t = 0; t < rows;) {
            const uint32_t R = rows - t >= 4 ? 4 : 1;
            const float *xt = x + (size_t)t * xrow;
            if (accurate) {
                double acc[4][4];
                for (uint32_t r = 0; r < R; r++)
                    for (uint32_t j = 0; j < 4; j++) acc[r][j] = (j < nc && bias) ? (double)bias[n0 + j] : 0.0;
                for (uint32_t k0 = 0; k0 < K; k0 += block) {
                    const uint32_t kl = K - k0 < block ? K - k0 : block;
                    if (chunked) gf_decode_panel(&w, n0, nc, k0, kl, panel);
                    if (R == 4)
                        gf_tile64_4(xt + k0, xrow, panel, kl, acc);
                    else
                        gf_tile64_1(xt + k0, panel, kl, acc[0]);
                }
                for (uint32_t r = 0; r < R; r++)
                    for (uint32_t j = 0; j < nc; j++) out[(size_t)(t + r) * N + n0 + j] = (float)acc[r][j];
            } else {
                float acc[4][4];
                for (uint32_t r = 0; r < R; r++)
                    for (uint32_t j = 0; j < 4; j++) acc[r][j] = (j < nc && bias) ? bias[n0 + j] : 0.0f;
                for (uint32_t k0 = 0; k0 < K; k0 += block) {
                    const uint32_t kl = K - k0 < block ? K - k0 : block;
                    if (chunked) gf_decode_panel(&w, n0, nc, k0, kl, panel);
                    if (R == 4)
                        gf_tile32_4(xt + k0, xrow, panel, kl, acc);
                    else
                        gf_tile32_1(xt + k0, panel, kl, acc[0]);
                }
                for (uint32_t r = 0; r < R; r++)
                    for (uint32_t j = 0; j < nc; j++) out[(size_t)(t + r) * N + n0 + j] = acc[r][j];
            }
            t += R;
        }
    }
}

/* Row-range operators split into GT_ROWS tasks of GT_CHUNK rows (each row is computed exactly as in one pass). */
typedef void (*GtRowsFn)(Context *c, const Op *o, uint32_t t0, uint32_t t1);
#define GT_CHUNK 32u
static void gt_rows(GtRowsFn fn, Context *c, const Op *o, uint32_t rows) {
    if (gt_threads() <= 1u || rows <= GT_CHUNK) {
        fn(c, o, 0, rows);
        return;
    }
    GtArgs args;
    args.p[0] = c;
    args.p[1] = o;
    args.p[2] = (const void *)fn;
    args.u[0] = rows;
    gt_parallel(GT_ROWS, &args, (rows + GT_CHUNK - 1u) / GT_CHUNK);
}

/* F_ADD: float32 sum of equal-shape tensors. */
static void gf_add_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const uint32_t C = c->t[o->out].c;
    const float *a = gf_tensor(c, o->a), *b = gf_tensor(c, o->b);
    float *z = gf_tensor(c, o->out);
    for (size_t i = (size_t)t0 * C; i < (size_t)t1 * C; i++) z[i] = a[i] + b[i];
}
static void gf_op_add(Context *c, const Op *o, uint32_t rows) { gt_rows(gf_add_rows, c, o, rows); }

/* F_NORM: p0 0 LayerNorm of every valid row over its C channels; p0 1 one group (GroupNorm with a single group)
   over all valid rows * C values with a per-channel affine. Means and variances are float64 sums in increasing
   (row, channel) order, the variance is the mean of squared deviations from the mean, the output is
   float32(((x - mean) * (1 / sqrt(var + eps))) * gamma + beta) evaluated in float64. */
static void gf_op_norm(Context *c, const Op *o, uint32_t rows) {
    const uint32_t C = c->t[o->a].c;
    const float *x = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    const GfF *gamma = (const GfF *)(c->blob + o->weights);
    const GfF *beta = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    const double eps = (double)*(const GfF *)(c->blob + o->mult);
#ifdef GUST_FLOAT32_T5
    if (o->p0 == 4) { /* RMS norm (T5): float32((x * (1 / sqrt(mean(x^2) + eps))) * gamma) evaluated in float64 */
        for (uint32_t t = 0; t < rows; t++) {
            const float *xr = x + (size_t)t * C;
            float *zr = z + (size_t)t * C;
            double sq = 0.0;
            for (uint32_t i = 0; i < C; i++) sq = sq + (double)xr[i] * (double)xr[i];
            const double inv = 1.0 / __builtin_sqrt(sq / (double)C + eps);
            for (uint32_t i = 0; i < C; i++) zr[i] = (float)(((double)xr[i] * inv) * (double)gamma[i]);
        }
        return;
    }
#endif
    if (o->p0 == 0) {
        for (uint32_t t = 0; t < rows; t++) {
            const float *xr = x + (size_t)t * C;
            float *zr = z + (size_t)t * C;
            double sum = 0.0;
            for (uint32_t i = 0; i < C; i++) sum = sum + (double)xr[i];
            const double mean = sum / (double)C;
            double var = 0.0;
            for (uint32_t i = 0; i < C; i++) {
                const double d = (double)xr[i] - mean;
                var = var + d * d;
            }
            var = var / (double)C;
            const double inv = 1.0 / __builtin_sqrt(var + eps);
            for (uint32_t i = 0; i < C; i++) {
                double y = (((double)xr[i] - mean) * inv) * (double)gamma[i];
                if (beta) y = y + (double)beta[i];
                zr[i] = (float)y;
            }
        }
        return;
    }
    const double count = (double)rows * (double)C;
    double sum = 0.0;
    for (size_t i = 0, n = (size_t)rows * C; i < n; i++) sum = sum + (double)x[i];
    const double mean = sum / count;
    double var = 0.0;
    for (size_t i = 0, n = (size_t)rows * C; i < n; i++) {
        const double d = (double)x[i] - mean;
        var = var + d * d;
    }
    var = var / count;
    const double inv = 1.0 / __builtin_sqrt(var + eps);
    for (uint32_t t = 0; t < rows; t++) {
        const float *xr = x + (size_t)t * C;
        float *zr = z + (size_t)t * C;
        for (uint32_t i = 0; i < C; i++) {
            double y = (((double)xr[i] - mean) * inv) * (double)gamma[i];
            if (beta) y = y + (double)beta[i];
            zr[i] = (float)y;
        }
    }
}

/* F_UNARY: p0 0 tanh, p0 1 GELU (exact erf form), 2 fast GELU, 3 ReLU. */
static void gf_unary_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const size_t C = c->t[o->out].c, n = (size_t)(t1 - t0) * C;
    const float *x = gf_tensor(c, o->a) + (size_t)t0 * C;
    float *z = gf_tensor(c, o->out) + (size_t)t0 * C;
    if (o->p0 == 0)
        for (size_t i = 0; i < n; i++) z[i] = gf_tanhf(x[i]);
    else if (o->p0 == 2) /* fast GELU (kernels.c): vector and scalar lanes give the same bits */
        gf_gelu_n(x, z, n);
#ifdef GUST_FLOAT32_T5
    else if (o->p0 == 3) /* ReLU (float32-t5-v1); -0 and NaN give 0 */
        for (size_t i = 0; i < n; i++) z[i] = x[i] > 0.0f ? x[i] : 0.0f;
#endif
    else
        for (size_t i = 0; i < n; i++) z[i] = gf_gelu(x[i]);
}
static void gf_op_unary(Context *c, const Op *o, uint32_t rows) { gt_rows(gf_unary_rows, c, o, rows); }

/* F_ROPE: per head, rotates the first kw dimensions as interleaved pairs with the stored tables; the remaining
   dimensions are copied. Row t uses table row t: out[2i] = x[2i] * cos + (-x[2i+1]) * sin,
   out[2i+1] = x[2i+1] * cos + x[2i] * sin, each product rounded to float32 before the sum. */
static void gf_rope_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads, rot = o->kw, half = rot / 2;
    const float *x = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    const GfF *cs = (const GfF *)(c->blob + o->weights), *sn = (const GfF *)(c->blob + o->bias);
    for (uint32_t t = t0; t < t1; t++) {
        const GfF *ct = cs + (size_t)t * half, *st = sn + (size_t)t * half;
        for (uint32_t h = 0; h < heads; h++) {
            const float *xr = x + (size_t)t * C + (size_t)h * D;
            float *zr = z + (size_t)t * C + (size_t)h * D;
            for (uint32_t i = 0; i < half; i++) {
                const float a = xr[2 * i], b = xr[2 * i + 1], cv = ct[i], sv = st[i];
                const float p = a * cv, q = (-b) * sv, r = b * cv, s = a * sv;
                zr[2 * i] = p + q;
                zr[2 * i + 1] = r + s;
            }
            for (uint32_t d = rot; d < D; d++) zr[d] = xr[d];
        }
    }
}
static void gf_op_rope(Context *c, const Op *o, uint32_t rows) { gt_rows(gf_rope_rows, c, o, rows); }

/* F_MHA: fused multi-head non-causal attention. Q is a, K is b, V is the tensor named by `weights`; heads = kh,
   head width D = C / heads, scale = float32 at `mult`. For every query row and head:
     score[j] = float32(dot64(q, k_j) * scale) for the rk valid keys (dot order in kernels.c),
     m = max score, e[j] = expf(score[j] - m) in float32, sum = float64 sum of e[j] in increasing j,
     out[d] = float32((float64 sum over j of e[j] * v[j][d]) / sum), keys in increasing order.
   The score row lives in the Context scratch (at most GUST_F32_MAX_KEYS keys); no score matrix is stored.
   p0 bit 0 (banded, sliding-window encoders such as ModernBERT): query row i sees only the keys j with
   |i - j| <= p1 (and j < rk); the loops are the same, run over that key range.
   `bias` (float32-t5-v1, F_MHA p0 bit 2 and F_MHA_PREFIX kw 1) is 0 or an additive score bias, one float32
   addition before the maximum: score[j] = float32(score[j] + bias[(h * bq + i) * bs + j]) (never banded). */
static void gf_op_mha_bias(Context *c, const Op *o, uint32_t rows, uint32_t rk, const float *bias, uint32_t bq,
                           uint32_t bs);
static void gf_op_mha(Context *c, const Op *o, uint32_t rows, uint32_t rk) {
#ifdef GUST_FLOAT32_T5
    if (o->code == GUST_F_MHA && (o->p0 & 4u)) {
        gf_op_mha_bias(c, o, rows, rk, gf_tensor(c, o->bias), c->t[o->bias].h / o->kh, c->t[o->bias].c);
        return;
    }
#endif
    gf_op_mha_bias(c, o, rows, rk, (const float *)0, 0, 0);
}
static void gf_op_mha_bias(Context *c, const Op *o, uint32_t rows, uint32_t rk, const float *bias, uint32_t bq,
                           uint32_t bs) {
    const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads;
    const float scale = *(const GfF *)(c->blob + o->mult);
    const float *Q = gf_tensor(c, o->a), *K = gf_tensor(c, o->b), *V = gf_tensor(c, o->weights);
    float *z = gf_tensor(c, o->out);
    float *score = gf_scratch(c);
    double q[GUST_F32_MAX_HEAD] __attribute__((aligned(16))), acc[GUST_F32_MAX_HEAD] __attribute__((aligned(16)));
    const uint32_t banded = o->code == GUST_F_MHA && (o->p0 & 1u), half = banded ? o->p1 : 0u;
    for (uint32_t i = 0; i < rows; i++) {
        uint32_t j0 = 0, n = rk;
        if (banded) {
            j0 = i > half ? i - half : 0u;
            const uint32_t j1 = (uint64_t)i + half + 1u < rk ? i + half + 1u : rk;
            n = j1 > j0 ? j1 - j0 : 0u;
        }
        for (uint32_t h = 0; h < heads; h++) {
            const float *qr = Q + (size_t)i * C + (size_t)h * D;
            const float *kh = K + (size_t)j0 * C + (size_t)h * D, *vh = V + (size_t)j0 * C + (size_t)h * D;
            for (uint32_t d = 0; d < D; d++) q[d] = (double)qr[d];
            float m = 0.0f;
            for (uint32_t j = 0; j < n; j++) {
                float s = (float)(gf_dot64(q, kh + (size_t)j * C, D) * (double)scale);
                if (bias) s = s + bias[((size_t)h * bq + i) * bs + j];
                score[j] = s;
                if (j == 0 || s > m) m = s;
            }
            double sum = 0.0;
            for (uint32_t j = 0; j < n; j++) {
                const float e = gf_expf(score[j] - m);
                score[j] = e;
                sum = sum + (double)e;
            }
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0;
            uint32_t j = 0;
            for (; j + 4 <= n; j += 4) {
                const double e4[4] = {(double)score[j], (double)score[j + 1], (double)score[j + 2],
                                      (double)score[j + 3]};
                gf_axpy64_4(acc, e4, vh + (size_t)j * C, C, D);
            }
            for (; j < n; j++) gf_axpy64_1(acc, (double)score[j], vh + (size_t)j * C, D);
            float *zr = z + (size_t)i * C + (size_t)h * D;
            for (uint32_t d = 0; d < D; d++) zr[d] = (float)(acc[d] / sum);
        }
    }
}

/* F_NORM p0 bit 1 (fast LayerNorm, layer mode only): float32 mean = sum32(x) / C, variance = sum32((x - mean)^2) / C
   (eight partial sums, kernels.c), inv = 1 / sqrtf(var + eps), y = ((x - mean) * inv) * gamma (+ beta), float32. */
static void gf_norm32_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const uint32_t C = c->t[o->a].c;
    const float *x = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    const GfF *gamma = (const GfF *)(c->blob + o->weights);
    const GfF *beta = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    const float eps = *(const GfF *)(c->blob + o->mult);
    for (uint32_t t = t0; t < t1; t++) {
        const float *xr = x + (size_t)t * C;
        float *zr = z + (size_t)t * C;
        const float mean = gf_sum32(xr, C, 0, 0.0f) / (float)C;
        const float var = gf_sum32(xr, C, 1, mean) / (float)C;
        const float inv = 1.0f / __builtin_sqrtf(var + eps);
        if (beta)
            for (uint32_t i = 0; i < C; i++) zr[i] = ((xr[i] - mean) * inv) * gamma[i] + beta[i];
        else
            for (uint32_t i = 0; i < C; i++) zr[i] = ((xr[i] - mean) * inv) * gamma[i];
    }
}
static void gf_op_norm32(Context *c, const Op *o, uint32_t rows) { gt_rows(gf_norm32_rows, c, o, rows); }

/* F_MHA p0 bit 1, tiled (head width a multiple of GF_TW and every thread's per-head panels fit Context.f32_act):
   per head (one GT_HEAD task each), K is packed as [key block][D][GF_TW] and V as [dim block][keys][GF_TW];
   GF_TR queries at a time get score[r][j] = float32 sum over d = 0 .. D-1 in increasing order of q_r[d] * k_j[d],
   times scale (gf_tile32w), then e = expf(score - m) and sum per query exactly as gf_softmax_exp over its keys, then
   out[r][d] = (sum over keys j increasing of e[r][j] * v_j[d]) / sum, also on gf_tile32w (keys of the tile outside a
   query's band carry e = 0). Banding as in gf_op_mha. */
/* Packed K ([key block][D][GF_TW]) and V ([dim block][keys][GF_TW]) of head h at kt / vp. */
static void gf_mha_pack(Context *c, const Op *o, uint32_t rk, uint32_t h, float *kt, float *vp) {
    const uint32_t C = c->t[o->a].c, D = C / o->kh, nb = (rk + GF_TW - 1u) / GF_TW, kpad = nb * GF_TW;
    const float *K = gf_tensor(c, o->b), *V = gf_tensor(c, o->weights);
    for (uint32_t b = 0; b < nb; b++)
        for (uint32_t d = 0; d < D; d++)
            for (uint32_t jj = 0; jj < GF_TW; jj++) {
                const uint32_t j = b * GF_TW + jj;
                kt[((size_t)b * D + d) * GF_TW + jj] = j < rk ? K[(size_t)j * C + h * D + d] : 0.0f;
            }
    for (uint32_t db = 0; db < D / GF_TW; db++)
        for (uint32_t j = 0; j < kpad; j++)
            for (uint32_t jj = 0; jj < GF_TW; jj++)
                vp[((size_t)db * kpad + j) * GF_TW + jj] = j < rk ? V[(size_t)j * C + h * D + db * GF_TW + jj] : 0.0f;
}

/* Query rows i0s .. i1s - 1 of head h against the packed kt / vp, scores in S ([GF_TR][kpad]). Tiles start at
   i0s, so callers start chunks at multiples of GF_TR to keep the single-pass row grouping (and bits). */
static void gf_mha_rows(Context *c, const Op *o, uint32_t rk, uint32_t h, uint32_t i0s, uint32_t i1s,
                        const float *kt, const float *vp, float *S) {
    const uint32_t C = c->t[o->a].c, D = C / o->kh, nb = (rk + GF_TW - 1u) / GF_TW, kpad = nb * GF_TW, srow = kpad;
    const float scale = *(const GfF *)(c->blob + o->mult);
    const float *Q = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    const uint32_t banded = o->p0 & 1u, half = o->p1;
    float acc[GF_TR * GF_TW] __attribute__((aligned(32)));
    for (uint32_t i0 = i0s; i0 < i1s;) {
        const uint32_t R = i1s - i0 >= GF_TR ? GF_TR : 1u;
        uint32_t lo[GF_TR], hi[GF_TR], tlo = rk, thi = 0;
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t i = i0 + r;
            lo[r] = 0;
            hi[r] = rk;
            if (banded) {
                lo[r] = i > half ? i - half : 0u;
                hi[r] = (uint64_t)i + half + 1u < rk ? i + half + 1u : rk;
                if (hi[r] < lo[r]) hi[r] = lo[r];
            }
            if (lo[r] < tlo) tlo = lo[r];
            if (hi[r] > thi) thi = hi[r];
        }
        const uint32_t b0 = tlo / GF_TW, b1 = (thi + GF_TW - 1u) / GF_TW, j0 = b0 * GF_TW, nk = (b1 - b0) * GF_TW;
        const float *qt = Q + (size_t)i0 * C + (size_t)h * D;
        for (uint32_t b = b0; b < b1; b++) {
            for (uint32_t u = 0; u < GF_TR * GF_TW; u++) acc[u] = 0.0f;
            if (R == GF_TR)
                gf_tile32w(qt, C, kt + (size_t)b * D * GF_TW, D, acc, GF_TR);
            else
                gf_tile32w(qt, C, kt + (size_t)b * D * GF_TW, D, acc, 1u);
            for (uint32_t r = 0; r < R; r++)
                for (uint32_t jj = 0; jj < GF_TW; jj++) S[(size_t)r * srow + (b - b0) * GF_TW + jj] = acc[r * GF_TW + jj] * scale;
        }
        float sum[GF_TR];
        for (uint32_t r = 0; r < R; r++) {
            float *sr = S + (size_t)r * srow;
            const uint32_t a = lo[r] - j0, e = hi[r] - j0;
            const float m = gf_maxf(sr + a, e - a);
            sum[r] = gf_softmax_exp(sr + a, e - a, m);
            for (uint32_t j = 0; j < a; j++) sr[j] = 0.0f;
            for (uint32_t j = e; j < nk; j++) sr[j] = 0.0f;
        }
        for (uint32_t db = 0; db < D / GF_TW; db++) {
            for (uint32_t u = 0; u < GF_TR * GF_TW; u++) acc[u] = 0.0f;
            const float *vb = vp + ((size_t)db * kpad + j0) * GF_TW;
            if (R == GF_TR)
                gf_tile32w(S, srow, vb, nk, acc, GF_TR);
            else
                gf_tile32w(S, srow, vb, nk, acc, 1u);
            for (uint32_t r = 0; r < R; r++) {
                float *zr = z + (size_t)(i0 + r) * C + (size_t)h * D + db * GF_TW;
                for (uint32_t jj = 0; jj < GF_TW; jj++) zr[jj] = acc[r * GF_TW + jj] / sum[r];
            }
        }
        i0 += R;
    }
}

/* GT_HEAD (panels per thread): pack head h and run all its queries. */
static void gf_mha_head(const GtArgs *ga, uint32_t h, uint32_t tid) {
    Context *c = (Context *)ga->p[0];
    const Op *o = (const Op *)ga->p[1];
    const uint32_t rows = ga->u[0], rk = ga->u[1], per = ga->u[2];
    const uint32_t D = c->t[o->a].c / o->kh, kpad = (rk + GF_TW - 1u) / GF_TW * GF_TW;
    float *kt = (float *)(void *)c->f32_act + (size_t)tid * per, *vp = kt + (size_t)D * kpad, *S = vp + (size_t)D * kpad;
    gf_mha_pack(c, o, rk, h, kt, vp);
    gf_mha_rows(c, o, rk, h, 0, rows, kt, vp, S);
}

#define GT_QROWS (8u * GF_TR) /* query rows per GT_QCHUNK task: a multiple of the tile height */

/* GT_PACK / GT_QCHUNK (panels of every head stored at once): ga->u[3] = floats per head's panels, ga->u[4] =
   offset of the per-thread score rows, ga->u[5] = query chunks per head. */
static void gf_mha_pack_task(const GtArgs *ga, uint32_t h) {
    Context *c = (Context *)ga->p[0];
    const Op *o = (const Op *)ga->p[1];
    const uint32_t rk = ga->u[1], ph = ga->u[3];
    const uint32_t D = c->t[o->a].c / o->kh, kpad = (rk + GF_TW - 1u) / GF_TW * GF_TW;
    float *kt = (float *)(void *)c->f32_act + (size_t)h * ph;
    gf_mha_pack(c, o, rk, h, kt, kt + (size_t)D * kpad);
}
static void gf_mha_chunk_task(const GtArgs *ga, uint32_t i, uint32_t tid) {
    Context *c = (Context *)ga->p[0];
    const Op *o = (const Op *)ga->p[1];
    const uint32_t rows = ga->u[0], rk = ga->u[1], ph = ga->u[3], so = ga->u[4], chunks = ga->u[5];
    const uint32_t h = i / chunks, q0 = (i % chunks) * GT_QROWS, q1 = q0 + GT_QROWS < rows ? q0 + GT_QROWS : rows;
    const uint32_t D = c->t[o->a].c / o->kh, kpad = (rk + GF_TW - 1u) / GF_TW * GF_TW;
    float *base = (float *)(void *)c->f32_act, *kt = base + (size_t)h * ph;
    gf_mha_rows(c, o, rk, h, q0, q1, kt, kt + (size_t)D * kpad, base + so + (size_t)tid * GF_TR * kpad);
}

static __attribute__((noinline)) int gf_mha_tiled(Context *c, const Op *o, uint32_t rows, uint32_t rk) {
    const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads;
    const uint32_t kpad = (rk + GF_TW - 1u) / GF_TW * GF_TW;
    if (D % GF_TW) return 0;
    const uint64_t cap = GUST_F32_ACT_SLOTS / 2u, T = gt_threads();
    const uint64_t ph = (2ull * D * kpad + 3u) & ~3ull, sc = ((uint64_t)GF_TR * kpad + 3u) & ~3ull;
    GtArgs args;
    args.p[0] = c;
    args.p[1] = o;
    args.u[0] = rows;
    args.u[1] = rk;
    if (T > 1u && rows > GT_QROWS && heads * ph + T * sc <= cap) { /* all heads packed, then query chunks */
        args.u[3] = (uint32_t)ph;
        args.u[4] = (uint32_t)(heads * ph);
        args.u[5] = (rows + GT_QROWS - 1u) / GT_QROWS;
        gt_parallel(GT_PACK, &args, heads);
        gt_parallel(GT_QCHUNK, &args, heads * args.u[5]);
        return 1;
    }
    if ((ph + sc) * T > cap) return 0;
    args.u[2] = (uint32_t)(ph + sc);
    gt_parallel(GT_HEAD, &args, heads);
    return 1;
}

static void gt_task(uint32_t kind, const GtArgs *a, uint32_t i, uint32_t tid) {
    switch (kind) {
    case GT_QUANT: gq_quant_task(a, i); break;
    case GT_SLAB: gq_slab_task(a, i, tid); break;
    case GT_QUANT8: gq8_quant_task(a, i); break;
    case GT_SLAB8: gq8_slab_task(a, i, tid); break;
    case GT_HEAD: gf_mha_head(a, i, tid); break;
    case GT_PACK: gf_mha_pack_task(a, i); break;
    case GT_QCHUNK: gf_mha_chunk_task(a, i, tid); break;
    case GT_ZERO: gf_zero_task(a, i); break;
    case GT_HOST: ((const GtHost *)a->p[0])->fn(((const GtHost *)a->p[0])->opaque, i, tid); break;
    case GT_COLS:
        gf_conv32_cols((Context *)a->p[0], (const Op *)a->p[1], (const GfWeights *)a->p[2], a->u[0], i * GF_TW,
                       (float *)(void *)((Context *)a->p[0])->f32_slab[tid]);
        break;
    default: { /* GT_ROWS */
        const uint32_t t0 = i * GT_CHUNK, t1 = t0 + GT_CHUNK < a->u[0] ? t0 + GT_CHUNK : a->u[0];
        ((GtRowsFn)a->p[2])((Context *)a->p[0], (const Op *)a->p[1], t0, t1);
    }
    }
}

/* F_MHA p0 bit 1 (fast attention): float32 throughout. score[j] = dot32(q, k_j) * scale (kernels.c), m = max,
   e[j] = expf(score[j] - m), sum = float32 sum of e[j] in increasing j, out[d] = (sum over increasing j of
   e[j] * v[j][d]) / sum. p0 bit 0 bands the keys exactly like gf_op_mha. */
static void gf_op_mha32(Context *c, const Op *o, uint32_t rows, uint32_t rk) {
    if (gf_mha_tiled(c, o, rows, rk)) return;
    const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads;
    const float scale = *(const GfF *)(c->blob + o->mult);
    const float *Q = gf_tensor(c, o->a), *K = gf_tensor(c, o->b), *V = gf_tensor(c, o->weights);
    float *z = gf_tensor(c, o->out);
    float *score = gf_scratch(c);
    float acc[GUST_F32_MAX_HEAD] __attribute__((aligned(32)));
    const uint32_t banded = o->p0 & 1u, half = o->p1;
    for (uint32_t i = 0; i < rows; i++) {
        uint32_t j0 = 0, n = rk;
        if (banded) {
            j0 = i > half ? i - half : 0u;
            const uint32_t j1 = (uint64_t)i + half + 1u < rk ? i + half + 1u : rk;
            n = j1 > j0 ? j1 - j0 : 0u;
        }
        for (uint32_t h = 0; h < heads; h++) {
            const float *qr = Q + (size_t)i * C + (size_t)h * D;
            const float *kh = K + (size_t)j0 * C + (size_t)h * D, *vh = V + (size_t)j0 * C + (size_t)h * D;
            float m = 0.0f;
            uint32_t j = 0;
            for (; j + 4 <= n; j += 4) {
                float s4[4];
                gf_dot32x4(qr, kh + (size_t)j * C, C, D, s4);
                for (uint32_t u = 0; u < 4; u++) {
                    const float s = s4[u] * scale;
                    score[j + u] = s;
                    if (j + u == 0 || s > m) m = s;
                }
            }
            for (; j < n; j++) {
                const float s = gf_dot32(qr, kh + (size_t)j * C, D) * scale;
                score[j] = s;
                if (j == 0 || s > m) m = s;
            }
            const float sum = gf_softmax_exp(score, n, m);
            for (uint32_t d = 0; d < D; d++) acc[d] = 0.0f;
            for (uint32_t j = 0; j < n; j++) gf_axpy32(acc, score[j], vh + (size_t)j * C, D);
            float *zr = z + (size_t)i * C + (size_t)h * D;
            for (uint32_t d = 0; d < D; d++) zr[d] = acc[d] / sum;
        }
    }
}
