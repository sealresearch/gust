/* float32-temporal-v1: operators 47 to 49 for temporal convolution networks (ConvNeXt-style blocks over time with a
   conditioning vector). Builds with GUST_FLOAT32_DECODER (like float32-t5-v1); see TEMPORAL.md.
   Part of the runtime/src/gust.c translation unit.

   Every operator fixes its operation order, uses IEEE float32 multiplies and adds (contraction off) and the
   deterministic tanh of math.c, so native and WebAssembly builds, SIMD or not, return identical bits. */
#pragma clang fp contract(off)

#define TP_MAX_TAPS    31u

/* Loader checks of one temporal operator; 0 or a GUST_* status. Liveness as for the decoder operators. */
static int tp_load_op(const Context *c, int32_t *first, int32_t *last, uint32_t index, const Op *o) {
    int s;
    if ((s = gd_read(c, first, last, o->a, index))) return s;
    if (o->code == GUST_F_FILM && (s = gd_read(c, first, last, o->b, index))) return s;
    if ((s = gd_write(c, first, last, o->out, index))) return s;
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    switch (o->code) {
    case GUST_F_DWCONV: {
        if (!same(a, z) || !o->kh || o->kh > TP_MAX_TAPS || !(o->kh & 1u) || !o->dy) return GUST_SHAPE;
        if ((uint64_t)o->dy * (o->kh - 1u) / 2u >= GUST_F32_MAX_DIM) return GUST_SHAPE;
        const uint64_t taps = (uint64_t)o->kh * a->c;
        if (!gf_blob(c, o->weights, 4 * taps, 4)) return GUST_FORMAT;
        if (!gf_floats_ok(c, o->weights, taps, 0)) return GUST_QUANTIZATION;
        if (o->bias) {
            if (!gf_blob(c, o->bias, 4ull * a->c, 4)) return GUST_FORMAT;
            if (!gf_floats_ok(c, o->bias, a->c, 0)) return GUST_QUANTIZATION;
        }
        return GUST_OK;
    }
    case GUST_F_FILM: {
        const Tensor *b = c->t + o->b;
        if (!same(a, z) || o->p0 > 1 || (uint64_t)b->c != 2ull * a->c || a->c > 16384u) return GUST_SHAPE;
        return GUST_OK;
    }
    default: /* F_MEAN */
        return z->h == 1 && z->c == a->c ? GUST_OK : GUST_SHAPE;
    }
}

/* F_DWCONV: depthwise convolution over rows (time) with zero padding at the valid-row edges. For output row t and
   channel j: y = bias[j] (or 0); for tap k = 0 .. kh-1 in order, s = t + (k - (kh-1)/2) * dy, and when 0 <= s < V:
   y = y + x[s][j] * w[k][j]. Weights are float32 [kh][C] (tap-major). */
static void tp_dwconv_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const uint32_t C = c->t[o->a].c, kh = o->kh, half = (kh - 1u) / 2u, dy = o->dy, V = c->packed_offsets[o->a];
    const float *x = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    const GfF *w = (const GfF *)(c->blob + o->weights);
    const GfF *bias = o->bias ? (const GfF *)(c->blob + o->bias) : (const GfF *)0;
    for (uint32_t t = t0; t < t1; t++) {
        float *zr = z + (size_t)t * C;
        if (bias)
            for (uint32_t j = 0; j < C; j++) zr[j] = bias[j];
        else
            for (uint32_t j = 0; j < C; j++) zr[j] = 0.0f;
        for (uint32_t k = 0; k < kh; k++) {
            const int64_t s = (int64_t)t + ((int64_t)k - (int64_t)half) * (int64_t)dy;
            if (s < 0 || s >= (int64_t)V) continue;
            const float *xr = x + (size_t)s * C;
            const GfF *wk = w + (size_t)k * C;
            uint32_t j = 0;
#ifdef GUST_F32_SIMD
            for (; j + 8 <= C; j += 8) *(gv8f *)(zr + j) = *(const gv8f *)(zr + j) + *(const gv8f *)(xr + j) * *(const gv8f *)(wk + j);
#endif
            for (; j < C; j++) zr[j] = zr[j] + xr[j] * wk[j];
        }
    }
}

/* F_FILM: z[t][j] = x[t][j] * g[j] + v[j] from row 0 of b = [u | v] ([.., 2C]): p0 0 g = 1 + u, p0 1 g = 1 + 0.5 * tanh(u)
   (tanh of math.c). The C gains are computed once per call, before any row. */
static void tp_film_rows(Context *c, const Op *o, uint32_t t0, uint32_t t1) {
    const uint32_t C = c->t[o->a].c;
    const float *x = gf_tensor(c, o->a), *g = gf_scratch(c), *v = gf_tensor(c, o->b) + C;
    float *z = gf_tensor(c, o->out);
    for (uint32_t t = t0; t < t1; t++) {
        const float *xr = x + (size_t)t * C;
        float *zr = z + (size_t)t * C;
        uint32_t j = 0;
#ifdef GUST_F32_SIMD
        for (; j + 8 <= C; j += 8) *(gv8f *)(zr + j) = *(const gv8f *)(xr + j) * *(const gv8f *)(g + j) + *(const gv8f *)(v + j);
#endif
        for (; j < C; j++) zr[j] = xr[j] * g[j] + v[j];
    }
}
static void tp_op_film(Context *c, const Op *o, uint32_t rows) {
    const uint32_t C = c->t[o->a].c;
    const float *u = gf_tensor(c, o->b);
    float *g = gf_scratch(c); /* the 64 KiB scratch holds the gains (the loader checks C <= 16384) */
    for (uint32_t j = 0; j < C; j++) g[j] = o->p0 ? 1.0f + 0.5f * gf_tanhf(u[j]) : 1.0f + u[j];
    gt_rows(tp_film_rows, c, o, rows);
}

/* F_MEAN: z[0][j] = float32(sum over valid rows t in increasing order of (double)x[t][j], divided by V). */
static void tp_op_mean(Context *c, const Op *o) {
    const uint32_t C = c->t[o->a].c, V = c->packed_offsets[o->a];
    const float *x = gf_tensor(c, o->a);
    float *z = gf_tensor(c, o->out);
    for (uint32_t j = 0; j < C; j++) {
        double sum = 0.0;
        for (uint32_t t = 0; t < V; t++) sum = sum + (double)x[(size_t)t * C + j];
        z[j] = (float)(sum / (double)V);
    }
}

static void tp_run_op(Context *c, const Op *o, uint32_t rows) {
    switch (o->code) {
    case GUST_F_DWCONV: gt_rows(tp_dwconv_rows, c, o, rows); break;
    case GUST_F_FILM: tp_op_film(c, o, rows); break;
    default: tp_op_mean(c, o); break;
    }
}
