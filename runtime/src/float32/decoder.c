/* float32-decoder-v1: operators 38 to 45 and persistent state for cached (KV) decoding. Builds with
   GUST_FLOAT32_DECODER only; every other build is byte-for-byte what it was without this file.

   Dynamic values (token, position, count, counter) are checked for the whole graph by gd_preflight BEFORE the first
   operator runs, so a refused call changes no arena, state, input or output byte. A step then runs its operators in
   artifact order; the last operator (F_COMMIT) advances the position counter, so a stop request between operators
   (GUST_BUDGET) leaves the counter where it was and a retry of the same position rewrites the same cache rows with
   the same values. Old cache rows are never touched, rows past the position are never read.

   Math: expf is the deterministic port in math.c, everything else is IEEE float32 add/multiply/divide, no host
   libm, no contraction. The attention loops are the ones of F_MHA (ops.c). */
#pragma clang fp contract(off)

static void gd_load_begin(GdLoad *g) {
    g->counter = -1;
    g->position = -1;
    g->commits = 0;
    for (uint32_t i = 0; i < 4096; i++) g->append_at[i] = 0xffffu;
}

static inline int gd_is_state(const Context *c, uint32_t t) {
    return (c->t[t].flags & GUST_F32_STATE) != 0;
}

/* The operands of a fixed-function (32 to 37) operator are never state. */
static int gd_ordinary_operands_ok(const Context *c, const Op *o) {
    if (gd_is_state(c, o->a) || gd_is_state(c, o->out)) return 0;
    if ((o->code == GUST_F_ADD || o->code == GUST_F_MHA) && gd_is_state(c, o->b)) return 0;
    if (o->code == GUST_F_MHA && gd_is_state(c, o->weights)) return 0;
    return 1;
}

/* A tensor read by an ordinary operand: it exists and is not state. */
static int gd_read(const Context *c, const int32_t *first, int32_t *last, uint32_t t, uint32_t index) {
    if (t >= c->header->nt || first[t] == -2 || gd_is_state(c, t)) return GUST_GRAPH;
    last[t] = (int32_t)index;
    return GUST_OK;
}
/* The output of an ordinary operand: not an input, not state, exactly one producer. */
static int gd_write(const Context *c, int32_t *first, int32_t *last, uint32_t t, uint32_t index) {
    if (t >= c->header->nt || (c->t[t].flags & (1u | GUST_F32_STATE)) || first[t] != -2) return GUST_GRAPH;
    first[t] = (int32_t)index;
    last[t] = (int32_t)index;
    return GUST_OK;
}
/* A [1,1] float32 input the host writes before the run: a token, a position or a count. */
static int gd_scalar_input(const Context *c, int32_t *last, uint32_t t, uint32_t index) {
    if (t >= c->header->nt || !(c->t[t].flags & 1u)) return GUST_GRAPH;
    if (c->t[t].h != 1 || c->t[t].c != 1) return GUST_SHAPE;
    last[t] = (int32_t)index;
    return GUST_OK;
}
/* All appends, the commit and self attention name the same position input. */
static int gd_same_position(GdLoad *g, uint32_t t) {
    if (g->position < 0) g->position = (int32_t)t;
    return g->position == (int32_t)t ? GUST_OK : GUST_GRAPH;
}
static int gd_same_counter(GdLoad *g, uint32_t t) {
    if (g->counter < 0) g->counter = (int32_t)t;
    return g->counter == (int32_t)t ? GUST_OK : GUST_GRAPH;
}

static int gd_load_op(const Context *c, GdLoad *g, int32_t *first, int32_t *last, uint32_t index, const Op *o) {
    const uint32_t nt = c->header->nt;
    int s;
    switch (o->code) {
    case GUST_F_MUL: {
        if (o->a >= nt || o->b >= nt || o->out >= nt) return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_read(c, first, last, o->b, index)) ||
            (s = gd_write(c, first, last, o->out, index)))
            return s;
        const Tensor *a = c->t + o->a, *b = c->t + o->b, *z = c->t + o->out;
        return same(a, b) && same(a, z) ? GUST_OK : GUST_SHAPE;
    }
    case GUST_F_SLICE: {
        if (o->a >= nt || o->out >= nt) return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_write(c, first, last, o->out, index))) return s;
        const Tensor *a = c->t + o->a, *z = c->t + o->out;
        return a->h == z->h && (uint64_t)o->p0 + z->c <= a->c ? GUST_OK : GUST_SHAPE;
    }
    case GUST_F_GATHER: {
        if (o->a >= nt || o->out >= nt) return GUST_GRAPH;
        /* tokens: an [H, 1] input (H = 1 is the decoder's single token), one output row per token */
        if (o->a >= nt || !(c->t[o->a].flags & 1u)) return GUST_GRAPH;
        if (c->t[o->a].c != 1) return GUST_SHAPE;
        last[o->a] = (int32_t)index;
        if ((s = gd_write(c, first, last, o->out, index))) return s;
        const Tensor *z = c->t + o->out;
        if (z->h != c->t[o->a].h) return GUST_SHAPE;
        if (o->p1 < 2 || o->p1 > 8) return GUST_FORMAT;
        if (!o->p0 || o->p0 > GUST_F32_MAX_DIM) return GUST_SHAPE;
        const uint64_t K = z->c, N = o->p0, codebits = N * K * o->p1, codebytes = (codebits + 7) / 8;
        if (!gf_blob(c, o->weights, codebytes, 4)) return GUST_FORMAT;
        if ((codebits & 7) && (c->blob[o->weights + codebytes - 1] >> (codebits & 7))) return GUST_FORMAT;
#ifdef GUST_FLOAT32_T5
        if (o->kh > 1) return GUST_FORMAT;
        if (o->kh) { /* float16 scale per 64 codes of a row: the F_CONV p0 bit 0 layout, so a tied head shares it */
            const uint64_t groups = N * ((K + 63) / 64);
            if (!gf_blob(c, o->mult, 2 * groups, 4)) return GUST_FORMAT;
            for (uint64_t gi = 0; gi < groups; gi++) {
                const uint16_t sc =
                    (uint16_t)(c->blob[o->mult + 2 * gi] | (uint32_t)c->blob[o->mult + 2 * gi + 1] << 8);
                if ((sc & 0x8000u) || ((sc >> 10) & 31u) == 31u || !(sc & 0x7fffu)) return GUST_QUANTIZATION;
            }
            return GUST_OK;
        }
#endif
        if (!gf_blob(c, o->mult, 4 * N, 4)) return GUST_FORMAT;
        return gf_floats_ok(c, o->mult, N, 1) ? GUST_OK : GUST_QUANTIZATION;
    }
    case GUST_F_SILU: {
        if (o->a >= nt || o->out >= nt) return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_write(c, first, last, o->out, index))) return s;
        return same(c->t + o->a, c->t + o->out) ? GUST_OK : GUST_SHAPE;
    }
    case GUST_F_ROPE_AT: {
        if (o->a >= nt || o->out >= nt) return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_scalar_input(c, last, o->b, index)) ||
            (s = gd_write(c, first, last, o->out, index)))
            return s;
        const Tensor *a = c->t + o->a, *z = c->t + o->out;
        if (a->h != 1 || !same(a, z) || !o->kh || a->c % o->kh) return GUST_SHAPE;
        const uint32_t D = a->c / o->kh;
        if (o->kw < 2 || o->kw % 2 || o->kw > D || o->a == o->out) return GUST_SHAPE;
        if (!o->p0 || o->p0 > GUST_F32_MAX_DIM) return GUST_SHAPE;
        const uint64_t table = (uint64_t)o->p0 * (o->kw / 2);
        if (!gf_blob(c, o->weights, 4 * table, 4) || !gf_blob(c, o->bias, 4 * table, 4)) return GUST_FORMAT;
        if (!gf_floats_ok(c, o->weights, table, 0) || !gf_floats_ok(c, o->bias, table, 0)) return GUST_QUANTIZATION;
        return GUST_OK;
    }
    case GUST_F_CACHE_APPEND: {
        if (o->a >= nt || o->out >= nt || o->weights >= nt) return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_scalar_input(c, last, o->b, index))) return s;
        if (!gd_is_state(c, o->out) || !gd_is_state(c, o->weights) || o->out == o->weights) return GUST_GRAPH;
        const Tensor *a = c->t + o->a, *z = c->t + o->out, *n = c->t + o->weights;
        if (a->h != 1 || z->c != a->c || !z->h || z->h > GUST_F32_STATE_ROWS || n->h != 1 || n->c != 1)
            return GUST_SHAPE;
        if (g->append_at[o->out] != 0xffffu) return GUST_GRAPH; /* exactly one append per cache and run */
        g->append_at[o->out] = (uint16_t)index;
        if ((s = gd_same_position(g, o->b)) || (s = gd_same_counter(g, o->weights))) return s;
        return GUST_OK;
    }
    case GUST_F_COMMIT: {
        if (o->a >= nt || o->out >= nt) return GUST_GRAPH;
        if (index + 1 != c->header->no || g->commits++) return GUST_GRAPH; /* one commit, the last operator */
        if ((s = gd_scalar_input(c, last, o->a, index))) return s;
        if (!gd_is_state(c, o->out)) return GUST_GRAPH;
        if (c->t[o->out].h != 1 || c->t[o->out].c != 1) return GUST_SHAPE;
        if ((s = gd_same_position(g, o->a)) || (s = gd_same_counter(g, o->out))) return s;
        return GUST_OK;
    }
    case GUST_F_MHA_CACHE: {
        /* queries a, own keys b / values weights (the query rows), cached keys bias / values shift (host-written
           inputs, the first `count` rows used), count = scalar input p0, p1 = band half width of the own keys */
        if (o->a >= nt || o->b >= nt || o->out >= nt || o->weights >= nt || o->bias >= nt || o->shift >= nt ||
            o->p0 >= nt)
            return GUST_GRAPH;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_read(c, first, last, o->b, index)) ||
            (s = gd_read(c, first, last, o->weights, index)) || (s = gd_read(c, first, last, o->bias, index)) ||
            (s = gd_read(c, first, last, o->shift, index)) || (s = gd_scalar_input(c, last, o->p0, index)) ||
            (s = gd_write(c, first, last, o->out, index)))
            return s;
        if (!(c->t[o->bias].flags & 1u) || !(c->t[o->shift].flags & 1u)) return GUST_GRAPH;
        const Tensor *a = c->t + o->a, *k = c->t + o->b, *v = c->t + o->weights, *kc = c->t + o->bias,
                     *vc = c->t + o->shift, *z = c->t + o->out;
        if (!same(a, z) || !same(a, k) || !same(a, v) || !same(kc, vc) || kc->c != a->c) return GUST_SHAPE;
        if (!o->kh || a->c % o->kh || a->c / o->kh > GUST_F32_MAX_HEAD) return GUST_SHAPE;
        if ((uint64_t)a->h + kc->h > GUST_F32_MAX_KEYS) return GUST_SHAPE;
        if (!gf_blob(c, o->mult, 4, 4)) return GUST_FORMAT;
        return gf_floats_ok(c, o->mult, 1, 1) ? GUST_OK : GUST_QUANTIZATION;
    }
    default: { /* GUST_F_MHA_PREFIX */
        if (o->a >= nt || o->b >= nt || o->out >= nt || o->weights >= nt || o->p0 >= nt) return GUST_GRAPH;
        if (o->p1 > 1) return GUST_FORMAT;
        if ((s = gd_read(c, first, last, o->a, index)) || (s = gd_scalar_input(c, last, o->p0, index)) ||
            (s = gd_write(c, first, last, o->out, index)))
            return s;
        if (o->p1 == 0) { /* count: the keys and values are ordinary tensors (the static cross attention cache) */
            if ((s = gd_read(c, first, last, o->b, index)) || (s = gd_read(c, first, last, o->weights, index)))
                return s;
        } else { /* position + 1: the keys and values are caches this very run already appended to */
            if (!gd_is_state(c, o->b) || !gd_is_state(c, o->weights) || o->b == o->weights) return GUST_GRAPH;
            if (g->append_at[o->b] >= index || g->append_at[o->weights] >= index) return GUST_GRAPH;
            if (g->position != (int32_t)o->p0) return GUST_GRAPH;
        }
        const Tensor *a = c->t + o->a, *k = c->t + o->b, *v = c->t + o->weights, *z = c->t + o->out;
        if (a->h != 1 || !same(a, z) || !o->kh || a->c % o->kh || a->c / o->kh > GUST_F32_MAX_HEAD) return GUST_SHAPE;
        if (a->c != k->c || !same(k, v) || k->h > GUST_F32_MAX_KEYS) return GUST_SHAPE;
#ifdef GUST_FLOAT32_T5
        if (o->kw > 1 || (!o->kw && o->bias)) return GUST_FORMAT;
        if (o->kw) { /* additive score bias [heads, key rows] the host writes for this step */
            if (o->bias >= nt) return GUST_GRAPH;
            if ((s = gd_read(c, first, last, o->bias, index))) return s;
            if (c->t[o->bias].h != o->kh || c->t[o->bias].c != k->h) return GUST_SHAPE;
        }
#endif
        if (!gf_blob(c, o->mult, 4, 4)) return GUST_FORMAT;
        return gf_floats_ok(c, o->mult, 1, 1) ? GUST_OK : GUST_QUANTIZATION;
    }
    }
}

static int gd_load_end(const Context *c, const GdLoad *g) {
    uint32_t states = 0;
    for (uint32_t t = 0; t < c->header->nt; t++) {
        if (!gd_is_state(c, t)) continue;
        states++;
        if ((int32_t)t == g->counter) {
            if (g->append_at[t] != 0xffffu) return GUST_GRAPH; /* the counter is not a cache */
        } else if (g->append_at[t] == 0xffffu) {
            return GUST_GRAPH; /* a state that nothing writes */
        }
    }
    if (states && (g->counter < 0 || g->commits != 1)) return GUST_GRAPH;
    return GUST_OK;
}

static void gd_zero_state(const Context *c) {
    for (uint32_t t = 0; t < c->header->nt; t++) {
        if (!gd_is_state(c, t)) continue;
        float *s = gf_tensor(c, t);
        for (size_t i = 0, n = (size_t)c->t[t].h * c->t[t].c; i < n; i++) s[i] = 0.0f;
    }
}

/* ---- run time ------------------------------------------------------------------------------------------------ */

static inline float gd_scalar(const Context *c, uint32_t t) {
    return gf_tensor(c, t)[0];
}

/* v is a finite non-negative exact integer below `limit` (a float32 below 2^24, so the comparison is exact); NaN,
   infinities, negative values (also -0), fractions and anything out of range are refused. */
static int gd_index(float v, uint32_t limit, uint32_t *out) {
    if (gf_asuint(v) >> 31) return 0;
    if (!(v < (float)limit)) return 0;
    const uint32_t i = (uint32_t)v;
    if ((float)i != v) return 0;
    *out = i;
    return 1;
}

static int gd_preflight(const Context *c) {
    const uint32_t no = c->header->no;
    uint32_t cap = GUST_F32_STATE_ROWS, now = 0; /* smallest cache capacity, the position counter */
    if (c->ops[no - 1].code == GUST_F_COMMIT) {
        if (!gd_index(gd_scalar(c, c->ops[no - 1].out), GUST_F32_STATE_ROWS + 1u, &now)) return GUST_SHAPE;
        for (uint32_t i = 0; i < no; i++)
            if (c->ops[i].code == GUST_F_CACHE_APPEND && c->t[c->ops[i].out].h < cap) cap = c->t[c->ops[i].out].h;
    }
    for (uint32_t i = 0; i < no; i++) {
        const Op *o = c->ops + i;
        uint32_t v;
        switch (o->code) {
        case GUST_F_GATHER: { /* every valid token row */
            const float *tok = gf_tensor(c, o->a);
            for (uint32_t r = 0, n = c->packed_offsets[o->a]; r < n; r++)
                if (!gd_index(tok[r], o->p0, &v)) return GUST_SHAPE;
            break;
        }
        case GUST_F_ROPE_AT:
            if (!gd_index(gd_scalar(c, o->b), o->p0, &v)) return GUST_SHAPE;
            break;
        case GUST_F_CACHE_APPEND:
            if (!gd_index(gd_scalar(c, o->b), c->t[o->out].h, &v) || v != now) return GUST_SHAPE;
            break;
        case GUST_F_COMMIT:
            if (!gd_index(gd_scalar(c, o->a), cap, &v) || v != now) return GUST_SHAPE;
            break;
        case GUST_F_MHA_CACHE: /* 0 .. cache rows */
            if (!gd_index(gd_scalar(c, o->p0), c->t[o->bias].h + 1u, &v)) return GUST_SHAPE;
            break;
        case GUST_F_MHA_PREFIX:
            if (o->p1 == 0) {
                if (!gd_index(gd_scalar(c, o->p0), c->t[o->b].h + 1u, &v) || !v) return GUST_SHAPE;
            } else if (!gd_index(gd_scalar(c, o->p0), c->t[o->b].h, &v) || v != now) {
                return GUST_SHAPE;
            }
            break;
        default: break;
        }
    }
    return GUST_OK;
}

/* Stable float32 SiLU. x >= 0 (also +0 and -0): x / (1 + exp(-x)); x < 0: (x * exp(x)) / (1 + exp(x)). Both forms
   only exponentiate a non-positive number, so nothing overflows and a finite input gives a finite output. */
static inline float gd_silu(float x) {
    if (x < 0.0f) {
        const float e = gf_expf(x);
        return (x * e) / (1.0f + e);
    }
    const float e = gf_expf(-x);
    return x / (1.0f + e);
}

/* One decoder operator. `rows` is the executor's valid row count of the output. */
static void gd_run_op(Context *c, const Op *o, uint32_t rows) {
    switch (o->code) {
    case GUST_F_MUL: {
        const float *a = gf_tensor(c, o->a), *b = gf_tensor(c, o->b);
        float *z = gf_tensor(c, o->out);
        for (size_t i = 0, n = (size_t)rows * c->t[o->out].c; i < n; i++) z[i] = a[i] * b[i];
        break;
    }
    case GUST_F_SLICE: {
        const uint32_t cin = c->t[o->a].c, cout = c->t[o->out].c;
        const float *a = gf_tensor(c, o->a);
        float *z = gf_tensor(c, o->out);
        for (uint32_t t = 0; t < rows; t++)
            for (uint32_t j = 0; j < cout; j++) z[(size_t)t * cout + j] = a[(size_t)t * cin + o->p0 + j];
        break;
    }
    case GUST_F_GATHER: {
        GfWeights w;
        w.code = c->blob + o->weights;
        w.scale = c->blob + o->mult;
        w.bits = o->p1;
        w.grouped = 0;
        w.K = c->t[o->out].c;
        w.ngroups = 0;
#ifdef GUST_FLOAT32_T5
        if (o->kh) {
            w.grouped = 1;
            w.ngroups = (uint32_t)((w.K + 63u) / 64u);
        }
#endif
        const uint32_t n = c->t[o->out].c;
        const float *tok = gf_tensor(c, o->a);
        for (uint32_t r = 0; r < rows; r++) {
            const uint32_t token = (uint32_t)tok[r];
            float *z = gf_tensor(c, o->out) + (size_t)r * n;
            switch (w.bits) {
            case 2: gf_decode_row(&w, 2, token, 0, n, z, 1); break;
            case 3: gf_decode_row(&w, 3, token, 0, n, z, 1); break;
            case 4: gf_decode_row(&w, 4, token, 0, n, z, 1); break;
            case 5: gf_decode_row(&w, 5, token, 0, n, z, 1); break;
            case 6: gf_decode_row(&w, 6, token, 0, n, z, 1); break;
            case 7: gf_decode_row(&w, 7, token, 0, n, z, 1); break;
            default: gf_decode_row(&w, 8, token, 0, n, z, 1); break;
            }
        }
        break;
    }
    case GUST_F_SILU: {
        const float *a = gf_tensor(c, o->a);
        float *z = gf_tensor(c, o->out);
        for (size_t i = 0, n = (size_t)rows * c->t[o->out].c; i < n; i++) z[i] = gd_silu(a[i]);
        break;
    }
    case GUST_F_ROPE_AT: {
        /* The row of gf_op_rope for table row `position`: out[2i] = a*cos + (-b)*sin, out[2i+1] = b*cos + a*sin. */
        const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads, rot = o->kw, half = rot / 2;
        const uint32_t pos = (uint32_t)gd_scalar(c, o->b);
        const float *x = gf_tensor(c, o->a);
        float *z = gf_tensor(c, o->out);
        const GfF *ct = (const GfF *)(c->blob + o->weights) + (size_t)pos * half;
        const GfF *st = (const GfF *)(c->blob + o->bias) + (size_t)pos * half;
        for (uint32_t h = 0; h < heads; h++) {
            const float *xr = x + (size_t)h * D;
            float *zr = z + (size_t)h * D;
            for (uint32_t i = 0; i < half; i++) {
                const float a = xr[2 * i], b = xr[2 * i + 1], cv = ct[i], sv = st[i];
                const float p = a * cv, q = (-b) * sv, r = b * cv, s = a * sv;
                zr[2 * i] = p + q;
                zr[2 * i + 1] = r + s;
            }
            for (uint32_t d = rot; d < D; d++) zr[d] = xr[d];
        }
        break;
    }
    case GUST_F_CACHE_APPEND: {
        const uint32_t C = c->t[o->a].c, pos = (uint32_t)gd_scalar(c, o->b);
        const float *a = gf_tensor(c, o->a);
        float *row = gf_tensor(c, o->out) + (size_t)pos * C;
        for (uint32_t j = 0; j < C; j++) row[j] = a[j];
        break;
    }
    case GUST_F_COMMIT: gf_tensor(c, o->out)[0] = (float)((uint32_t)gd_scalar(c, o->a) + 1u); break;
    case GUST_F_MHA_CACHE: { /* per query row and head: scores over the own keys (banded by p1), then the first n
                                cached keys; one float32 softmax (gf_softmax_exp); values summed in that key order */
        const uint32_t C = c->t[o->a].c, heads = o->kh, D = C / heads, n = (uint32_t)gd_scalar(c, o->p0);
        if (!n) { /* empty cache: exactly F_MHA fast attention (same kernels, same bits as a plain encoder graph) */
            Op m = *o;
            m.p0 = 2u | (o->p1 ? 1u : 0u);
            gf_op_mha32(c, &m, rows, rows);
            break;
        }
        const float scale = *(const GfF *)(c->blob + o->mult);
        const float *Q = gf_tensor(c, o->a), *K = gf_tensor(c, o->b), *V = gf_tensor(c, o->weights);
        const float *KC = gf_tensor(c, o->bias), *VC = gf_tensor(c, o->shift);
        float *z = gf_tensor(c, o->out), *score = gf_scratch(c);
        float acc[GUST_F32_MAX_HEAD] __attribute__((aligned(32)));
        for (uint32_t i = 0; i < rows; i++) {
            uint32_t j0 = 0, j1 = rows;
            if (o->p1) {
                j0 = i > o->p1 ? i - o->p1 : 0u;
                j1 = (uint64_t)i + o->p1 + 1u < rows ? i + o->p1 + 1u : rows;
            }
            for (uint32_t h = 0; h < heads; h++) {
                const float *qr = Q + (size_t)i * C + (size_t)h * D;
                uint32_t idx = 0;
                float m = 0.0f;
                /* gf_dot32x4 equals gf_dot32 per key in every build; four keys per call */
                for (uint32_t pass = 0; pass < 2; pass++) {
                    const float *kb = pass ? KC : K;
                    const uint32_t a = pass ? 0u : j0, e = pass ? n : j1;
                    uint32_t j = a;
                    for (; j + 4 <= e; j += 4, idx += 4) {
                        float s4[4];
                        gf_dot32x4(qr, kb + (size_t)j * C + (size_t)h * D, C, D, s4);
                        for (uint32_t u = 0; u < 4; u++) {
                            const float s = s4[u] * scale;
                            score[idx + u] = s;
                            if (!(idx + u) || s > m) m = s;
                        }
                    }
                    for (; j < e; j++, idx++) {
                        const float s = gf_dot32(qr, kb + (size_t)j * C + (size_t)h * D, D) * scale;
                        score[idx] = s;
                        if (!idx || s > m) m = s;
                    }
                }
                const float sum = gf_softmax_exp(score, idx, m);
                for (uint32_t d = 0; d < D; d++) acc[d] = 0.0f;
                idx = 0;
                for (uint32_t pass = 0; pass < 2; pass++) {
                    const float *vb = pass ? VC : V;
                    const uint32_t a = pass ? 0u : j0, e = pass ? n : j1;
                    uint32_t j = a;
                    /* four keys per sweep of acc: per element the same additions in the same order as four
                       gf_axpy32 calls, so every build keeps its bits */
                    for (; j + 4 <= e; j += 4, idx += 4) {
                        const float e0 = score[idx], e1 = score[idx + 1], e2 = score[idx + 2], e3 = score[idx + 3];
                        const float *v0 = vb + (size_t)j * C + (size_t)h * D, *v1 = v0 + C, *v2 = v1 + C, *v3 = v2 + C;
                        for (uint32_t d = 0; d < D; d++) {
                            float x = acc[d];
                            x = x + e0 * v0[d];
                            x = x + e1 * v1[d];
                            x = x + e2 * v2[d];
                            x = x + e3 * v3[d];
                            acc[d] = x;
                        }
                    }
                    for (; j < e; j++, idx++) gf_axpy32(acc, score[idx], vb + (size_t)j * C + (size_t)h * D, D);
                }
                float *zr = z + (size_t)i * C + (size_t)h * D;
                for (uint32_t d = 0; d < D; d++) zr[d] = acc[d] / sum;
            }
        }
        break;
    }
    default: { /* GUST_F_MHA_PREFIX: F_MHA over the first `count` keys and values of one query row */
        const uint32_t v = (uint32_t)gd_scalar(c, o->p0);
#ifdef GUST_FLOAT32_T5
        if (o->kw) {
            gf_op_mha_bias(c, o, 1, o->p1 ? v + 1u : v, gf_tensor(c, o->bias), 1, c->t[o->bias].c);
            break;
        }
#endif
        gf_op_mha(c, o, 1, o->p1 ? v + 1u : v);
        break;
    }
    }
}

/* Clears every state tensor (key/value caches and the position counter); inputs, outputs and the context stay
   usable. NULL or a context that is not loaded: GUST_GRAPH; a loaded INT8 graph: GUST_OPERATOR; nothing else is
   written in either case. A float32 graph without state has nothing to clear and returns GUST_OK. */
int gust_reset(void *ctx) {
    if (!ctx) return GUST_GRAPH;
    Context *c = (Context *)ctx;
    if (!c->ready) return GUST_GRAPH;
    if (c->ready != GUST_F32_READY) return GUST_OPERATOR;
    gd_zero_state(c);
    return GUST_OK;
}
