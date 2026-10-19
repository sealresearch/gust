/* float32-graph-v1 execution: valid-row propagation, preflight, operator dispatch and zeroed tails.

   Valid rows. `valid` counts the real rows of tensor 0 (samples for a waveform input); 0 or tensor 0's height means
   all rows. Before anything is written, every operator's valid output rows are derived from its inputs':
     F_CONV    V < kh -> 0 (refused), else floor((V - kh) / sy) + 1
     F_ADD     both operands must have the same V
     F_MHA     queries V(a); keys and values must have the same V
     F_MEAN    1 (float32-temporal-v1)
     others    V of the input
   Other inputs have all rows unless the caller supplies a per-input row array. A zero anywhere is GUST_SHAPE and the call
   leaves the arena untouched. The valid rows of every output equal those of a graph compiled for exactly that
   many input rows, bit for bit, whatever the padding holds; the rows past them are written as zero. Inputs and
   their padding are never modified. The row map lives in Context.packed_offsets (4096 entries). */
#pragma clang fp contract(off)

static void gf_zero_tail(const Context *c, uint32_t t, uint32_t rows) {
    const Tensor *z = c->t + t;
    float *y = gf_tensor(c, t);
    const size_t start = (size_t)rows * z->c, n = (size_t)z->h * z->c;
    if (gt_threads() > 1u && n - start >= 65536u) {
        GtArgs args;
        args.p[0] = y + start;
        args.u[0] = (uint32_t)(n - start); /* arena is at most 64 MiB */
        gt_parallel(GT_ZERO, &args, (args.u[0] + 16383u) / 16384u);
        return;
    }
    for (size_t i = start; i < n; i++) y[i] = 0.0f;
}

/* A later ordinary operator writes its destination's valid rows AND tail. If that whole interval contains this
   tail, clearing it now is redundant: no operator reads padding, and the loader forbids aliasing live tensors.
   This preserves every final arena byte, not just the public outputs. Checked runs and dynamic cache graphs
   retain eager clearing because callbacks / counted attention may observe intermediate padding. */
static int gf_tail_overwritten(const Context *c, uint32_t oi, uint32_t rows) {
    const Tensor *z = c->t + c->ops[oi].out;
    if (rows == z->h) return 1;
    const uint32_t begin = z->offset + rows * z->c * 4u, end = z->offset + z->bytes;
    for (uint32_t j = oi + 1u; j < c->header->no; j++) {
        const Tensor *next = c->t + c->ops[j].out;
        if (next->offset <= begin && next->offset + next->bytes >= end) return 1;
    }
    return 0;
}

/* Fills V for every tensor; returns GUST_OK or the error for the first operator whose rows are inconsistent. */
static int gf_valid_rows(Context *c, uint32_t valid, const uint32_t *input_rows) {
    uint32_t *V = c->packed_offsets;
    uint32_t input = 0;
    for (uint32_t i = 0; i < c->header->nt; i++) {
        uint32_t rows = 0;
        if (c->t[i].flags & 1u) {
            rows = input_rows ? input_rows[input++] : c->t[i].h;
            if (!rows || rows > c->t[i].h) return GUST_SHAPE;
        }
        V[i] = rows;
    }
    if (!input_rows) V[0] = valid;
    for (uint32_t i = 0; i < c->header->no; i++) {
        const Op *o = c->ops + i;
        const uint32_t va = V[o->a];
        uint32_t out = va;
        switch (o->code) {
        case GUST_F_CONV: out = va < o->kh ? 0u : (va - o->kh) / o->sy + 1u; break;
        case GUST_F_ADD:
            if (V[o->b] != va) return GUST_SHAPE;
            break;
        case GUST_F_MHA:
            if (V[o->b] != V[o->weights]) return GUST_SHAPE;
            if (!V[o->b]) return GUST_SHAPE;
            if ((o->p0 & 1u) && V[o->b] != va) return GUST_SHAPE; /* banded: every query row has its own key */
            break;
#ifdef GUST_FLOAT32_DECODER
        /* Decoder operators (decoder.c). The explicit scalars (token, position, counts) and the cross-attention
           keys never decide a row count: F_MHA_PREFIX takes its query's rows (1) whatever the cache holds, a cache
           append or commit reports its whole state tensor, which is never zeroed. */
        case GUST_F_MUL: /* b may have more valid rows (e.g. a host-written gate input): the first va rows are used */
            if (V[o->b] < va) return GUST_SHAPE;
            break;
        case GUST_F_MHA_CACHE: /* own keys and values have the query rows; the cache rows come from the count */
            if (V[o->b] != va || V[o->weights] != va) return GUST_SHAPE;
            break;
        case GUST_F_CACHE_APPEND:
        case GUST_F_COMMIT: out = c->t[o->out].h; break;
        case GUST_F_MEAN: out = 1u; break; /* float32-temporal-v1: one row, the mean of the valid rows */
#endif
        default: break;
        }
        if (!out || out > c->t[o->out].h) return GUST_SHAPE;
        V[o->out] = out;
    }
    return GUST_OK;
}

static int float_run_inputs(Context *c, gust_stop_callback stop, void *opaque, uint32_t valid,
                            const uint32_t *input_rows, gust_trace_callback observer, void *trace_opaque) {
#ifdef GUST_FLOAT32_DECODER
    if (input_rows)
        for (uint32_t oi = 0; oi < c->header->no; oi++)
            if (c->ops[oi].code > GUST_F_SILU && c->ops[oi].code < GUST_F_DWCONV) return GUST_OPERATOR;
#endif
    if (valid > c->t[0].h) return GUST_SHAPE;
    if (!valid) valid = c->t[0].h;
    const int status = gf_valid_rows(c, valid, input_rows);
    if (status) return status;
#ifdef GUST_FLOAT32_DECODER
    {
        const int checked = gd_preflight(c); /* every dynamic value of the graph, before any arena write */
        if (checked) return checked;
    }
#endif
    const uint32_t *V = c->packed_offsets;
    int trim_tails = !stop;
#ifdef GUST_FLOAT32_DECODER
    for (uint32_t oi = 0; oi < c->header->no; oi++)
        if (c->ops[oi].code > GUST_F_SILU && c->ops[oi].code < GUST_F_DWCONV) trim_tails = 0;
#endif
    for (uint32_t oi = 0; oi < c->header->no; oi++) {
        if (stop && stop(opaque)) return GUST_BUDGET;
        const Op *o = c->ops + oi;
        const uint32_t rows = V[o->out];
        trace_emit(observer, trace_opaque, GUST_TRACE_OP_BEGIN, c, oi, rows, GUST_OK);
        GP_START(gp0);
        switch (o->code) {
        case GUST_F_CONV: gf_op_conv(c, o, rows); break;
        case GUST_F_ADD: gf_op_add(c, o, rows); break;
        case GUST_F_NORM:
            if (o->p0 & 2u)
                gf_op_norm32(c, o, rows);
            else
                gf_op_norm(c, o, rows);
            break;
        case GUST_F_UNARY: gf_op_unary(c, o, rows); break;
        case GUST_F_ROPE: gf_op_rope(c, o, rows); break;
        case GUST_F_MHA:
            if (o->p0 & 2u)
                gf_op_mha32(c, o, rows, V[o->b]);
            else
                gf_op_mha(c, o, rows, V[o->b]);
            break;
#ifdef GUST_FLOAT32_DECODER
        case GUST_F_MUL:
        case GUST_F_SLICE:
        case GUST_F_GATHER:
        case GUST_F_SILU:
        case GUST_F_ROPE_AT:
        case GUST_F_CACHE_APPEND:
        case GUST_F_COMMIT:
        case GUST_F_MHA_PREFIX:
        case GUST_F_MHA_CACHE: gd_run_op(c, o, rows); break;
        case GUST_F_DWCONV:
        case GUST_F_FILM:
        case GUST_F_MEAN: tp_run_op(c, o, rows); break;
#endif
        default: break;
        }
        GP_ADD(o->code, gp0);
#ifdef GUST_FLOAT32_DECODER
        if (c->t[o->out].flags & GUST_F32_STATE) { /* a cache keeps its rows past the position */
            trace_emit(observer, trace_opaque, GUST_TRACE_OP_END, c, oi, rows, GUST_OK);
            continue;
        }
#endif
        GP_START(gpz);
        if (!trim_tails || !gf_tail_overwritten(c, oi, rows)) gf_zero_tail(c, o->out, rows);
        GP_ADD(4, gpz);
        trace_emit(observer, trace_opaque, GUST_TRACE_OP_END, c, oi, rows, GUST_OK);
    }
    return GUST_OK;
}

static int float_run(Context *c, gust_stop_callback stop, void *opaque, uint32_t valid,
                     gust_trace_callback observer, void *trace_opaque) {
    return float_run_inputs(c, stop, opaque, valid, NULL, observer, trace_opaque);
}
