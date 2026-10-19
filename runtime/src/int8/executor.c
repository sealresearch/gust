/* Graph execution and tensor access. Operators run in artifact order over the
   caller's arena; gust_init has already validated every index and bound.
   Part of the runtime/gust.c translation unit. */

#ifdef GUST_PROFILE
/* Native diagnostic build only; shipping native/WASM modules omit hooks. */
extern void gust_profile_begin(uint32_t operation);
extern void gust_profile_end(uint32_t operation);
#define PROFILE_BEGIN(i) gust_profile_begin(i)
#define PROFILE_END(i)   gust_profile_end(i)
#else
#define PROFILE_BEGIN(i) ((void)0)
#define PROFILE_END(i)   ((void)0)
#endif

#ifdef GUST_ATTENTION
/* Real rows of tensor t when the first `valid` of tensor 0's H rows are real:
   ceil(valid * h / H), i.e. ceil(valid / stride) for a tensor at temporal stride H/h. */
static uint32_t real_rows(const Context *c, uint32_t t, uint32_t valid) {
    const uint32_t full = c->t[0].h, h = c->t[t].h;
    return valid >= full ? h : (uint32_t)(((uint64_t)valid * h + full - 1) / full);
}

/* Zero rows r0 .. r1 - 1 of tensor t, without memset (freestanding builds). */
static void zero_rows(const Context *c, uint32_t t, uint32_t r0, uint32_t r1) {
    const Tensor *z = c->t + t;
    int8_t *y = c->arena + z->offset;
    for (uint32_t i = r0 * z->w * z->c, e = r1 * z->w * z->c; i < e; i++) y[i] = 0;
}
/* Zero the rows of tensor t at or after `rows`. */
static void zero_tail(const Context *c, uint32_t t, uint32_t rows) { zero_rows(c, t, rows, c->t[t].h); }
/* Rows of input a that operator o reads for its first `rows` output rows: convolutions reach (kh - 1) * dy past the
   strided last row (less top padding), nearest resizes reach floor((rows - 1) * h_a / h_out). Every other
   operator reads only real input rows. */
static uint32_t read_rows(const Context *c, const Op *o, uint32_t rows) {
    const uint32_t ha = c->t[o->a].h;
    if (!rows) return 0;
    int64_t n = 0;
    if (o->code == GUST_CONV || o->code == GUST_DEPTHWISE)
        n = (int64_t)(rows - 1) * o->sy - o->py + (int64_t)(o->kh - 1) * o->dy + 1;
    else if (o->code == GUST_RESIZE)
        n = (int64_t)(rows - 1) * ha / c->t[o->out].h + 1;
    return n < 0 ? 0 : n > ha ? ha : (uint32_t)n;
}
#endif

/* valid = 0 runs every row; otherwise only the first `valid` rows of tensor 0
   are real (gust_run_valid) and the caller has checked valid < tensor 0 rows. outputs_only (gust_run_valid_outputs):
   only output tensors get zero tails; other tensors keep unspecified bytes past their real rows, except the rows a
   convolution or resize reads past its input's real rows, which are zeroed before it runs. */
static int run_checked(void *ctx, gust_stop_callback stop, void *opaque, uint32_t valid,
                       gust_trace_callback observer, void *trace_opaque, int outputs_only) {
    if (!ctx) return GUST_GRAPH;
    Context *c = (Context *)ctx;
    if (!c->ready) return GUST_GRAPH;
#ifdef GUST_FLOAT32
    if (c->ready == GUST_F32_READY) { /* src/float32/executor.c */
        if (outputs_only) return GUST_OPERATOR;
        return float_run(c, stop, opaque, valid, observer, trace_opaque);
    }
#endif
    if (valid > c->t[0].h) return GUST_SHAPE;
    if (valid == c->t[0].h) valid = 0;
    if (valid) {
#ifdef GUST_ATTENTION
        if (!c->valid_ok) return GUST_OPERATOR;
        if (!(c->t[0].flags & 1)) return GUST_SHAPE;
#else
        return GUST_OPERATOR;
#endif
    }
    if (!gather_preflight(c)) return GUST_SHAPE;
#ifdef GUST_ATTENTION
    if (valid)
        for (uint32_t i = 0; i < c->header->nt; i++)
            if (c->t[i].flags & 1) zero_tail(c, i, real_rows(c, i, valid));
#endif
    for (uint32_t oi = 0; oi < c->header->no; oi++) {
        if (stop && stop(opaque)) return GUST_BUDGET;
        PROFILE_BEGIN(oi);
        const Op *o = c->ops + oi;
        uint32_t rows = c->t[o->out].h, input_rows = c->t[o->a].h;
#ifdef GUST_ATTENTION
        if (valid) {
            rows = real_rows(c, o->out, valid);
            input_rows = real_rows(c, o->a, valid);
            if (!outputs_only || (c->t[o->out].flags & 2)) zero_tail(c, o->out, rows);
            if (outputs_only) {
                const uint32_t need = read_rows(c, o, rows);
                if (need > input_rows) zero_rows(c, o->a, input_rows, need);
            }
        }
#endif
        trace_emit(observer, trace_opaque, GUST_TRACE_OP_BEGIN, c, oi, rows, GUST_OK);
        switch (o->code) {
        case GUST_CONV:
        case GUST_DEPTHWISE:
#ifdef GUST_PACKED_WEIGHTS
            if (o->p1) { /* parameter1 3 or 4 (gust_init accepts nothing else here): packed weights, ops/packed.c */
                op_conv_packed(c, o, rows);
                break;
            }
#endif
            op_conv(c, o, rows);
            break;
        case GUST_GATHER: op_gather(c, o); break;
        case GUST_POOL: op_pool(c, o, input_rows); break;
        case GUST_TOPK_POOL: op_topk_pool(c, o); break;
        case GUST_WEIGHTED_POOL:
            if (!op_weighted_pool(c, o)) {
                trace_emit(observer, trace_opaque, GUST_TRACE_OP_END, c, oi, rows, GUST_QUANTIZATION);
                return GUST_QUANTIZATION;
            }
            break;
        case GUST_REQUANTIZE: op_requantize(c, o, rows); break;
        case GUST_ADD: op_add(c, o, rows); break;
        case GUST_RESIZE: op_resize(c, o, rows); break;
        case GUST_CONCAT: op_concat(c, o, rows); break;
        case GUST_SLICE: op_slice(c, o, rows); break;
#ifdef GUST_ATTENTION
        case GUST_MATMUL: op_matmul(c, o, rows, valid ? real_rows(c, o->b, valid) : c->t[o->b].h); break;
        case GUST_ATTEND: op_attend(c, o, rows, valid ? real_rows(c, o->b, valid) : c->t[o->b].h); break;
#endif
#ifdef GUST_CONFORMER
        case GUST_NORM: op_norm(c, o, rows); break;
        case GUST_LUT: op_lut(c, o, rows); break;
        case GUST_MUL: op_mul(c, o, rows); break;
#endif
        default: break;
        }
        PROFILE_END(oi);
        trace_emit(observer, trace_opaque, GUST_TRACE_OP_END, c, oi, rows, GUST_OK);
    }
    return GUST_OK;
}

int gust_run(void *ctx) {
    return run_checked(ctx, NULL, NULL, 0, NULL, NULL, 0);
}

int gust_run_valid(void *ctx, uint32_t valid_frames) {
    return run_checked(ctx, NULL, NULL, valid_frames, NULL, NULL, 0);
}

int gust_run_valid_outputs(void *ctx, uint32_t valid_frames) {
    return run_checked(ctx, NULL, NULL, valid_frames, NULL, NULL, 1);
}

static int run_inputs(void *ctx, const uint32_t *input_rows, uint32_t input_count,
                      gust_trace_callback observer, void *trace_opaque) {
    if (!ctx) return GUST_GRAPH;
    Context *c = (Context *)ctx;
    if (!c->ready) return GUST_GRAPH;
#ifdef GUST_FLOAT32
    if (c->ready == GUST_F32_READY) {
        if (!input_rows || input_count != c->header->ni) return GUST_SHAPE;
        const uintptr_t base = (uintptr_t)c, rows = (uintptr_t)input_rows;
        if ((rows >= base && rows - base < sizeof *c) ||
            (rows < base && base - rows < (uintptr_t)input_count * sizeof *input_rows)) return GUST_MEMORY;
        return float_run_inputs(c, NULL, NULL, 0, input_rows, observer, trace_opaque);
    }
#else
    (void)input_rows;
    (void)input_count;
    (void)observer;
    (void)trace_opaque;
#endif
    return GUST_OPERATOR;
}

int gust_run_valid_inputs(void *ctx, const uint32_t *input_rows, uint32_t input_count) {
    return run_inputs(ctx, input_rows, input_count, NULL, NULL);
}

int gust_run_traced(void *ctx, uint32_t valid_rows, gust_trace_callback observer, void *opaque) {
    trace_emit(observer, opaque, GUST_TRACE_RUN_BEGIN, NULL, 0, 0, GUST_OK);
    const int status = run_checked(ctx, NULL, NULL, valid_rows, observer, opaque, 0);
    trace_emit(observer, opaque, GUST_TRACE_RUN_END, NULL, 0, 0, status);
    return status;
}

int gust_run_valid_inputs_traced(void *ctx, const uint32_t *input_rows, uint32_t input_count,
                                 gust_trace_callback observer, void *opaque) {
    trace_emit(observer, opaque, GUST_TRACE_RUN_BEGIN, NULL, 0, 0, GUST_OK);
    const int status = run_inputs(ctx, input_rows, input_count, observer, opaque);
    trace_emit(observer, opaque, GUST_TRACE_RUN_END, NULL, 0, 0, status);
    return status;
}

int gust_run_checked(void *ctx, gust_stop_callback stop, void *opaque) {
    return run_checked(ctx, stop, opaque, 0, NULL, NULL, 0);
}

void *gust_tensor(void *ctx, uint32_t index) {
    Context *c = (Context *)ctx;
    return c && c->ready && index < c->header->nt ? c->arena + c->t[index].offset : NULL;
}

uint32_t gust_tensor_bytes(void *ctx, uint32_t index) {
    Context *c = (Context *)ctx;
    return c && c->ready && index < c->header->nt ? c->t[index].bytes : 0;
}
