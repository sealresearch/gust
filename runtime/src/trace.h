/* Per-run observers, shared by both graph executors. No retained state, clocks, formatting or allocation. */
static void trace_emit(gust_trace_callback observer, void *opaque, uint32_t kind, const Context *c,
                       uint32_t oi, uint32_t rows, int status) {
    if (!observer) return;
    gust_trace_event e = {kind, UINT32_MAX, 0, UINT32_MAX, 0, 0, 0, 0, status};
    if (kind == GUST_TRACE_OP_BEGIN || kind == GUST_TRACE_OP_END) {
        const Op *o = c->ops + oi;
        const Tensor *t = c->t + o->out;
        e.operator_index = oi;
        e.opcode = o->code;
        e.output_tensor = o->out;
        e.rows = rows;
        e.height = t->h;
        e.width = t->w;
        e.channels = t->c;
    }
    observer(opaque, &e);
}
