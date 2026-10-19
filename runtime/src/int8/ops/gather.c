/* GUST_GATHER: embedding-table lookup from raw little-endian uint16 indices
   (HxWx2), optionally weighted by a signed INT8 coefficient (HxWx3). Index
   65535 (2-channel form) or coefficient 0 (3-channel form) yields zeros.
   Part of the runtime/gust.c translation unit. */

/* Gather indices are inputs, so preflight every lookup before any output
   write. A bad input is recoverable and cannot expose a partial new result. */
static int gather_preflight(const Context *c) {
    for (uint32_t oi = 0; oi < c->header->no; oi++) {
        const Op *o = c->ops + oi;
        if (o->code == GUST_GATHER) {
            const Tensor *a = c->t + o->a;
            const uint8_t *x = (const uint8_t *)(c->arena + a->offset);
            for (uint32_t p = 0; p < a->h * a->w; p++) {
                uint32_t index = (uint32_t)x[a->c * p] | (uint32_t)x[a->c * p + 1] << 8;
                int active = a->c == 3 ? (int8_t)x[3 * p + 2] != 0 : index != 65535;
                if (active && index >= o->p0) return 0;
            }
        }
    }
    return 1;
}

static void op_gather(const Context *c, const Op *o) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const int32_t *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    const uint8_t *indices = (const uint8_t *)x;
    const int8_t *weights = (const int8_t *)(c->blob + o->weights);
    for (uint32_t p = 0; p < a->h * a->w; p++) {
        uint32_t index = (uint32_t)indices[a->c * p] | (uint32_t)indices[a->c * p + 1] << 8;
        int coefficient = a->c == 3 ? x[3 * p + 2] : index == 65535 ? 0 : 1;
        for (uint32_t ch = 0; ch < z->c; ch++)
            y[p * z->c + ch] = coefficient == 0 ? 0 : quant(weights[ch * o->p0 + index] * coefficient, m[ch], s[ch], o);
    }
}
