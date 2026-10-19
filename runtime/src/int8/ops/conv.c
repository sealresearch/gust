/* GUST_CONV and GUST_DEPTHWISE: a sparse full-field dense classifier, otherwise the convolution driver (ops/gemm.c
   conv_rows) over byte weights or decoded packed codes. Every path produces identical results. Only the first
   `rows` output rows are computed (z->h unless gust_run_valid shortens them); the sparse classifier always has a
   single output row. Part of the runtime/gust.c translation unit. */

static void op_conv(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    const int8_t *w = (const int8_t *)(c->blob + o->weights);
    if (o->code == GUST_CONV && z->h == 1 && z->w == 1 && o->kh == a->h && o->kw == a->w && o->dy == 1 && o->dx == 1 &&
        o->py == 0 && o->px == 0 && o->sy == 1 && o->sx == 1 && a->bytes >= 8192 && a->bytes <= 65536) {
        uint16_t indices[512];
        uint32_t used = 0;
        if (sparse_indices(x, a->bytes, indices, &used)) {
            for (uint32_t oc = 0; oc < z->c; oc++) {
                int32_t acc = bias[oc];
                const int8_t *weights = w + oc * a->bytes;
                for (uint32_t k = 0; k < used; k++) acc += (int32_t)x[indices[k]] * weights[indices[k]];
                y[oc] = quant(acc, m[oc], s[oc], o);
            }
            return;
        }
    }
    const uint32_t oi = (uint32_t)(o - c->ops);
    conv_rows(c, o, rows, w, bias, m, s, z->c, y, z->c, (int)((c->small[oi >> 5] >> (oi & 31)) & 1u));
}

#ifdef GUST_PACKED_WEIGHTS
/* A packed convolution as byte convolutions over blocks of `block` output channels (a whole depthwise filter at
   once), the codes decoded into one stack buffer. Out of line, so the 1x1 GEMM path does not carry the buffer. */
static __attribute__((noinline)) void packed_blocks(const Context *c, const Op *o, uint32_t rows, uint32_t block) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    int8_t *y = c->arena + z->offset;
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    const uint8_t *w = c->blob + o->weights;
    const uint32_t row = o->kh * o->kw * (o->code == GUST_CONV ? a->c : 1);
    int8_t codes[PACKED_BLOCK] __attribute__((aligned(64))); /* >= PACKED_SCRATCH, a whole depthwise filter */
    if (o->code == GUST_DEPTHWISE) {
        packed_expand(w, 0, row * z->c, o->p1, codes);
        conv_rows(c, o, rows, codes, bias, m, s, z->c, y, z->c, 1);
        return;
    }
    /* a multiple of 32 channels (one VNNI tile block) when possible, at least 16 (row <= PACKED_DENSE_TERMS) */
    block = block >= 32 ? block & ~31u : block & ~7u;
    for (uint32_t oc = 0; oc < z->c; oc += block) {
        const uint32_t nb = z->c - oc < block ? z->c - oc : block;
        packed_expand(w, (uint64_t)oc * row, nb * row, o->p1, codes);
        conv_rows(c, o, rows, codes, bias + oc, m + oc, s + oc, nb, y + (size_t)oc, z->c, 1);
    }
}

/* GUST_CONV and GUST_DEPTHWISE with parameter1 = 3 or 4 (ops/packed.c). Every output equals
   quant(bias + sum(input * weight)) exactly as the byte path computes it; gust_init has validated the payload range
   and the packed limits. */
static void op_conv_packed(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const uint32_t row = o->kh * o->kw * (o->code == GUST_CONV ? a->c : 1);
#ifdef GEMM
    if (o->code == GUST_CONV && o->kh == 1 && o->kw == 1 && o->py == 0 && o->px == 0 && rows * z->w >= GEMM_MIN &&
        gemm_fits(row)) {
        /* 1x1: every pixel is one contiguous dot, so the GEMM tiles decode each channel block while packing it and
           run once over all channels. */
        const int8_t *x = c->arena + a->offset, *xs[GEMM_CHUNK];
        int8_t *y = c->arena + z->offset, *ys[GEMM_CHUNK];
        const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
        const uint32_t *s = (const uint32_t *)(c->blob + o->shift), npv = rows * z->w;
        for (uint32_t pb = 0; pb < npv; pb += GEMM_CHUNK) {
            const uint32_t cnt = npv - pb < GEMM_CHUNK ? npv - pb : GEMM_CHUNK;
            for (uint32_t p = 0; p < cnt; p++) {
                const uint32_t q = pb + p, hy = q / z->w, wx = q % z->w;
                xs[p] = x + ((size_t)hy * o->sy * a->w + (size_t)wx * o->sx) * a->c;
                ys[p] = y + (size_t)q * z->c;
            }
            gemm_par(xs, ys, cnt, NULL, row, row, z->c, bias, m, s, 0, 1, c->blob + o->weights, o);
        }
        return;
    }
#endif
#ifdef GEMM128
    if (o->code == GUST_CONV && o->kh == 1 && o->kw == 1 && o->py == 0 && o->px == 0) {
        const int8_t *x = c->arena + a->offset, *xs[NG_CHUNK];
        int8_t *y = c->arena + z->offset, *ys[NG_CHUNK];
        const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
        const uint32_t *s = (const uint32_t *)(c->blob + o->shift), npv = rows * z->w;
        for (uint32_t pb = 0; pb < npv; pb += NG_CHUNK) {
            const uint32_t cnt = npv - pb < NG_CHUNK ? npv - pb : NG_CHUNK;
            for (uint32_t p = 0; p < cnt; p++) {
                const uint32_t q = pb + p, hy = q / z->w, wx = q % z->w;
                xs[p] = x + ((size_t)hy * o->sy * a->w + (size_t)wx * o->sx) * a->c;
                ys[p] = y + (size_t)q * z->c;
            }
            g128_gemm_packed(xs, ys, cnt, c->blob + o->weights, o->p1, row, z->c, bias, m, s, o);
        }
        return;
    }
#endif
    packed_blocks(c, o, rows, o->code == GUST_CONV ? PACKED_BLOCK / row : z->c);
}
#endif /* GUST_PACKED_WEIGHTS */
