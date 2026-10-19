/* Parameter-free and per-channel tensor operators: global average pool,
   requantize, residual add, nearest-neighbour resize, channel concat and
   channel slice. Part of the runtime/gust.c translation unit. */

/* Global mean over the first `rows` input rows. The stored multiplier holds
   1/T for the compiled T = H*W positions, so a shorter sum of n positions is
   first rescaled to round_half_away(sum * T / n) in int64 (|result| <= 128*T,
   inside the gust_init bound); with n = T the sum is untouched. */
static void op_pool(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    const uint64_t n = (uint64_t)rows * a->w;
#ifdef GUST_ATTENTION
    const uint64_t total = (uint64_t)a->h * a->w;
#endif
    for (uint32_t ch = 0; ch < z->c; ch++) {
        int64_t sum = 0;
        for (uint64_t p = 0; p < n; p++) sum += x[p * a->c + ch];
#ifdef GUST_ATTENTION
        if (n != total) {
            const uint64_t u = (uint64_t)(sum < 0 ? -sum : sum) * total, r = (2 * u + n) / (2 * n);
            sum = sum < 0 ? -(int64_t)r : (int64_t)r;
        }
#endif
        y[ch] = quant((int32_t)(bias[ch] + sum), m[ch], s[ch], o);
    }
}

/* GUST_TOPK_POOL: per channel, the sum of the k largest values (exact
   counting selection; ties have equal values), requantized once. Scratch is
   bounded to 1 KiB. */
static void op_topk_pool(const Context *c, const Op *o) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    uint32_t histogram[256];
    for (uint32_t ch = 0; ch < z->c; ch++) {
        for (uint32_t v = 0; v < 256; v++) histogram[v] = 0;
        for (uint32_t p = 0; p < a->h * a->w; p++) histogram[(int32_t)x[p * a->c + ch] + 128]++;
        int32_t acc = bias[ch];
        uint32_t remaining = o->p0;
        for (int v = 255; v >= 0 && remaining; v--) {
            uint32_t take = histogram[v] < remaining ? histogram[v] : remaining;
            acc += (v - 128) * (int32_t)take;
            remaining -= take;
        }
        y[ch] = quant(acc, m[ch], s[ch], o);
    }
}

/* GUST_WEIGHTED_POOL: per channel, (bias + sum(x * w)) * multiplier divided
   by sum(w) << shift, rounded once to nearest (ties away from zero). Weights
   are input B (HxWx1, 0..127); a negative weight or an all-zero mask fails. */
static int op_weighted_pool(const Context *c, const Op *o) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    const int8_t *weights = c->arena + c->t[o->b].offset;
    uint32_t count = 0;
    for (uint32_t p = 0; p < a->h * a->w; p++) {
        if (weights[p] < 0) return 0;
        count += (uint32_t)weights[p];
    }
    if (!count) return 0;
    for (uint32_t ch = 0; ch < z->c; ch++) {
        int32_t acc = bias[ch];
        for (uint32_t p = 0; p < a->h * a->w; p++) acc += (int32_t)x[p * a->c + ch] * weights[p];
        int64_t product = (int64_t)acc * m[ch];
        uint64_t denominator = (uint64_t)count << s[ch];
        uint64_t absolute = (uint64_t)(product < 0 ? -product : product),
                 rounded = (absolute + denominator / 2) / denominator;
        y[ch] = clamp(product < 0 ? -(int64_t)rounded : (int64_t)rounded, o);
    }
    return 1;
}

static void op_requantize(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *z = c->t + o->out;
    const int8_t *x = c->arena + c->t[o->a].offset;
    int8_t *y = c->arena + z->offset;
    const uint32_t npv = rows * z->w;
    uint32_t table_offset = c->packed_offsets[o - c->ops];
    if (table_offset != UINT32_MAX) {
        const int8_t *table = (const int8_t *)(c->packed_weights + table_offset);
        for (uint32_t j = 0; j < npv * z->c; j++) y[j] = table[(int32_t)x[j] + 128];
        return;
    }
    const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
#ifdef GUST_INT8_VNNI
    for (uint32_t p = 0; p < npv; p++)
        for (uint32_t ch = 0; ch < z->c; ch += 16) {
            const __mmask16 t = z->c - ch >= 16 ? 0xFFFF : (__mmask16)((1u << (z->c - ch)) - 1);
            __m512i v = _mm512_add_epi32(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(t, x + p * z->c + ch)),
                                         _mm512_maskz_loadu_epi32(t, bias + ch));
            _mm_mask_storeu_epi8(
                y + p * z->c + ch, t,
                vnni_quant(v, _mm512_maskz_loadu_epi32(t, m + ch), _mm512_maskz_loadu_epi32(t, s + ch), o->lo, o->hi));
        }
#else
    uint32_t ch = 0;
#if defined(__wasm_simd128__)
    const v128_t lo = wasm_f64x2_splat(o->lo), hi = wasm_f64x2_splat(o->hi);
    for (; ch + 16 <= z->c; ch += 16) { /* |x + bias| fits int32 (gust_init bound) */
        v128_t b[4], f[8], r[4];
        for (uint32_t j = 0; j < 4; j++) b[j] = wasm_v128_load(bias + ch + 4 * j);
        for (uint32_t j = 0; j < 8; j++)
            f[j] = wasm_f64x2_make(qscale(m[ch + 2 * j], s[ch + 2 * j]), qscale(m[ch + 2 * j + 1], s[ch + 2 * j + 1]));
        for (uint32_t p = 0; p < npv; p++) {
            v128_t v = wasm_v128_load(x + p * z->c + ch), l = wasm_i16x8_extend_low_i8x16(v),
                   h = wasm_i16x8_extend_high_i8x16(v);
            r[0] = wasm_i32x4_extend_low_i16x8(l);
            r[1] = wasm_i32x4_extend_high_i16x8(l);
            r[2] = wasm_i32x4_extend_low_i16x8(h);
            r[3] = wasm_i32x4_extend_high_i16x8(h);
            wasm_v128_store(y + p * z->c + ch, wasm_quant16(r, b, f, lo, hi));
        }
    }
#elif defined(__AVX2__)
    const __m256d lo = _mm256_set1_pd(o->lo), hi = _mm256_set1_pd(o->hi);
    for (; ch + 16 <= z->c; ch += 16) { /* |x + bias| fits int32 (gust_init bound) */
        __m128i b[4], r[4];
        __m256d f[4];
        for (uint32_t j = 0; j < 4; j++) {
            b[j] = _mm_loadu_si128((const __m128i *)(bias + ch + 4 * j));
            f[j] = avx_qf(m + ch + 4 * j, s + ch + 4 * j);
        }
        for (uint32_t p = 0; p < npv; p++) {
            __m128i v = _mm_loadu_si128((const __m128i *)(x + p * z->c + ch));
            r[0] = _mm_cvtepi8_epi32(v);
            r[1] = _mm_cvtepi8_epi32(_mm_srli_si128(v, 4));
            r[2] = _mm_cvtepi8_epi32(_mm_srli_si128(v, 8));
            r[3] = _mm_cvtepi8_epi32(_mm_srli_si128(v, 12));
            _mm_storeu_si128((__m128i *)(y + p * z->c + ch), avx_quant16(r, b, f, lo, hi));
        }
    }
#endif
    for (uint32_t p = 0; p < npv; p++)
        for (uint32_t j = ch; j < z->c; j++) y[p * z->c + j] = quant((int32_t)x[p * z->c + j] + bias[j], m[j], s[j], o);
#endif
}

static void op_add(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *z = c->t + o->out;
    clampcopy(c->arena + z->offset, c->arena + c->t[o->a].offset, c->arena + c->t[o->b].offset, rows * z->w * z->c, o);
}

static void op_resize(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    for (uint32_t hy = 0; hy < rows; hy++)
        for (uint32_t wx = 0; wx < z->w; wx++)
            clampcopy(y + (hy * z->w + wx) * z->c,
                      x + (((uint64_t)hy * a->h / z->h) * a->w + (uint64_t)wx * a->w / z->w) * a->c, NULL, z->c, o);
}

static void op_concat(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out, *bt = c->t + o->b;
    const int8_t *x = c->arena + a->offset, *b = c->arena + bt->offset;
    int8_t *y = c->arena + z->offset;
    const uint32_t npv = rows * a->w;
#ifdef GUST_INT8_VNNI
    if (o->lo == -128 && o->hi == 127) {
        copy_rows(y, z->c, x, a->c, a->c, npv);
        copy_rows(y + a->c, z->c, b, bt->c, bt->c, npv);
        return;
    }
#endif
    for (uint32_t p = 0; p < npv; p++) {
        clampcopy(y + p * z->c, x + p * a->c, NULL, a->c, o);
        clampcopy(y + p * z->c + a->c, b + p * bt->c, NULL, bt->c, o);
    }
}

static void op_slice(const Context *c, const Op *o, uint32_t rows) {
    const Tensor *a = c->t + o->a, *z = c->t + o->out;
    const int8_t *x = c->arena + a->offset;
    int8_t *y = c->arena + z->offset;
    const uint32_t npv = rows * a->w;
#ifdef GUST_INT8_VNNI
    if (o->lo == -128 && o->hi == 127) {
        copy_rows(y, z->c, x + o->p0, a->c, z->c, npv);
        return;
    }
#endif
    for (uint32_t p = 0; p < npv; p++) clampcopy(y + p * z->c, x + p * a->c + o->p0, NULL, z->c, o);
}
