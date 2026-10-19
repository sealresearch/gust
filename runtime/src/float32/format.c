/* float32-graph-v1 loader (Header.flags == 1). A sibling of gust_init's legacy path: gust_init calls it after the
   magic check and it validates everything itself, once, before it returns success. What a refusal promises:
   the arena and the blob are never written, and the Context is not runnable (gust_init clears `ready` first and
   it is set only at the very end). It does not promise an untouched Context: after the geometry checks the loader
   stages the header, tensor, operator, blob and arena pointers in the Context while it validates the rest, so
   those fields may hold values from a refused artifact; nothing reads them while `ready` is 0. The legacy loader
   and every legacy operator are untouched.

   Error codes: GUST_FORMAT header/table/field/offset/trailing-bit violations, GUST_CHECKSUM CRC, GUST_SHAPE
   dimensions and operator geometry, GUST_MEMORY arena/context placement and alignment, GUST_OPERATOR an opcode
   outside 32..37, GUST_QUANTIZATION a non-finite or non-positive scale, bias or constant, GUST_GRAPH producers,
   operands, input/output roles and aliasing. */
#pragma clang fp contract(off)

/* CRC32 (0xEDB88320), nibble table, bytes 16 to 19 treated as zero: same value as the legacy bitwise crc(). */
static const uint32_t gf_crc_nibble[16] = {0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu, 0x76DC4190u, 0x6B6B51F4u,
                                           0x4DB26158u, 0x5005713Cu, 0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
                                           0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu};
static uint32_t gf_crc(const uint8_t *p, uint32_t n) {
    uint32_t c = 0xffffffffu;
    for (uint32_t i = 0; i < n; i++) {
        c ^= (i >= 16u && i < 20u) ? 0u : p[i];
        c = (c >> 4) ^ gf_crc_nibble[c & 15u];
        c = (c >> 4) ^ gf_crc_nibble[c & 15u];
    }
    return ~c;
}

/* A blob range [off, off + bytes) inside the artifact's parameter payload with the given alignment. */
static int gf_blob(const Context *c, uint32_t off, uint64_t bytes, uint32_t align) {
    return off >= c->header->data && off % align == 0 && range(off, bytes, c->header->total);
}
/* Every float32 in the range is finite (and positive when `positive`). */
static int gf_floats_ok(const Context *c, uint32_t off, uint64_t count, int positive) {
    const GfF *p = (const GfF *)(c->blob + off);
    for (uint64_t i = 0; i < count; i++)
        if (positive ? !gf_positive(p[i]) : !gf_finite(p[i])) return 0;
    return 1;
}

/* Operand fields that an operator must leave zero. Bit i is Op field i in order: opcode, input_a, input_b,
   output, weights, bias, multiplier, shift, kh, kw, stride_y, stride_x, pad_y, pad_x, dilation_y, dilation_x,
   parameter0, parameter1, clamp_min, clamp_max. */
#define GF_F(i) (1u << (i))
static uint32_t gf_used(uint32_t code) {
    switch (code) {
    case GUST_F_CONV:
        return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(6) | GF_F(8) | GF_F(9) | GF_F(10) | GF_F(11) |
               GF_F(12) | GF_F(13) | GF_F(14) | GF_F(15) | GF_F(16) | GF_F(17);
    case GUST_F_ADD: return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3);
    case GUST_F_NORM: return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(6) | GF_F(16);
    case GUST_F_UNARY: return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(16);
    case GUST_F_ROPE: return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(8) | GF_F(9);
#ifdef GUST_FLOAT32_DECODER
    case GUST_F_MUL: return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3);
    case GUST_F_SLICE: return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(16);
    case GUST_F_GATHER: /* kh = 1 (float32-t5-v1): float16 scales per 64 codes of a row */
        return GF_F(0) | GF_F(1) | GF_F(3) | GF_F(4) | GF_F(6) | GF_F(8) | GF_F(16) | GF_F(17);
    case GUST_F_SILU: return GF_F(0) | GF_F(1) | GF_F(3);
    case GUST_F_ROPE_AT:
        return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(8) | GF_F(9) | GF_F(16);
    case GUST_F_CACHE_APPEND: return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4);
    case GUST_F_COMMIT: return GF_F(0) | GF_F(1) | GF_F(3);
    case GUST_F_MHA_PREFIX: /* kw = 1 (float32-t5-v1): `bias` is the tensor index of an additive score bias */
        return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(6) | GF_F(8) | GF_F(9) | GF_F(16) |
               GF_F(17);
    case GUST_F_MHA_CACHE:
        return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(6) | GF_F(7) | GF_F(8) | GF_F(16) |
               GF_F(17);
    case GUST_F_DWCONV: return TP_USED_DWCONV; /* float32-temporal-v1 (temporal.c) */
    case GUST_F_FILM: return TP_USED_FILM;
    case GUST_F_MEAN: return TP_USED_MEAN;
    default: /* F_MHA; p0 bit 2 (float32-t5-v1): `bias` is the tensor index of an additive score bias */
        return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(6) | GF_F(8) | GF_F(16) | GF_F(17);
#else
    default: return GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(4) | GF_F(6) | GF_F(8) | GF_F(16) | GF_F(17); /* F_MHA */
#endif
    }
}

static int float_init(Context *c, const void *blob, uint32_t bytes, void *arena, uint32_t capacity) {
    const Header *h = (const Header *)blob;
    if (h->version != 1 || h->total != bytes || h->flags != GUST_F32_FLAGS || h->r0 || h->r1 || !h->nt ||
        h->nt > 4096 || !h->no || h->no > 4096)
        return GUST_FORMAT;
    if (h->to != 64 || h->oo != 64ull + 32ull * h->nt || h->data != h->oo + 80ull * h->no || h->data > bytes ||
        !range(h->to, 32ull * h->nt, bytes) || !range(h->oo, 80ull * h->no, bytes))
        return GUST_FORMAT;
    if (gf_crc((const uint8_t *)blob, bytes) != h->crc) return GUST_CHECKSUM;
    if (!arena || (uintptr_t)arena % 16 || !h->arena || h->arena > 64u * 1024u * 1024u || capacity < h->arena)
        return GUST_MEMORY;
    uintptr_t bp = (uintptr_t)blob, ap = (uintptr_t)arena, cp = (uintptr_t)c;
    if ((bp < ap + capacity && ap < bp + bytes) || (cp < ap + capacity && ap < cp + sizeof(Context)) ||
        (cp < bp + bytes && bp < cp + sizeof(Context)))
        return GUST_MEMORY;

    /* Stage the artifact in the context. `ready` stays 0 until every check below passed. */
    c->header = h;
    c->t = (const Tensor *)((const uint8_t *)blob + h->to);
    c->ops = (const Op *)((const uint8_t *)blob + h->oo);
    c->blob = (const uint8_t *)blob;
    c->arena = (int8_t *)arena;

    uint32_t ni = 0, nout = 0;
    int32_t first[4096], last[4096];
#ifdef GUST_FLOAT32_DECODER
    GdLoad gd;
    gd_load_begin(&gd);
#endif
    for (uint32_t i = 0; i < h->nt; i++) {
        const Tensor *t = c->t + i;
        if (t->w != 1 || !t->h || !t->c || t->h > GUST_F32_MAX_DIM || t->c > GUST_F32_MAX_DIM) return GUST_SHAPE;
        const uint64_t n = (uint64_t)t->h * t->c;
        if ((uint64_t)t->bytes != 4 * n || !range(t->offset, 4 * n, h->arena)) return GUST_SHAPE;
        if (t->offset % 16) return GUST_MEMORY;
        if (gf_asuint(t->scale) != 0x3f800000u) return GUST_QUANTIZATION;
#ifdef GUST_FLOAT32_DECODER
        /* float32 (4) plus optionally state (8, never together with input or output); at most 256 rows of state */
        if (t->reserved || !(t->flags & GUST_F32_TENSOR) || (t->flags & ~15u) || (t->flags & 3u) == 3u ||
            ((t->flags & GUST_F32_STATE) && (t->flags & 3u)))
            return GUST_FORMAT;
        if ((t->flags & GUST_F32_STATE) && t->h > GUST_F32_STATE_ROWS) return GUST_SHAPE;
#else
        if (t->reserved || !(t->flags & GUST_F32_TENSOR) || (t->flags & ~7u) || (t->flags & 3u) == 3u)
            return GUST_FORMAT;
#endif
        ni += (t->flags & 1u) != 0;
        nout += (t->flags & 2u) != 0;
#ifdef GUST_FLOAT32_DECODER
        first[i] = (t->flags & (1u | GUST_F32_STATE)) ? -1 : -2; /* state exists before the first operator */
#else
        first[i] = (t->flags & 1u) ? -1 : -2;
#endif
        last[i] = first[i];
    }
    if (!ni || !nout || ni != h->ni || nout != h->nout || !(c->t[0].flags & 1u)) return GUST_GRAPH;

    for (uint32_t i = 0; i < h->no; i++) {
        const Op *o = c->ops + i;
#ifdef GUST_FLOAT32_DECODER
        if (o->code < GUST_F_CONV || o->code > GUST_F_MEAN) return GUST_OPERATOR;
#else
        if (o->code < GUST_F_CONV || o->code > GUST_F_MHA) return GUST_OPERATOR;
#endif
        const uint32_t *fields = (const uint32_t *)o, used = gf_used(o->code);
        for (uint32_t f = 0; f < 20; f++)
            if (!(used & GF_F(f)) && fields[f]) return GUST_FORMAT;
#ifdef GUST_FLOAT32_DECODER
        if (o->code >= GUST_F_DWCONV) { /* float32-temporal-v1 operators validate their own operands (temporal.c) */
            const int status = tp_load_op(c, first, last, i, o);
            if (status) return status;
            continue;
        }
        if (o->code > GUST_F_MHA) { /* decoder operators validate their own operands, state and liveness */
            const int status = gd_load_op(c, &gd, first, last, i, o);
            if (status) return status;
            continue;
        }
#endif
        const int binary = o->code == GUST_F_ADD || o->code == GUST_F_MHA;
        if (o->a >= h->nt || o->out >= h->nt || (binary && o->b >= h->nt)) return GUST_GRAPH;
#ifdef GUST_FLOAT32_DECODER
        if (o->code == GUST_F_MHA && o->weights >= h->nt) return GUST_GRAPH;
        if (!gd_ordinary_operands_ok(c, o)) return GUST_GRAPH; /* fixed-function operators never touch state */
#endif
        if (o->code == GUST_F_MHA && o->weights >= h->nt) return GUST_GRAPH;
        const Tensor *a = c->t + o->a, *z = c->t + o->out;
        if (z->flags & 1u) return GUST_GRAPH;
        if (first[o->out] != -2 || first[o->a] == -2 || (binary && first[o->b] == -2) ||
            (o->code == GUST_F_MHA && first[o->weights] == -2))
            return GUST_GRAPH;
        first[o->out] = (int32_t)i;
        last[o->out] = (int32_t)i;
        last[o->a] = (int32_t)i;
        if (binary) last[o->b] = (int32_t)i;
        if (o->code == GUST_F_MHA) last[o->weights] = (int32_t)i;
#ifdef GUST_FLOAT32_T5
        if (o->code == GUST_F_MHA && !(o->p0 & 4u) && o->bias) return GUST_FORMAT;
        if (o->code == GUST_F_MHA && (o->p0 & 4u)) { /* the bias: an ordinary tensor written before, never state */
            if (o->bias >= h->nt || first[o->bias] == -2) return GUST_GRAPH;
            if (c->t[o->bias].flags & GUST_F32_STATE) return GUST_GRAPH;
            last[o->bias] = (int32_t)i;
        }
#endif

        switch (o->code) {
        case GUST_F_CONV: {
            if (o->kw != 1 || o->sx != 1 || o->py || o->px || o->dy != 1 || o->dx != 1) return GUST_SHAPE;
            if (!o->kh || !o->sy || a->h < o->kh) return GUST_SHAPE;
            if ((a->h - o->kh) / o->sy + 1u != z->h) return GUST_SHAPE;
#ifdef GUST_FLOAT32_T5
            /* p0 bit 4 (float32-t5-v1): grouped partial sums, only with bit 0 (4-bit group64 codes, kh 1, K a
               multiple of 64, float32 sums) */
            if ((o->p0 & 16u) && (o->p0 != 17u || o->p1 != 4 || o->kh != 1 || a->c % 64u)) return GUST_FORMAT;
#ifdef GUST_FLOAT32_TEMPORAL
            /* p1 32 (float32-temporal-v1): float32 weights with a float32 scale per row, float32 or accurate sums only */
            if (o->p1 == 32 && (o->p0 & ~2u)) return GUST_FORMAT;
            if ((o->p1 < 2 || o->p1 > 8) && o->p1 != 32) return GUST_FORMAT;
            if ((o->p0 & ~31u) || ((o->p0 & 1u) && o->kh != 1)) return GUST_FORMAT;
#else
            if (o->p1 < 2 || o->p1 > 8 || (o->p0 & ~31u) || ((o->p0 & 1u) && o->kh != 1)) return GUST_FORMAT;
#endif
#else
            if (o->p1 < 2 || o->p1 > 8 || (o->p0 & ~15u) || ((o->p0 & 1u) && o->kh != 1)) return GUST_FORMAT;
#endif
            /* p0 bit 3 (with bit 2): int8 activations, per-row scales (int8-activation convolution) */
            if ((o->p0 & 8u) && !(o->p0 & 4u)) return GUST_FORMAT;
            const uint64_t K = (uint64_t)o->kh * a->c, N = z->c;
            /* p0 bit 4: int16 activations; per-row scales, unaccurate mode, codes of at most 4 bits (float32-temporal-v1:
               at most 8 bits, with activation codes bounded so the int32 sums cannot overflow, see TEMPORAL.md) */
#ifdef GUST_FLOAT32_TEMPORAL
            const uint32_t q_bits = (o->p0 & 8u) ? 4u : 8u;
#else
            const uint32_t q_bits = 4u;
#endif
            if ((o->p0 & 4u) && ((o->p0 & 3u) || o->p1 > q_bits || K > GQ_MAX_K || z->h > GQ_MAX_ROWS ||
                                 (uint64_t)z->h * ((K + 1u) & ~1ull) > GUST_F32_ACT_SLOTS))
                return GUST_FORMAT;
            if (K > (1ull << 28) || N > (1ull << 28)) return GUST_SHAPE;
            const uint64_t codebits = N * K * o->p1, codebytes = (codebits + 7) / 8;
            if (!gf_blob(c, o->weights, codebytes, 4)) return GUST_FORMAT;
            /* Unused high bits of the last byte are zero: one encoding per weight set. */
            if ((codebits & 7) && (c->blob[o->weights + codebytes - 1] >> (codebits & 7))) return GUST_FORMAT;
#ifdef GUST_FLOAT32_TEMPORAL
            if (o->p1 == 32 && !gf_floats_ok(c, o->weights, N * K, 0)) return GUST_QUANTIZATION;
#endif
            if (o->bias) {
                if (!gf_blob(c, o->bias, 4 * N, 4)) return GUST_FORMAT;
                if (!gf_floats_ok(c, o->bias, N, 0)) return GUST_QUANTIZATION;
            }
            if (o->p0 & 1u) {
                const uint64_t groups = N * ((K + 63) / 64);
                if (!gf_blob(c, o->mult, 2 * groups, 4)) return GUST_FORMAT;
                for (uint64_t g = 0; g < groups; g++) {
                    const uint16_t s =
                        (uint16_t)(c->blob[o->mult + 2 * g] | (uint32_t)c->blob[o->mult + 2 * g + 1] << 8);
                    /* positive, finite float16 */
                    if ((s & 0x8000u) || ((s >> 10) & 31u) == 31u || !(s & 0x7fffu)) return GUST_QUANTIZATION;
                }
            } else {
                if (!gf_blob(c, o->mult, 4 * N, 4)) return GUST_FORMAT;
                if (!gf_floats_ok(c, o->mult, N, 1)) return GUST_QUANTIZATION;
            }
            break;
        }
        case GUST_F_ADD: {
            const Tensor *b = c->t + o->b;
            if (!same(a, b) || !same(a, z)) return GUST_SHAPE;
            break;
        }
        case GUST_F_NORM:
#ifdef GUST_FLOAT32_T5
            /* 0 layer, 1 group, 2 fast float32 layer norm, 4 RMS norm (float32-t5-v1, no beta) */
            if (!same(a, z) || (o->p0 > 2 && o->p0 != 4)) return GUST_SHAPE;
            if (o->p0 == 4 && o->bias) return GUST_FORMAT;
#else
            if (!same(a, z) || o->p0 > 2) return GUST_SHAPE; /* 0 layer, 1 group, 2 fast float32 layer norm */
#endif
            if (!gf_blob(c, o->weights, 4ull * a->c, 4) || !gf_blob(c, o->mult, 4, 4)) return GUST_FORMAT;
            if (!gf_floats_ok(c, o->weights, a->c, 0) || !gf_floats_ok(c, o->mult, 1, 1)) return GUST_QUANTIZATION;
            if (o->bias) {
                if (!gf_blob(c, o->bias, 4ull * a->c, 4)) return GUST_FORMAT;
                if (!gf_floats_ok(c, o->bias, a->c, 0)) return GUST_QUANTIZATION;
            }
            break;
        case GUST_F_UNARY:
#ifdef GUST_FLOAT32_T5
            if (!same(a, z) || o->p0 > 3) return GUST_SHAPE; /* 0 tanh, 1 GELU, 2 fast GELU, 3 ReLU (float32-t5-v1) */
#else
            if (!same(a, z) || o->p0 > 2) return GUST_SHAPE; /* 0 tanh, 1 GELU, 2 fast GELU */
#endif
            break;
        case GUST_F_ROPE: {
            if (!same(a, z) || !o->kh || a->c % o->kh) return GUST_SHAPE;
            const uint32_t D = a->c / o->kh;
            if (o->kw < 2 || o->kw % 2 || o->kw > D || o->a == o->out) return GUST_SHAPE;
            const uint64_t table = (uint64_t)a->h * (o->kw / 2);
            if (!gf_blob(c, o->weights, 4 * table, 4) || !gf_blob(c, o->bias, 4 * table, 4)) return GUST_FORMAT;
            if (!gf_floats_ok(c, o->weights, table, 0) || !gf_floats_ok(c, o->bias, table, 0)) return GUST_QUANTIZATION;
            break;
        }
        default: { /* F_MHA */
            const Tensor *k = c->t + o->b, *v = c->t + o->weights;
            if (!o->kh || a->c % o->kh || a->c / o->kh > GUST_F32_MAX_HEAD) return GUST_SHAPE;
            if (!same(a, z) || a->c != k->c || !same(k, v) || k->h > GUST_F32_MAX_KEYS) return GUST_SHAPE;
            /* p0 0: every key; p0 1: banded, keys within p1 rows of the query (self attention only) */
            /* p0 bit 1: float32 (fast) attention */
#ifdef GUST_FLOAT32_T5
            /* p0 bit 2 (float32-t5-v1): additive score bias [heads * query rows, key rows] at tensor `bias`, row
               (head * query rows + query); exact attention, not banded */
            if (o->p0 > 4 || (!(o->p0 & 1u) && o->p1)) return GUST_FORMAT;
            if ((o->p0 & 4u) &&
                (c->t[o->bias].h != (uint64_t)o->kh * a->h || c->t[o->bias].c != k->h))
                return GUST_SHAPE;
#else
            if (o->p0 > 3 || (!(o->p0 & 1u) && o->p1)) return GUST_FORMAT;
#endif
            if ((o->p0 & 1u) && !same(a, k)) return GUST_SHAPE;
            if (!gf_blob(c, o->mult, 4, 4)) return GUST_FORMAT;
            if (!gf_floats_ok(c, o->mult, 1, 1)) return GUST_QUANTIZATION;
            break;
        }
        }
    }
#ifdef GUST_FLOAT32_DECODER
    {
        const int status = gd_load_end(c, &gd);
        if (status) return status;
    }
#endif
    for (uint32_t i = 0; i < h->nt; i++) {
        if (first[i] == -2) return GUST_GRAPH;
#ifdef GUST_FLOAT32_DECODER
        if (c->t[i].flags & (3u | GUST_F32_STATE)) last[i] = (int32_t)h->no; /* state lives for the whole model */
#else
        if (c->t[i].flags & 3u) last[i] = (int32_t)h->no;
#endif
    }
    /* Tensors that are live at the same time never share arena bytes (also: no destination aliases an operand). */
    for (uint32_t i = 0; i < h->nt; i++)
        for (uint32_t j = 0; j < i; j++)
            if (first[i] <= last[j] && first[j] <= last[i] &&
                overlap(c->t[i].offset, c->t[i].bytes, c->t[j].offset, c->t[j].bytes))
                return GUST_MEMORY;
#ifdef GUST_FLOAT32_DECODER
    gd_zero_state(c); /* the only arena write of a load, after every check passed */
#endif
    c->valid_ok = 1;
    c->ready = GUST_F32_READY;
    return GUST_OK;
}
