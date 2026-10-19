/* Artifact loader: header, tensor and operator validation, liveness/overlap
   checks and context setup. Every check runs once in gust_init so that
   gust_run can execute without bounds checks. Part of the runtime/gust.c
   translation unit. */

static const uint8_t magic[8] = {'G', 'U', 'S', 'T', 0, 0, 1, 0};
static int range(uint32_t off, uint64_t bytes, uint32_t total) {
    return off <= total && bytes <= total - off;
}
static int overlap(uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    return (uint64_t)a < (uint64_t)b + bn && (uint64_t)b < (uint64_t)a + an;
}
static uint32_t crc(const uint8_t *p, uint32_t n) {
    uint32_t c = 0xffffffffu;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t v = (i >= 16 && i < 20) ? 0 : p[i];
        c ^= v;
        for (int j = 0; j < 8; j++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return ~c;
}
static int same(const Tensor *a, const Tensor *b) {
    return a->h == b->h && a->w == b->w && a->c == b->c;
}
static int qtable(const Context *c, const Op *o, uint32_t channels) {
    uint32_t offsets[3] = {o->bias, o->mult, o->shift};
    for (int k = 0; k < 3; k++)
        if (offsets[k] < c->header->data || offsets[k] % 4 || !range(offsets[k], 4ull * channels, c->header->total))
            return 0;
    const int32_t *m = (const int32_t *)(c->blob + o->mult);
    const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
    for (uint32_t i = 0; i < channels; i++)
        if (m[i] <= 0 || s[i] > 30) return 0;
    return 1;
}
#ifdef GUST_ATTENTION
/* MATMUL/ATTEND carry one multiplier/shift for the whole tensor and no bias. ATTEND's weights field is its exp
   table: 256 uint32 entries, lut[0] = 2^15, non-increasing. */
static int qscalar(const Context *c, const Op *o) {
    const uint32_t total = c->header->total, data = c->header->data;
    if (o->bias || o->mult < data || o->mult % 4 || !range(o->mult, 4, total) || o->shift < data || o->shift % 4 ||
        !range(o->shift, 4, total))
        return 0;
    if (*(const int32_t *)(c->blob + o->mult) <= 0 || *(const uint32_t *)(c->blob + o->shift) > 30) return 0;
    if (o->code != GUST_ATTEND) return 1;
    if (o->weights < data || o->weights % 4 || !range(o->weights, 1024, total)) return 0;
    const uint32_t *lut = (const uint32_t *)(c->blob + o->weights);
    if (lut[0] != 32768u) return 0;
    for (uint32_t d = 1; d < 256; d++)
        if (lut[d] > lut[d - 1]) return 0;
    return 1;
}
#endif
uint32_t gust_context_bytes(void) {
    return (uint32_t)sizeof(Context);
}
#if defined(GUST_CONFORMER)
#define GUST_LAST_OPERATOR GUST_MUL
#elif defined(GUST_ATTENTION)
#define GUST_LAST_OPERATOR GUST_ATTEND
#else
#define GUST_LAST_OPERATOR GUST_WEIGHTED_POOL
#endif
uint32_t gust_capabilities(void) {
#if defined(GUST_CONFORMER)
    const uint32_t extra = 32u | 64u | 128u;
#elif defined(GUST_ATTENTION)
    const uint32_t extra = 32u | 64u;
#else
    const uint32_t extra = 0;
#endif
#ifdef GUST_ATTENTION
    const uint32_t outputs = 16384u; /* valid-outputs-v1 (gust_run_valid_outputs) */
#else
    const uint32_t outputs = 0;
#endif
#ifdef GUST_PACKED_WEIGHTS
    const uint32_t packed = 256u; /* packed-conv-weights-v1, independent of the attention features */
#else
    const uint32_t packed = 0;
#endif
#ifdef GUST_FLOAT32
    const uint32_t floating = GUST_F32_CAPABILITY; /* float32-graph-v1: operators 32 to 37, Header.flags 1 */
#else
    const uint32_t floating = 0;
#endif
#if defined(__wasm_simd128__) || defined(__SSE2__) || defined(__aarch64__)
    return 31u | extra | packed | floating | outputs;
#else
    return 30u | extra | packed | floating | outputs;
#endif
}
int gust_init(void *ctx, const void *blob, uint32_t bytes, void *arena, uint32_t capacity) {
    if (!ctx || (uintptr_t)ctx % 8) return GUST_MEMORY;
    Context *c = (Context *)ctx;
    c->ready = 0;
    if (!blob || (uintptr_t)blob % 4 || bytes < 64 || bytes > 64u * 1024u * 1024u) return GUST_FORMAT;
    const Header *h = (const Header *)blob;
    for (int i = 0; i < 8; i++)
        if (h->magic[i] != magic[i]) return GUST_FORMAT;
#ifdef GUST_FLOAT32
#ifdef GUST_SHARED_SCRATCH
    {
        static int16_t act[GUST_F32_ACT_SLOTS] __attribute__((aligned(16)));
        static int16_t slab[16][32768] __attribute__((aligned(16)));
        c->f32_act = act;
        c->f32_slab = slab;
    }
#endif
    /* float32-graph-v1 (src/float32/format.c) validates itself; flags 1 is refused by every other build below. */
    if (h->flags == GUST_F32_FLAGS) return float_init(c, blob, bytes, arena, capacity);
#endif
    if (h->version != 1 || h->total != bytes || h->flags || h->r0 || h->r1 || !h->nt || h->nt > 4096 || !h->no ||
        h->no > 4096)
        return GUST_FORMAT;
    if (h->to != 64 || h->oo != 64ull + 32ull * h->nt || h->data != h->oo + 80ull * h->no || h->data > bytes ||
        !range(h->to, 32ull * h->nt, bytes) || !range(h->oo, 80ull * h->no, bytes))
        return GUST_FORMAT;
    if (crc((const uint8_t *)blob, bytes) != h->crc) return GUST_CHECKSUM;
    if (!arena || !h->arena || h->arena > 64u * 1024u * 1024u || capacity < h->arena) return GUST_MEMORY;
    uintptr_t bp = (uintptr_t)blob, ap = (uintptr_t)arena, cp = (uintptr_t)ctx;
    if ((bp < ap + capacity && ap < bp + bytes) || (cp < ap + capacity && ap < cp + sizeof(Context)) ||
        (cp < bp + bytes && bp < cp + sizeof(Context)))
        return GUST_MEMORY;
    c->header = h;
    c->t = (const Tensor *)((const uint8_t *)blob + h->to);
    c->ops = (const Op *)((const uint8_t *)blob + h->oo);
    c->blob = (const uint8_t *)blob;
    c->arena = (int8_t *)arena;
    uint32_t ni = 0, nout = 0;
    int32_t first[4096], last[4096];
    for (uint32_t i = 0; i < 128; i++) c->nonneg[i] = c->small[i] = 0;
    for (uint32_t i = 0; i < h->nt; i++) {
        const Tensor *t = c->t + i;
        uint64_t n = (uint64_t)t->h * t->w * t->c;
        if (!t->h || !t->w || !t->c || t->h > 65536 || t->w > 65536 || t->c > 65536 || n != t->bytes ||
            !range(t->offset, n, h->arena))
            return GUST_SHAPE;
        if (!(t->scale > 0 && t->scale <= 1e20f) || t->reserved || (t->flags & ~3u)) return GUST_QUANTIZATION;
        ni += (t->flags & 1) != 0;
        nout += (t->flags & 2) != 0;
        first[i] = (t->flags & 1) ? -1 : -2;
        last[i] = first[i];
    }
    if (!ni || !nout || ni != h->ni || nout != h->nout) return GUST_GRAPH;
    for (uint32_t i = 0; i < h->no; i++) {
        const Op *o = c->ops + i;
        if (o->code < GUST_CONV || o->code > GUST_LAST_OPERATOR) return GUST_OPERATOR;
        int binary = o->code == GUST_ADD || o->code == GUST_CONCAT || o->code == GUST_WEIGHTED_POOL ||
                     o->code == GUST_MATMUL || o->code == GUST_ATTEND || o->code == GUST_MUL;
        int scored = o->code == GUST_MATMUL || o->code == GUST_ATTEND;
        /* Convolution parameter1: 0 byte INT8 weights, 3 or 4 packed signed codes (ops/packed.c); else refused. */
        int packed_conv = (o->code == GUST_CONV || o->code == GUST_DEPTHWISE) && (o->p1 == 3 || o->p1 == 4);
        if (o->a >= h->nt || o->out >= h->nt || (binary && o->b >= h->nt)) return GUST_GRAPH;
        const Tensor *a = c->t + o->a, *z = c->t + o->out;
        if (z->flags & 1 || o->lo < -128 || o->hi > 127 || o->lo > o->hi ||
            (o->p0 && o->code != GUST_SLICE && o->code != GUST_GATHER && o->code != GUST_TOPK_POOL &&
             o->code != GUST_MATMUL && o->code != GUST_NORM) ||
            (o->p1 && !scored && !packed_conv))
            return GUST_GRAPH;
        if (first[o->out] != -2 || first[o->a] == -2 || (binary && first[o->b] == -2)) return GUST_GRAPH;
        first[o->out] = (int32_t)i;
        last[o->out] = (int32_t)i;
        last[o->a] = (int32_t)i;
        if (binary) last[o->b] = (int32_t)i;
        { /* Every operator clamps to [lo, hi]; data movement preserves the sign of its input. */
            const uint32_t na = (c->nonneg[o->a >> 5] >> (o->a & 31)) & 1u,
                           nb = o->code == GUST_CONCAT ? (c->nonneg[o->b >> 5] >> (o->b & 31)) & 1u : 0;
            if (o->lo >= 0 || ((o->code == GUST_RESIZE || o->code == GUST_SLICE) && na) ||
                (o->code == GUST_CONCAT && na && nb))
                c->nonneg[o->out >> 5] |= 1u << (o->out & 31);
        }
        if (packed_conv) c->small[i >> 5] |= 1u << (i & 31); /* 3/4-bit codes lie in [-8, 7] */
        if (o->code != GUST_CONV && o->code != GUST_DEPTHWISE &&
            ((o->weights && o->code != GUST_GATHER && o->code != GUST_ATTEND && o->code != GUST_NORM &&
              o->code != GUST_LUT) ||
             o->kh || o->kw || o->sy || o->sx || o->py || o->px || o->dy || o->dx))
            return GUST_FORMAT;
        if (o->code != GUST_CONV && o->code != GUST_DEPTHWISE && o->code != GUST_POOL && o->code != GUST_REQUANTIZE &&
            o->code != GUST_GATHER && o->code != GUST_TOPK_POOL && o->code != GUST_WEIGHTED_POOL && !scored &&
            o->code != GUST_NORM && o->code != GUST_MUL && (o->bias || o->mult || o->shift))
            return GUST_FORMAT;
        uint64_t terms = 0, weightbytes = 0;
        switch (o->code) {
        case GUST_CONV:
        case GUST_DEPTHWISE: {
            if (!o->kh || !o->kw || o->kh > 64 || o->kw > 64 || !o->sy || !o->sx || o->sy > 64 || o->sx > 64 ||
                !o->dy || !o->dx || o->dy > 64 || o->dx > 64 || o->py > 4096 || o->px > 4096)
                return GUST_SHAPE;
            int64_t eh = (int64_t)a->h + 2 * o->py - (int64_t)o->dy * (o->kh - 1) - 1;
            int64_t ew = (int64_t)a->w + 2 * o->px - (int64_t)o->dx * (o->kw - 1) - 1;
            if (eh < 0 || ew < 0 || (uint64_t)(eh / o->sy + 1) != z->h || (uint64_t)(ew / o->sx + 1) != z->w ||
                (o->code == GUST_DEPTHWISE && a->c != z->c))
                return GUST_SHAPE;
            terms = (uint64_t)o->kh * o->kw * (o->code == GUST_CONV ? a->c : 1);
            weightbytes = terms * z->c;
            if (packed_conv) {
#ifdef GUST_PACKED_WEIGHTS
                /* Supported size: dense row terms and depthwise filters are bounded so decoding needs only a
                   fixed stack tile; bigger packed operators are refused here, never decoded wholesale. */
                if (o->code == GUST_CONV ? terms > PACKED_DENSE_TERMS : weightbytes > PACKED_DEPTHWISE_WEIGHTS)
                    return GUST_SHAPE;
                const uint64_t codebits = weightbytes * o->p1; /* <= 2^28 * 2^16 * 4 */
                weightbytes = (codebits + 7) / 8;
                if (o->weights < h->data || !range(o->weights, weightbytes, bytes)) return GUST_FORMAT;
                /* The unused high bits of the last byte are zero, so each logical weight set has one encoding. */
                if ((codebits & 7) && (c->blob[o->weights + weightbytes - 1] >> (codebits & 7))) return GUST_FORMAT;
#else
                return GUST_OPERATOR; /* build without packed-conv-weights-v1 */
#endif
                break;
            }
            if (o->weights < h->data || !range(o->weights, weightbytes, bytes)) return GUST_FORMAT;
#if defined(INT8_TILE) && defined(__AVX2__) && !defined(GUST_INT8_VNNI)
            if (o->code == GUST_CONV) { /* the AVX2 offset-weight tiles take signed inputs only with |w| <= 63 */
                const int8_t *wt = (const int8_t *)(c->blob + o->weights);
                int ok = 1;
                for (uint64_t k = 0; k < weightbytes; k++) ok &= wt[k] >= -63 && wt[k] <= 63;
                if (ok) c->small[i >> 5] |= 1u << (i & 31);
            }
#endif
            break;
        }
        case GUST_ADD: {
            const Tensor *b = c->t + o->b;
            if (!same(a, b) || !same(a, z) || a->scale != b->scale || a->scale != z->scale) return GUST_SHAPE;
            break;
        }
        case GUST_POOL:
            if (z->h != 1 || z->w != 1 || z->c != a->c) return GUST_SHAPE;
            terms = (uint64_t)a->h * a->w;
            break;
        case GUST_TOPK_POOL:
            if (!o->p0 || o->p0 > (uint64_t)a->h * a->w || z->h != 1 || z->w != 1 || z->c != a->c) return GUST_SHAPE;
            terms = o->p0;
            break;
        case GUST_WEIGHTED_POOL: {
            const Tensor *b = c->t + o->b;
            if (b->h != a->h || b->w != a->w || b->c != 1 || b->scale != 1.f || z->h != 1 || z->w != 1 || z->c != a->c)
                return GUST_SHAPE;
            terms = (uint64_t)a->h * a->w * 127;
            break;
        }
        case GUST_RESIZE:
            if (z->c != a->c || z->scale != a->scale) return GUST_SHAPE;
            break;
        case GUST_CONCAT: {
            const Tensor *b = c->t + o->b;
            if (a->h != b->h || a->w != b->w || z->h != a->h || z->w != a->w || z->c != a->c + b->c ||
                a->scale != b->scale || a->scale != z->scale)
                return GUST_SHAPE;
            break;
        }
        case GUST_REQUANTIZE:
            if (!same(a, z)) return GUST_SHAPE;
            terms = 1;
            break;
        case GUST_SLICE:
            if (a->h != z->h || a->w != z->w || (uint64_t)o->p0 + z->c > a->c || a->scale != z->scale)
                return GUST_SHAPE;
            break;
        case GUST_GATHER:
            if ((a->c != 2 && a->c != 3) || a->scale != 1 || !(a->flags & 1) || a->h != z->h || a->w != z->w ||
                !o->p0 || o->p0 > (a->c == 3 ? 65536u : 65535u) || o->b || o->lo > 0 || o->hi < 0)
                return GUST_SHAPE;
            if (o->weights < h->data || !range(o->weights, (uint64_t)o->p0 * z->c, bytes)) return GUST_FORMAT;
            terms = 1;
            break;
#ifdef GUST_ATTENTION
        case GUST_MATMUL: { /* a [Ta,1,C] . b [Tb,1,C]^T -> [Ta,1,Tb]; p1 = 1: keys s > floor(t*Tb/Ta) + p0 are -128 */
            const Tensor *b = c->t + o->b;
            if (a->w != 1 || b->w != 1 || z->w != 1 || a->c != b->c || z->h != a->h || z->c != b->h) return GUST_SHAPE;
            if (o->p1 > 1 || (!o->p1 && o->p0) || o->p0 > 65536 || (o->p1 && o->lo < -127)) return GUST_FORMAT;
            if (!qscalar(c, o) || (uint64_t)a->c * 16384 > INT32_MAX) return GUST_QUANTIZATION;
            break;
        }
        case GUST_ATTEND: { /* softmax(a [T,1,S]) . b [S,1,C] -> [T,1,C]; p1 = 1: a == -128 has weight 0 */
            const Tensor *b = c->t + o->b;
            if (a->w != 1 || b->w != 1 || z->w != 1 || a->c != b->h || z->h != a->h || z->c != b->c) return GUST_SHAPE;
            if (o->p1 > 1) return GUST_FORMAT;
            if (!qscalar(c, o)) return GUST_QUANTIZATION;
            break;
        }
#endif
#ifdef GUST_CONFORMER
        case GUST_NORM: { /* per-position LayerNorm over C <= 4096 channels; weights: C int8 gamma codes, bias: C
                             beta codes (units of 2^-15 * the channel's weight scale), p0: epsilon term E */
            if (!same(a, z) || a->c > 4096) return GUST_SHAPE;
            if (o->weights < h->data || !range(o->weights, a->c, bytes)) return GUST_FORMAT;
            if (!qtable(c, o, z->c)) return GUST_QUANTIZATION;
            uint64_t r = 1;
            while (r * r < a->c) r++; /* ceil(sqrt(C)) */
            const int32_t *beta = (const int32_t *)(c->blob + o->bias);
            const uint64_t bound = r * 32768 * 128;
            for (uint32_t k = 0; k < z->c; k++) {
                int64_t v = beta[k];
                if (bound + (uint64_t)(v < 0 ? -v : v) > INT32_MAX) return GUST_QUANTIZATION;
            }
            break;
        }
        case GUST_LUT: { /* z = table[a + 128]: 256 int8 entries at weights, each inside [lo, hi]; no bias/mult/shift */
            if (!same(a, z)) return GUST_SHAPE;
            if (o->weights < h->data || !range(o->weights, 256, bytes)) return GUST_FORMAT;
            const int8_t *table = (const int8_t *)(c->blob + o->weights);
            int nonneg = 1;
            for (uint32_t k = 0; k < 256; k++) {
                if (table[k] < o->lo || table[k] > o->hi) return GUST_QUANTIZATION;
                nonneg &= table[k] >= 0;
            }
            if (nonneg) c->nonneg[o->out >> 5] |= 1u << (o->out & 31); /* table entries or zero tails */
            break;
        }
        case GUST_MUL: /* z = quant(a * b): equal shapes, one multiplier/shift, no bias */
            if (!same(a, c->t + o->b) || !same(a, z)) return GUST_SHAPE;
            if (!qscalar(c, o)) return GUST_QUANTIZATION;
            break;
#endif
        default: return GUST_OPERATOR;
        }
        if (terms) {
            if (!qtable(c, o, z->c)) return GUST_QUANTIZATION;
            const int32_t *bias = (const int32_t *)(c->blob + o->bias);
            uint64_t bound =
                terms * (o->code == GUST_CONV || o->code == GUST_DEPTHWISE || (o->code == GUST_GATHER && a->c == 3)
                             ? 16384
                             : 128);
            for (uint32_t k = 0; k < z->c; k++) {
                int64_t v = bias[k];
                uint64_t ab = (uint64_t)(v < 0 ? -v : v);
                if (bound + ab > INT32_MAX) return GUST_QUANTIZATION;
                if (o->code == GUST_GATHER && v) return GUST_QUANTIZATION;
            }
        }
    }
    for (uint32_t i = 0; i < h->nt; i++) {
        if (first[i] == -2) return GUST_GRAPH;
        if (c->t[i].flags & 3) last[i] = (int32_t)h->no;
    }
    for (uint32_t i = 0; i < h->nt; i++) {
        for (uint32_t j = 0; j < i; j++) {
            if (first[i] <= last[j] && first[j] <= last[i] &&
                overlap(c->t[i].offset, c->t[i].bytes, c->t[j].offset, c->t[j].bytes))
                return GUST_MEMORY;
        }
    }
    /* Uniform requantize operators keep an exact 256-value table in the packed-weight storage. */
    uint32_t packed = 0;
    for (uint32_t i = 0; i < h->no; i++) c->packed_offsets[i] = UINT32_MAX;
    for (uint32_t i = 0; i < h->no && packed + 128 <= 32768u; i++) {
        const Op *o = c->ops + i;
        if (o->code != GUST_REQUANTIZE) continue;
        const Tensor *z = c->t + o->out;
        const int32_t *bias = (const int32_t *)(c->blob + o->bias), *m = (const int32_t *)(c->blob + o->mult);
        const uint32_t *s = (const uint32_t *)(c->blob + o->shift);
        int uniform = 1;
        for (uint32_t ch = 1; ch < z->c; ch++) uniform &= bias[ch] == bias[0] && m[ch] == m[0] && s[ch] == s[0];
        if (uniform) {
            c->packed_offsets[i] = packed;
            int8_t *table = (int8_t *)(c->packed_weights + packed);
            for (int j = 0; j < 256; j++) table[j] = quant(j - 128 + bias[0], m[0], s[0], o);
            packed += 128;
        }
    }
    /* gust_run_valid supports operators 1-8 and 12-16 (the gather and masked/top-k pools read whole inputs). */
    c->valid_ok = 1;
    for (uint32_t i = 0; i < h->no; i++)
        if (c->ops[i].code == GUST_GATHER || c->ops[i].code == GUST_TOPK_POOL || c->ops[i].code == GUST_WEIGHTED_POOL)
            c->valid_ok = 0;
    c->ready = 1;
    return GUST_OK;
}
