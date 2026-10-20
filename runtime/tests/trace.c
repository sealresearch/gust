/* Observer contract: full arena parity, error boundaries, active rows, NULL observers and isolated sinks.
   Build with gust.c's feature flags. Optional argument: a float32 artifact to exercise real operators. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/gust.c"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const Context *ctx; unsigned events, completed; int in_op, ended, status, operator_status; } sink;
static void observe(void *opaque, const gust_trace_event *e) {
    sink *s = opaque;
    assert(!s->ended);
    if (!s->events++) assert(e->kind == GUST_TRACE_RUN_BEGIN);
    else if (e->kind == GUST_TRACE_RUN_END) {
        assert(!s->in_op);
        s->ended = 1;
        s->status = e->status;
    } else {
        assert(s->ctx && e->operator_index == s->completed);
        const Op *o = s->ctx->ops + e->operator_index;
        const Tensor *t = s->ctx->t + o->out;
        assert(e->opcode == o->code && e->output_tensor == o->out);
        assert(e->height == t->h && e->width == t->w && e->channels == t->c);
        assert(e->rows && e->rows <= t->h);
        if (e->kind == GUST_TRACE_OP_BEGIN) { assert(!s->in_op); s->in_op = 1; }
        else { assert(e->kind == GUST_TRACE_OP_END && s->in_op); s->in_op = 0; s->completed++; s->operator_status = e->status; }
    }
    if (e->kind == GUST_TRACE_RUN_BEGIN || e->kind == GUST_TRACE_RUN_END)
        assert(e->operator_index == UINT32_MAX && e->output_tensor == UINT32_MAX);
}
static void complete(const sink *s, int status, unsigned ops) {
    assert(s->ended && !s->in_op && s->status == status && s->completed == ops);
    assert(s->events == 2u + 2u * ops);
}
static void seed(Context *c) {
    memset(c->arena, 0xa5, c->header->arena);
    for (uint32_t i = 0; i < c->header->nt; i++) {
        const Tensor *t = c->t + i;
        if (!(t->flags & 1u)) continue;
#ifdef GUST_FLOAT32
        if (c->ready == GUST_F32_READY) {
            float *x = (float *)(c->arena + t->offset);
            for (uint32_t j = 0; j < t->bytes / 4u; j++) x[j] = t->c == 1 ? (float)(j % 4u) : (float)(j % 11u) * .01f;
        } else
#endif
            memset(c->arena + t->offset, (int)i + 1, t->bytes);
    }
}
static void parity(Context *c, int per_input) {
    uint32_t counts[4096], n = 0;
    for (uint32_t i = 0; i < c->header->nt; i++)
        if (c->t[i].flags & 1u) {
            counts[n] = c->t[i].h;
#ifdef GUST_ATTENTION
            if (counts[n] > 3u) counts[n] = 3u;
#endif
            n++;
        }
    uint8_t *expected = malloc(c->header->arena);
    assert(expected);
    seed(c);
    int status = per_input ? gust_run_valid_inputs(c, counts, n) : gust_run_valid(c, counts[0]);
    assert(status == GUST_OK);
    memcpy(expected, c->arena, c->header->arena);
    seed(c);
    sink first = {.ctx=c};
    status = per_input ? gust_run_valid_inputs_traced(c, counts, n, observe, &first) :
                        gust_run_traced(c, counts[0], observe, &first);
    assert(status == GUST_OK);
    complete(&first, status, c->header->no);
    assert(!memcmp(expected, c->arena, c->header->arena));
    seed(c);
    status = per_input ? gust_run_valid_inputs_traced(c, counts, n, NULL, NULL) : gust_run_traced(c, counts[0], NULL, NULL);
    assert(status == GUST_OK && !memcmp(expected, c->arena, c->header->arena));
    /* A separate observer must not inherit the first call's state. */
    seed(c);
    sink second = {.ctx=c};
    assert(gust_run_traced(c, counts[0], observe, &second) == GUST_OK);
    complete(&second, GUST_OK, c->header->no);
    complete(&first, GUST_OK, c->header->no);
    memcpy(expected, c->arena, c->header->arena);
    sink bad = {.ctx=c};
    assert(gust_run_traced(c, c->t[0].h + 1u, observe, &bad) == GUST_SHAPE);
    complete(&bad, GUST_SHAPE, 0);
    assert(!memcmp(expected, c->arena, c->header->arena));
    bad = (sink){.ctx=c};
    const int refusal = per_input ? GUST_SHAPE : GUST_OPERATOR;
    assert(gust_run_valid_inputs_traced(c, NULL, 0, observe, &bad) == refusal);
    complete(&bad, refusal, 0);
    assert(!memcmp(expected, c->arena, c->header->arena));
    if (per_input) {
        bad = (sink){.ctx=c};
        assert(gust_run_valid_inputs_traced(c, (uint32_t *)c, n, observe, &bad) == GUST_MEMORY);
        complete(&bad, GUST_MEMORY, 0);
        assert(!memcmp(expected, c->arena, c->header->arena));
    }
    free(expected);
}
static void *aligned_memory(size_t bytes, void **owned) {
    *owned = malloc(bytes + 63u);
    assert(*owned);
    return (void *)(((uintptr_t)*owned + 63u) & ~(uintptr_t)63u);
}
int main(int argc, char **argv) {
    sink invalid = {0};
    assert(gust_run_traced(NULL, 0, observe, &invalid) == GUST_GRAPH);
    complete(&invalid, GUST_GRAPH, 0);
    invalid = (sink){0};
    assert(gust_run_valid_inputs_traced(NULL, NULL, 0, observe, &invalid) == GUST_GRAPH);
    complete(&invalid, GUST_GRAPH, 0);
    void *context_memory, *arena_memory;
    Context *c = aligned_memory(gust_context_bytes(), &context_memory);
    memset(c, 0, gust_context_bytes());
    invalid = (sink){0};
    assert(gust_run_traced(c, 0, observe, &invalid) == GUST_GRAPH);
    complete(&invalid, GUST_GRAPH, 0);
    /* A real, loader-validated INT8 ADD graph, independent of Relay. */
    uint8_t blob[264] __attribute__((aligned(8))) = {0};
    Header *h = (Header *)blob;
    memcpy(h->magic, magic, 8);
    h->version = 1; h->total = sizeof blob; h->nt = 3; h->no = 1;
    h->to = 64; h->oo = 160; h->data = 240; h->arena = 64; h->ni = 2; h->nout = 1;
    Tensor *t = (Tensor *)(blob + h->to);
    for (unsigned i = 0; i < 3; i++) t[i] = (Tensor){4, 1, 2, i * 16u, 8, 1.f, i == 2 ? 2u : 1u, 0};
    Op *o = (Op *)(blob + h->oo);
    o->code = GUST_ADD; o->a = 0; o->b = 1; o->out = 2; o->lo = -128; o->hi = 127;
    h->crc = crc(blob, sizeof blob);
    void *arena = aligned_memory(h->arena, &arena_memory);
    assert(gust_init(c, blob, sizeof blob, arena, h->arena) == GUST_OK);
    parity(c, 0);
    assert(((int8_t *)gust_tensor(c, 2))[0] == 3);
    /* Negative weighted-pool inputs fail inside an operator, after OP_BEGIN. */
    t[1].c = 1; t[1].bytes = 4; t[2].h = 1; t[2].bytes = 2;
    o->code = GUST_WEIGHTED_POOL; o->bias = 240; o->mult = 248; o->shift = 256;
    ((int32_t *)(blob + 248))[0] = 1; ((int32_t *)(blob + 248))[1] = 1;
    h->crc = crc(blob, sizeof blob);
    assert(gust_init(c, blob, sizeof blob, arena, h->arena) == GUST_OK);
    seed(c);
    memset(gust_tensor(c, 1), 0xff, 4);
    sink failed = {.ctx=c};
    assert(gust_run_traced(c, 0, observe, &failed) == GUST_QUANTIZATION);
    complete(&failed, GUST_QUANTIZATION, 1);
    assert(failed.operator_status == GUST_QUANTIZATION);
    free(arena_memory);
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        assert(f && !fseek(f, 0, SEEK_END));
        const long size = ftell(f);
        assert(size > 64 && !fseek(f, 0, SEEK_SET));
        void *blob_memory;
        uint8_t *model = aligned_memory((size_t)size, &blob_memory);
        assert(fread(model, 1, (size_t)size, f) == (size_t)size);
        fclose(f);
        h = (Header *)model;
        arena = aligned_memory(h->arena, &arena_memory);
        assert(gust_init(c, model, (uint32_t)size, arena, h->arena) == GUST_OK);
        parity(c, 1);
        free(arena_memory); free(blob_memory);
    }
    free(context_memory);
    puts("trace: arena parity, event order/metadata, isolated sinks and refusal boundaries pass");
    return 0;
}
