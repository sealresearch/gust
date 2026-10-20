/* The Gust graphs of Relay (encoder, route, anchor, args) and the host's key . query
   dot products (port of the GustNet class of web/relay.mjs). Head values are float32; the arithmetic the JavaScript
   host does on them (float64 sums of float32 products, Math.fround) is reproduced exactly. */
#include "relay_host.h"
#include "gust.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Row capacities of the head graphs (relay2/gust_build.py); larger sets are split into several runs. */
enum { TOOLS = 128, INSTS = 32, ROWS = 32, OPTS = 64 };

/* Internal JSON event writer: a newline ends an event; the next write adds its array separator. */
void rh_tracef(sb *trace, const char *format, ...) {
    if (!trace) return;
    va_list ap;
    va_start(ap, format);
    va_list copy;
    va_copy(copy, ap);
    const int n = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (n >= 0) {
        char local[256], *s = (size_t)n < sizeof local ? local : rh_malloc((size_t)n + 1);
        vsnprintf(s, (size_t)n + 1, format, ap);
        if (trace->n && trace->s[trace->n - 1] == '\n') sb_str(trace, ",");
        sb_put(trace, s, (size_t)n);
        if (s != local) free(s);
    }
    va_end(ap);
}
void rh_trace_name(sb *trace, const jnode *name) {
    if (trace) { rv v = rv_json(name); rh_js_stringify(trace, &v); }
}

/* ---------------------------------------------------------------- one Gust graph */
struct rh_gmodel {
    void *mem; /* owns the blob, context and, unless borrowed, arena */
    uint8_t *blob;
    void *ctx, *arena;
    size_t bytes;
    uint32_t nt;
    rh_gtensor *t;
    int *in;
    int nin;
    rh_arena names;
};

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static size_t up64(size_t n) { return (n + 63u) & ~(size_t)63u; }

/* Runs of these graphs are serialized, and the host copies every live output before the next graph. */
size_t rh_gmodel_share_bytes(const void *bytes, size_t n) {
    const uint8_t *b = bytes;
    if (!b || n < 68 || n > 0x7fffffffu || rd32(b + 52) != 1u) return 0;
    const uint32_t nt = rd32(b + 20), no = rd32(b + 24), arena = rd32(b + 40);
    if (!nt || nt > 4096 || !no || no > 4096 || !arena || arena > 64u * 1024u * 1024u) return 0;
    if (rd32(b + 28) != 64 || rd32(b + 32) != 64ull + 32ull * nt ||
        64ull + 32ull * nt + 80ull * no > n) return 0;
    for (uint32_t i = 0; i < nt; i++) if (rd32(b + 64 + 32 * i + 24) & 8u) return 0;
    for (uint32_t i = 0; i < no; i++) {
        const uint32_t code = rd32(b + rd32(b + 32) + 80 * i);
        if (code < 32u || code > 41u) return 0;
    }
    /* This is eligibility only. gust_init still validates the entire artifact and arena independently. */
    return up64(arena);
}
rh_gmodel *rh_gmodel_load_shared(const void *bytes, size_t n, int *err, void *shared, size_t shared_bytes) {
    *err = GUST_FORMAT;
    if (n < 68 || n > 0x7fffffffu) return NULL;
    const uint8_t *b = bytes;
    const uint32_t arena = rd32(b + 40), nt = rd32(b + 20);
    if (!arena) return NULL;
    rh_gmodel *m = rh_malloc(sizeof *m);
    memset(m, 0, sizeof *m);
    const size_t cb = up64(gust_context_bytes()), bb = up64(n), ab = up64(arena);
    if (shared && (shared_bytes < ab || !rh_gmodel_share_bytes(bytes, n))) { free(m); return NULL; }
    m->bytes = bb + cb + (shared ? 0 : ab);
    m->mem = rh_malloc(m->bytes + 64);
    uint8_t *base = (uint8_t *)(((uintptr_t)m->mem + 63u) & ~(uintptr_t)63u);
    m->blob = base;
    m->ctx = base + bb;
    m->arena = shared ? shared : base + bb + cb;
    memcpy(m->blob, b, n);
    const int s = gust_init(m->ctx, m->blob, (uint32_t)n, m->arena, (uint32_t)ab);
    if (s) { *err = s; rh_gmodel_free(m); return NULL; }
    /* manifest JSON before the trailing length: tensor names */
    const uint32_t ml = rd32(b + n - 4);
    if (ml > n - 4) { rh_gmodel_free(m); return NULL; }
    const jnode *man = rh_json_parse(&m->names, (const char *)b + n - 4 - ml, ml);
    const jnode *ts = rh_json_get(man, "tensors");
    if (!ts || ts->t != J_ARR || ts->count != nt) { rh_gmodel_free(m); return NULL; }
    m->nt = nt;
    m->t = rh_malloc((nt + 1) * sizeof *m->t);
    m->in = rh_malloc((nt + 1) * sizeof(int));
    const jnode *e = ts->first;
    for (uint32_t i = 0; i < nt; i++, e = e->next) {
        const uint8_t *r = b + 64 + 32 * i;
        rh_gtensor *t = &m->t[i];
        const jnode *nm = rh_json_get(e, "name");
        if (!nm || nm->t != J_STR) { rh_gmodel_free(m); return NULL; }
        char *name = rh_arena_alloc(&m->names, nm->len + 1);
        memcpy(name, nm->s, nm->len);
        name[nm->len] = 0;
        t->name = name;
        t->h = rd32(r);
        t->w = rd32(r + 4);
        t->c = rd32(r + 8);
        t->bytes = rd32(r + 16);
        t->flags = rd32(r + 24);
        t->p = gust_tensor(m->ctx, i);
        if (t->flags & 1u) m->in[m->nin++] = (int)i;
    }
    *err = 0;
    return m;
}
rh_gmodel *rh_gmodel_load(const void *bytes, size_t n, int *err) {
    return rh_gmodel_load_shared(bytes, n, err, NULL, 0);
}
void rh_gmodel_free(rh_gmodel *m) {
    if (!m) return;
    free(m->mem);
    free(m->t);
    free(m->in);
    rh_arena_free(&m->names);
    free(m);
}
size_t rh_gmodel_bytes(const rh_gmodel *m) { return m->bytes; }
rh_gtensor *rh_gm_tensor(rh_gmodel *m, const char *name) {
    for (uint32_t i = 0; i < m->nt; i++) if (!strcmp(m->t[i].name, name)) return &m->t[i];
    return NULL;
}

typedef struct { const char *name; const float *v; size_t n; } feed;
/* Only the independent-row request heads use per-input lengths. GroupNorm couples rows; dynamic/count ops
   have their own row semantics and retain the original execution path. The validated operator table is immutable. */
static int head_rows_independent(const rh_gmodel *m) {
    if (m->nin < 2 || m->nin > 6 || strcmp(m->t[0].name, "req")) return 0;
    const uint32_t no = rd32(m->blob + 24), off = rd32(m->blob + 32);
    for (uint32_t i = 0; i < no; i++) {
        const uint8_t *op = m->blob + off + (size_t)i * 80u;
        const uint32_t code = rd32(op);
        if (code > 41u || (code == 34u && (rd32(op + 64) & 1u))) return 0;
    }
    return 1;
}
/* _run: every input zeroed, then the fed ones written (truncated to the tensor); `valid` real rows of tensor 0 */
static int run(rh_gmodel *m, const feed *f, int nf, int valid, gust_trace_callback observer, void *opaque) {
    if (valid < 0 || (uint32_t)valid > m->t[0].h) return GUST_SHAPE;
    const int narrow = head_rows_independent(m);
    uint32_t lengths[6];
    for (int i = 0; i < m->nin; i++) {
        rh_gtensor *t = &m->t[m->in[i]];
        if (narrow) lengths[i] = t->h;
        memset(t->p, 0, t->bytes);
        for (int k = 0; k < nf; k++) {
            if (strcmp(f[k].name, t->name)) continue;
            const size_t cap = t->bytes / 4;
            const size_t count = f[k].n < cap ? f[k].n : cap;
            memcpy(t->p, f[k].v, count * sizeof(float));
            if (narrow) lengths[i] = count ? (uint32_t)((count + t->c - 1u) / t->c) : 1u;
        }
    }
    if (narrow) {
        lengths[0] = valid ? (uint32_t)valid : m->t[0].h;
        return observer ? gust_run_valid_inputs_traced(m->ctx, lengths, (uint32_t)m->nin, observer, opaque) :
                          gust_run_valid_inputs(m->ctx, lengths, (uint32_t)m->nin);
    }
    return observer ? gust_run_traced(m->ctx, (uint32_t)valid, observer, opaque) : gust_run_valid(m->ctx, (uint32_t)valid);
}
static const float *out_of(rh_gmodel *m, const char *name) {
    rh_gtensor *t = rh_gm_tensor(m, name);
    return t ? t->p : NULL;
}

/* ---------------------------------------------------------------- the net */
struct rh_net {
    rh_gmodel *enc, *route, *anchor, *args;
    int L, d, dk, K1, nOps, nAux;
    float scale;
    sb *trace; /* borrowed for one diagnostic call */
    int trace_ops;
};
void rh_net_set_trace(rh_net *n, sb *trace, int operators) { n->trace = trace; n->trace_ops = operators; }
sb *rh_net_trace(const rh_net *n) { return n->trace; }

/* Relay adds graph labels/JSON to model-independent Gust events. */
typedef struct { rh_net *net; const char *name; int valid; } relay_graph_trace;
static void graph_event(void *opaque, const gust_trace_event *e) {
    const relay_graph_trace *g = opaque;
    sb *trace = g->net->trace;
    if (e->kind == GUST_TRACE_RUN_BEGIN || e->kind == GUST_TRACE_RUN_END) {
        rh_tracef(trace, "{\"event\":\"%s\",\"name\":\"%s\",\"valid_rows\":%d,\"status\":%d}\n",
                  e->kind == GUST_TRACE_RUN_BEGIN ? "graph_begin" : "graph", g->name, g->valid, e->status);
    } else if (g->net->trace_ops) {
        rh_tracef(trace, "{\"event\":\"%s\",\"graph\":\"%s\",\"index\":%u,\"opcode\":%u,\"output\":%u,"
                  "\"rows\":%u,\"shape\":[%u,%u,%u],\"status\":%d}\n",
                  e->kind == GUST_TRACE_OP_BEGIN ? "operator_begin" : "operator_end", g->name,
                  e->operator_index, e->opcode, e->output_tensor, e->rows, e->height, e->width, e->channels, e->status);
    }
}
static int net_run(rh_net *n, rh_gmodel *m, const char *name, const feed *f, int nf, int valid) {
    relay_graph_trace trace = {n, name, valid};
    return run(m, f, nf, valid, n->trace ? graph_event : NULL, &trace);
}

static int shape(rh_gmodel *m, const char *name, int dim, int *out) {
    rh_gtensor *t = m ? rh_gm_tensor(m, name) : NULL;
    if (!t) return 0;
    *out = (int)(dim == 0 ? t->h : dim == 1 ? t->w : t->c);
    return 1;
}

rh_net *rh_net_open(rh_gmodel *encoder, rh_gmodel *route, rh_gmodel *anchor, rh_gmodel *args, char *err, size_t errlen) {
    rh_net *n = rh_malloc(sizeof *n);
    memset(n, 0, sizeof *n);
    n->enc = encoder;
    n->route = route;
    n->anchor = anchor;
    n->args = args;
    int ok = encoder && route && anchor && args;
    ok = ok && shape(encoder, "tokens", 0, &n->L) && shape(encoder, "hidden", 2, &n->d) &&
         shape(anchor, "anchor_k", 2, &n->dk) && shape(route, "tool_out", 2, &n->K1) && shape(args, "op", 2, &n->nOps) &&
         shape(args, "aux", 2, &n->nAux);
    if (!ok) {
        snprintf(err, errlen, "the graphs lack an expected tensor (tokens, hidden, anchor_k, tool_out, op, aux)");
        free(n);
        return NULL;
    }
    if (n->nOps != RH_N_OPS) { snprintf(err, errlen, "args graph has %d ops, the host knows %d", n->nOps, RH_N_OPS); free(n); return NULL; }
    n->scale = (float)sqrt((double)n->dk); /* f32(dk ** 0.5) */
    return n;
}

void rh_net_close(rh_net *n) { free(n); }
int rh_net_d(const rh_net *n) { return n->d; }

const float *rh_net_hidden(rh_net *n, rh_example *ex, const int32_t *ids, int len, rh_arena *A) {
    if (ex) {
        if (ex->H) return ex->H;
        ids = ex->ids;
        len = ex->n;
    }
    if (len > n->L) return NULL; /* sequence exceeds the encoder */
    float *tok = rh_arena_alloc(A, (size_t)(len + 1) * sizeof(float));
    for (int i = 0; i < len; i++) tok[i] = (float)ids[i];
    const feed f = {"tokens", tok, (size_t)len};
    if (net_run(n, n->enc, "encoder", &f, 1, len)) return NULL;
    float *H = rh_arena_alloc(A, (size_t)len * n->d * sizeof(float) + 4);
    memcpy(H, out_of(n->enc, "hidden"), (size_t)len * n->d * sizeof(float));
    if (ex) ex->H = H;
    return H;
}

static double dot(const float *a, const float *b, int n) {
    double s = 0; /* float64 accumulation of exact float32 products */
    for (int i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}
static float scaled(const rh_net *n, double x) { return (float)((double)(float)x / (double)n->scale); }

static float *rows_of(const float *H, int d, const int *pos, int np, rh_arena *A) {
    float *o = rh_arena_alloc(A, (size_t)(np ? np : 1) * d * sizeof(float) + 4);
    for (int i = 0; i < np; i++) memcpy(o + (size_t)i * d, H + (size_t)pos[i] * d, (size_t)d * sizeof(float));
    return o;
}

int rh_net_route(rh_net *n, rh_example *ex, float *tool, float **count, int *ncount, float *none, rh_arena *A) {
    const float *H = rh_net_hidden(n, ex, NULL, 0, A);
    if (!H) return 0;
    const int d = n->d, K1 = n->K1, nt = ex->ntools;
    *ncount = K1 - 1;
    for (int s = 0; s < (nt > 1 ? nt : 1); s += TOOLS) {
        const int m = nt - s < TOOLS ? nt - s : TOOLS;
        feed f[2];
        f[1] = (feed){"cls", H, (size_t)d};
        if (m > 0) {
            int pos[TOOLS];
            for (int i = 0; i < m; i++) pos[i] = ex->tools[s + i].pos;
            f[0] = (feed){"tools", rows_of(H, d, pos, m, A), (size_t)m * d};
            if (net_run(n, n->route, "route", f, 2, m)) return 0;
        } else {
            float *z = rh_arena_alloc(A, (size_t)d * sizeof(float));
            memset(z, 0, (size_t)d * sizeof(float));
            f[0] = (feed){"tools", z, (size_t)d};
            if (net_run(n, n->route, "route", f, 2, 1)) return 0;
        }
        const float *to = out_of(n->route, "tool_out");
        for (int i = 0; i < m; i++) {
            tool[s + i] = to[i * K1];
            count[s + i] = rh_arena_alloc(A, (size_t)K1 * sizeof(float));
            memcpy(count[s + i], to + i * K1 + 1, (size_t)(K1 - 1) * sizeof(float));
        }
        *none = out_of(n->route, "none")[0];
    }
    return 1;
}

int rh_net_anchors(rh_net *n, rh_example *ex, const int (*insts)[2], int ninst, float *out, rh_arena *A) {
    const float *H = rh_net_hidden(n, ex, NULL, 0, A);
    if (!H) return 0;
    const int d = n->d, dk = n->dk, nq = ex->nq;
    const float *req = H + (size_t)ex->q0 * d;
    for (int s = 0; s < ninst; s += INSTS) {
        const int m = ninst - s < INSTS ? ninst - s : INSTS;
        int pos[INSTS];
        float slot[INSTS];
        for (int i = 0; i < m; i++) { pos[i] = ex->tools[insts[s + i][0]].pos; slot[i] = (float)insts[s + i][1]; }
        const feed f[3] = {{"req", req, (size_t)nq * d}, {"inst", rows_of(H, d, pos, m, A), (size_t)m * d}, {"inst_slot", slot, (size_t)m}};
        if (net_run(n, n->anchor, "anchor", f, 3, nq)) return 0;
        const float *aq = out_of(n->anchor, "anchor_q"), *ak = out_of(n->anchor, "anchor_k");
        for (int i = 0; i < m; i++)
            for (int q = 0; q < nq; q++) out[(size_t)(s + i) * nq + q] = scaled(n, dot(aq + (size_t)i * dk, ak + (size_t)q * dk, dk));
    }
    return 1;
}

/* ---------------------------------------------------------------- args */
typedef struct { const float *op, *aux, *sq, *tq, *oq, *nul; int p; float *spans, *tags, *en; } argrow;
struct rh_args {
    rh_net *n;
    rh_arena *A;
    argrow *r;
    int nrows, nq, E;
    float *sk, *tk;
    const float **ok;
    int *nopt, *ostart;
};

static float *copy_out(rh_gmodel *m, const char *name, size_t cnt, rh_arena *A) {
    float *c = rh_arena_alloc(A, cnt * sizeof(float) + 4);
    memcpy(c, out_of(m, name), cnt * sizeof(float));
    return c;
}

rh_args *rh_net_args(rh_net *n, rh_example *ex, const rh_rowreq *rows, int nrows, rh_arena *A) {
    const float *H = rh_net_hidden(n, ex, NULL, 0, A);
    if (!H) return NULL;
    const int d = n->d, dk = n->dk, nq = ex->nq;
    const float *req = H + (size_t)ex->q0 * d;
    /* global parameter / option numbering of the example (collate order) */
    int np = 0, no = 0;
    for (int t = 0; t < ex->ntools; t++) { np += ex->tools[t].nparams; for (int j = 0; j < ex->tools[t].nparams; j++) no += ex->tools[t].params[j].nopts; }
    int *pbase = rh_arena_alloc(A, (size_t)(ex->ntools + 1) * sizeof(int)), *nopt = rh_arena_alloc(A, (size_t)(np + 1) * sizeof(int)),
        *ostart = rh_arena_alloc(A, (size_t)(np + 1) * sizeof(int)), *optPos = rh_arena_alloc(A, (size_t)(no + 1) * sizeof(int));
    int pi = 0, oi = 0, E = 1;
    for (int t = 0; t < ex->ntools; t++) {
        pbase[t] = pi;
        for (int j = 0; j < ex->tools[t].nparams; j++, pi++) {
            const rh_pslot *ps = &ex->tools[t].params[j];
            ostart[pi] = oi;
            nopt[pi] = ps->nopts;
            if (ps->nopts > E) E = ps->nopts;
            for (int o = 0; o < ps->nopts; o++) optPos[oi++] = ps->opts[o];
        }
    }
    int *prow = rh_arena_alloc(A, (size_t)(nrows + 1) * sizeof(int));
    char *seen = rh_arena_alloc(A, (size_t)np + 1);
    memset(seen, 0, (size_t)np + 1);
    for (int i = 0; i < nrows; i++) { prow[i] = pbase[rows[i].ti] + rows[i].pj; seen[prow[i]] = 1; }
    int *opts = rh_arena_alloc(A, (size_t)(no + 1) * sizeof(int)), nop = 0;
    for (int p = 0; p < np; p++) if (seen[p]) for (int o = ostart[p]; o < ostart[p] + nopt[p]; o++) opts[nop++] = o;
    rh_args *a = rh_arena_alloc(A, sizeof *a);
    memset(a, 0, sizeof *a);
    a->n = n;
    a->A = A;
    a->nrows = nrows;
    a->nq = nq;
    a->E = E;
    a->nopt = nopt;
    a->ostart = ostart;
    a->ok = rh_arena_alloc(A, (size_t)(no + 1) * sizeof(float *));
    a->r = rh_arena_alloc(A, (size_t)(nrows + 1) * sizeof(argrow));
    memset(a->r, 0, (size_t)(nrows + 1) * sizeof(argrow));
    float *zero = rh_arena_alloc(A, (size_t)d * sizeof(float));
    memset(zero, 0, (size_t)d * sizeof(float));
    for (int s = 0; s < nrows; s += ROWS) {
        const int m = nrows - s < ROWS ? nrows - s : ROWS, first = s == 0;
        float *ha = rh_arena_alloc(A, (size_t)m * d * sizeof(float));
        memset(ha, 0, (size_t)m * d * sizeof(float));
        int hp[ROWS], ht[ROWS];
        float slot[ROWS];
        for (int i = 0; i < m; i++) {
            const rh_rowreq *r = &rows[s + i];
            if (r->anchor >= 0) memcpy(ha + (size_t)i * d, H + (size_t)(ex->q0 + r->anchor) * d, (size_t)d * sizeof(float));
            hp[i] = ex->tools[r->ti].params[r->pj].pos;
            ht[i] = ex->tools[r->ti].pos;
            slot[i] = (float)r->k;
        }
        int c0 = nop < OPTS ? nop : OPTS, opos[OPTS];
        for (int i = 0; i < c0; i++) opos[i] = optPos[opts[i]];
        feed f[6] = {{"req", req, (size_t)nq * d}, {"hp", rows_of(H, d, hp, m, A), (size_t)m * d},
                     {"ht", rows_of(H, d, ht, m, A), (size_t)m * d}, {"slot", slot, (size_t)m}, {"ha", ha, (size_t)m * d},
                     {"opt", first && nop ? rows_of(H, d, opos, c0, A) : zero, first && nop ? (size_t)c0 * d : (size_t)d}};
        if (net_run(n, n->args, "args", f, 6, nq)) return NULL;
#define COPY(name, cnt) copy_out(n->args, name, (size_t)(cnt), A)
        if (first) {
            a->sk = COPY("span_k", (size_t)nq * 4 * dk);
            a->tk = COPY("tag_k", (size_t)nq * 3 * dk);
            if (nop) {
                float *okc = COPY("opt_k", (size_t)c0 * dk);
                for (int i = 0; i < c0; i++) a->ok[opts[i]] = okc + (size_t)i * dk;
            }
        }
        float *op = COPY("op", (size_t)m * n->nOps), *aux = COPY("aux", (size_t)m * n->nAux), *nul = COPY("null", (size_t)m * 4),
              *sq = COPY("span_q", (size_t)m * 4 * dk), *tq = COPY("tag_q", (size_t)m * 3 * dk), *oq = COPY("opt_q", (size_t)m * dk);
        if (first)
            for (int k = OPTS; k < nop; k += OPTS) {
                const int c = nop - k < OPTS ? nop - k : OPTS;
                for (int i = 0; i < c; i++) opos[i] = optPos[opts[k + i]];
                f[5] = (feed){"opt", rows_of(H, d, opos, c, A), (size_t)c * d};
                if (net_run(n, n->args, "args", f, 6, nq)) return NULL;
                float *okc = COPY("opt_k", (size_t)c * dk);
                for (int i = 0; i < c; i++) a->ok[opts[k + i]] = okc + (size_t)i * dk;
            }
#undef COPY
        for (int i = 0; i < m; i++) {
            argrow *r = &a->r[s + i];
            r->op = op + (size_t)i * n->nOps;
            r->aux = aux + (size_t)i * n->nAux;
            r->nul = nul + (size_t)i * 4;
            r->sq = sq + (size_t)i * 4 * dk;
            r->tq = tq + (size_t)i * 3 * dk;
            r->oq = oq + (size_t)i * dk;
            r->p = prow[s + i];
        }
    }
    return a;
}

const float *rh_args_op(rh_args *a, int i) { return a->r[i].op; }
const float *rh_args_aux(rh_args *a, int i) { return a->r[i].aux; }
int rh_args_naux(rh_args *a) { return a->n->nAux; }

const float *rh_args_spans(rh_args *a, int i) {
    argrow *r = &a->r[i];
    if (!r->spans) {
        const int nq = a->nq, dk = a->n->dk;
        r->spans = rh_arena_alloc(a->A, (size_t)4 * (nq + 1) * sizeof(float));
        for (int k = 0; k < 4; k++) {
            float *v = r->spans + (size_t)k * (nq + 1);
            for (int q = 0; q < nq; q++) v[q] = scaled(a->n, dot(a->sk + (size_t)(q * 4 + k) * dk, r->sq + (size_t)k * dk, dk));
            v[nq] = r->nul[k];
        }
    }
    return r->spans;
}
const float *rh_args_tags(rh_args *a, int i) {
    argrow *r = &a->r[i];
    if (!r->tags) {
        const int nq = a->nq, dk = a->n->dk;
        r->tags = rh_arena_alloc(a->A, (size_t)3 * (nq + 1) * sizeof(float));
        for (int q = 0; q < nq; q++)
            for (int c = 0; c < 3; c++) r->tags[q * 3 + c] = scaled(a->n, dot(a->tk + (size_t)(q * 3 + c) * dk, r->tq + (size_t)c * dk, dk));
    }
    return r->tags;
}
const float *rh_args_enum(rh_args *a, int i, int *E) {
    argrow *r = &a->r[i];
    *E = a->E;
    if (!r->en) {
        const int dk = a->n->dk;
        r->en = rh_arena_alloc(a->A, (size_t)a->E * sizeof(float));
        for (int e = 0; e < a->E; e++) r->en[e] = -1e4f;
        for (int e = 0; e < a->nopt[r->p]; e++) r->en[e] = scaled(a->n, dot(a->ok[a->ostart[r->p] + e], r->oq, dk));
    }
    return r->en;
}
