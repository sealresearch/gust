/* The Relay host: model loading, the retrieval prefilter (port of web/prefilter.mjs), the ALL-CAPS fix, decoding and
   requireArgs (web/relay.mjs), behind the small API of relay.h. */
#include "relay.h"
#include "relay_host.h"
#include "gust.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- files */
#ifndef RELAY_NO_FILES
static char *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = len >= 0 ? rh_malloc((size_t)len + 1) : NULL;
    if (!b || fread(b, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(b); return NULL; }
    fclose(f);
    b[len] = 0;
    *n = (size_t)len;
    return b;
}
static char *join(const char *dir, const char *name) {
    const size_t a = strlen(dir), b = strlen(name);
    char *p = rh_malloc(a + b + 2);
    memcpy(p, dir, a);
    p[a] = '/';
    memcpy(p + a + 1, name, b + 1);
    return p;
}
#endif

/* ---------------------------------------------------------------- prefilter */
typedef struct { uint16_t *w; int o, n; } ten; /* original FP16 coefficients, [o, n] row-major (bias: n = 1) */
typedef struct pvec { int32_t *ids; int n; float *v; struct pvec *next; } pvec;
typedef struct {
    int dim;
    ten t0w, t0b, t2w, t2b, r0w, r0b, r2w, r2b;
    int k, minTools, budget, minKeep;
    pvec *cache[1024];
    size_t cached;
} prefilter;

static float half(uint16_t h) { /* exact FP16 values, including subnormals and signed zero; no FP16 ISA needed */
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 31u, m = h & 1023u, bits;
    if (e == 31u) bits = m ? 0x7fc00000u : sign | 0x7f800000u;
    else if (!e) {
        if (!m) bits = sign;
        else {
            e = 113u;
            while (!(m & 1024u)) { m <<= 1; e--; }
            bits = sign | (e << 23) | ((m & 1023u) << 13);
        }
    } else bits = sign | ((e + 112u) << 23) | (m << 13);
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static prefilter *prefilter_load(const unsigned char *b, size_t n, char *err, size_t errlen) {
    prefilter *p = rh_malloc(sizeof *p);
    memset(p, 0, sizeof *p);
    const uint32_t hl = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    rh_arena A = {0};
    const jnode *meta = hl + 4 <= n ? rh_json_parse(&A, (const char *)b + 4, hl) : NULL;
    const jnode *ts = rh_json_get(meta, "tensors");
    size_t off = 4 + hl;
    int ok = ts && ts->t == J_ARR;
    for (const jnode *t = ok ? ts->first : NULL; t && ok; t = t->next) {
        const jnode *nm = t->first, *sh = nm ? nm->next : NULL;
        if (!nm || nm->t != J_STR || !sh || sh->t != J_ARR) { ok = 0; break; }
        size_t len = 1;
        int dims[2] = {1, 1}, nd = 0;
        for (const jnode *d = sh->first; d; d = d->next, nd++) { len *= (size_t)d->num; if (nd < 2) dims[nd] = (int)d->num; }
        if (off + 2 * len > n) { ok = 0; break; }
        ten x = {rh_malloc(len * sizeof(uint16_t)), dims[0], nd > 1 ? dims[1] : 1};
        for (size_t i = 0; i < len; i++) x.w[i] = (uint16_t)(b[off + 2 * i] | b[off + 2 * i + 1] << 8);
        off += 2 * len;
        char name[64];
        snprintf(name, sizeof name, "%.*s", (int)nm->len, nm->s);
        ten *dst = !strcmp(name, "t.net.0.weight") ? &p->t0w : !strcmp(name, "t.net.0.bias") ? &p->t0b
                 : !strcmp(name, "t.net.2.weight") ? &p->t2w : !strcmp(name, "t.net.2.bias") ? &p->t2b
                 : !strcmp(name, "r.net.0.weight") ? &p->r0w : !strcmp(name, "r.net.0.bias") ? &p->r0b
                 : !strcmp(name, "r.net.2.weight") ? &p->r2w : !strcmp(name, "r.net.2.bias") ? &p->r2b : NULL;
        if (dst) *dst = x;
        else free(x.w);
    }
    rh_arena_free(&A);
    if (!ok || !p->t0w.w || !p->t2w.w || !p->r0w.w || !p->r2w.w || !p->t0b.w || !p->t2b.w || !p->r0b.w || !p->r2b.w) {
        snprintf(err, errlen, "relay-prefilter.bin: not a Relay prefilter head");
        free(p);
        return NULL;
    }
    p->dim = p->t2w.o;
    p->k = 10;
    p->minTools = 13;
    p->budget = 600;
    p->minKeep = 3;
    return p;
}
static void prefilter_free(prefilter *p) {
    if (!p) return;
    for (int i = 0; i < 1024; i++)
        for (pvec *e = p->cache[i], *x; e; e = x) { x = e->next; free(e->ids); free(e->v); free(e); }
    ten *t[] = {&p->t0w, &p->t0b, &p->t2w, &p->t2b, &p->r0w, &p->r0b, &p->r2w, &p->r2b};
    for (int i = 0; i < 8; i++) free(t[i]->w);
    free(p);
}

static void linear(const float *x, const ten *W, const ten *b, float *y) {
    for (int i = 0; i < W->o; i++) {
        double s = half(b->w[i]);
        const uint16_t *row = W->w + (size_t)i * W->n;
        for (int j = 0; j < W->n; j++) s += (double)half(row[j]) * (double)x[j];
        y[i] = (float)s;
    }
}
static double erf_as(double x) { /* Abramowitz-Stegun 7.1.26, as the JavaScript host */
    const double s = x > 0 ? 1 : x < 0 ? -1 : x, a = fabs(x), t = 1 / (1 + 0.3275911 * a);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
                        rh_js_exp(-a * a));
}
/* normalize(linear(gelu(linear(x)))) */
static void side(const prefilter *p, int tool, const float *x, float *out) {
    const ten *w0 = tool ? &p->t0w : &p->r0w, *b0 = tool ? &p->t0b : &p->r0b, *w2 = tool ? &p->t2w : &p->r2w,
              *b2 = tool ? &p->t2b : &p->r2b;
    float *h = rh_malloc((size_t)w0->o * sizeof(float) + 4);
    linear(x, w0, b0, h);
    for (int i = 0; i < w0->o; i++) { const double v = h[i]; h[i] = (float)(0.5 * v * (1 + erf_as(v / 1.4142135623730951))); }
    linear(h, w2, b2, out);
    double s = 0;
    for (int i = 0; i < w2->o; i++) s += (double)out[i] * (double)out[i];
    s = sqrt(s);
    if (s == 0 || s != s) s = 1;
    for (int i = 0; i < w2->o; i++) out[i] = (float)((double)out[i] / s);
    free(h);
}

struct relay {
    relay_options o;
    rh_tok *tok;
    rh_net *net;
    rh_gmodel *g[4];
    void *shared_memory, *shared_arena;
    size_t shared_bytes;
    prefilter *pre;
};

/* first-token state and mean state of one encoder pass */
static int feat(relay *r, const int32_t *ids, int n, float *x, rh_arena *A) {
    const float *H = rh_net_hidden(r->net, NULL, ids, n, A);
    if (!H) return 0;
    const int d = rh_net_d(r->net);
    float *mean = x + d;
    memcpy(x, H, (size_t)d * sizeof(float));
    memset(mean, 0, (size_t)d * sizeof(float));
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) mean[j] = (float)((double)mean[j] + (double)H[(size_t)i * d + j]);
    for (int j = 0; j < d; j++) mean[j] = (float)((double)mean[j] / n);
    return 1;
}

static uint64_t ids_hash(const int32_t *ids, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) h = (h ^ (uint32_t)ids[i]) * 1099511628211ull;
    return h;
}
/* the tool's vector (cached by its segment); NULL when the segment cannot be built */
static const float *tool_vector(relay *r, const jnode *tool, int *ntok) {
    /* Only the projected vector survives this tool. Do not retain every temporary encoder output until
       the entire request ends: a cold list of tool schemas can otherwise hold many full hidden states. */
    rh_arena scratch = {0};
    rh_arena *A = &scratch;
    int32_t *seg;
    int np, *pp, *no, **op;
    const int n = rh_tool_segment(r->tok, tool, A, &seg, &np, &pp, &no, &op);
    if (n < 0) { rh_arena_free(A); return NULL; }
    *ntok = n;
    const int m = n < RH_MAX_LEN / 2 ? n : RH_MAX_LEN / 2;
    const uint64_t h = ids_hash(seg, m);
    pvec **slot = &r->pre->cache[h & 1023];
    for (pvec *e = *slot; e; e = e->next)
        if (e->n == m && !memcmp(e->ids, seg, (size_t)m * sizeof(int32_t))) {
            sb *trace = rh_net_trace(r->net);
            rh_tracef(trace, "{\"event\":\"tool_vector\",\"tool\":");
            rh_trace_name(trace, rh_json_get(tool, "name"));
            rh_tracef(trace, ",\"tokens\":%d,\"encoded_tokens\":%d,\"cached\":true}\n", n, m);
            rh_arena_free(A);
            return e->v;
        }
    const int d = rh_net_d(r->net);
    sb *trace = rh_net_trace(r->net);
    rh_tracef(trace, "{\"event\":\"tool_vector\",\"tool\":");
    rh_trace_name(trace, rh_json_get(tool, "name"));
    rh_tracef(trace, ",\"tokens\":%d,\"encoded_tokens\":%d,\"cached\":false}\n", n, m);
    float *x = rh_arena_alloc(A, (size_t)2 * d * sizeof(float));
    if (!feat(r, seg, m, x, A)) { rh_arena_free(A); return NULL; }
    pvec *e = rh_malloc(sizeof *e);
    e->ids = rh_malloc((size_t)(m + 1) * sizeof(int32_t));
    memcpy(e->ids, seg, (size_t)m * sizeof(int32_t));
    e->n = m;
    e->v = rh_malloc((size_t)r->pre->dim * sizeof(float));
    side(r->pre, 1, x, e->v);
    e->next = *slot;
    *slot = e;
    rh_arena_free(A);
    return e->v;
}

/* Prefilter.filter: the tool list cut to the top k (and, with a budget, to the best tools that fit it); unchanged
   below minTools tools when within the budget; keeps the original order. Returns the kept count. */
static int prefilter_tools_scratch(relay *r, const jnode *rec, const uint32_t *text, size_t tn, const jnode **tools, int nt, rh_arena *A) {
    prefilter *p = r->pre;
    sb *trace = rh_net_trace(r->net);
    const jnode *kind = rh_json_get(rec, "kind");
    if (kind && kind->t == J_STR && kind->len == 7 && !memcmp(kind->s, "extract", 7)) {
        rh_tracef(trace, "{\"event\":\"prefilter_skip\",\"reason\":\"extract\"}\n");
        return nt;
    }
    int *ntok = rh_arena_alloc(A, (size_t)(nt + 1) * sizeof(int));
    const float **vec = rh_arena_alloc(A, (size_t)(nt + 1) * sizeof(float *));
    long total = 0;
    for (int i = 0; i < nt; i++) {
        int32_t *seg;
        int np, *pp, *no, **op;
        rh_arena scratch = {0};
        ntok[i] = rh_tool_segment(r->tok, tools[i], &scratch, &seg, &np, &pp, &no, &op);
        rh_arena_free(&scratch);
        if (ntok[i] < 0) return nt; /* the JavaScript host throws here */
        total += ntok[i];
    }
    const int over = p->budget > 0 && total > p->budget;
    if (nt < p->minTools && !over) {
        rh_tracef(trace, "{\"event\":\"prefilter_skip\",\"reason\":\"within_limits\",\"tool_tokens\":%ld}\n", total);
        return nt;
    }
    rh_tracef(trace, "{\"event\":\"prefilter_rank\",\"tools\":%d,\"tool_tokens\":%ld,\"k\":%d,\"budget\":%d}\n", nt, total, p->k, p->budget);
    /* rank: the request's vector against every tool's */
    int32_t *q;
    long(*offs)[2];
    const int nq = (int)rh_tok_request(r->tok, text, tn, RH_LIM_REQUEST, A, &q, &offs);
    int32_t *ids = rh_arena_alloc(A, (size_t)(nq + 2) * sizeof(int32_t));
    ids[0] = rh_tok_marker(r->tok, 0);
    memcpy(ids + 1, q, (size_t)nq * sizeof(int32_t));
    ids[nq + 1] = rh_tok_marker(r->tok, 1);
    const int d = rh_net_d(r->net);
    float *x = rh_arena_alloc(A, (size_t)2 * d * sizeof(float)), *rv_ = rh_arena_alloc(A, (size_t)p->dim * sizeof(float));
    if (!feat(r, ids, nq + 2, x, A)) return nt;
    side(p, 0, x, rv_);
    double *score = rh_arena_alloc(A, (size_t)(nt + 1) * sizeof(double));
    int *ord = rh_arena_alloc(A, (size_t)(nt + 1) * sizeof(int));
    for (int i = 0; i < nt; i++) {
        int k;
        vec[i] = tool_vector(r, tools[i], &k);
        if (!vec[i]) return nt;
        double s = 0;
        for (int j = 0; j < p->dim; j++) s += (double)vec[i][j] * (double)rv_[j];
        score[i] = s;
        int at = i; /* stable sort, score descending */
        while (at > 0 && score[ord[at - 1]] < s) { ord[at] = ord[at - 1]; at--; }
        ord[at] = i;
    }
    char *keep = rh_arena_alloc(A, (size_t)nt + 1);
    memset(keep, 0, (size_t)nt + 1);
    int kept = 0;
    long used = 0;
    const char *stop = "exhausted";
    for (int i = 0; i < nt; i++) {
        if (kept >= p->k) { stop = "k"; break; }
        const int t = ord[i];
        if (p->budget > 0 && kept >= p->minKeep && used + ntok[t] > p->budget) { stop = "budget"; break; }
        keep[t] = 1;
        kept++;
        used += ntok[t];
    }
    int w = 0;
    rh_tracef(trace, "{\"event\":\"prefilter_result\",\"kept\":%d,\"tool_tokens\":%ld,\"stop\":\"%s\"}\n", kept, used, stop);
    if (trace) for (int i = 0; i < nt; i++) {
        const int t = ord[i];
        rh_tracef(trace, "{\"event\":\"candidate\",\"tool\":");
        rh_trace_name(trace, rh_json_get(tools[t], "name"));
        rh_tracef(trace, ",\"rank\":%d,\"score\":%.17g,\"tokens\":%d,\"kept\":%s}\n", i + 1, score[t], ntok[t], keep[t] ? "true" : "false");
    }
    for (int i = 0; i < nt; i++) if (keep[i]) tools[w++] = tools[i];
    return w;
}

/* Ranking temporaries do not escape; tool pointers belong to the caller's parsed record. */
static int prefilter_tools(relay *r, const jnode *rec, const uint32_t *text, size_t tn, const jnode **tools, int nt) {
    rh_arena scratch = {0};
    const int kept = prefilter_tools_scratch(r, rec, text, tn, tools, nt, &scratch);
    rh_arena_free(&scratch);
    return kept;
}

/* ---------------------------------------------------------------- API */
void relay_default_options(relay_options *o) {
    memset(o, 0, sizeof *o);
    o->gate = 0;
    o->gate2 = 2;
    o->require = 1;
    o->fixes = 1;
    o->prefilter = 1;
    o->prefilter_k = 10;
    o->prefilter_budget = 600;
    o->threads = 1;
}

/* File-backed input buffers may be released as soon as their independent copy is built. */
static void release_source(char **owned, int index) {
    if (owned) { free(owned[index]); owned[index] = NULL; }
}

static relay *relay_open_impl(const relay_files *f, const relay_options *opt, char *err, size_t errlen, char **owned) {
    relay *r = rh_malloc(sizeof *r);
    memset(r, 0, sizeof *r);
    if (opt) r->o = *opt;
    else relay_default_options(&r->o);
    static const char *names[4] = {"relay.gust", "relay-route.gust", "relay-anchor.gust", "relay-args.gust"};
    const void *const bufs[4] = {f->encoder, f->route, f->anchor, f->args};
    const size_t lens[4] = {f->encoder_len, f->route_len, f->anchor_len, f->args_len};
    /* Stateful or unsupported graphs keep independent arenas. */
    size_t shared_bytes = 0;
    for (int i = 0; i < 4; i++) {
        const size_t required = rh_gmodel_share_bytes(bufs[i], lens[i]);
        if (!required) { shared_bytes = 0; break; }
        if (required > shared_bytes) shared_bytes = required;
    }
    if (shared_bytes) {
        r->shared_memory = rh_malloc(shared_bytes + 64);
        r->shared_arena = (void *)(((uintptr_t)r->shared_memory + 63u) & ~(uintptr_t)63u);
        r->shared_bytes = shared_bytes;
    }
    int fail = 0;
    for (int i = 0; i < 4 && !fail; i++) {
        if (!bufs[i]) { snprintf(err, errlen, "%s is missing", names[i]); fail = 1; break; }
        int e;
        if (!(r->g[i] = rh_gmodel_load_shared(bufs[i], lens[i], &e, r->shared_arena, r->shared_bytes))) { snprintf(err, errlen, "%s: Gust error %d", names[i], e); fail = 1; }
        release_source(owned, i);
    }
    if (!fail) {
        r->net = rh_net_open(r->g[0], r->g[1], r->g[2], r->g[3], err, errlen);
        fail = !r->net;
    }
    if (!fail) {
        if (!f->tokenizer) { snprintf(err, errlen, "no tokenizer"); fail = 1; }
        else if (!(r->tok = rh_tok_open(f->tokenizer, f->tokenizer_len))) { snprintf(err, errlen, "invalid tokenizer"); fail = 1; }
    }
    release_source(owned, 4);
    if (!fail && r->o.prefilter && f->prefilter) {
        r->pre = prefilter_load(f->prefilter, f->prefilter_len, err, errlen);
        if (!r->pre) fail = 1;
        else {
            r->pre->k = r->o.prefilter_k;
            r->pre->budget = r->o.prefilter_budget;
        }
    }
    release_source(owned, 5);
    if (fail) { relay_close(r); return NULL; }
    rh_set_fixes(r->o.fixes);
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
    if (r->o.threads > 1) gust_threads_start((uint32_t)r->o.threads);
#endif
    return r;
}

relay *relay_open_mem(const relay_files *f, const relay_options *opt, char *err, size_t errlen) {
    return relay_open_impl(f, opt, err, errlen, NULL);
}

#ifndef RELAY_NO_FILES
relay *relay_open(const char *dir, const relay_options *opt, char *err, size_t errlen) {
    static const char *names[6] = {"relay.gust", "relay-route.gust", "relay-anchor.gust", "relay-args.gust",
                                   "relay-tokenizer.json", "relay-prefilter.bin"};
    char *buf[6] = {0};
    size_t len[6] = {0};
    for (int i = 0; i < 6; i++) {
        char *p = i == 4 && opt && opt->tokenizer ? NULL : join(dir, names[i]);
        buf[i] = read_file(p ? p : opt->tokenizer, &len[i]);
        free(p);
    }
    relay *r = NULL;
    if (!buf[0] || !buf[1] || !buf[2] || !buf[3])
        snprintf(err, errlen, "%s: relay.gust, relay-route.gust, relay-anchor.gust and relay-args.gust are needed", dir);
    else if (!buf[4]) snprintf(err, errlen, "cannot read the tokenizer %s", opt && opt->tokenizer ? opt->tokenizer : "relay-tokenizer.json");
    else {
        relay_files f = {buf[0], len[0], buf[1], len[1], buf[2], len[2], buf[3], len[3], buf[4], len[4], buf[5], len[5]};
        r = relay_open_impl(&f, opt, err, errlen, buf);
    }
    for (int i = 0; i < 6; i++) free(buf[i]);
    return r;
}
#endif

void relay_close(relay *r) {
    if (!r) return;
    prefilter_free(r->pre);
    rh_tok_close(r->tok);
    rh_net_close(r->net);
    for (int i = 0; i < 4; i++) rh_gmodel_free(r->g[i]);
    free(r->shared_memory);
    free(r);
}
void relay_free(void *p) { free(p); }
size_t relay_model_bytes(const relay *r) {
    size_t n = r->shared_bytes;
    for (int i = 0; i < 4; i++) if (r->g[i]) n += rh_gmodel_bytes(r->g[i]);
    return n;
}

/* calls of one parsed record, appended to `out` as JSON (gauge_run form; explain: relay_explain's form) */
static void record_json(relay *r, const jnode *rec, int explain, sb *out) {
    rh_arena A = {0};
    rh_set_fixes(r->o.fixes);
    const jnode *tx = rh_json_get(rec, "text"), *tl = rh_json_get(rec, "tools");
    rh_pred pred = {0};
    if (tx && tx->t == J_STR) {
        uint32_t *text;
        size_t tn = rh_utf8_decode(tx->s, tx->len, &text);
        int nt = tl && tl->t == J_ARR ? (int)tl->count : 0;
        const jnode **tools = rh_arena_alloc(&A, (size_t)(nt + 1) * sizeof(jnode *));
        nt = 0;
        if (tl && tl->t == J_ARR) for (const jnode *t = tl->first; t; t = t->next) tools[nt++] = t;
        sb *trace = rh_net_trace(r->net);
        if (trace) rh_tracef(trace, "{\"event\":\"runtime\",\"isa\":\"%s\",\"graph_bytes\":%zu}\n", gust_variant(), relay_model_bytes(r));
        rh_tracef(trace, "{\"event\":\"request\",\"tools\":%d,\"prefilter\":%s,\"gate\":%.17g,\"gate2\":%.17g,\"require\":%s,\"fixes\":%s}\n",
                  nt, r->pre ? "true" : "false", r->o.gate, r->o.gate2, r->o.require ? "true" : "false", r->o.fixes ? "true" : "false");
        if (r->pre) nt = prefilter_tools(r, rec, text, tn, tools, nt);
        rh_tracef(trace, "{\"event\":\"selected_tools\",\"tools\":[");
        if (trace) for (int i = 0; i < nt; i++) {
            if (i) sb_str(trace, ",");
            rh_trace_name(trace, rh_json_get(tools[i], "name"));
        }
        rh_tracef(trace, "]}\n");
        if (FIX.caps) { /* ALL-CAPS requests -> lower case (the cased tokenizer splits capitals into fragments) */
            size_t letters = 0, upper = 0;
            for (size_t i = 0; i < tn; i++) {
                if (!(uc_flags(text[i]) & RH_UC_L)) continue;
                letters++;
                uint32_t *l;
                const size_t ln = uc_lower_str(text + i, 1, &l);
                upper += !(ln == 1 && l[0] == text[i]);
                free(l);
            }
            if (letters >= 6 && (double)upper / (double)letters > 0.7) {
                rh_tracef(trace, "{\"event\":\"text_fix\",\"name\":\"lowercase_all_caps\"}\n");
                uint32_t *l;
                const size_t ln = uc_lower_str(text, tn, &l);
                free(text);
                text = l;
                tn = ln;
            }
        }
        rh_tok_set_trace(r->tok, trace);
        rh_predict_record(r->net, r->tok, rec, text, tn, tools, nt, r->o.gate, r->o.gate2, explain, &A, &pred);
        rh_tok_set_trace(r->tok, NULL);
        rh_pred all = pred;
        if (explain) {
            all.calls = rh_arena_alloc(&A, ((size_t)pred.n + 1) * sizeof(rh_call));
            memcpy(all.calls, pred.calls, (size_t)pred.n * sizeof(rh_call));
        }
        if (r->o.require) rh_require_args(&pred, tools, nt);
        rh_tracef(trace, "{\"event\":\"require_args\",\"enabled\":%s,\"before\":%d,\"after\":%d}\n", r->o.require ? "true" : "false", all.n, pred.n);
        if (explain) rh_dump_explain(out, &pred, &all, text, tn);
        else rh_dump_calls(out, &pred);
        free(text);
    } else sb_str(out, explain ? "{\"calls\": [], \"dropped\": [], \"chunks\": []}" : "[]");
    rh_arena_free(&A);
}
int rh_relay_record(relay *r, const jnode *rec, sb *out) {
    record_json(r, rec, 0, out);
    return 1;
}

static char *call_json(relay *r, const char *record, size_t len, int explain) {
    if (len == (size_t)-1) len = strlen(record);
    rh_arena A = {0};
    const jnode *rec = rh_json_parse(&A, record, len);
    if (!rec) { rh_arena_free(&A); return NULL; }
    sb out = {0};
    record_json(r, rec, explain, &out);
    rh_arena_free(&A);
    if (!out.s) sb_str(&out, "");
    return out.s;
}
char *relay_call(relay *r, const char *record, size_t len) { return call_json(r, record, len, 0); }
char *relay_explain(relay *r, const char *record, size_t len) { return call_json(r, record, len, 1); }
static char *trace_json(relay *r, const char *record, size_t len, int operators) {
    sb events = {0}, out = {0};
    rh_net_set_trace(r->net, &events, operators);
    char *result = call_json(r, record, len, 1);
    rh_net_set_trace(r->net, NULL, 0);
    rh_tok_set_trace(r->tok, NULL);
    if (result) {
        sb_str(&out, "{\"events\":[");
        if (events.s) sb_put(&out, events.s, events.n);
        sb_str(&out, "],\"result\":");
        sb_str(&out, result);
        sb_str(&out, "}");
    }
    free(result);
    sb_free(&events);
    return out.s;
}

char *relay_trace(relay *r, const char *record, size_t len) { return trace_json(r, record, len, 0); }
char *relay_trace_ops(relay *r, const char *record, size_t len) { return trace_json(r, record, len, 1); }
