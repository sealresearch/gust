/* Relay-2 inputs (port of web/features.mjs): one joint sequence per example,
       [CLS] request [SEP] [T] tool text [P] param text [O] option ... [P] ... [T] ... [SEP]
   with long tool lists split into several sequences (eval_examples). Request offsets are code point indices. */
#include "relay_host.h"
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- tokenizer with an encoding cache */
typedef struct { uint64_t h; uint32_t n; char *text; size_t len; int32_t *ids; size_t nids; } centry;
struct rh_tok {
    rh_tokenizer *tk;
    int32_t mark[6];
    centry *tab; /* open addressing, cap a power of two */
    size_t cap, used;
    rh_ivec ids, offs;
    sb *trace;
    const jnode *trace_tool, *trace_param;
    const char *trace_scope;
    int trace_request_done;
};
void rh_tok_set_trace(rh_tok *t, sb *trace) { t->trace = trace; t->trace_request_done = 0; }

rh_tok *rh_tok_open(const char *json, size_t len) {
    rh_tokenizer *tk = rh_tok_load(json, len);
    if (!tk) return NULL;
    rh_tok *t = rh_malloc(sizeof *t);
    memset(t, 0, sizeof *t);
    t->tk = tk;
    static const char *names[6] = {"[CLS]", "[SEP]", "[PAD]", "[unused0]", "[unused1]", "[unused2]"};
    for (int i = 0; i < 6; i++) t->mark[i] = rh_tok_id(tk, names[i]);
    return t;
}
static void cache_clear(rh_tok *t) {
    for (size_t i = 0; i < t->cap; i++) if (t->tab[i].text) { free(t->tab[i].text); free(t->tab[i].ids); }
    free(t->tab);
    t->tab = NULL;
    t->cap = t->used = 0;
}
void rh_tok_close(rh_tok *t) {
    if (!t) return;
    cache_clear(t);
    rh_ivec_free(&t->ids);
    rh_ivec_free(&t->offs);
    rh_tok_free(t->tk);
    free(t);
}
int32_t rh_tok_marker(const rh_tok *t, int which) { return t->mark[which]; }

static uint64_t fnv(const char *s, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
    return h;
}
static void encode_utf8(rh_tok *t, const char *s, size_t len) {
    uint32_t *cps;
    const size_t n = rh_utf8_decode(s, len, &cps);
    t->ids.n = t->offs.n = 0;
    rh_tok_encode(t->tk, cps, n, &t->ids, &t->offs);
    free(cps);
}

size_t rh_tok_ids(rh_tok *t, const char *s, size_t len, size_t n, const int32_t **ids) {
    if (t->trace) {
        encode_utf8(t, s, len);
        if (t->ids.n > n) {
            rh_tracef(t->trace, "{\"event\":\"token_limit\",\"tool\":");
            rh_trace_name(t->trace, rh_json_get(t->trace_tool, "name"));
            rh_tracef(t->trace, ",\"param\":");
            rh_trace_name(t->trace, rh_json_get(t->trace_param, "name"));
            rh_tracef(t->trace, ",\"scope\":\"%s\",\"tokens\":%zu,\"kept\":%zu}\n", t->trace_scope, t->ids.n, n);
        }
    }
    const uint64_t h = fnv(s, len, 1469598103934665603ull ^ (uint64_t)n * 0x9e3779b97f4a7c15ull);
    if (t->cap) {
        for (size_t i = h & (t->cap - 1);; i = (i + 1) & (t->cap - 1)) {
            centry *e = &t->tab[i];
            if (!e->text) break;
            if (e->h == h && e->n == n && e->len == len && !memcmp(e->text, s, len)) { *ids = e->ids; return e->nids; }
        }
    }
    encode_utf8(t, s, len);
    const size_t k = t->ids.n < n ? t->ids.n : n;
    if (t->used >= 300000) cache_clear(t);
    if (2 * (t->used + 1) > t->cap) { /* grow (rehash) */
        const size_t cap = t->cap ? 2 * t->cap : 1024;
        centry *tab = rh_malloc(cap * sizeof *tab);
        memset(tab, 0, cap * sizeof *tab);
        for (size_t i = 0; i < t->cap; i++) {
            if (!t->tab[i].text) continue;
            size_t j = t->tab[i].h & (cap - 1);
            while (tab[j].text) j = (j + 1) & (cap - 1);
            tab[j] = t->tab[i];
        }
        free(t->tab);
        t->tab = tab;
        t->cap = cap;
    }
    size_t j = h & (t->cap - 1);
    while (t->tab[j].text) j = (j + 1) & (t->cap - 1);
    centry *e = &t->tab[j];
    e->h = h;
    e->n = (uint32_t)n;
    e->text = rh_malloc(len + 1);
    memcpy(e->text, s, len);
    e->len = len;
    e->ids = rh_malloc((k + 1) * sizeof(int32_t));
    if (k) memcpy(e->ids, t->ids.v, k * sizeof(int32_t));
    e->nids = k;
    t->used++;
    *ids = e->ids;
    return k;
}

size_t rh_tok_request(rh_tok *t, const uint32_t *cps, size_t ncp, size_t n, rh_arena *A, int32_t **ids, long (**offs)[2]) {
    t->ids.n = t->offs.n = 0;
    rh_tok_encode(t->tk, cps, ncp, &t->ids, &t->offs);
    const size_t k = t->ids.n < n ? t->ids.n : n;
    *ids = rh_arena_alloc(A, (k + 1) * sizeof(int32_t));
    if (t->trace && !t->trace_request_done) {
        rh_tracef(t->trace, "{\"event\":\"request_tokens\",\"tokens\":%zu,\"kept\":%zu,\"limit\":%zu}\n", t->ids.n, k, n);
        t->trace_request_done = 1;
    }
    *offs = rh_arena_alloc(A, (k + 1) * sizeof **offs);
    for (size_t i = 0; i < k; i++) {
        (*ids)[i] = t->ids.v[i];
        long a = t->offs.v[2 * i], b = t->offs.v[2 * i + 1];
        while (a < b && rh_py_isspace(cps[a])) a++;
        (*offs)[i][0] = a;
        (*offs)[i][1] = b;
    }
    return k;
}

/* ---------------------------------------------------------------- texts */
static void put_str(sb *b, const jnode *j) { rv v = rv_json(j); py_str(b, &v); }
static int py_truthy(const jnode *j) { rv v = rv_json(j); return rv_truthy(&v); }

/* humanize: playlist_owner -> 'playlist owner'; restaurantName -> 'restaurant name' */
static void humanize(sb *out, const jnode *name) {
    sb s = {0};
    put_str(&s, name);
    uint32_t *c;
    const size_t n = rh_utf8_decode(s.s ? s.s : "", s.n, &c);
    sb_free(&s);
    uint32_t *t = rh_malloc((2 * n + 1) * sizeof(uint32_t));
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t p = i ? c[i - 1] : 0;
        if (i && ((p >= 'a' && p <= 'z') || (p >= '0' && p <= '9')) && c[i] >= 'A' && c[i] <= 'Z') t[m++] = ' ';
        if (c[i] == '_' || c[i] == '-' || c[i] == '.') {
            if (!m || t[m - 1] != 0xFFFFFFFFu) t[m++] = 0xFFFFFFFFu; /* one space per run of separators */
            continue;
        }
        t[m++] = c[i];
    }
    for (size_t i = 0; i < m; i++) if (t[i] == 0xFFFFFFFFu) t[i] = ' ';
    size_t a = 0, b = m;
    while (a < b && rh_py_isspace(t[a])) a++;
    while (b > a && rh_py_isspace(t[b - 1])) b--;
    uint32_t *low;
    const size_t ln = uc_lower_str(t + a, b - a, &low);
    for (size_t i = 0; i < ln; i++) sb_cp(out, low[i]);
    free(low);
    free(t);
    free(c);
}

static void tool_text(sb *b, const jnode *t) {
    put_str(b, rh_json_get(t, "name"));
    sb_str(b, " (");
    humanize(b, rh_json_get(t, "name"));
    sb_str(b, ") : ");
    const jnode *d = rh_json_get(t, "description");
    if (py_truthy(d)) put_str(b, d);
}

static void param_text(sb *b, const jnode *p) {
    const jnode *raw = rh_json_get(p, "raw_type"), *ty = rh_json_get(p, "type");
    put_str(b, rh_json_get(p, "name"));
    sb_str(b, " (");
    if (py_truthy(raw)) put_str(b, raw);
    else if (ty) put_str(b, ty);
    else sb_str(b, "string");
    if (py_truthy(rh_json_get(p, "required"))) sb_str(b, ", required");
    sb_str(b, ") : ");
    const jnode *d = rh_json_get(p, "description");
    if (py_truthy(d)) put_str(b, d);
    const jnode *df = rh_json_get(p, "default");
    if (df) { sb_str(b, " default: "); put_str(b, df); }
    const jnode *it = rh_json_get(p, "items");
    if (ty && ty->t == J_STR && ty->len == 5 && !memcmp(ty->s, "array", 5) && py_truthy(it)) {
        sb_str(b, " items: ");
        const jnode *ir = rh_json_get(it, "raw_type");
        put_str(b, py_truthy(ir) ? ir : rh_json_get(it, "type"));
    }
}

/* options(p): [list, kind] */
int rh_options(const jnode *p, rh_arena *A, rv **out, int *is_enum) {
    const jnode *en = rh_json_get(p, "enum");
    if (py_truthy(en)) {
        *is_enum = 1;
        if (en->t == J_ARR) {
            rv *o = rh_arena_alloc(A, (en->count + 1) * sizeof(rv));
            int k = 0;
            for (const jnode *c = en->first; c; c = c->next, k++) {
                sb b = {0};
                put_str(&b, c);
                char *s = rh_arena_alloc(A, b.n + 1);
                memcpy(s, b.s ? b.s : "", b.n);
                s[b.n] = 0;
                o[k] = rv_str(s, b.n);
                sb_free(&b);
            }
            *out = o;
            return k;
        }
        if (en->t == J_STR) { /* characters */
            uint32_t *c;
            const size_t n = rh_utf8_decode(en->s, en->len, &c);
            rv *o = rh_arena_alloc(A, (n + 1) * sizeof(rv));
            for (size_t i = 0; i < n; i++) {
                char *s = rh_arena_alloc(A, 5);
                const size_t l = rh_utf8_put(s, c[i]);
                s[l] = 0;
                o[i] = rv_str(s, l);
            }
            free(c);
            *out = o;
            return (int)n;
        }
        if (en->t == J_OBJ) { /* keys */
            const jnode **m;
            const size_t n = rh_js_members(en, A, &m);
            rv *o = rh_arena_alloc(A, (n + 1) * sizeof(rv));
            for (size_t i = 0; i < n; i++) o[i] = rv_str(m[i]->key, m[i]->klen);
            *out = o;
            return (int)n;
        }
        return -1; /* not iterable */
    }
    *is_enum = 0;
    rv *o = rh_arena_alloc(A, 64 * sizeof(rv));
    *out = o;
    return (int)rh_desc_options(p, A, o);
}

/* ---------------------------------------------------------------- spans */
static const uint32_t OPEN[] = {'"', '\'', 0x201c, 0x2018, '(', '[', '{', '`'};
static const uint32_t CLOSE[] = {'"', '\'', 0x201d, 0x2019, ')', ']', '}', '`'};
static int idx_of(const uint32_t *set, uint32_t c) { for (int i = 0; i < 8; i++) if (set[i] == c) return i; return -1; }
static int within(const uint32_t *cps, uint32_t ch, long x, long y) { for (long i = x; i < y; i++) if (cps[i] == ch) return 1; return 0; }

void rh_trim(const uint32_t *cps, long *pa, long *pb) {
    long a = *pa, b = *pb;
    int k;
    while (a < b && (k = idx_of(OPEN, cps[a])) >= 0 && !within(cps, CLOSE[k], a + 1, b)) a++;
    while (b > a && (k = idx_of(CLOSE, cps[b - 1])) >= 0 && !within(cps, OPEN[k], a, b - 1)) b--;
    *pa = a;
    *pb = b;
}

/* ---------------------------------------------------------------- examples */
static void push_ids(rh_ivec *v, rh_tok *tk, sb *text, size_t n) {
    const int32_t *ids;
    const size_t k = rh_tok_ids(tk, text->s ? text->s : "", text->n, n, &ids);
    for (size_t i = 0; i < k; i++) rh_ivec_push(v, ids[i]);
}

int rh_tool_segment(rh_tok *tk, const jnode *tool, rh_arena *A, int32_t **seg, int *nparams, int **ppos, int **nopts, int ***opos) {
    if (tk->trace) { tk->trace_tool = tool; tk->trace_param = NULL; tk->trace_scope = "tool"; }
    if (!tool || tool->t == J_NULL) return -1;
    const jnode *ps = rh_json_get(tool, "params");
    if (!ps || ps->t != J_ARR) return -1;
    rh_ivec v = {0};
    rh_ivec_push(&v, tk->mark[3]);
    sb b = {0};
    tool_text(&b, tool);
    push_ids(&v, tk, &b, RH_LIM_TOOL);
    const int np = (int)(ps->count < RH_LIM_PARAMS ? ps->count : RH_LIM_PARAMS);
    *nparams = np;
    *ppos = rh_arena_alloc(A, (size_t)(np + 1) * sizeof(int));
    *nopts = rh_arena_alloc(A, (size_t)(np + 1) * sizeof(int));
    *opos = rh_arena_alloc(A, (size_t)(np + 1) * sizeof(int *));
    const jnode *p = ps->first;
    for (int j = 0; j < np; j++, p = p->next) {
        if (tk->trace) { tk->trace_param = p; tk->trace_scope = "param"; }
        if (p->t == J_NULL) { rh_ivec_free(&v); sb_free(&b); return -1; }
        (*ppos)[j] = (int)v.n;
        rh_ivec_push(&v, tk->mark[4]);
        b.n = 0;
        param_text(&b, p);
        push_ids(&v, tk, &b, RH_LIM_PARAM);
        rv *opts;
        int ke;
        const int no = rh_options(p, A, &opts, &ke);
        if (no < 0) { rh_ivec_free(&v); sb_free(&b); return -1; }
        const int k = no < RH_LIM_OPTIONS ? no : RH_LIM_OPTIONS;
        (*nopts)[j] = k;
        (*opos)[j] = rh_arena_alloc(A, (size_t)(k + 1) * sizeof(int));
        for (int o = 0; o < k; o++) {
            if (tk->trace) tk->trace_scope = "option";
            (*opos)[j][o] = (int)v.n;
            rh_ivec_push(&v, tk->mark[5]);
            const int32_t *ids;
            const size_t n = rh_tok_ids(tk, opts[o].s, opts[o].len, RH_LIM_OPTION, &ids);
            for (size_t i = 0; i < n; i++) rh_ivec_push(&v, ids[i]);
        }
    }
    sb_free(&b);
    *seg = rh_arena_alloc(A, (v.n + 1) * sizeof(int32_t));
    memcpy(*seg, v.v, v.n * sizeof(int32_t));
    const int n = (int)v.n;
    rh_ivec_free(&v);
    return n;
}

typedef struct { const jnode *tool; int32_t *seg; int n, np; int *ppos, *nopts, **opos; } tseg;

/* JavaScript Map key equality (SameValueZero) of two tool names */
static int same_name(const jnode *a, const jnode *b) {
    if (!a || !b) return a == b;
    if (a->t != b->t) return 0;
    if (a->t == J_STR) return a->len == b->len && !memcmp(a->s, b->s, a->len);
    if (a->t == J_NUM) { rv x = rv_json(a), y = rv_json(b); return x.num == y.num || (x.num != x.num && y.num != y.num); }
    if (a->t == J_ARR || a->t == J_OBJ) return a == b;
    return 1;
}

/* build_example(train=False): NULL when unusable */
static rh_example *build_example(rh_tok *tk, const jnode *rec, const uint32_t *text, size_t tn, tseg *segs, int nt, rh_arena *A) {
    if (!nt) return NULL;
    int32_t *q;
    long(*offs)[2];
    const int nq = (int)rh_tok_request(tk, text, tn, RH_LIM_REQUEST, A, &q, &offs);
    if (!nq) return NULL;
    const int budget = RH_MAX_LEN - nq - 3;
    int total = 0;
    for (int i = 0; i < nt; i++) total += segs[i].n;
    if (total > budget) return NULL;
    /* keep.sort(key=order[name]), order = last index of each name: duplicate names move together (stable) */
    int *order = rh_arena_alloc(A, (size_t)nt * sizeof(int)), *keep = rh_arena_alloc(A, (size_t)nt * sizeof(int));
    for (int i = 0; i < nt; i++) {
        order[i] = i;
        const jnode *ni = rh_json_get(segs[i].tool, "name");
        for (int j = i + 1; j < nt; j++) if (same_name(ni, rh_json_get(segs[j].tool, "name"))) order[i] = j;
    }
    for (int i = 0; i < nt; i++) {
        int j = i;
        while (j > 0 && order[keep[j - 1]] > order[i]) { keep[j] = keep[j - 1]; j--; }
        keep[j] = i;
    }
    rh_example *ex = rh_arena_alloc(A, sizeof *ex);
    memset(ex, 0, sizeof *ex);
    ex->ids = rh_arena_alloc(A, (size_t)(nq + total + 3) * sizeof(int32_t));
    int n = 0;
    ex->ids[n++] = tk->mark[0];
    for (int i = 0; i < nq; i++) ex->ids[n++] = q[i];
    ex->ids[n++] = tk->mark[1];
    ex->tools = rh_arena_alloc(A, (size_t)nt * sizeof(rh_tslot));
    for (int k = 0; k < nt; k++) {
        const tseg *s = &segs[keep[k]];
        const int base = n;
        memcpy(ex->ids + n, s->seg, (size_t)s->n * sizeof(int32_t));
        n += s->n;
        rh_tslot *t = &ex->tools[k];
        t->tool = s->tool;
        t->name = rh_json_get(s->tool, "name");
        t->pos = base;
        t->nparams = s->np;
        t->params = rh_arena_alloc(A, (size_t)(s->np + 1) * sizeof(rh_pslot));
        const jnode *p = rh_json_get(s->tool, "params")->first;
        for (int j = 0; j < s->np; j++, p = p->next) {
            rh_pslot *ps = &t->params[j];
            ps->pos = base + s->ppos[j];
            ps->nopts = s->nopts[j];
            ps->opts = rh_arena_alloc(A, (size_t)(ps->nopts + 1) * sizeof(int));
            for (int o = 0; o < ps->nopts; o++) ps->opts[o] = base + s->opos[j][o];
            ps->param = p;
        }
    }
    ex->ids[n++] = tk->mark[1];
    ex->n = n;
    ex->q0 = 1;
    ex->nq = nq;
    ex->offs = offs;
    ex->cps = text;
    ex->ncp = tn;
    ex->ntools = nt;
    ex->rec = rec;
    const jnode *kind = rh_json_get(rec, "kind");
    ex->extract = kind && kind->t == J_STR && kind->len == 7 && !memcmp(kind->s, "extract", 7);
    return ex;
}

int rh_eval_examples(rh_tok *tk, const jnode *rec, const uint32_t *text, size_t tn, const jnode *const *tools,
                     int ntools, rh_arena *A, rh_example **out) {
    int32_t *q;
    long(*offs)[2];
    const int nq = (int)rh_tok_request(tk, text, tn, RH_LIM_REQUEST, A, &q, &offs);
    const int budget = RH_MAX_LEN - nq - 3;
    tseg *segs = rh_arena_alloc(A, (size_t)(ntools + 1) * sizeof(tseg));
    for (int i = 0; i < ntools; i++) {
        tseg *s = &segs[i];
        s->tool = tools[i];
        s->n = rh_tool_segment(tk, tools[i], A, &s->seg, &s->np, &s->ppos, &s->nopts, &s->opos);
        if (s->n < 0) return -1;
    }
    int *start = rh_arena_alloc(A, (size_t)(ntools + 2) * sizeof(int)); /* chunk c = tools [start[c], start[c + 1]) */
    int nc = 0, cur = 0, used = 0;
    for (int i = 0; i < ntools; i++) {
        if (cur && used + segs[i].n > budget) { start[nc++] = i - cur; cur = 0; used = 0; }
        cur++;
        used += segs[i].n;
    }
    if (cur) start[nc++] = ntools - cur;
    start[nc] = ntools;
    if (FIX.chunks && nc > 1) { /* same number of chunks, filled evenly (no lone tool in a last chunk) */
        double total = 0;
        for (int i = 0; i < ntools; i++) total += segs[i].n;
        const double target = total / nc;
        int ne = 0, c = 0, u = 0;
        int *even = rh_arena_alloc(A, (size_t)(ntools + 2) * sizeof(int));
        for (int i = 0; i < ntools; i++) {
            if (c && (u + segs[i].n > budget || (u >= target && ne < nc - 1))) { even[ne++] = i - c; c = 0; u = 0; }
            c++;
            u += segs[i].n;
        }
        if (c) even[ne++] = ntools - c;
        even[ne] = ntools;
        start = even;
        nc = ne;
    }
    rh_example *exs = rh_arena_alloc(A, (size_t)(nc + 1) * sizeof(rh_example));
    int k = 0;
    for (int c = 0; c < nc; c++) {
        rh_example *ex = build_example(tk, rec, text, tn, segs + start[c], start[c + 1] - start[c], A);
        if (!ex) continue;
        ex->chunk = c;
        exs[k++] = *ex;
    }
    *out = exs;
    return k;
}
