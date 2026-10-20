/* Relay-2 outputs -> value programs -> calls (port of web/decode.mjs: predictExample, predictRecord, makeProgram,
   bestSpan, allowedOps; requireArgs of web/relay.mjs). Head values are float32; the float32 arithmetic PyTorch does
   on them (sums, softmax, masking) is reproduced by rounding to float exactly where the JavaScript host calls
   Math.fround, so that argmax ties and saturated confidences break the same way. */
#include "relay_host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SPAN 96

static int is_str(const jnode *j, const char *s) { return j && j->t == J_STR && j->len == strlen(s) && !memcmp(j->s, s, j->len); }
static int argmax(const float *a, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (a[i] > a[best]) best = i;
    return best;
}

/* allowed_ops: additive mask over OPS (0 or -1e4) */
static void allowed_ops(const jnode *param, int nOpts, float *mask) {
    static const int BOOL[] = {OP_ABSENT, OP_BOOL_TRUE, OP_BOOL_FALSE, OP_DEFAULT, -1},
                     INT[] = {OP_ABSENT, OP_NUMBER, OP_COPY, OP_TIMEPART, OP_DEFAULT, OP_ENUM, OP_DIGITS, -1},
                     NUM[] = {OP_ABSENT, OP_NUMBER, OP_COPY, OP_DEFAULT, OP_ENUM, -1},
                     ARR[] = {OP_ABSENT, OP_LIST, OP_DEFAULT, -1}, OBJ[] = {OP_ABSENT, OP_DEFAULT, -1};
    const jnode *t = rh_json_get(param, "type");
    const int *list = NULL;
    if (!t || is_str(t, "string")) list = NULL;
    else if (is_str(t, "boolean")) list = BOOL;
    else if (is_str(t, "integer")) list = INT;
    else if (is_str(t, "number")) list = NUM;
    else if (is_str(t, "array")) list = ARR;
    else if (is_str(t, "object")) list = OBJ;
    int ok[RH_N_OPS];
    for (int o = 0; o < RH_N_OPS; o++) ok[o] = !list && o != OP_LIST;
    if (list) for (int i = 0; list[i] >= 0; i++) ok[list[i]] = 1;
    if (!nOpts) ok[OP_ENUM] = 0;
    const jnode *en = rh_json_get(param, "enum");
    if (FIX.enumOnly && nOpts && en && rh_js_truthy(en)) {
        size_t len = 0; /* param.enum.length (UTF-16 units for a string) */
        if (en->t == J_ARR) len = en->count;
        else if (en->t == J_STR) {
            uint32_t *c;
            const size_t n = rh_utf8_decode(en->s, en->len, &c);
            for (size_t i = 0; i < n; i++) len += c[i] > 0xffff ? 2 : 1;
            free(c);
        }
        if (len) {
            static const int DROP[] = {OP_COPY, OP_SELECT, OP_LOOKUP, OP_AFFIX, OP_DIGITS, OP_REF};
            for (int i = 0; i < 6; i++) ok[DROP[i]] = 0; /* closed enum: no copied words; bool / number stay */
        }
    }
    if (!rh_json_get(param, "default")) ok[OP_DEFAULT] = 0;
    for (int o = 0; o < RH_N_OPS; o++) mask[o] = ok[o] ? 0.0f : -1e4f;
}

/* torch.softmax on a float32 vector: the maximum's probability, exp(0) * (1 / sum) */
static float softmax_max(const float *x, int n) {
    double m = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)(float)rh_js_exp((double)(float)((double)x[i] - m));
    return (float)(1.0 / (double)(float)sum);
}

/* best_span: joint argmax of start + end over j >= i, j - i < MAX_SPAN; 0 if the null slot (index n) wins */
static int best_span(const float *s, const float *e, int n, int *a, int *b) {
    double best = -INFINITY;
    *a = *b = 0;
    for (int i = 0; i < n; i++) {
        const int lim = n < i + MAX_SPAN ? n : i + MAX_SPAN;
        const double si = s[i];
        for (int j = i; j < lim; j++) {
            const double v = (float)(si + (double)e[j]);
            if (v > best) { best = v; *a = i; *b = j; }
        }
    }
    if ((double)(float)((double)s[n] + (double)e[n]) > best) return 0;
    return 1;
}

static void char_span(const rh_example *ex, int t0, int t1, long *sp) {
    sp[0] = ex->offs[t0][0];
    sp[1] = ex->offs[t1][1];
    rh_trim(ex->cps, &sp[0], &sp[1]);
}

/* make_program: head outputs of one argument row -> value program (code point spans) or NULL */
static rh_prog *make_program(int name, rh_args *a, int row, const rh_example *ex, const jnode *param, rh_arena *A) {
    const int n = ex->nq;
    const float *aux = rh_args_aux(a, row);
    const int naux = rh_args_naux(a);
    rh_prog *p = rh_arena_alloc(A, sizeof *p);
    memset(p, 0, sizeof *p);
    if (name == OP_COPY || name == OP_NUMBER || name == OP_LOOKUP || name == OP_TIMEPART || name == OP_DIGITS || name == OP_AFFIX) {
        const float *sp = rh_args_spans(a, row);
        int t0, t1;
        if (!best_span(sp, sp + (n + 1), n, &t0, &t1)) return NULL;
        char_span(ex, t0, t1, p->span);
        p->has_span = 1;
        if (name == OP_COPY) { p->op = PR_COPY; return p; }
        if (name == OP_DIGITS) { p->op = PR_DIGITS; return p; }
        if (name == OP_NUMBER) {
            p->op = PR_NUMBER;
            p->scale = RH_NUM_SCALES[argmax(aux, RH_N_NUM_SCALES < naux ? RH_N_NUM_SCALES : naux)];
            p->is_int = is_str(rh_json_get(param, "type"), "integer");
            return p;
        }
        if (name == OP_TIMEPART) { p->op = PR_TIMEPART; p->part_hour = argmax(aux, 2) == 0; return p; }
        if (name == OP_AFFIX) {
            const int m = (int)rh_affixes()->count < naux ? (int)rh_affixes()->count : naux;
            if (!m) return NULL;
            p->op = PR_AFFIX;
            p->index = argmax(aux, m);
            return p;
        }
        p->op = PR_LOOKUP;
        p->table = argmax(aux, (int)rh_n_tables());
        return p;
    }
    if (name == OP_DATETIME) {
        const float *sp = rh_args_spans(a, row);
        int d0, d1, t0, t1;
        const int hd = best_span(sp, sp + (n + 1), n, &d0, &d1), ht = best_span(sp + 2 * (n + 1), sp + 3 * (n + 1), n, &t0, &t1);
        if (!hd && !ht) return NULL;
        p->op = PR_DATETIME;
        if ((p->has_date = hd)) char_span(ex, d0, d1, p->date);
        if ((p->has_time = ht)) char_span(ex, t0, t1, p->time);
        p->format = RH_DATE_FORMATS[argmax(aux, RH_N_DATE_FORMATS)];
        return p;
    }
    if (name == OP_ENUM) {
        rv *opts;
        int is_enum;
        const int no = rh_options(param, A, &opts, &is_enum);
        if (no <= 0) return NULL;
        int E;
        const float *en = rh_args_enum(a, row, &E);
        const int i = argmax(en, E);
        p->op = is_enum ? PR_ENUM : PR_OPTION;
        p->index = i < no - 1 ? i : no - 1;
        return p;
    }
    if (name == OP_BOOL_TRUE || name == OP_BOOL_FALSE) { p->op = PR_BOOL; p->value = name == OP_BOOL_TRUE; return p; }
    if (name == OP_DEFAULT) { p->op = PR_DEFAULT; return p; }
    if (name == OP_REF) { p->op = PR_REF; p->call = argmax(aux, RH_K); return p; }
    if (name == OP_LIST || name == OP_SELECT) {
        const float *tg = rh_args_tags(a, row);
        int (*sp)[2] = rh_arena_alloc(A, (size_t)(n + 1) * sizeof *sp), ns = 0, cur = 0, c0 = 0, c1 = 0;
        for (int i = 0; i < n; i++) {
            const int t = argmax(tg + 3 * i, 3);
            if (t == 1) {
                if (cur) { sp[ns][0] = c0; sp[ns][1] = c1; ns++; }
                cur = 1;
                c0 = c1 = i;
            } else if (t == 2 && cur) c1 = i;
            else {
                if (cur) { sp[ns][0] = c0; sp[ns][1] = c1; ns++; }
                cur = 0;
            }
        }
        if (cur) { sp[ns][0] = c0; sp[ns][1] = c1; ns++; }
        if (!ns) return NULL;
        if (name == OP_SELECT) { /* the spans joined by spaces, no space before punctuation */
            sb v = {0};
            for (int k = 0; k < ns; k++) {
                long c[2];
                char_span(ex, sp[k][0], sp[k][1], c);
                if (k) sb_str(&v, " ");
                for (long i = c[0]; i < c[1]; i++) sb_cp(&v, ex->cps[i]);
            }
            uint32_t *cp;
            const size_t m = rh_utf8_decode(v.s ? v.s : "", v.n, &cp);
            sb o = {0};
            for (size_t i = 0; i < m;) {
                if (rh_py_isspace(cp[i])) {
                    size_t j = i;
                    while (j < m && rh_py_isspace(cp[j])) j++;
                    if (j < m && strchr(",.;:!?", (int)cp[j]) && cp[j] < 128) { i = j; continue; }
                    for (; i < j; i++) sb_cp(&o, cp[i]);
                    continue;
                }
                sb_cp(&o, cp[i++]);
            }
            free(cp);
            sb_free(&v);
            char *s = rh_arena_alloc(A, o.n + 1);
            memcpy(s, o.s ? o.s : "", o.n);
            s[o.n] = 0;
            p->op = PR_CONST;
            p->cvalue = s;
            p->clen = o.n;
            sb_free(&o);
            return p;
        }
        p->op = PR_LIST;
        p->nitems = ns;
        p->items = rh_arena_alloc(A, (size_t)ns * sizeof(rh_prog));
        memset(p->items, 0, (size_t)ns * sizeof(rh_prog));
        for (int k = 0; k < ns; k++) {
            p->items[k].op = PR_COPY;
            p->items[k].has_span = 1;
            char_span(ex, sp[k][0], sp[k][1], p->items[k].span);
        }
        return p;
    }
    return NULL;
}

/* String(param.name): the property key the JavaScript host stores the argument under */
static const char *prop_key(const jnode *name, rh_arena *A, size_t *len) {
    if (name && name->t == J_STR) { *len = name->len; return name->s; }
    sb b = {0};
    if (!name) sb_str(&b, "undefined");
    else if (name->t == J_NULL) sb_str(&b, "null");
    else if (name->t == J_TRUE || name->t == J_FALSE) sb_str(&b, name->t == J_TRUE ? "true" : "false");
    else if (name->t == J_NUM) { rv v = rv_json(name); if (v.isfloat && isfinite(v.num) && v.num == floor(v.num)) py_float(&b, v.num); else js_num(&b, v.num); }
    else if (name->t == J_OBJ) sb_str(&b, "[object Object]");
    else { rv v = rv_json(name); rh_js_stringify(&b, &v); }
    char *s = rh_arena_alloc(A, b.n + 1);
    memcpy(s, b.s ? b.s : "", b.n);
    s[b.n] = 0;
    *len = b.n;
    sb_free(&b);
    return s;
}

typedef struct {
    double conf;
    int idx;
    const char *key;
    size_t klen;
    rv v;
    long span[2];
    int has_span, is_ref, has_where;
    long where;
    const rh_prog *prog; /* explain mode */
    double opt;
    int has_opt;
} cand;
typedef struct { rh_call c; cand *cands; int nc, cap; } callb;

static int cmp_cand(const void *x, const void *y) {
    const cand *a = x, *b = y;
    if (a->conf != b->conf) return a->conf < b->conf ? 1 : -1;
    return (a->idx > b->idx) - (a->idx < b->idx);
}

static void set_arg(rh_call *c, int *cap, const char *key, size_t klen, rv v, rh_arena *A) {
    for (int i = 0; i < c->nargs; i++)
        if (c->args[i].klen == klen && !memcmp(c->args[i].key, key, klen)) { c->args[i].v = v; return; }
    if (c->nargs == *cap) {
        const int nc = *cap ? 2 * *cap : 8;
        rh_kv *na = rh_arena_alloc(A, (size_t)nc * sizeof(rh_kv));
        if (c->nargs) memcpy(na, c->args, (size_t)c->nargs * sizeof(rh_kv));
        c->args = na;
        *cap = nc;
    }
    c->args[c->nargs++] = (rh_kv){key, klen, v};
}

static void set_why(rh_call *c, int *cap, const cand *x, rh_arena *A) {
    const rh_why w = {x->key, x->klen, x->prog, x->conf, x->opt, x->has_opt};
    for (int i = 0; i < c->nwhy; i++)
        if (c->why[i].klen == x->klen && !memcmp(c->why[i].key, x->key, x->klen)) { c->why[i] = w; return; }
    if (c->nwhy == *cap) {
        const int nc = *cap ? 2 * *cap : 8;
        rh_why *nw = rh_arena_alloc(A, (size_t)nc * sizeof(rh_why));
        if (c->nwhy) memcpy(nw, c->why, (size_t)c->nwhy * sizeof(rh_why));
        c->why = nw;
        *cap = nc;
    }
    c->why[c->nwhy++] = w;
}

static int ordered_before(const rh_call *x, const rh_call *y) {
    if (x->ref != y->ref) return x->ref < y->ref;
    if (x->pos != y->pos) return x->pos < y->pos;
    if (FIX.orderTie && x->first != y->first) return x->first < y->first;
    if (x->ti != y->ti) return x->ti < y->ti;
    return x->k < y->k;
}

/* JSON.stringify(object) in JavaScript property order (integer keys first) */
static void stringify_args(sb *b, const rh_call *c, int dumps);

static void predict_example(rh_net *net, rh_example *ex, double gate, double gate2, int explain, rh_arena *A, rh_pred *out) {
    const int nt = ex->ntools;
    memset(out, 0, sizeof *out);
    if (!nt) { out->margin = -1e9f; return; }
    float *tool = rh_arena_alloc(A, (size_t)nt * sizeof(float)), none;
    float **count = rh_arena_alloc(A, (size_t)nt * sizeof(float *));
    int ncount;
    if (!rh_net_route(net, ex, tool, count, &ncount, &none, A)) { out->margin = -1e9f; return; }
    float *margin = rh_arena_alloc(A, (size_t)nt * sizeof(float));
    for (int i = 0; i < nt; i++) margin[i] = (float)((double)tool[i] - (double)none);
    const double g2 = gate > gate2 ? gate : gate2;
    int *counts = rh_arena_alloc(A, (size_t)nt * sizeof(int)), ninst = 0;
    const int j = argmax(margin, nt);
    for (int i = 0; i < nt; i++) {
        int called = margin[i] > g2;
        if (i == j) called = called || margin[j] > gate || ex->extract;
        counts[i] = called ? argmax(count[i], ncount) + 1 : 0;
        ninst += counts[i];
    }
    out->margin = margin[j];
    if (explain) { out->tools = ex->tools; out->margins = margin; out->counts = counts; out->ntools = nt; }
    if (!ninst) return;
    int (*insts)[2] = rh_arena_alloc(A, (size_t)ninst * sizeof *insts), q = 0;
    for (int ti = 0; ti < nt; ti++) for (int k = 0; k < counts[ti]; k++) { insts[q][0] = ti; insts[q][1] = k; q++; }
    const int nq = ex->nq;
    float *al = rh_arena_alloc(A, (size_t)ninst * nq * sizeof(float));
    if (!rh_net_anchors(net, ex, (const int(*)[2])insts, ninst, al, A)) return;
    int *anchor = rh_arena_alloc(A, (size_t)ninst * sizeof(int));
    float *lg = rh_arena_alloc(A, (size_t)nq * sizeof(float));
    for (int i = 0; i < ninst; i++) { /* instances of one tool take distinct anchors */
        memcpy(lg, al + (size_t)i * nq, (size_t)nq * sizeof(float));
        for (int p = 0; p < i; p++) if (insts[p][0] == insts[i][0]) lg[anchor[p]] = -1e9f;
        anchor[i] = argmax(lg, nq);
    }
    int nrows = 0;
    for (int i = 0; i < ninst; i++) nrows += ex->tools[insts[i][0]].nparams;
    rh_rowreq *rows = rh_arena_alloc(A, (size_t)(nrows + 1) * sizeof(rh_rowreq));
    int *rinst = rh_arena_alloc(A, (size_t)(nrows + 1) * sizeof(int));
    nrows = 0;
    for (int i = 0; i < ninst; i++)
        for (int pj = 0; pj < ex->tools[insts[i][0]].nparams; pj++) {
            rows[nrows] = (rh_rowreq){insts[i][0], insts[i][1], pj, anchor[i]};
            rinst[nrows++] = i;
        }
    rh_args *a = nrows ? rh_net_args(net, ex, rows, nrows, A) : NULL;
    if (nrows && !a) return;
    callb *calls = rh_arena_alloc(A, (size_t)ninst * sizeof(callb));
    memset(calls, 0, (size_t)ninst * sizeof(callb));
    for (int i = 0; i < ninst; i++) {
        calls[i].c.ti = insts[i][0];
        calls[i].c.k = insts[i][1];
        calls[i].c.tool = ex->tools[insts[i][0]].name;
        calls[i].c.pos = anchor[i];
        calls[i].c.margin = margin[insts[i][0]];
    }
    const jnode *ctx = rh_json_get(ex->rec, "context");
    float mask[RH_N_OPS], logits[RH_N_OPS];
    for (int jr = 0; jr < nrows; jr++) {
        const rh_rowreq *rw = &rows[jr];
        const rh_pslot *ps = &ex->tools[rw->ti].params[rw->pj];
        const jnode *param = ps->param;
        callb *c = &calls[rinst[jr]];
        allowed_ops(param, ps->nopts, mask);
        const float *op = rh_args_op(a, jr);
        for (int o = 0; o < RH_N_OPS; o++) logits[o] = (float)((double)op[o] + (double)mask[o]);
        const int o = argmax(logits, RH_N_OPS);
        if (o == OP_ABSENT) continue;
        double conf = softmax_max(logits, RH_N_OPS), opt = 0;
        rh_prog *prog = make_program(o, a, jr, ex, param, A);
        if (!prog) continue;
        rv v;
        if (!rh_run(prog, ex->cps, ex->ncp, param, ctx, A, &v)) continue;
        if (v.t == RV_NULL || v.t == RV_UNDEF) continue;
        const jnode *ty = rh_json_get(param, "type");
        const int ptype_str = !ty || is_str(ty, "string");
        if (FIX.numReject && (is_str(ty, "integer") || is_str(ty, "number")) && v.t != RV_NUM) continue;
        if (explain && (prog->op == PR_ENUM || prog->op == PR_OPTION)) {
            int E;
            const float *en = rh_args_enum(a, jr, &E);
            opt = softmax_max(en, E);
        }
        if (prog->has_span) {
            const float *sp = rh_args_spans(a, jr);
            conf *= (double)(float)((double)softmax_max(sp, nq + 1) * (double)softmax_max(sp + (nq + 1), nq + 1));
        }
        /* only text-copying ops compete for a stretch of text; '10:15' may give both an hour and a minute */
        const int compete = ((prog->op == PR_COPY || prog->op == PR_LOOKUP || prog->op == PR_DIGITS || prog->op == PR_AFFIX) && ptype_str) ||
                            (FIX.numSpan && prog->op == PR_NUMBER);
        if (c->nc == c->cap) {
            const int nc = c->cap ? 2 * c->cap : 8;
            cand *na = rh_arena_alloc(A, (size_t)nc * sizeof(cand));
            if (c->nc) memcpy(na, c->cands, (size_t)c->nc * sizeof(cand));
            c->cands = na;
            c->cap = nc;
        }
        cand *x = &c->cands[c->nc];
        memset(x, 0, sizeof *x);
        x->conf = conf;
        x->idx = c->nc++;
        x->key = prop_key(rh_json_get(param, "name"), A, &x->klen);
        x->v = v;
        if ((x->has_span = compete)) { x->span[0] = prog->span[0]; x->span[1] = prog->span[1]; }
        x->is_ref = prog->op == PR_REF;
        if (prog->has_span) { x->has_where = 1; x->where = prog->span[0]; }
        else if (prog->has_date) { x->has_where = 1; x->where = prog->date[0]; }
        else if (prog->has_time) { x->has_where = 1; x->where = prog->time[0]; }
        x->prog = prog;
        x->opt = opt;
        x->has_opt = prog->op == PR_ENUM || prog->op == PR_OPTION;
    }
    for (int i = 0; i < ninst; i++) {
        callb *c = &calls[i];
        qsort(c->cands, (size_t)c->nc, sizeof(cand), cmp_cand);
        long (*taken)[2] = rh_arena_alloc(A, (size_t)(c->nc + 1) * sizeof *taken);
        int nk = 0, cap = 0, wcap = 0;
        c->c.first = INFINITY;
        for (int k = 0; k < c->nc; k++) {
            const cand *x = &c->cands[k];
            if (x->has_span) {
                int hit = 0;
                for (int t = 0; t < nk && !hit; t++) hit = x->span[0] < taken[t][1] && taken[t][0] < x->span[1];
                if (hit) continue;
                taken[nk][0] = x->span[0];
                taken[nk][1] = x->span[1];
                nk++;
            }
            set_arg(&c->c, &cap, x->key, x->klen, x->v, A);
            if (explain) set_why(&c->c, &wcap, x, A);
            if (x->has_where && (double)x->where < c->c.first) c->c.first = (double)x->where;
            c->c.ref = c->c.ref || x->is_ref;
        }
    }
    rh_call *res = rh_arena_alloc(A, (size_t)ninst * sizeof(rh_call));
    int nres = 0;
    for (int i = 0; i < ninst; i++) { /* stable insertion by the call order */
        int p = nres++;
        while (p > 0 && ordered_before(&calls[i].c, &res[p - 1])) { res[p] = res[p - 1]; p--; }
        res[p] = calls[i].c;
    }
    if (FIX.dedupe) {
        char **keys = rh_arena_alloc(A, (size_t)nres * sizeof(char *));
        size_t *kl = rh_arena_alloc(A, (size_t)nres * sizeof(size_t));
        int w = 0;
        for (int i = 0; i < nres; i++) {
            sb b = {0};
            size_t tl;
            const char *tn = prop_key(res[i].tool, A, &tl); /* c.tool + JSON.stringify(c.args) */
            sb_put(&b, tn, tl);
            stringify_args(&b, &res[i], 0);
            int dup = 0;
            for (int k = 0; k < w && !dup; k++) dup = kl[k] == b.n && !memcmp(keys[k], b.s, b.n);
            if (!dup) {
                keys[w] = rh_arena_alloc(A, b.n + 1);
                memcpy(keys[w], b.s, b.n);
                kl[w] = b.n;
                res[w++] = res[i];
            }
            sb_free(&b);
        }
        nres = w;
    }
    out->calls = res;
    out->n = nres;
}

int rh_predict_record(rh_net *net, rh_tok *tk, const jnode *rec, const uint32_t *text, size_t tn,
                      const jnode *const *tools, int ntools, double gate, double gate2, int explain, rh_arena *A,
                      rh_pred *out) {
    rh_example *exs;
    int ne = rh_eval_examples(tk, rec, text, tn, tools, ntools, A, &exs);
    sb *trace = rh_net_trace(net);
    rh_tracef(trace, "{\"event\":\"sequence_build\",\"chunks\":%d,\"status\":\"%s\"}\n", ne < 0 ? 0 : ne,
              ne < 0 ? "invalid_schema" : ne ? "ok" : "no_usable_sequence");
    if (ne < 0) ne = 0; /* an unbuildable record simply gets no calls */
    rh_pred *preds = rh_arena_alloc(A, (size_t)(ne + 1) * sizeof(rh_pred));
    int *ord = rh_arena_alloc(A, (size_t)(ne + 1) * sizeof(int));
    for (int i = 0; i < ne; i++) {
        rh_tracef(trace, "{\"event\":\"chunk\",\"index\":%d,\"tokens\":%d,\"request_tokens\":%d,\"tools\":[", i, exs[i].n, exs[i].nq);
        if (trace) for (int j = 0; j < exs[i].ntools; j++) {
            if (j) sb_str(trace, ",");
            rh_trace_name(trace, exs[i].tools[j].name);
        }
        rh_tracef(trace, "]}\n");
        predict_example(net, &exs[i], gate, gate2, explain, A, &preds[i]);
        rh_tracef(trace, "{\"event\":\"chunk_result\",\"index\":%d,\"margin\":%.9g,\"calls\":%d}\n", i, preds[i].margin, preds[i].n);
        int p = i; /* stable: margin descending */
        while (p > 0 && (double)preds[ord[p - 1]].margin < (double)preds[i].margin) { ord[p] = ord[p - 1]; p--; }
        ord[p] = i;
    }
    const double g2 = gate > gate2 ? gate : gate2;
    int total = 0;
    for (int i = 0; i < ne; i++) total += preds[i].n;
    memset(out, 0, sizeof *out);
    out->calls = rh_arena_alloc(A, (size_t)(total + 1) * sizeof(rh_call));
    if (explain) { out->chunks = preds; out->nchunks = ne; }
    out->margin = ne ? preds[ord[0]].margin : -1e9f;
    if (!ne) return 0;
    for (int k = 0; k < preds[ord[0]].n; k++) out->calls[out->n++] = preds[ord[0]].calls[k];
    /* FIX.chunks: a later chunk's best tool must also come within 2 of the best chunk's (it used to need only gate2) */
    double bar = g2;
    if (FIX.chunks) { const double b2 = (double)preds[ord[0]].margin - 2; bar = g2 > b2 ? g2 : b2; }
    if (!is_str(rh_json_get(rec, "kind"), "extract"))
        for (int i = 1; i < ne; i++) {
            const rh_pred *p = &preds[ord[i]];
            if ((double)p->margin > bar) for (int k = 0; k < p->n; k++) out->calls[out->n++] = p->calls[k];
        }
    return ne;
}

/* ---------------------------------------------------------------- requireArgs */
static const char *const PROTO[] = {"constructor", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable",
    "toLocaleString", "toString", "valueOf", "__proto__", "__defineGetter__", "__defineSetter__", "__lookupGetter__",
    "__lookupSetter__"};

static int same_name(const jnode *a, const jnode *b) { /* Map key equality (SameValueZero) */
    if (!a || !b) return a == b;
    if (a->t != b->t) return 0;
    if (a->t == J_STR) return a->len == b->len && !memcmp(a->s, b->s, a->len);
    if (a->t == J_NUM) { rv x = rv_json(a), y = rv_json(b); return x.num == y.num || (x.num != x.num && y.num != y.num); }
    if (a->t == J_ARR || a->t == J_OBJ) return a == b;
    return 1;
}

static int incomplete(const rh_call *c, const jnode *const *tools, int ntools) {
    const jnode *t = NULL;
    for (int i = 0; i < ntools; i++) if (same_name(rh_json_get(tools[i], "name"), c->tool)) t = tools[i];
    if (!t) return 0;
    const jnode *ps = rh_json_get(t, "params");
    if (!ps || ps->t != J_ARR) return 0;
    rh_arena A = {0};
    int bad = 0;
    for (const jnode *p = ps->first; p && !bad; p = p->next) {
        if (!rh_js_truthy(rh_json_get(p, "required"))) continue;
        size_t kl;
        const char *k = prop_key(rh_json_get(p, "name"), &A, &kl);
        int in = 0;
        for (int i = 0; i < c->nargs && !in; i++) in = c->args[i].klen == kl && !memcmp(c->args[i].key, k, kl);
        for (size_t i = 0; i < sizeof PROTO / sizeof PROTO[0] && !in; i++) in = strlen(PROTO[i]) == kl && !memcmp(PROTO[i], k, kl);
        bad = !in;
    }
    rh_arena_free(&A);
    return bad;
}

void rh_require_args(rh_pred *p, const jnode *const *tools, int ntools) {
    if (FIX.perCallHold) { /* hold only the incomplete calls */
        int w = 0;
        for (int i = 0; i < p->n; i++) if (!incomplete(&p->calls[i], tools, ntools)) p->calls[w++] = p->calls[i];
        p->n = w;
        return;
    }
    for (int i = 0; i < p->n; i++) if (incomplete(&p->calls[i], tools, ntools)) { p->n = 0; return; }
}

/* ---------------------------------------------------------------- serialization */
static void stringify_json(sb *b, const jnode *j) {
    if (j->t == J_ARR) {
        sb_str(b, "[");
        for (const jnode *c = j->first; c; c = c->next) { if (c != j->first) sb_str(b, ","); stringify_json(b, c); }
        sb_str(b, "]");
    } else if (j->t == J_OBJ) {
        rh_arena A = {0};
        const jnode **m;
        const size_t n = rh_js_members(j, &A, &m);
        sb_str(b, "{");
        for (size_t k = 0; k < n; k++) {
            if (k) sb_str(b, ",");
            js_string(b, m[k]->key, m[k]->klen);
            sb_str(b, ":");
            stringify_json(b, m[k]);
        }
        sb_str(b, "}");
        rh_arena_free(&A);
    } else {
        rv v = rv_json(j);
        rh_js_stringify(b, &v);
    }
}
void rh_js_stringify(sb *b, const rv *v) {
    switch (v->t) {
    case RV_UNDEF: case RV_NULL: sb_str(b, "null"); return;
    case RV_BOOL: sb_str(b, v->num ? "true" : "false"); return;
    case RV_NUM: if (isfinite(v->num)) js_num(b, v->num); else sb_str(b, "null"); return;
    case RV_STR: js_string(b, v->s, v->len); return;
    case RV_JSON: stringify_json(b, v->j); return;
    case RV_LIST:
        sb_str(b, "[");
        for (size_t i = 0; i < v->n; i++) { if (i) sb_str(b, ","); rh_js_stringify(b, &v->items[i]); }
        sb_str(b, "]");
        return;
    }
}

static int index_key(const char *k, size_t n, double *v) {
    if (!n || n > 10 || (k[0] == '0' && n > 1)) return 0;
    double x = 0;
    for (size_t i = 0; i < n; i++) { if (k[i] < '0' || k[i] > '9') return 0; x = x * 10 + (k[i] - '0'); }
    if (x >= 4294967295.0) return 0;
    *v = x;
    return 1;
}
static void stringify_args(sb *b, const rh_call *c, int dumps) {
    int *ord = malloc(((size_t)c->nargs + 1) * sizeof(int)), n = 0;
    double v, w;
    for (int pass = 0; pass < 2; pass++) /* integer keys first (ascending), then insertion order */
        for (int i = 0; i < c->nargs; i++) {
            const int ik = index_key(c->args[i].key, c->args[i].klen, &v);
            if (pass == 0 && ik) {
                int p = n++;
                while (p > 0 && (index_key(c->args[ord[p - 1]].key, c->args[ord[p - 1]].klen, &w), w > v)) { ord[p] = ord[p - 1]; p--; }
                ord[p] = i;
            } else if (pass == 1 && !ik) ord[n++] = i;
        }
    sb_str(b, "{");
    for (int k = 0; k < n; k++) {
        const rh_kv *a = &c->args[ord[k]];
        if (k) sb_str(b, dumps ? ", " : ",");
        js_string(b, a->key, a->klen);
        sb_str(b, dumps ? ": " : ":");
        if (dumps) rv_dumps(b, &a->v);
        else rh_js_stringify(b, &a->v);
    }
    sb_str(b, "}");
    free(ord);
}

void rh_dump_calls(sb *b, const rh_pred *p) {
    sb_str(b, "[");
    for (int i = 0; i < p->n; i++) {
        if (i) sb_str(b, ", ");
        sb_str(b, "{\"tool\": ");
        rv t = rv_json(p->calls[i].tool);
        rv_dumps(b, &t);
        sb_str(b, ", \"args\": ");
        stringify_args(b, &p->calls[i], 1);
        sb_str(b, "}");
    }
    sb_str(b, "]");
}

/* ---------------------------------------------------------------- explain */
static const char *const PR_NAMES[] = {"default", "bool", "enum", "option", "const", "digits", "affix", "ref", "copy",
    "number", "timepart", "datetime", "lookup", "list"};

static void utf16_span(sb *b, const long *sp, const long *u16) {
    char t[64];
    snprintf(t, sizeof t, "[%ld, %ld]", u16[sp[0]], u16[sp[1]]);
    sb_str(b, t);
}
static void dump_prog(sb *b, const rh_prog *p, const long *u16) {
    sb_str(b, "{\"op\": \"");
    sb_str(b, PR_NAMES[p->op]);
    sb_str(b, "\"");
    if (p->has_span) { sb_str(b, ", \"span\": "); utf16_span(b, p->span, u16); }
    if (p->has_date) { sb_str(b, ", \"date\": "); utf16_span(b, p->date, u16); }
    if (p->has_time) { sb_str(b, ", \"time\": "); utf16_span(b, p->time, u16); }
    if (p->op == PR_LIST) {
        sb_str(b, ", \"items\": [");
        for (int i = 0; i < p->nitems; i++) { if (i) sb_str(b, ", "); dump_prog(b, &p->items[i], u16); }
        sb_str(b, "]");
    }
    sb_str(b, "}");
}
static void dump_explained(sb *b, const rh_call *cs, int n, const long *u16) {
    sb_str(b, "[");
    for (int i = 0; i < n; i++) {
        const rh_call *c = &cs[i];
        if (i) sb_str(b, ", ");
        sb_str(b, "{\"tool\": ");
        rv t = rv_json(c->tool);
        rh_js_stringify(b, &t);
        sb_str(b, ", \"args\": ");
        stringify_args(b, c, 0);
        sb_str(b, ", \"why\": {");
        for (int k = 0; k < c->nwhy; k++) {
            const rh_why *w = &c->why[k];
            if (k) sb_str(b, ", ");
            js_string(b, w->key, w->klen);
            sb_str(b, ": {\"op\": \"");
            sb_str(b, PR_NAMES[w->prog->op]);
            sb_str(b, "\", \"conf\": ");
            js_num(b, w->conf);
            sb_str(b, ", \"span\": ");
            if (w->prog->has_span) utf16_span(b, w->prog->span, u16);
            else sb_str(b, "null");
            sb_str(b, ", \"prog\": ");
            dump_prog(b, w->prog, u16);
            sb_str(b, ", \"optProb\": ");
            if (w->has_opt) js_num(b, w->opt);
            else sb_str(b, "null");
            sb_str(b, "}");
        }
        char t2[48];
        snprintf(t2, sizeof t2, "}, \"anchor\": %d, \"margin\": ", c->pos);
        sb_str(b, t2);
        js_num(b, c->margin);
        sb_str(b, "}");
    }
    sb_str(b, "]");
}
void rh_dump_explain(sb *b, const rh_pred *kept, const rh_pred *all, const uint32_t *text, size_t tn) {
    long *u16 = malloc((tn + 1) * sizeof(long)); /* code point index -> UTF-16 offset (the page indexes strings) */
    u16[0] = 0;
    for (size_t i = 0; i < tn; i++) u16[i + 1] = u16[i] + (text[i] > 0xFFFF ? 2 : 1);
    sb_str(b, "{\"calls\": ");
    dump_explained(b, kept->calls, kept->n, u16);
    sb_str(b, ", \"dropped\": ");
    rh_call *dropped = malloc(((size_t)all->n + 1) * sizeof(rh_call));
    int nd = 0;
    for (int i = 0, k = 0; i < all->n; i++) { /* requireArgs keeps the order: kept is a subsequence */
        if (k < kept->n && !memcmp(&kept->calls[k], &all->calls[i], sizeof(rh_call))) k++;
        else dropped[nd++] = all->calls[i];
    }
    dump_explained(b, dropped, nd, u16);
    free(dropped);
    sb_str(b, ", \"chunks\": [");
    for (int c = 0; c < all->nchunks; c++) {
        const rh_pred *p = &all->chunks[c];
        if (c) sb_str(b, ", ");
        sb_str(b, "{\"margin\": ");
        js_num(b, p->margin);
        sb_str(b, ", \"margins\": [");
        for (int i = 0; i < p->ntools; i++) {
            if (i) sb_str(b, ", ");
            sb_str(b, "{\"tool\": ");
            rv t = rv_json(p->tools[i].name);
            rh_js_stringify(b, &t);
            sb_str(b, ", \"margin\": ");
            js_num(b, p->margins[i]);
            char t3[32];
            snprintf(t3, sizeof t3, ", \"count\": %d}", p->counts[i]);
            sb_str(b, t3);
        }
        sb_str(b, "]}");
    }
    sb_str(b, "]}");
    free(u16);
}
