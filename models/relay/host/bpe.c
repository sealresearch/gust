/* Byte-level BPE tokenizer (port of web/tokenizer.mjs): added tokens (leftmost-longest, lstrip), NFC with the
   alignment rules of HF tokenizers, the GPT-2 pre-tokenizer regex as a scanner, BPE with merge ranks.
   Works on code points; offsets are code point indices into the input text. */
#include "relay_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- small utilities */
void *rh_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "relay: out of memory\n"); abort(); }
    return p;
}
void *rh_realloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) { fprintf(stderr, "relay: out of memory\n"); abort(); }
    return p;
}
struct rh_block { rh_block *next; size_t used, cap; };
void *rh_arena_alloc(rh_arena *a, size_t n) {
    n = (n + 7u) & ~(size_t)7u;
    if (!a->head || a->head->used + n > a->head->cap) {
        const size_t cap = n > 65536 ? n : 65536;
        rh_block *b = rh_malloc(sizeof(rh_block) + cap);
        b->next = a->head;
        b->used = 0;
        b->cap = cap;
        a->head = b;
    }
    void *p = (char *)(a->head + 1) + a->head->used;
    a->head->used += n;
    return p;
}
void rh_arena_free(rh_arena *a) {
    while (a->head) { rh_block *n = a->head->next; free(a->head); a->head = n; }
}
void rh_ivec_push(rh_ivec *a, int32_t x) {
    if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 16; a->v = rh_realloc(a->v, a->cap * sizeof(int32_t)); }
    a->v[a->n++] = x;
}
void rh_ivec_free(rh_ivec *a) { free(a->v); a->v = NULL; a->n = a->cap = 0; }

size_t rh_utf8_decode(const char *s, size_t n, uint32_t **out) {
    uint32_t *o = rh_malloc((n + 1) * sizeof(uint32_t));
    size_t m = 0;
    for (size_t i = 0; i < n;) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t k;
        if (c < 0x80) { cp = c; k = 1; }
        else if ((c >> 5) == 6 && i + 1 < n) { cp = ((c & 31u) << 6) | ((unsigned char)s[i + 1] & 63u); k = 2; }
        else if ((c >> 4) == 14 && i + 2 < n) {
            cp = ((c & 15u) << 12) | (((unsigned char)s[i + 1] & 63u) << 6) | ((unsigned char)s[i + 2] & 63u); k = 3;
        } else if ((c >> 3) == 30 && i + 3 < n) {
            cp = ((c & 7u) << 18) | (((unsigned char)s[i + 1] & 63u) << 12) | (((unsigned char)s[i + 2] & 63u) << 6) |
                 ((unsigned char)s[i + 3] & 63u);
            k = 4;
        } else { cp = 0xfffd; k = 1; }
        o[m++] = cp;
        i += k;
    }
    *out = o;
    return m;
}

/* Python str.isspace / Rust char::is_whitespace / Oniguruma \s (the tokenizer regex) */
int rh_py_isspace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
static int rust_ws(uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || c == 0x20 || c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
           c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
static int re_ws(uint32_t c) { return (c >= 0x09 && c <= 0x0d) || c == 0x85 || (uc_flags(c) & RH_UC_Z); }
static int re_l(uint32_t c) { return (uc_flags(c) & RH_UC_L) != 0; }
static int re_n(uint32_t c) { return (uc_flags(c) & RH_UC_N) != 0; }
static int re_o(uint32_t c) { return !re_ws(c) && !re_l(c) && !re_n(c); }
static int attach(uint32_t c) { return (uc_flags(c) & RH_UC_M) || (c >= 0x1160 && c <= 0x11ff) || (c >= 0xd7b0 && c <= 0xd7ff); }

/* ---------------------------------------------------------------- hash maps */
typedef struct { char *key; uint32_t klen; int32_t val; } skv;
typedef struct { skv *slot; uint32_t cap; } smap; /* string -> int, open addressing */
static uint32_t fnv(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 16777619u;
    return h;
}
static void smap_put(smap *m, const char *k, size_t n, int32_t v) {
    uint32_t i = fnv(k, n) & (m->cap - 1);
    while (m->slot[i].key) {
        if (m->slot[i].klen == n && !memcmp(m->slot[i].key, k, n)) return; /* first wins */
        i = (i + 1) & (m->cap - 1);
    }
    m->slot[i].key = rh_malloc(n + 1);
    memcpy(m->slot[i].key, k, n);
    m->slot[i].key[n] = 0;
    m->slot[i].klen = (uint32_t)n;
    m->slot[i].val = v;
}
static int32_t smap_get(const smap *m, const char *k, size_t n) {
    uint32_t i = fnv(k, n) & (m->cap - 1);
    while (m->slot[i].key) {
        if (m->slot[i].klen == n && !memcmp(m->slot[i].key, k, n)) return m->slot[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}
typedef struct { uint32_t key; int32_t rank, id; } mkv; /* (a << 16 | b) + 1 -> merge */
typedef struct { mkv *slot; uint32_t cap; } mmap_;
static uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
static void mmap_put(mmap_ *m, uint32_t key, int32_t rank, int32_t id) {
    uint32_t i = mix(key) & (m->cap - 1);
    while (m->slot[i].key) {
        if (m->slot[i].key == key) return; /* first (lowest rank) wins */
        i = (i + 1) & (m->cap - 1);
    }
    m->slot[i].key = key; m->slot[i].rank = rank; m->slot[i].id = id;
}
static const mkv *mmap_get(const mmap_ *m, uint32_t key) {
    uint32_t i = mix(key) & (m->cap - 1);
    while (m->slot[i].key) {
        if (m->slot[i].key == key) return &m->slot[i];
        i = (i + 1) & (m->cap - 1);
    }
    return NULL;
}

/* ---------------------------------------------------------------- tokenizer */
typedef struct { uint32_t *cps; size_t n; int32_t id; int lstrip, normalized; } added_tok;

struct rh_tokenizer {
    smap vocab, added;
    mmap_ merges;
    int32_t byte_id[256];
    added_tok *tok;
    size_t ntok;
};

static uint32_t byte_char[256];
static void init_byte_char(void) {
    int n = 0;
    int is[256] = {0};
    for (int b = 33; b <= 126; b++) is[b] = 1;
    for (int b = 161; b <= 172; b++) is[b] = 1;
    for (int b = 174; b <= 255; b++) is[b] = 1;
    for (int b = 0; b < 256; b++) byte_char[b] = is[b] ? (uint32_t)b : (uint32_t)(256 + n++);
}

static uint32_t pow2_above(size_t n) { uint32_t c = 16; while (c < n * 2) c <<= 1; return c; }

rh_tokenizer *rh_tok_load(const char *text, size_t len) {
    init_byte_char();
    rh_arena a = {0};
    jnode *root = rh_json_parse(&a, text, len);
    jnode *model = rh_json_get(root, "model");
    jnode *vocab = rh_json_get(model, "vocab"), *merges = rh_json_get(model, "merges");
    if (!vocab || !merges) { rh_arena_free(&a); return NULL; }
    rh_tokenizer *t = rh_malloc(sizeof *t);
    memset(t, 0, sizeof *t);
    t->vocab.cap = pow2_above(vocab->count);
    t->vocab.slot = calloc(t->vocab.cap, sizeof(skv));
    for (jnode *v = vocab->first; v; v = v->next) smap_put(&t->vocab, v->key, v->klen, (int32_t)v->num);
    for (int b = 0; b < 256; b++) {
        char u[4];
        const size_t k = rh_utf8_put(u, byte_char[b]);
        t->byte_id[b] = smap_get(&t->vocab, u, k);
    }
    t->merges.cap = pow2_above(merges->count);
    t->merges.slot = calloc(t->merges.cap, sizeof(mkv));
    int32_t rank = 0;
    char buf[1024];
    for (jnode *m = merges->first; m; m = m->next, rank++) {
        const char *x, *y;
        size_t xl, yl;
        if (m->t == J_ARR) { x = m->first->s; xl = m->first->len; y = m->first->next->s; yl = m->first->next->len; }
        else { const char *sp = memchr(m->s, ' ', m->len); x = m->s; xl = (size_t)(sp - m->s); y = sp + 1; yl = m->len - xl - 1; }
        const int32_t ia = smap_get(&t->vocab, x, xl), ib = smap_get(&t->vocab, y, yl);
        if (xl + yl >= sizeof buf) continue;
        memcpy(buf, x, xl);
        memcpy(buf + xl, y, yl);
        const int32_t id = smap_get(&t->vocab, buf, xl + yl);
        if (ia < 0 || ib < 0 || id < 0) continue;
        mmap_put(&t->merges, ((uint32_t)ia << 16 | (uint32_t)ib) + 1u, rank, id);
    }
    jnode *added = rh_json_get(root, "added_tokens");
    t->added.cap = pow2_above(added ? added->count + 1 : 1);
    t->added.slot = calloc(t->added.cap, sizeof(skv));
    if (added) {
        t->tok = rh_malloc(added->count * sizeof(added_tok));
        for (jnode *x = added->first; x; x = x->next) {
            jnode *c = rh_json_get(x, "content"), *id = rh_json_get(x, "id"), *ls = rh_json_get(x, "lstrip"),
                  *nm = rh_json_get(x, "normalized");
            added_tok *k = &t->tok[t->ntok++];
            k->n = rh_utf8_decode(c->s, c->len, &k->cps);
            k->id = (int32_t)id->num;
            k->lstrip = ls && ls->t == J_TRUE;
            k->normalized = nm && nm->t == J_TRUE;
            smap_put(&t->added, c->s, c->len, k->id);
        }
    }
    rh_arena_free(&a);
    return t;
}

void rh_tok_free(rh_tokenizer *t) {
    if (!t) return;
    for (uint32_t i = 0; i < t->vocab.cap; i++) free(t->vocab.slot[i].key);
    for (uint32_t i = 0; i < t->added.cap; i++) free(t->added.slot[i].key);
    free(t->vocab.slot); free(t->added.slot); free(t->merges.slot);
    for (size_t i = 0; i < t->ntok; i++) free(t->tok[i].cps);
    free(t->tok);
    free(t);
}

int32_t rh_tok_id(const rh_tokenizer *t, const char *name) {
    const size_t n = strlen(name);
    const int32_t a = smap_get(&t->added, name, n);
    return a >= 0 ? a : smap_get(&t->vocab, name, n);
}

/* Leftmost-longest added-token matches among tokens with the given `normalized` flag:
   out[k] = {start, end, token index}; returns the count. */
static size_t matches(const rh_tokenizer *t, int normalized, const uint32_t *s, size_t n, size_t (*out)[3]) {
    size_t cnt = 0, last = 0;
    for (size_t i = 0; i < n;) {
        const added_tok *hit = NULL;
        size_t hi = 0;
        for (size_t k = 0; k < t->ntok; k++) { /* longest content first, ties: first listed */
            const added_tok *a = &t->tok[k];
            if (a->normalized != normalized || !a->n || a->n > n - i || a->cps[0] != s[i]) continue;
            if (memcmp(a->cps, s + i, a->n * sizeof(uint32_t))) continue;
            if (!hit || a->n > hit->n) { hit = a; hi = k; }
        }
        if (!hit) { i++; continue; }
        size_t start = i;
        if (hit->lstrip) while (start > last && rust_ws(s[start - 1])) start--;
        out[cnt][0] = start; out[cnt][1] = i + hit->n; out[cnt][2] = hi;
        cnt++;
        i += hit->n;
        last = i;
    }
    return cnt;
}

/* ---- BPE of one pre-token (bytes) -> (id, byte start, byte end) triples appended to res ---- */
typedef struct { int32_t id; int32_t s, e, prev, next, len; } sym;
typedef struct { int32_t pos, rank, id; } hent;
static int hlt(const hent *x, const hent *y) { return x->rank < y->rank || (x->rank == y->rank && x->pos < y->pos); }
static void hpush(hent *h, size_t *n, hent x) {
    size_t i = (*n)++;
    h[i] = x;
    while (i > 0) { size_t p = (i - 1) >> 1; if (hlt(&h[i], &h[p])) { hent t = h[i]; h[i] = h[p]; h[p] = t; i = p; } else break; }
}
static hent hpop(hent *h, size_t *n) {
    hent top = h[0];
    h[0] = h[--(*n)];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && hlt(&h[l], &h[m])) m = l;
        if (r < *n && hlt(&h[r], &h[m])) m = r;
        if (m == i) break;
        hent t = h[i]; h[i] = h[m]; h[m] = t; i = m;
    }
    return top;
}
static size_t bpe(const rh_tokenizer *t, const unsigned char *bytes, size_t nb, int32_t (*res)[3]) {
    sym *s = rh_malloc((nb + 1) * sizeof(sym));
    size_t ns = 0;
    for (size_t i = 0; i < nb; i++) {
        const int32_t id = t->byte_id[bytes[i]];
        if (id < 0) continue;
        s[ns] = (sym){id, (int32_t)i, (int32_t)i + 1, (int32_t)ns - 1, (int32_t)ns + 1, 1};
        ns++;
    }
    if (ns) s[ns - 1].next = -1;
    hent *h = rh_malloc((4 * nb + 4) * sizeof(hent));
    size_t nh = 0, hcap = 4 * nb + 4;
    for (size_t i = 0; i + 1 < ns; i++) {
        const mkv *m = mmap_get(&t->merges, ((uint32_t)s[i].id << 16 | (uint32_t)s[i + 1].id) + 1u);
        if (m) hpush(h, &nh, (hent){(int32_t)i, m->rank, m->id});
    }
    while (nh) {
        const hent top = hpop(h, &nh);
        sym *cur = &s[top.pos];
        if (cur->len == 0 || cur->next == -1) continue;
        sym *right = &s[cur->next];
        const mkv *m = mmap_get(&t->merges, ((uint32_t)cur->id << 16 | (uint32_t)right->id) + 1u);
        if (!m || m->id != top.id) continue;
        cur->id = top.id;
        cur->e = right->e;
        cur->len += right->len;
        right->len = 0;
        cur->next = right->next;
        if (cur->next >= 0) s[cur->next].prev = top.pos;
        if (nh + 2 > hcap) { hcap *= 2; h = rh_realloc(h, hcap * sizeof(hent)); }
        if (cur->prev >= 0) {
            const mkv *pm = mmap_get(&t->merges, ((uint32_t)s[cur->prev].id << 16 | (uint32_t)cur->id) + 1u);
            if (pm) hpush(h, &nh, (hent){cur->prev, pm->rank, pm->id});
        }
        if (cur->next >= 0) {
            const mkv *nm = mmap_get(&t->merges, ((uint32_t)cur->id << 16 | (uint32_t)s[cur->next].id) + 1u);
            if (nm) hpush(h, &nh, (hent){top.pos, nm->rank, nm->id});
        }
    }
    size_t k = 0;
    for (size_t i = 0; i < ns; i++)
        if (s[i].len > 0) { res[k][0] = s[i].id; res[k][1] = s[i].s; res[k][2] = s[i].e; k++; }
    free(h);
    free(s);
    return k;
}

/* GPT-2 pre-tokenizer: length of the piece starting at i (scanner equivalent of the regex alternation). */
static size_t piece(const uint32_t *c, size_t i, size_t n) {
    if (c[i] == '\'' && i + 1 < n) {
        const uint32_t x = c[i + 1];
        if (x == 's' || x == 't' || x == 'm' || x == 'd') return 2;
        if (i + 2 < n && ((x == 'r' && c[i + 2] == 'e') || (x == 'v' && c[i + 2] == 'e') || (x == 'l' && c[i + 2] == 'l')))
            return 3;
    }
    int (*cls[3])(uint32_t) = {re_l, re_n, re_o};
    for (int k = 0; k < 3; k++) {
        size_t j = i;
        if (c[j] == ' ' && j + 1 < n && cls[k](c[j + 1])) j++;
        if (cls[k](c[j])) {
            while (j < n && cls[k](c[j])) j++;
            return j - i;
        }
    }
    /* whitespace run; \s+(?!\S) leaves the last space before a non-space for the next piece */
    size_t j = i;
    while (j < n && re_ws(c[j])) j++;
    const size_t run = j - i;
    if (j < n && run >= 2) return run - 1;
    return run ? run : 1;
}

/* Pre-tokenize and BPE one normalized stretch cps[0..n) whose characters map to original ranges al[k]. */
static void model(const rh_tokenizer *t, const uint32_t *cps, const int32_t (*al)[2], size_t n, rh_ivec *ids,
                  rh_ivec *offs) {
    unsigned char *bytes = rh_malloc(4 * n + 4);
    int32_t *borig = rh_malloc((4 * n + 4) * sizeof(int32_t));
    int32_t (*res)[3] = rh_malloc((4 * n + 4) * sizeof *res);
    for (size_t i = 0; i < n;) {
        const size_t len = piece(cps, i, n);
        size_t nb = 0;
        for (size_t c = i; c < i + len; c++) {
            char u[4];
            const size_t k = rh_utf8_put(u, cps[c]);
            for (size_t q = 0; q < k; q++) { bytes[nb] = (unsigned char)u[q]; borig[nb] = (int32_t)c; nb++; }
        }
        const size_t nr = bpe(t, bytes, nb, res);
        for (size_t r = 0; r < nr; r++) {
            rh_ivec_push(ids, res[r][0]);
            rh_ivec_push(offs, al[borig[res[r][1]]][0]);
            rh_ivec_push(offs, al[borig[res[r][2] - 1]][1]);
        }
        i += len;
    }
    free(bytes);
    free(borig);
    free(res);
}

/* NFC of src[0..n) with HF alignments (each output character -> original [start, end) at base + index). */
static size_t nfc_aligned(const uint32_t *src, size_t n, size_t base, uint32_t **ocps, int32_t (**oal)[2]) {
    uint32_t *norm;
    const size_t nn = uc_nfc(src, n, &norm);
    if (nn == n && !memcmp(norm, src, n * sizeof(uint32_t))) {
        int32_t (*al)[2] = rh_malloc((n + 1) * sizeof *al);
        for (size_t i = 0; i < n; i++) { al[i][0] = (int32_t)(base + i); al[i][1] = (int32_t)(base + i + 1); }
        *ocps = norm;
        *oal = al;
        return n;
    }
    free(norm);
    uint32_t *out = rh_malloc((n * 4 + 4) * sizeof(uint32_t));
    int32_t (*al)[2] = rh_malloc((n * 4 + 4) * sizeof *al);
    size_t m = 0;
    for (size_t s = 0; s < n;) {
        size_t e = s + 1;
        while (e < n && attach(src[e])) e++;
        uint32_t *nc;
        const size_t nl = uc_nfc(src + s, e - s, &nc);
        /* decomposed pieces of every source character, each remembering its source */
        uint32_t *dec = rh_malloc(((e - s) * 18 + 1) * sizeof(uint32_t));
        size_t *dsrc = rh_malloc(((e - s) * 18 + 1) * sizeof(size_t));
        char *used = NULL;
        size_t nd = 0;
        for (size_t j = s; j < e; j++) {
            uint32_t *d;
            const size_t k = uc_nfd(src + j, 1, &d);
            for (size_t q = 0; q < k; q++) { dec[nd] = d[q]; dsrc[nd] = j - s; nd++; }
            free(d);
        }
        used = calloc(nd + 1, 1);
        size_t prev = 0;
        for (size_t o = 0; o < nl; o++) {
            uint32_t *d;
            const size_t k = uc_nfd(nc + o, 1, &d);
            size_t src_i = prev;
            for (size_t q = 0; q < nd; q++) if (!used[q] && dec[q] == d[0]) { src_i = dsrc[q]; break; }
            for (size_t x = 0; x < k; x++)
                for (size_t q = 0; q < nd; q++) if (!used[q] && dec[q] == d[x]) { used[q] = 1; break; }
            free(d);
            out[m] = nc[o];
            al[m][0] = (int32_t)(base + s + src_i);
            al[m][1] = (int32_t)(base + s + src_i + 1);
            m++;
            prev = src_i;
        }
        free(used); free(dec); free(dsrc); free(nc);
        s = e;
    }
    *ocps = out;
    *oal = al;
    return m;
}

void rh_tok_encode(rh_tokenizer *t, const uint32_t *text, size_t n, rh_ivec *ids, rh_ivec *offs) {
    size_t (*sp)[3] = rh_malloc((n + 1) * sizeof *sp);
    const size_t ns = matches(t, 0, text, n, sp);
    size_t pos = 0;
    for (size_t k = 0; k <= ns; k++) {
        const size_t s = k < ns ? sp[k][0] : n;
        if (s > pos) { /* a stretch of normal text: NFC, normalized added tokens, then the model */
            uint32_t *nc;
            int32_t (*al)[2];
            const size_t nl = nfc_aligned(text + pos, s - pos, pos, &nc, &al);
            size_t (*mp)[3] = rh_malloc((nl + 1) * sizeof *mp);
            const size_t nm = matches(t, 1, nc, nl, mp);
            size_t p = 0;
            for (size_t q = 0; q <= nm; q++) {
                const size_t a = q < nm ? mp[q][0] : nl;
                if (a > p) model(t, nc + p, (const int32_t (*)[2])(al + p), a - p, ids, offs);
                if (q < nm) {
                    rh_ivec_push(ids, t->tok[mp[q][2]].id);
                    rh_ivec_push(offs, al[mp[q][0]][0]);
                    rh_ivec_push(offs, al[mp[q][1] - 1][1]);
                    p = mp[q][1];
                }
            }
            free(mp); free(nc); free(al);
        }
        if (k < ns) {
            rh_ivec_push(ids, t->tok[sp[k][2]].id);
            rh_ivec_push(offs, (int32_t)sp[k][0]);
            rh_ivec_push(offs, (int32_t)sp[k][1]);
            pos = sp[k][1];
        }
    }
    free(sp);
}
