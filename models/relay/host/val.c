/* Values of the native Relay host (port of py.mjs semantics): a small tagged value for program results and
   arguments, Python str()/repr() of values, JavaScript number formatting (shortest round trip), Python float repr,
   Python float()/int() parsing, and the output serializer (Python json.dumps separators, as web/gauge_run.mjs). */
#include "relay_host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- string builder */
void sb_put(sb *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->s = rh_realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}
void sb_str(sb *b, const char *s) { sb_put(b, s, strlen(s)); }
void sb_cp(sb *b, uint32_t cp) { char u[4]; sb_put(b, u, rh_utf8_put(u, cp)); }
void sb_free(sb *b) { free(b->s); b->s = NULL; b->n = b->cap = 0; }

/* ---------------------------------------------------------------- numbers */
/* Shortest digits d (no point) and decimal exponent n with value = 0.d * 10^n (JavaScript / Python repr). */
static void shortest(double v, char *digits, int *nexp) {
    char buf[40];
    for (int p = 1; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p - 1, v);
        if (strtod(buf, NULL) == v) break;
    }
    /* buf = d.ddddde±XX */
    int k = 0;
    const char *e = strchr(buf, 'e');
    for (const char *q = buf; q < e; q++) if (*q >= '0' && *q <= '9') digits[k++] = *q;
    while (k > 1 && digits[k - 1] == '0') k--; /* %.*e keeps trailing zeros only when needed: trim */
    digits[k] = 0;
    *nexp = atoi(e + 1) + 1;
}

/* Number.prototype.toString(10) */
void js_num(sb *b, double v) {
    if (isnan(v)) { sb_str(b, "NaN"); return; }
    if (isinf(v)) { sb_str(b, v < 0 ? "-Infinity" : "Infinity"); return; }
    if (v == 0) { sb_str(b, "0"); return; }
    if (v < 0) { sb_str(b, "-"); v = -v; }
    char d[24];
    int n;
    shortest(v, d, &n);
    const int k = (int)strlen(d);
    if (k <= n && n <= 21) {
        sb_str(b, d);
        for (int i = 0; i < n - k; i++) sb_str(b, "0");
    } else if (0 < n && n <= 21) {
        sb_put(b, d, (size_t)n);
        sb_str(b, ".");
        sb_str(b, d + n);
    } else if (-6 < n && n <= 0) {
        sb_str(b, "0.");
        for (int i = 0; i < -n; i++) sb_str(b, "0");
        sb_str(b, d);
    } else {
        sb_put(b, d, 1);
        if (k > 1) { sb_str(b, "."); sb_str(b, d + 1); }
        char e[16];
        snprintf(e, sizeof e, "e%c%d", n - 1 < 0 ? '-' : '+', abs(n - 1));
        sb_str(b, e);
    }
}

/* Python repr(float) (py.mjs floatRepr) */
void py_float(sb *b, double v) {
    if (isnan(v)) { sb_str(b, "nan"); return; }
    if (isinf(v)) { sb_str(b, v < 0 ? "-inf" : "inf"); return; }
    if (v == 0) { sb_str(b, signbit(v) ? "-0.0" : "0.0"); return; }
    if (v < 0) { sb_str(b, "-"); v = -v; }
    char d[24];
    int n;
    shortest(v, d, &n);
    const int e = n - 1, k = (int)strlen(d);
    if (e >= -4 && e < 16) {
        if (e >= 0) {
            if (k > e + 1) { sb_put(b, d, (size_t)e + 1); sb_str(b, "."); sb_str(b, d + e + 1); }
            else { sb_str(b, d); for (int i = 0; i < e + 1 - k; i++) sb_str(b, "0"); sb_str(b, ".0"); }
        } else {
            sb_str(b, "0.");
            for (int i = 0; i < -e - 1; i++) sb_str(b, "0");
            sb_str(b, d);
        }
        return;
    }
    sb_put(b, d, 1);
    if (k > 1) { sb_str(b, "."); sb_str(b, d + 1); }
    char x[16];
    snprintf(x, sizeof x, "e%c%02d", e < 0 ? '-' : '+', abs(e));
    sb_str(b, x);
}

/* ---------------------------------------------------------------- values */
rv rv_undef(void) { rv v = {0}; v.t = RV_UNDEF; return v; }
rv rv_null(void) { rv v = {0}; v.t = RV_NULL; return v; }
rv rv_bool(int b) { rv v = {0}; v.t = RV_BOOL; v.num = b ? 1 : 0; return v; }
rv rv_int(double x) { rv v = {0}; v.t = RV_NUM; v.num = x; v.isfloat = 0; return v; }
/* pyFloat(v): a computed Python float (integral values keep their float-ness) */
rv rv_float(double x) { rv v = {0}; v.t = RV_NUM; v.num = x; v.isfloat = 1; return v; }
/* a plain JavaScript number: a float only when not integral */
rv rv_jsnum(double x) { rv v = {0}; v.t = RV_NUM; v.num = x; v.isfloat = !(isfinite(x) && x == floor(x)); return v; }
rv rv_str(const char *s, size_t n) { rv v = {0}; v.t = RV_STR; v.s = s; v.len = n; return v; }

/* A JSON node as a value (parseJSON: fraction/exponent literals are Python floats). */
rv rv_json(const jnode *j) {
    if (!j) return rv_undef();
    switch (j->t) {
    case J_NULL: return rv_null();
    case J_TRUE: return rv_bool(1);
    case J_FALSE: return rv_bool(0);
    case J_STR: return rv_str(j->s, j->len);
    case J_NUM: {
        int fl = 0;
        for (size_t i = 0; i < j->len; i++) if (strchr(".eE", j->s[i])) fl = 1;
        if (!strcmp(j->s, "NaN") || strstr(j->s, "Infinity")) {
            rv v = rv_float(!strcmp(j->s, "NaN") ? NAN : (j->s[0] == '-' ? -INFINITY : INFINITY));
            return v;
        }
        return fl ? rv_float(j->num) : rv_jsnum(j->num);
    }
    default: { rv v = {0}; v.t = RV_JSON; v.j = j; return v; }
    }
}

int rv_is_pyfloat(const rv *v) { return v->t == RV_NUM && v->isfloat; }

int rv_truthy(const rv *v) {
    switch (v->t) {
    case RV_UNDEF: case RV_NULL: return 0;
    case RV_BOOL: return v->num != 0;
    case RV_NUM: return v->num != 0 && !isnan(v->num) ? 1 : (isnan(v->num) ? 1 : 0);
    case RV_STR: return v->len > 0;
    case RV_JSON: return v->j->count > 0;
    case RV_LIST: return v->n > 0;
    }
    return 0;
}

/* repr(str) (py.mjs strRepr) */
static void str_repr(sb *b, const char *s, size_t n) {
    int sq = 0, dq = 0;
    for (size_t i = 0; i < n; i++) { if (s[i] == '\'') sq = 1; if (s[i] == '"') dq = 1; }
    const char q = sq && !dq ? '"' : '\'';
    sb_put(b, &q, 1);
    uint32_t *cps;
    const size_t m = rh_utf8_decode(s, n, &cps);
    for (size_t i = 0; i < m; i++) {
        const uint32_t c = cps[i];
        char e[16];
        if (c == '\\') sb_str(b, "\\\\");
        else if (c == (uint32_t)q) { sb_str(b, "\\"); sb_put(b, &q, 1); }
        else if (c == '\t') sb_str(b, "\\t");
        else if (c == '\n') sb_str(b, "\\n");
        else if (c == '\r') sb_str(b, "\\r");
        else if (c < 0x20 || c == 0x7f || (c != ' ' && (uc_flags(c) & (RH_UC_C | RH_UC_Z)))) {
            if (c < 0x100) snprintf(e, sizeof e, "\\x%02x", c);
            else if (c < 0x10000) snprintf(e, sizeof e, "\\u%04x", c);
            else snprintf(e, sizeof e, "\\U%08x", c);
            sb_str(b, e);
        } else sb_cp(b, c);
    }
    free(cps);
    sb_put(b, &q, 1);
}

/* ---------------------------------------------------------------- JavaScript object semantics */
static int index_key(const jnode *c, double *v) { /* canonical array index: "0" or [1-9][0-9]*, below 2^32 - 1 */
    if (!c->klen || c->klen > 10 || (c->key[0] == '0' && c->klen > 1)) return 0;
    double x = 0;
    for (size_t i = 0; i < c->klen; i++) {
        if (c->key[i] < '0' || c->key[i] > '9') return 0;
        x = x * 10 + (c->key[i] - '0');
    }
    if (x >= 4294967295.0) return 0;
    *v = x;
    return 1;
}
size_t rh_js_members(const jnode *obj, rh_arena *A, const jnode ***out) {
    const jnode **m = rh_arena_alloc(A, (obj->count + 1) * sizeof *m);
    size_t n = 0, nidx = 0;
    for (const jnode *c = obj->first; c; c = c->next) {
        size_t k = 0;
        while (k < n && !(m[k]->klen == c->klen && !memcmp(m[k]->key, c->key, c->klen))) k++;
        if (k < n) m[k] = c; /* duplicate key: first position, last value */
        else m[n++] = c;
    }
    double v;
    for (size_t k = 0; k < n; k++) nidx += index_key(m[k], &v);
    if (nidx) { /* integer keys first, ascending; the rest keep their order */
        const jnode **o = rh_arena_alloc(A, (n + 1) * sizeof *o);
        size_t a = 0, b = nidx;
        for (size_t k = 0; k < n; k++) {
            if (!index_key(m[k], &v)) { o[b++] = m[k]; continue; }
            size_t i = a++;
            double w;
            while (i > 0 && (index_key(o[i - 1], &w), w > v)) { o[i] = o[i - 1]; i--; }
            o[i] = m[k];
        }
        m = o;
    }
    *out = m;
    return n;
}
int rh_js_truthy(const jnode *j) {
    if (!j) return 0;
    switch (j->t) {
    case J_NULL: case J_FALSE: return 0;
    case J_STR: return j->len > 0;
    case J_NUM: { rv v = rv_json(j); return v.num != 0 && !isnan(v.num); }
    default: return 1;
    }
}

/* Math.exp: V8 runs fdlibm's exp (base/ieee754.cc); ported so that every platform rounds like the JavaScript host */
double rh_js_exp(double x) {
    static const double one = 1.0, halF[2] = {0.5, -0.5}, o_threshold = 7.09782712893383973096e+02,
        u_threshold = -7.45133219101941108420e+02, ln2HI[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01},
        ln2LO[2] = {1.90821492927058770002e-10, -1.90821492927058770002e-10}, invln2 = 1.44269504088896338700e+00,
        P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
        P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08, E = 2.718281828459045;
    static volatile double huge = 1.0e+300, twom1000 = 9.33263618503218878990e-302, two1023 = 8.988465674311579539e307;
    double y, hi = 0.0, lo = 0.0, c, t, twopk;
    int32_t k = 0, xsb;
    uint64_t bits;
    memcpy(&bits, &x, 8);
    uint32_t hx = (uint32_t)(bits >> 32);
    xsb = (int32_t)((hx >> 31) & 1);
    hx &= 0x7FFFFFFF;
    if (hx >= 0x40862E42) {
        if (hx >= 0x7FF00000) {
            const uint32_t lx = (uint32_t)bits;
            if (((hx & 0xFFFFF) | lx) != 0) return x + x;
            return xsb == 0 ? x : 0.0;
        }
        if (x > o_threshold) return huge * huge;
        if (x < u_threshold) return twom1000 * twom1000;
    }
    if (hx > 0x3FD62E42) {
        if (hx < 0x3FF0A2B2) {
            if (x == 1.0) return E;
            hi = x - ln2HI[xsb];
            lo = ln2LO[xsb];
            k = 1 - xsb - xsb;
        } else {
            k = (int32_t)(invln2 * x + halF[xsb]);
            t = k;
            hi = x - t * ln2HI[0];
            lo = t * ln2LO[0];
        }
        x = hi - lo;
    } else if (hx < 0x3E300000) {
        if (huge + x > one) return one + x;
    } else k = 0;
    t = x * x;
    const uint64_t tw = (uint64_t)(uint32_t)(0x3FF00000 + ((k >= -1021 ? k : k + 1000) * (1 << 20))) << 32;
    memcpy(&twopk, &tw, 8);
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return one - ((x * c) / (c - 2.0) - x);
    y = one - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021) {
        if (k == 1024) return y * 2.0 * two1023;
        return y * twopk;
    }
    return y * twopk * twom1000;
}

static void json_repr(sb *b, const jnode *j);
static void num_str(sb *b, const rv *v) {
    if (v->isfloat) py_float(b, v->num);
    else js_num(b, v->num); /* String(int) */
}
static void scalar_repr(sb *b, const jnode *j) {
    rv v = rv_json(j);
    if (v.t == RV_STR) str_repr(b, v.s, v.len);
    else if (v.t == RV_JSON) json_repr(b, j);
    else py_str(b, &v);
}
static void json_repr(sb *b, const jnode *j) {
    if (j->t == J_ARR) {
        sb_str(b, "[");
        for (const jnode *c = j->first; c; c = c->next) { if (c != j->first) sb_str(b, ", "); scalar_repr(b, c); }
        sb_str(b, "]");
    } else if (j->t == J_OBJ) {
        rh_arena A = {0};
        const jnode **m;
        const size_t n = rh_js_members(j, &A, &m);
        sb_str(b, "{");
        for (size_t k = 0; k < n; k++) {
            if (k) sb_str(b, ", ");
            str_repr(b, m[k]->key, m[k]->klen);
            sb_str(b, ": ");
            scalar_repr(b, m[k]);
        }
        sb_str(b, "}");
        rh_arena_free(&A);
    } else scalar_repr(b, j);
}

/* str() of a value (py.mjs pyStr) */
void py_str(sb *b, const rv *v) {
    switch (v->t) {
    case RV_UNDEF: case RV_NULL: sb_str(b, "None"); return;
    case RV_BOOL: sb_str(b, v->num ? "True" : "False"); return;
    case RV_NUM: num_str(b, v); return;
    case RV_STR: sb_put(b, v->s, v->len); return;
    case RV_JSON: json_repr(b, v->j); return;
    case RV_LIST:
        sb_str(b, "[");
        for (size_t i = 0; i < v->n; i++) {
            if (i) sb_str(b, ", ");
            if (v->items[i].t == RV_STR) str_repr(b, v->items[i].s, v->items[i].len);
            else py_str(b, &v->items[i]);
        }
        sb_str(b, "]");
        return;
    }
}

/* ---------------------------------------------------------------- output (gauge_run dumps) */
void js_string(sb *b, const char *s, size_t n) {
    sb_str(b, "\"");
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        char e[8];
        if (c == '"') sb_str(b, "\\\"");
        else if (c == '\\') sb_str(b, "\\\\");
        else if (c == '\n') sb_str(b, "\\n");
        else if (c == '\r') sb_str(b, "\\r");
        else if (c == '\t') sb_str(b, "\\t");
        else if (c == '\b') sb_str(b, "\\b");
        else if (c == '\f') sb_str(b, "\\f");
        else if (c < 0x20) { snprintf(e, sizeof e, "\\u%04x", c); sb_str(b, e); }
        else if (c == 0xed && i + 2 < n && ((unsigned char)s[i + 1] & 0xe0) == 0xa0) { /* lone surrogate */
            const uint32_t cp = ((c & 15u) << 12) | (((unsigned char)s[i + 1] & 63u) << 6) | ((unsigned char)s[i + 2] & 63u);
            snprintf(e, sizeof e, "\\u%04x", cp);
            sb_str(b, e);
            i += 2;
        } else sb_put(b, (const char *)&s[i], 1);
    }
    sb_str(b, "\"");
}

static void dumps_json(sb *b, const jnode *j);
void rv_dumps(sb *b, const rv *v) {
    switch (v->t) {
    case RV_UNDEF: case RV_NULL: sb_str(b, "null"); return;
    case RV_BOOL: sb_str(b, v->num ? "true" : "false"); return;
    case RV_NUM:
        if (!isfinite(v->num)) { sb_str(b, "null"); return; }
        js_num(b, v->num);
        if (v->isfloat && v->num == floor(v->num)) sb_str(b, ".0");
        return;
    case RV_STR: js_string(b, v->s, v->len); return;
    case RV_JSON: dumps_json(b, v->j); return;
    case RV_LIST:
        sb_str(b, "[");
        for (size_t i = 0; i < v->n; i++) { if (i) sb_str(b, ", "); rv_dumps(b, &v->items[i]); }
        sb_str(b, "]");
        return;
    }
}
static void dumps_json(sb *b, const jnode *j) {
    if (j->t == J_ARR) {
        sb_str(b, "[");
        for (const jnode *c = j->first; c; c = c->next) { if (c != j->first) sb_str(b, ", "); dumps_json(b, c); }
        sb_str(b, "]");
    } else if (j->t == J_OBJ) {
        rh_arena A = {0};
        const jnode **m;
        const size_t n = rh_js_members(j, &A, &m);
        sb_str(b, "{");
        for (size_t k = 0; k < n; k++) {
            if (k) sb_str(b, ", ");
            js_string(b, m[k]->key, m[k]->klen);
            sb_str(b, ": ");
            dumps_json(b, m[k]);
        }
        sb_str(b, "}");
        rh_arena_free(&A);
    } else {
        rv v = rv_json(j);
        rv_dumps(b, &v);
    }
}

/* ---------------------------------------------------------------- Python parsing helpers */
/* Unicode decimal digits -> ASCII (asciiDigits) */
size_t py_ascii_digits(const uint32_t *s, size_t n, uint32_t *out) {
    for (size_t i = 0; i < n; i++) {
        uint32_t c = s[i];
        if (c >= 0x80 && (uc_flags(c) & RH_UC_ND)) {
            uint32_t z = c;
            while (uc_flags(z - 1) & RH_UC_ND) z--;
            c = '0' + (c - z) % 10;
        }
        out[i] = c;
    }
    return n;
}

static size_t strip_bounds(const uint32_t *s, size_t n, size_t *a) {
    size_t i = 0, j = n;
    while (i < j && rh_py_isspace(s[i])) i++;
    while (j > i && rh_py_isspace(s[j - 1])) j--;
    *a = i;
    return j - i;
}

/* float(str) -> 1 and *out, or 0 (ValueError) */
int py_float_parse(const uint32_t *s0, size_t n0, double *out) {
    uint32_t *t = rh_malloc((n0 + 1) * sizeof(uint32_t));
    py_ascii_digits(s0, n0, t);
    size_t a;
    const size_t n = strip_bounds(t, n0, &a);
    char *c = rh_malloc(n + 1);
    for (size_t i = 0; i < n; i++) c[i] = t[a + i] < 128 ? (char)t[a + i] : '?';
    c[n] = 0;
    free(t);
    /* inf / nan */
    const char *p = c + ((c[0] == '+' || c[0] == '-') ? 1 : 0);
    char low[16] = {0};
    for (int i = 0; p[i] && i < 15; i++) low[i] = (char)(p[i] >= 'A' && p[i] <= 'Z' ? p[i] + 32 : p[i]);
    if (strlen(p) < 16 && (!strcmp(low, "inf") || !strcmp(low, "infinity"))) { *out = c[0] == '-' ? -INFINITY : INFINITY; free(c); return 1; }
    if (strlen(p) < 16 && !strcmp(low, "nan")) { *out = NAN; free(c); return 1; }
    /* ^[+-]?(?:D(?:\.(?:D)?)?|\.D)(?:[eE][+-]?D)?$ with D = \d(?:_?\d)* */
    size_t i = (c[0] == '+' || c[0] == '-') ? 1 : 0;
    int digits = 0, ok = 1;
#define DIGRUN()                                                                          \
    do {                                                                                  \
        int d_ = 0;                                                                       \
        while (c[i] >= '0' && c[i] <= '9') {                                              \
            d_++; i++;                                                                    \
            if (c[i] == '_' && c[i + 1] >= '0' && c[i + 1] <= '9') i++;                   \
        }                                                                                 \
        digits = d_;                                                                      \
    } while (0)
    DIGRUN();
    const int intd = digits;
    if (c[i] == '.') {
        i++;
        DIGRUN();
        if (!intd && !digits) ok = 0;
    } else if (!intd) ok = 0;
    if (ok && (c[i] == 'e' || c[i] == 'E')) {
        i++;
        if (c[i] == '+' || c[i] == '-') i++;
        DIGRUN();
        if (!digits) ok = 0;
    }
#undef DIGRUN
    if (!ok || c[i]) { free(c); return 0; }
    /* drop underscores, then strtod (Number) */
    size_t w = 0;
    for (size_t k = 0; c[k]; k++) if (c[k] != '_') c[w++] = c[k];
    c[w] = 0;
    *out = strtod(c, NULL);
    free(c);
    return 1;
}

/* int(str) for captured digit strings -> 1 and *out, or 0 */
int py_int_parse(const uint32_t *s0, size_t n0, double *out) {
    uint32_t *t = rh_malloc((n0 + 1) * sizeof(uint32_t));
    py_ascii_digits(s0, n0, t);
    size_t a;
    size_t n = strip_bounds(t, n0, &a);
    char *c = rh_malloc(n + 1);
    size_t w = 0;
    for (size_t i = 0; i < n; i++) if (t[a + i] != '_') c[w++] = t[a + i] < 128 ? (char)t[a + i] : '?';
    c[w] = 0;
    free(t);
    size_t i = (c[0] == '+' || c[0] == '-') ? 1 : 0;
    if (!c[i]) { free(c); return 0; }
    for (size_t k = i; c[k]; k++) if (c[k] < '0' || c[k] > '9') { free(c); return 0; }
    *out = strtod(c, NULL); /* parseInt(t, 10) */
    free(c);
    return 1;
}

/* round() half to even; int(float) truncation */
double py_round(double x) {
    const double f = floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return fmod(f, 2) == 0 ? f : f + 1;
}
