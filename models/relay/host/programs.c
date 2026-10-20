/* Value programs (port of web/programs.mjs): number / time / date parsing, description options, coercion to the
   parameter type, and the program interpreter. Text is code points; spans are code point [start, end) indices.
   Functions that raise in Python / throw in JavaScript return 0 (the decoder then drops the argument). */
#include "relay_host.h"
#include "tables_data.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

rh_fix FIX;
void rh_set_fixes(int all) {
    rh_fix f = {0};
    if (all) f = (rh_fix){1, 1, 1, 1, 0, 1, 1, 1, 1}; /* orderTie measured no gain */
    FIX = f;
}

const double RH_NUM_SCALES[RH_N_NUM_SCALES] = {1, 0.01, 100, 1000, 1e6, 1e9, -1, 60, 3600, 1.0 / 60, 1e-3, 86400, 7, 0.5, 2};
const char *const RH_DATE_FORMATS[RH_N_DATE_FORMATS] = {"%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d", "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M", "%H:%M", "%H:%M:%S", "%I:%M %p", "%B %d, %Y", "%d %B %Y", "%Y%m%d", "%Y-%m"};

/* ---------------------------------------------------------------- helpers on code point strings */
typedef struct { uint32_t *v; size_t n; } cstr; /* owned code point string */

static cstr cs_from(const uint32_t *s, size_t n) {
    cstr c = {rh_malloc((n + 1) * sizeof(uint32_t)), n};
    if (n) memcpy(c.v, s, n * sizeof(uint32_t));
    return c;
}
static cstr cs_utf8(const char *s, size_t n) { cstr c; c.n = rh_utf8_decode(s, n, &c.v); return c; }
static void cs_free(cstr *c) { free(c->v); c->v = NULL; c->n = 0; }
static cstr cs_lower(const uint32_t *s, size_t n) { cstr c; c.n = uc_lower_str(s, n, &c.v); return c; }
static cstr cs_strip(const uint32_t *s, size_t n) {
    size_t i = 0, j = n;
    while (i < j && rh_py_isspace(s[i])) i++;
    while (j > i && rh_py_isspace(s[j - 1])) j--;
    return cs_from(s + i, j - i);
}
static int cs_eq_ascii(const uint32_t *s, size_t n, const char *a) {
    const size_t m = strlen(a);
    if (m != n) return 0;
    for (size_t i = 0; i < n; i++) if (s[i] != (unsigned char)a[i]) return 0;
    return 1;
}
static int cs_eq(const cstr *a, const cstr *b) { return a->n == b->n && !memcmp(a->v, b->v, a->n * sizeof(uint32_t)); }
static int cs_has(const uint32_t *s, size_t n, uint32_t c) { for (size_t i = 0; i < n; i++) if (s[i] == c) return 1; return 0; }
static int cs_ends(const uint32_t *s, size_t n, const char *suf) {
    const size_t m = strlen(suf);
    if (m > n) return 0;
    for (size_t i = 0; i < m; i++) if (s[n - m + i] != (unsigned char)suf[i]) return 0;
    return 1;
}
/* remove every occurrence of code point c */
static void cs_drop(cstr *c, uint32_t x) {
    size_t w = 0;
    for (size_t i = 0; i < c->n; i++) if (c->v[i] != x) c->v[w++] = c->v[i];
    c->n = w;
}
static char *cs_to_utf8(rh_arena *A, const uint32_t *s, size_t n, size_t *len) {
    char *o = rh_arena_alloc(A, 4 * n + 1);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) k += rh_utf8_put(o + k, s[i]);
    o[k] = 0;
    *len = k;
    return o;
}
static char *sb_to_arena(rh_arena *A, sb *b, size_t *len) {
    char *o = rh_arena_alloc(A, b->n + 1);
    memcpy(o, b->s ? b->s : "", b->n);
    o[b->n] = 0;
    *len = b->n;
    return o;
}
static int cap_ok(const long *caps, int g) { return caps[2 * g] >= 0; }

/* ---------------------------------------------------------------- compiled patterns */
typedef struct { const char *src; int flags; rh_rx *rx; } pat;
static rh_rx *P(pat *p) { if (!p->rx) p->rx = rh_rx_compile(p->src, p->flags); return p->rx; }

static pat SUF_PCT = {"^-?[\\d.]+\\s?percent$", RX_PY}, SUF_P = {"^-?[\\d.]+\\s?%$", RX_PY},
           SUF_BN = {"^-?[\\d.]+\\s?bn$", RX_PY}, SUF_K = {"^-?[\\d.]+\\s?k$", RX_PY}, SUF_M = {"^-?[\\d.]+\\s?m$", RX_PY},
           SPLIT_WORDS = {"[\\s-]+", RX_PY}, DIGITS_STRIP = {"[^\\d+]", RX_PY},
           QUOTED = {"[\"'\xe2\x80\x9c\xe2\x80\x98`]([^\"'\xe2\x80\x9d\xe2\x80\x99`\\n]{1,40})[\"'\xe2\x80\x9d\xe2\x80\x99`]", RX_PY},
           DASHED = {"(?:^|\\s)-\\s*[\\\"']?([\\w/.*+\\-]{1,40})[\\\"']?\\s*:", RX_PY},
           L_CUR = {"[\xe2\x82\xac\xc2\xa3\xc2\xa5\xe2\x82\xb9$]|\\b(?:eur|usd|gbp|jpy|euros?|dollars?|pounds?|yen|bucks|quid)\\b", 0},
           L_THOU = {"(\\d)[\\s\xc2\xa0\xe2\x80\xaf](?=\\d{3}\\b)", 0}, L_DOTS = {"^-?\\d{1,3}(\\.\\d{3})+,\\d+$", 0},
           L_COMMA = {"^-?\\d+,\\d{1,2}$", 0}, L_HALF = {"^(.+?)\\s+and\\s+a\\s+half$", 0},
           L_POINT = {"^(.+?)\\s+point\\s+([a-z\\s-]+)$", 0}, L_SPLIT = {"[\\s-]+", 0},
           NOW1 = {"^(\\d{4})-(\\d{2})-(\\d{2})(?:([^]))?([^]*)$", 0}, NOW2 = {"^(\\d{4})(\\d{2})(\\d{2})(?:([^]))?([^]*)$", 0},
           NOWT = {"^(\\d{2})(?::?(\\d{2})(?::?(\\d{2})(?:[.,](\\d+))?)?)?(?:[+-](\\d{2})(?::?(\\d{2})(?::?(\\d{2})(?:[.,]\\d+)?)?)?)?$", 0};

/* split s by a separator regex (JavaScript String.prototype.split with a regex, no captures) */
static size_t rx_split(rh_rx *rx, const uint32_t *s, size_t n, size_t (*parts)[2], size_t maxp) {
    long caps[8];
    size_t k = 0, pos = 0, from = 0;
    if (n == 0) { parts[0][0] = parts[0][1] = 0; return 1; }
    while (from < n && rh_rx_search(rx, s, n, from, caps)) {
        const size_t a = (size_t)caps[0], b = (size_t)caps[1];
        if (b == a) { from = a + 1; if (a >= n) break; if (a == pos) continue; }
        if (a == n) break;
        if (k < maxp) { parts[k][0] = pos; parts[k][1] = a; k++; }
        pos = b;
        from = b > a ? b : a + 1;
    }
    if (k < maxp) { parts[k][0] = pos; parts[k][1] = n; k++; }
    return k;
}
/* s.replace(rx_global, "") */
static cstr rx_remove(rh_rx *rx, const uint32_t *s, size_t n) {
    cstr o = cs_from(s, 0);
    o.v = rh_realloc(o.v, (n + 1) * sizeof(uint32_t));
    long caps[8];
    size_t pos = 0, from = 0;
    while (from <= n && rh_rx_search(rx, s, n, from, caps)) {
        const size_t a = (size_t)caps[0], b = (size_t)caps[1];
        for (size_t i = pos; i < a; i++) o.v[o.n++] = s[i];
        pos = b;
        from = b > a ? b : a + 1;
    }
    for (size_t i = pos; i < n; i++) o.v[o.n++] = s[i];
    return o;
}

/* ---------------------------------------------------------------- numbers */
static const char *UNITS[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                              "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                              "nineteen"};
static const struct { const char *w; double v; } TENS[] = {{"twenty", 20}, {"thirty", 30}, {"forty", 40}, {"fifty", 50},
    {"sixty", 60}, {"seventy", 70}, {"eighty", 80}, {"ninety", 90}};
static const struct { const char *w; double v; } SCALES[] = {{"hundred", 100}, {"thousand", 1000}, {"million", 1e6},
    {"billion", 1e9}, {"trillion", 1e12}};
static const struct { const char *w; double v; } SPECIAL[] = {{"a dozen", 12}, {"dozen", 12}, {"half", 0.5},
    {"a half", 0.5}, {"quarter", 0.25}, {"a quarter", 0.25}, {"once", 1}, {"twice", 2}, {"thrice", 3}, {"single", 1},
    {"a couple", 2}, {"couple", 2}, {"pair", 2}, {"a pair", 2}, {"first", 1}, {"second", 2}, {"third", 3},
    {"fourth", 4}, {"fifth", 5}, {"sixth", 6}, {"seventh", 7}, {"eighth", 8}, {"ninth", 9}, {"tenth", 10},
    {"noon", 12}, {"midnight", 0}, {"none", 0}, {"no", 0}, {"a", 1}, {"an", 1}};
static const char *DIGIT_WORDS[] = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
static const int DIGIT_VALS[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

static int word_unit(const uint32_t *w, size_t n) {
    for (int i = 0; i < 20; i++) if (cs_eq_ascii(w, n, UNITS[i])) return i;
    return -1;
}

static int parse_number_(const uint32_t *s, size_t n, double *out);
static int locale_inner = 0;

/* FIX.locale (parseLocale): 1 = value in *out, 0 = rules do not apply */
static int parse_locale(const uint32_t *s, size_t n, double *out) {
    cstr low = cs_lower(s, n);
    cstr t0 = rx_remove(P(&L_CUR), low.v, low.n);
    cstr t = cs_strip(t0.v, t0.n);
    cs_free(&t0);
    /* t.replace(/(\d)[\s  ](?=\d{3}\b)/g, '$1') */
    {
        cstr o = cs_from(t.v, 0);
        o.v = rh_realloc(o.v, (t.n + 1) * sizeof(uint32_t));
        long caps[8];
        size_t pos = 0, from = 0;
        while (from <= t.n && rh_rx_search(P(&L_THOU), t.v, t.n, from, caps)) {
            const size_t a = (size_t)caps[0], b = (size_t)caps[1];
            for (size_t i = pos; i < a; i++) o.v[o.n++] = t.v[i];
            for (long i = caps[2]; i < caps[3]; i++) o.v[o.n++] = t.v[i];
            pos = b;
            from = b > a ? b : a + 1;
        }
        for (size_t i = pos; i < t.n; i++) o.v[o.n++] = t.v[i];
        cs_free(&t);
        t = o;
    }
    long caps[16];
    int found = 0;
    if (rh_rx_search(P(&L_DOTS), t.v, t.n, 0, caps)) { /* 1.250,00 */
        char *b = rh_malloc(t.n + 1);
        size_t k = 0;
        int comma = 0;
        for (size_t i = 0; i < t.n; i++) {
            if (t.v[i] == '.') continue;
            b[k++] = (t.v[i] == ',' && !comma) ? (comma = 1, '.') : (char)t.v[i];
        }
        b[k] = 0;
        *out = strtod(b, NULL);
        free(b);
        found = 1;
    } else if (rh_rx_search(P(&L_COMMA), t.v, t.n, 0, caps)) { /* 23,40 */
        char *b = rh_malloc(t.n + 1);
        size_t k = 0;
        int comma = 0;
        for (size_t i = 0; i < t.n; i++) b[k++] = (t.v[i] == ',' && !comma) ? (comma = 1, '.') : (char)t.v[i];
        b[k] = 0;
        *out = strtod(b, NULL);
        free(b);
        found = 1;
    } else if (rh_rx_search(P(&L_HALF), t.v, t.n, 0, caps)) {
        double v;
        if (parse_number_(t.v + caps[2], (size_t)(caps[3] - caps[2]), &v)) { *out = v + 0.5; found = 1; }
        else found = -1;
    } else if (rh_rx_search(P(&L_POINT), t.v, t.n, 0, caps)) {
        const uint32_t *w = t.v + caps[4];
        const size_t wn = (size_t)(caps[5] - caps[4]);
        size_t parts[64][2];
        const size_t np = rx_split(P(&L_SPLIT), w, wn, parts, 64);
        int all = 1;
        sb digs = {0};
        for (size_t p = 0; p < np && all; p++) {
            int hit = -1;
            for (int d = 0; d < 11; d++) if (cs_eq_ascii(w + parts[p][0], parts[p][1] - parts[p][0], DIGIT_WORDS[d])) hit = d;
            if (hit < 0) all = 0;
            else { char c = (char)('0' + DIGIT_VALS[hit]); sb_put(&digs, &c, 1); }
        }
        if (all) {
            double v;
            if (!parse_number_(t.v + caps[2], (size_t)(caps[3] - caps[2]), &v)) found = -1;
            else {
                sb num = {0};
                js_num(&num, v);
                sb_str(&num, ".");
                sb_str(&num, digs.s ? digs.s : "");
                /* Number(string): JavaScript numeric literal grammar (a malformed one is NaN) */
                char *e;
                const double r = strtod(num.s, &e);
                *out = *e ? NAN : r;
                found = 1;
                sb_free(&num);
            }
        }
        sb_free(&digs);
    }
    if (!found) {
        cstr ls = cs_strip(low.v, low.n);
        if (!cs_eq(&t, &ls)) {
            double v;
            if (parse_number_(t.v, t.n, &v)) { *out = v; found = 1; }
            else found = -1;
        }
        cs_free(&ls);
    }
    cs_free(&low);
    cs_free(&t);
    return found == 1;
}

/* parse_number: 1 and *out, or 0 (None) */
static int parse_number_(const uint32_t *s, size_t n, double *out) {
    if (FIX.locale && !locale_inner) {
        locale_inner = 1;
        double v;
        const int ok = parse_locale(s, n, &v);
        locale_inner = 0;
        if (ok && isfinite(v)) { *out = v; return 1; }
    }
    cstr low = cs_lower(s, n);
    cs_drop(&low, ',');
    cs_drop(&low, '$');
    cstr t = cs_strip(low.v, low.n);
    cs_free(&low);
    double mult = 1;
    const int money = cs_has(s, n, '$') || cs_has(s, n, 0x20ac) || cs_has(s, n, 0xa3);
    static const struct { const char *suf; double m; pat *rx; } SUF[] = {
        {"percent", 0.01, &SUF_PCT}, {"%", 0.01, &SUF_P}, {"bn", 1e9, &SUF_BN}, {"k", 1e3, &SUF_K}, {"m", 1e6, &SUF_M}};
    for (int k = 0; k < 5; k++) {
        double m = SUF[k].m;
        if (!strcmp(SUF[k].suf, "m") && !money) m = 1;
        long caps[4];
        if (cs_ends(t.v, t.n, SUF[k].suf) && rh_rx_search(P(SUF[k].rx), t.v, t.n, 0, caps)) {
            cstr r = cs_strip(t.v, t.n - strlen(SUF[k].suf));
            cs_free(&t);
            t = r;
            mult = (strcmp(SUF[k].suf, "percent") && strcmp(SUF[k].suf, "%")) ? m : 1;
            break;
        }
    }
    double f;
    if (py_float_parse(t.v, t.n, &f)) { *out = f * mult; cs_free(&t); return 1; }
    for (size_t k = 0; k < sizeof SPECIAL / sizeof SPECIAL[0]; k++)
        if (cs_eq_ascii(t.v, t.n, SPECIAL[k].w)) { *out = SPECIAL[k].v; cs_free(&t); return 1; }
    double total = 0, cur = 0;
    size_t parts[256][2];
    const size_t np = rx_split(P(&SPLIT_WORDS), t.v, t.n, parts, 256);
    for (size_t p = 0; p < np; p++) {
        const uint32_t *w = t.v + parts[p][0];
        const size_t wn = parts[p][1] - parts[p][0];
        int u;
        if (cs_eq_ascii(w, wn, "and")) continue;
        if ((u = word_unit(w, wn)) >= 0) { cur += u; continue; }
        int hit = 0;
        for (int k = 0; k < 8; k++) if (cs_eq_ascii(w, wn, TENS[k].w)) { cur += TENS[k].v; hit = 1; break; }
        if (hit) continue;
        if (cs_eq_ascii(w, wn, "hundred")) { cur = (cur > 1 ? cur : 1) * 100; continue; }
        for (int k = 0; k < 5; k++)
            if (cs_eq_ascii(w, wn, SCALES[k].w)) { total += (cur > 1 ? cur : 1) * SCALES[k].v; cur = 0; hit = 1; break; }
        if (hit) continue;
        cs_free(&t);
        return 0;
    }
    cs_free(&t);
    *out = total + cur;
    return 1;
}
int rh_parse_number(const uint32_t *s, size_t n, double *out) { return parse_number_(s, n, out); }

/* ---------------------------------------------------------------- dates and times */
static const char *MONTH_NAMES[] = {"january", "february", "march", "april", "may", "june", "july", "august",
                                    "september", "october", "november", "december"};
static const char *WEEKDAY_NAMES[] = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"};

static int month_of(const uint32_t *s, size_t n) { /* MONTHS map: full names, 3-letter, sept */
    for (int i = 0; i < 12; i++) {
        if (cs_eq_ascii(s, n, MONTH_NAMES[i])) return i + 1;
        char ab[4] = {MONTH_NAMES[i][0], MONTH_NAMES[i][1], MONTH_NAMES[i][2], 0};
        if (cs_eq_ascii(s, n, ab)) return i + 1;
    }
    if (cs_eq_ascii(s, n, "sept")) return 9;
    return 0;
}
static int weekday_of(const uint32_t *s, size_t n) {
    for (int i = 0; i < 7; i++) {
        if (cs_eq_ascii(s, n, WEEKDAY_NAMES[i])) return i;
        char ab[4] = {WEEKDAY_NAMES[i][0], WEEKDAY_NAMES[i][1], WEEKDAY_NAMES[i][2], 0};
        if (cs_eq_ascii(s, n, ab)) return i;
    }
    return -1;
}

/* the regex sources of programs.mjs (built in code: the alternations are sorted by length) */
static char *alt_sorted(const char **words, int n, int extra_dot) {
    int order[64];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) { /* stable insertion sort by length, longest first */
        const int x = order[i];
        int j = i;
        while (j > 0 && strlen(words[order[j - 1]]) < strlen(words[x])) { order[j] = order[j - 1]; j--; }
        order[j] = x;
    }
    sb b = {0};
    sb_str(&b, "(?:");
    for (int i = 0; i < n; i++) { if (i) sb_str(&b, "|"); sb_str(&b, words[order[i]]); }
    sb_str(&b, extra_dot ? ")\\.?" : ")");
    return b.s;
}
static const char *MONTH_KEYS[] = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
                                   "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
                                   "sep", "oct", "nov", "dec", "sept"};
static const char *WD_KEYS[] = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mon",
                                "tue", "wed", "thu", "fri", "sat", "sun"};
static const char *NUMWORD_KEYS[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                     "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                     "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                                     "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                                     "billion", "trillion"};

typedef struct { const char *kind; char *src; int flags; rh_rx *full; } dtpat;
static dtpat DATE[7], TIME[6];
static rh_rx *EVENING, *MORNING;
static int dt_ready = 0;

static char *cat(const char *a, const char *b, const char *c, const char *d, const char *e, const char *f,
                 const char *g, const char *h, const char *i) {
    sb s = {0};
    const char *x[] = {a, b, c, d, e, f, g, h, i};
    for (int k = 0; k < 9; k++) if (x[k]) sb_str(&s, x[k]);
    return s.s;
}
static void dt_init(void) {
    if (dt_ready) return;
    dt_ready = 1;
    char *MON = alt_sorted(MONTH_KEYS, 24, 1), *WD = alt_sorted(WD_KEYS, 14, 0), *NW = alt_sorted(NUMWORD_KEYS, 33, 0);
    const char *AP = "(a\\.?m\\.?|p\\.?m\\.?)";
    DATE[0] = (dtpat){"mdy", cat("\\b(", MON, ")\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b", 0, 0, 0, 0, 0, 0), RX_I};
    DATE[1] = (dtpat){"dmy", cat("\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(", MON, ")(?:,?\\s+(\\d{4}))?\\b", 0, 0, 0, 0, 0, 0), RX_I};
    DATE[2] = (dtpat){"iso", cat("\\b(\\d{4})-(\\d{1,2})-(\\d{1,2})\\b", 0, 0, 0, 0, 0, 0, 0, 0), 0};
    DATE[3] = (dtpat){"slash", cat("\\b(\\d{1,2})/(\\d{1,2})/(\\d{2,4})\\b", 0, 0, 0, 0, 0, 0, 0, 0), 0};
    DATE[4] = (dtpat){"rel", cat("\\b(today|tonight|tomorrow|day after tomorrow|yesterday)\\b", 0, 0, 0, 0, 0, 0, 0, 0), RX_I};
    DATE[5] = (dtpat){"wd", cat("\\b(?:(this|next|coming|this coming|on)\\s+)?(", WD, ")\\b", 0, 0, 0, 0, 0, 0), RX_I};
    DATE[6] = (dtpat){"in", cat("\\bin\\s+(\\w+)\\s+(day|days|week|weeks)\\b", 0, 0, 0, 0, 0, 0, 0, 0), RX_I};
    TIME[0] = (dtpat){"hm_ap", cat("\\b(\\d{1,2})(?::(\\d{2}))?\\s*", AP, "(?![\\w])", 0, 0, 0, 0, 0, 0), RX_I};
    TIME[1] = (dtpat){"hm24", cat("\\b(\\d{1,2}):(\\d{2})(?::\\d{2})?\\b(?!\\s*(?:a\\.?m|p\\.?m))", 0, 0, 0, 0, 0, 0, 0, 0), RX_I};
    TIME[2] = (dtpat){"word", cat("\\b(noon|midday|midnight)\\b", 0, 0, 0, 0, 0, 0, 0, 0), RX_I};
    TIME[3] = (dtpat){"past", cat("\\b(half|quarter|\\d{1,2}|", NW, ")\\s+(past|to|after|before)\\s+(\\d{1,2}|", NW,
                                  ")(?:\\s*", AP, ")?", 0, 0), RX_I};
    TIME[4] = (dtpat){"oclock", cat("\\b(\\d{1,2}|", NW, ")\\s+o'?\\s?clock(?:\\s*", AP, ")?", 0, 0, 0, 0), RX_I};
    TIME[5] = (dtpat){"wordhm", cat("\\b(", NW, ")\\s+(", NW, "(?:[\\s-]", NW, ")?)\\s*", AP, 0), RX_I};
    for (int k = 0; k < 7; k++) {
        char *full = cat("^(?:", DATE[k].src, ")$", 0, 0, 0, 0, 0, 0);
        DATE[k].full = rh_rx_compile(full, DATE[k].flags | RX_PY);
        free(full);
    }
    for (int k = 0; k < 6; k++) {
        char *full = cat("^(?:", TIME[k].src, ")$", 0, 0, 0, 0, 0, 0);
        TIME[k].full = rh_rx_compile(full, TIME[k].flags | RX_PY);
        free(full);
    }
    EVENING = rh_rx_compile("\\b(tonight|evening|this evening|in the evening|pm|night|afternoon|dinner)\\b", RX_I | RX_PY);
    MORNING = rh_rx_compile("\\b(morning|a\\.?m\\.?|breakfast)\\b", RX_I | RX_PY);
    free(MON); free(WD); free(NW);
}

static int small_(const uint32_t *s, size_t n, double *out) { /* small(): int(parse_number) */
    double v;
    if (!parse_number_(s, n, &v) || !isfinite(v)) return 0;
    *out = trunc(v) + 0.0;
    return 1;
}
static int starts_ascii(const uint32_t *s, size_t n, const char *p) {
    const size_t m = strlen(p);
    if (m > n) return 0;
    for (size_t i = 0; i < m; i++) if (s[i] != (unsigned char)p[i]) return 0;
    return 1;
}

/* resolve_time -> 1 with *h, *mi, or 0 (None) */
static int resolve_time(const uint32_t *span, size_t sn, const uint32_t *ctx, size_t cn, double *H, double *MI) {
    dt_init();
    cstr low = cs_lower(span, sn);
    cstr s = cs_strip(low.v, low.n);
    cs_free(&low);
    long c[32];
    int ok = 0;
    for (int k = 0; k < 6; k++) {
        if (!rh_rx_search(TIME[k].full, s.v, s.n, 0, c)) continue;
        const char *kind = TIME[k].kind;
#define G(i) (s.v + c[2 * (i)]), ((size_t)(c[2 * (i) + 1] - c[2 * (i)]))
        double h = NAN, mi = NAN;
        const uint32_t *ap = NULL;
        size_t apn = 0;
        if (!strcmp(kind, "hm_ap")) {
            if (!py_int_parse(G(1), &h)) break;
            if (cap_ok(c, 2)) { if (!py_int_parse(G(2), &mi)) break; } else mi = 0;
            ap = s.v + c[6]; apn = (size_t)(c[7] - c[6]);
        } else if (!strcmp(kind, "hm24")) {
            if (!py_int_parse(G(1), &h) || !py_int_parse(G(2), &mi)) break;
        } else if (!strcmp(kind, "word")) {
            if (cs_eq_ascii(G(1), "midnight")) { *H = 0; *MI = 0; } else { *H = 12; *MI = 0; }
            ok = 1;
            break;
        } else if (!strcmp(kind, "past")) {
            double base, mins;
            const int bok = small_(G(3), &base);
            if (cap_ok(c, 4)) { ap = s.v + c[8]; apn = (size_t)(c[9] - c[8]); }
            int mok;
            if (cs_eq_ascii(G(1), "half")) { mins = 30; mok = 1; }
            else if (cs_eq_ascii(G(1), "quarter")) { mins = 15; mok = 1; }
            else mok = small_(G(1), &mins);
            if (!bok || !mok) break;
            if (cs_eq_ascii(G(2), "past") || cs_eq_ascii(G(2), "after")) { h = base; mi = mins; }
            else { h = fmod(fmod(base - 1, 24) + 24, 24); mi = 60 - mins; }
        } else if (!strcmp(kind, "oclock")) {
            if (!small_(G(1), &h)) break;
            mi = 0;
            if (cap_ok(c, 2)) { ap = s.v + c[4]; apn = (size_t)(c[5] - c[4]); }
        } else { /* wordhm */
            if (!small_(G(1), &h) || !small_(G(2), &mi)) break;
            ap = s.v + c[6]; apn = (size_t)(c[7] - c[6]);
        }
#undef G
        if (!(h >= 0 && h <= 24 && mi >= 0 && mi < 60)) break;
        long cc[8];
        if (ap && apn) h = fmod(h, 12) + (ap[0] == 'p' ? 12 : 0);
        else if (h < 12 && strcmp(kind, "hm24") && rh_rx_search(EVENING, ctx, cn, 0, cc) &&
                 !rh_rx_search(MORNING, ctx, cn, 0, cc)) h += 12;
        else if (h < 7 && !strcmp(kind, "past") && !rh_rx_search(MORNING, ctx, cn, 0, cc)) h += 12;
        *H = fmod(h, 24);
        *MI = mi;
        ok = 1;
        break;
    }
    cs_free(&s);
    return ok;
}

typedef struct { int y, m, d; } date;
static int is_leap(int y) { return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0); }
static int dim(int y, int m) { static const int D[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; return m == 2 && is_leap(y) ? 29 : D[m]; }
static int mk_date(double y, double m, double d, date *o) {
    if (!(y >= 1 && y <= 9999) || !(m >= 1 && m <= 12) || !(d >= 1 && d <= dim((int)y, (int)m))) return 0;
    o->y = (int)y; o->m = (int)m; o->d = (int)d;
    return 1;
}
static long ordinal(date x) {
    const long py = x.y - 1;
    long n = py * 365 + py / 4 - py / 100 + py / 400;
    for (int k = 1; k < x.m; k++) n += dim(x.y, k);
    return n + x.d;
}
static int from_ordinal(long n, date *o) {
    if (n < 1 || n > 3652059) return 0;
    int y = (int)floor((double)n / 365.2425) - 1;
    if (y < 1) y = 1;
    while (ordinal((date){y + 1, 1, 1}) <= n) y++;
    while (ordinal((date){y, 1, 1}) > n) y--;
    long r = n - ordinal((date){y, 1, 1}) + 1;
    int m = 1;
    while (r > dim(y, m)) { r -= dim(y, m); m++; }
    *o = (date){y, m, (int)r};
    return 1;
}

static int resolve_date(const uint32_t *span, size_t sn, const date *now, date *out) {
    dt_init();
    cstr st = cs_strip(span, sn);
    cstr s = cs_lower(st.v, st.n);
    cs_free(&st);
    long c[32];
    int ok = 0;
    for (int k = 0; k < 7; k++) {
        if (!rh_rx_search(DATE[k].full, s.v, s.n, 0, c)) continue;
        const char *kind = DATE[k].kind;
#define G(i) (s.v + c[2 * (i)]), ((size_t)(c[2 * (i) + 1] - c[2 * (i)]))
        if (!strcmp(kind, "mdy") || !strcmp(kind, "dmy")) {
            const int mi = !strcmp(kind, "mdy") ? 1 : 2, di = !strcmp(kind, "mdy") ? 2 : 1;
            const uint32_t *mon = s.v + c[2 * mi];
            size_t mn = (size_t)(c[2 * mi + 1] - c[2 * mi]);
            while (mn && mon[mn - 1] == '.') mn--;
            const int month = month_of(mon, mn);
            if (!month) break;
            double day, year;
            if (!py_int_parse(G(di), &day)) break;
            if (cap_ok(c, 3)) { if (!py_int_parse(G(3), &year)) break; }
            else {
                if (!now) break;
                year = now->y;
                date t;
                if (!mk_date(year, month, day, &t)) break;
                if (ordinal(t) < ordinal(*now)) year += 1;
            }
            ok = mk_date(year, month, day, out);
            break;
        }
        if (!strcmp(kind, "iso")) {
            double y, m, d;
            ok = py_int_parse(G(1), &y) && py_int_parse(G(2), &m) && py_int_parse(G(3), &d) && mk_date(y, m, d, out);
            break;
        }
        if (!strcmp(kind, "slash")) {
            double y, m, d;
            if (!py_int_parse(G(3), &y) || !py_int_parse(G(1), &m) || !py_int_parse(G(2), &d)) break;
            if (y < 100) y += 2000;
            ok = mk_date(y, m, d, out);
            break;
        }
        if (!now) break;
        if (!strcmp(kind, "rel")) {
            int off;
            if (cs_eq_ascii(G(1), "today") || cs_eq_ascii(G(1), "tonight")) off = 0;
            else if (cs_eq_ascii(G(1), "tomorrow")) off = 1;
            else if (cs_eq_ascii(G(1), "day after tomorrow")) off = 2;
            else if (cs_eq_ascii(G(1), "yesterday")) off = -1;
            else break;
            ok = from_ordinal(ordinal(*now) + off, out);
            break;
        }
        if (!strcmp(kind, "wd")) {
            const int target = weekday_of(G(2));
            if (target < 0) break;
            const int wd = (int)((ordinal(*now) + 6) % 7);
            int ahead = ((target - wd) % 7 + 7) % 7;
            if (cap_ok(c, 1) && starts_ascii(G(1), "next") && ahead == 0) ahead = 7;
            ok = from_ordinal(ordinal(*now) + ahead, out);
            break;
        }
        if (!strcmp(kind, "in")) {
            double n;
            if (!small_(G(1), &n) || fabs(n) > 4e6) break; /* beyond date.max: OverflowError / ValueError */
            const uint32_t *u = s.v + c[4];
            ok = from_ordinal(ordinal(*now) + (long)(n * (u[0] == 'w' ? 7 : 1)), out);
            break;
        }
#undef G
    }
    cs_free(&s);
    return ok;
}

static int parse_now(const jnode *ctx, date *out) {
    if (!ctx || ctx->t != J_OBJ) return 0;
    const jnode *now = rh_json_get(ctx, "now");
    rv nv = rv_json(now);
    if (!now || !rv_truthy(&nv)) return 0;
    sb b = {0};
    py_str(&b, &nv);
    cstr s = cs_utf8(b.s, b.n);
    sb_free(&b);
    cs_drop(&s, 'Z');
    long c[16];
    int ok = 0;
    if (rh_rx_search(P(&NOW1), s.v, s.n, 0, c) || rh_rx_search(P(&NOW2), s.v, s.n, 0, c)) {
        int good = 1;
        if (cap_ok(c, 4)) {
            long t[20];
            const uint32_t *r = s.v + c[10];
            const size_t rn = (size_t)(c[11] - c[10]);
            if (!rh_rx_search(P(&NOWT), r, rn, 0, t)) good = 0;
            else {
                double a, x;
                if (py_int_parse(r + t[2], (size_t)(t[3] - t[2]), &a) && a > 23) good = 0;
                if (t[4] >= 0 && py_int_parse(r + t[4], (size_t)(t[5] - t[4]), &x) && x > 59) good = 0;
                if (t[6] >= 0 && py_int_parse(r + t[6], (size_t)(t[7] - t[6]), &x) && x > 59) good = 0;
            }
        }
        double y, m, d;
        if (good && py_int_parse(s.v + c[2], 4, &y) && py_int_parse(s.v + c[4], 2, &m) && py_int_parse(s.v + c[6], 2, &d))
            ok = mk_date(y, m, d, out);
    }
    cs_free(&s);
    return ok;
}

static const char *MONTH_TITLE[] = {"January", "February", "March", "April", "May", "June", "July", "August",
                                    "September", "October", "November", "December"};
static void strftime_(sb *b, const char *fmt, date d, int h, int mi) {
    char t[16];
    for (const char *p = fmt; *p; p++) {
        if (*p != '%' || !p[1] || !strchr("YmdHMSIpB%", p[1])) { sb_put(b, p, 1); continue; }
        switch (*++p) {
        case 'Y': snprintf(t, sizeof t, "%04d", d.y); sb_str(b, t); break;
        case 'm': snprintf(t, sizeof t, "%02d", d.m); sb_str(b, t); break;
        case 'd': snprintf(t, sizeof t, "%02d", d.d); sb_str(b, t); break;
        case 'H': snprintf(t, sizeof t, "%02d", h); sb_str(b, t); break;
        case 'M': snprintf(t, sizeof t, "%02d", mi); sb_str(b, t); break;
        case 'S': sb_str(b, "00"); break;
        case 'I': snprintf(t, sizeof t, "%02d", h % 12 ? h % 12 : 12); sb_str(b, t); break;
        case 'p': sb_str(b, h < 12 ? "AM" : "PM"); break;
        case 'B': sb_str(b, MONTH_TITLE[d.m - 1]); break;
        default: sb_str(b, "%");
        }
    }
}

/* ---------------------------------------------------------------- tables, options */
static rh_arena tables_arena;
static const jnode *tables_root;
static const jnode *tables(void) {
    if (!tables_root) tables_root = rh_json_parse(&tables_arena, RELAY_TABLES_JSON, strlen(RELAY_TABLES_JSON));
    return tables_root;
}
const jnode *rh_tables(void) { return rh_json_get(tables(), "tables"); }
const jnode *rh_affixes(void) { return rh_json_get(tables(), "affixes"); }
size_t rh_n_tables(void) { return rh_tables()->count; }

/* desc_options: values quoted or listed in the description, first appearance order, at most 32 (UTF-8 strings) */
size_t rh_desc_options(const jnode *param, rh_arena *A, rv *out) {
    const jnode *raw = rh_json_get(param, "description");
    rv rd = rv_json(raw);
    sb d = {0};
    if (raw && rv_truthy(&rd)) py_str(&d, &rd);
    cstr s = cs_utf8(d.s ? d.s : "", d.n);
    sb_free(&d);
    size_t n = 0;
    pat *rxs[2] = {&QUOTED, &DASHED};
    for (int k = 0; k < 2; k++) {
        long c[8];
        size_t from = 0;
        while (from <= s.n && rh_rx_search(P(rxs[k]), s.v, s.n, from, c)) {
            cstr v = cs_strip(s.v + c[2], (size_t)(c[3] - c[2]));
            if (v.n) {
                size_t len;
                char *u = cs_to_utf8(A, v.v, v.n, &len);
                int dup = 0;
                for (size_t q = 0; q < n; q++) if (out[q].len == len && !memcmp(out[q].s, u, len)) dup = 1;
                if (!dup && n < 64) out[n++] = rv_str(u, len);
            }
            cs_free(&v);
            from = c[1] > c[0] ? (size_t)c[1] : (size_t)c[1] + 1;
        }
    }
    cs_free(&s);
    return n < 32 ? n : 32;
}

/* ---------------------------------------------------------------- coercion and interpreter */
static int is_type(const rv *typ, const char *name) { return typ->t == RV_STR && typ->len == strlen(name) && !memcmp(typ->s, name, typ->len); }

int rh_coerce(rv v, const jnode *typnode, rh_arena *A, rv *out) {
    rv typ = typnode ? rv_json(typnode) : rv_undef();
    *out = v;
    if (v.t == RV_NULL || v.t == RV_UNDEF || typ.t == RV_NULL || typ.t == RV_UNDEF || is_type(&typ, "any") ||
        is_type(&typ, "array") || is_type(&typ, "object"))
        return 1;
    if ((is_type(&typ, "integer") || is_type(&typ, "number")) && v.t == RV_STR) {
        cstr c = cs_utf8(v.s, v.len);
        double f;
        const int ok = parse_number_(c.v, c.n, &f);
        cs_free(&c);
        if (!ok) return 1;
        if (is_type(&typ, "integer")) {
            if (!isfinite(f)) return 0; /* round(inf): OverflowError */
            if (fabs(f - py_round(f)) < 1e-9) *out = rv_int(py_round(f));
            return 1;
        }
        *out = rv_float(f);
        return 1;
    }
    if (is_type(&typ, "integer") && v.t == RV_NUM && v.isfloat && isfinite(v.num) && v.num == floor(v.num)) {
        *out = rv_int(v.num);
        return 1;
    }
    if (is_type(&typ, "boolean") && v.t == RV_STR) {
        cstr c = cs_utf8(v.s, v.len), l = cs_lower(c.v, c.n), t = cs_strip(l.v, l.n);
        const int tr = cs_eq_ascii(t.v, t.n, "true"), fa = cs_eq_ascii(t.v, t.n, "false");
        cs_free(&c); cs_free(&l); cs_free(&t);
        if (tr || fa) *out = rv_bool(tr);
        return 1;
    }
    if (is_type(&typ, "string") && (v.t == RV_BOOL || v.t == RV_NUM)) {
        if (v.t == RV_BOOL) { *out = rv_str(v.num ? "true" : "false", v.num ? 4 : 5); return 1; }
        sb b = {0};
        py_str(&b, &v);
        size_t len;
        char *s = sb_to_arena(A, &b, &len);
        sb_free(&b);
        *out = rv_str(s, len);
    }
    return 1;
}

/* pyIndex(seq, i) */
static int py_index(const jnode *seq, double i, rh_arena *A, rv *out) {
    if (!seq || i < 0) return 0;
    if (seq->t == J_STR) {
        uint32_t *cps;
        const size_t n = rh_utf8_decode(seq->s, seq->len, &cps);
        if (i >= (double)n) { free(cps); return 0; }
        size_t len;
        char *u = cs_to_utf8(A, cps + (size_t)i, 1, &len);
        free(cps);
        *out = rv_str(u, len);
        return 1;
    }
    if (seq->t != J_ARR || i >= (double)seq->count) return 0;
    const jnode *c = seq->first;
    for (size_t k = 0; k < (size_t)i; k++) c = c->next;
    *out = rv_json(c);
    return 1;
}

static int run_(const rh_prog *p, const uint32_t *text, size_t tn, const jnode *param, const jnode *ctx, rh_arena *A, rv *out);

int rh_run(const rh_prog *p, const uint32_t *text, size_t tn, const jnode *param, const jnode *ctx, rh_arena *A, rv *out) {
    rv v;
    if (!run_(p, text, tn, param, ctx, A, &v)) return 0;
    return rh_coerce(v, rh_json_get(param, "type"), A, out);
}

static int run_(const rh_prog *p, const uint32_t *text, size_t tn, const jnode *param, const jnode *ctx, rh_arena *A, rv *out) {
    const uint32_t *sp = text + p->span[0];
    const size_t sn = (size_t)(p->span[1] - p->span[0]);
    size_t len;
    switch (p->op) {
    case PR_DEFAULT: { const jnode *d = rh_json_get(param, "default"); *out = d ? rv_json(d) : rv_undef(); return 1; }
    case PR_BOOL: *out = rv_bool(p->value); return 1;
    case PR_ENUM: return py_index(rh_json_get(param, "enum"), p->index, A, out);
    case PR_OPTION: {
        rv opts[64];
        const size_t n = rh_desc_options(param, A, opts);
        if (p->index < 0 || (size_t)p->index >= n) return 0;
        *out = opts[p->index];
        return 1;
    }
    case PR_CONST: *out = rv_str(p->cvalue, p->clen); return 1;
    case PR_DIGITS: {
        cstr r = rx_remove(P(&DIGITS_STRIP), sp, sn);
        const char *u = cs_to_utf8(A, r.v, r.n, &len);
        *out = rv_str(u, len);
        cs_free(&r);
        return 1;
    }
    case PR_AFFIX: {
        const jnode *af = rh_affixes();
        if (p->index < 0 || (size_t)p->index >= af->count) return 0;
        const jnode *e = af->first;
        for (int k = 0; k < p->index; k++) e = e->next;
        sb b = {0};
        sb_put(&b, e->first->s, e->first->len);
        size_t l2;
        char *mid = cs_to_utf8(A, sp, sn, &l2);
        sb_put(&b, mid, l2);
        sb_put(&b, e->first->next->s, e->first->next->len);
        char *s = sb_to_arena(A, &b, &len);
        sb_free(&b);
        *out = rv_str(s, len);
        return 1;
    }
    case PR_REF: {
        char t[24];
        snprintf(t, sizeof t, "#%d", p->call);
        char *s = rh_arena_alloc(A, strlen(t) + 1);
        strcpy(s, t);
        *out = rv_str(s, strlen(s));
        return 1;
    }
    case PR_COPY: { const char *u = cs_to_utf8(A, sp, sn, &len); *out = rv_str(u, len); return 1; }
    case PR_NUMBER: {
        double v;
        if (!parse_number_(sp, sn, &v)) return 0;
        v = v * p->scale;
        if (p->is_int && !isfinite(v)) return 0; /* round(inf / nan) raises */
        *out = p->is_int && fabs(v - py_round(v)) < 1e-9 ? rv_int(py_round(v)) : rv_float(v);
        return 1;
    }
    case PR_TIMEPART: {
        double h, mi;
        if (!resolve_time(sp, sn, text, tn, &h, &mi)) return 0;
        *out = rv_int(p->part_hour ? h : mi);
        return 1;
    }
    case PR_DATETIME: {
        date now, d;
        const int have_now = parse_now(ctx, &now);
        int dok;
        if (p->has_date) dok = resolve_date(text + p->date[0], (size_t)(p->date[1] - p->date[0]), have_now ? &now : NULL, &d);
        else { dok = have_now; d = now; }
        double h = 0, mi = 0;
        int tok = 1;
        if (p->has_time) tok = resolve_time(text + p->time[0], (size_t)(p->time[1] - p->time[0]), text, tn, &h, &mi);
        if (!dok || !tok) { *out = rv_null(); return 1; }
        sb b = {0};
        strftime_(&b, p->format, d, (int)h, (int)mi);
        char *s = sb_to_arena(A, &b, &len);
        sb_free(&b);
        *out = rv_str(s, len);
        return 1;
    }
    case PR_LOOKUP: {
        const jnode *tabs = rh_tables(), *tab = tabs->first;
        for (int k = 0; k < p->table && tab; k++) tab = tab->next;
        if (!tab) return 0;
        cstr st = cs_strip(sp, sn), key = cs_lower(st.v, st.n);
        char *u = cs_to_utf8(A, key.v, key.n, &len);
        cs_free(&st); cs_free(&key);
        for (const jnode *e = tab->first; e; e = e->next)
            if (e->klen == len && !memcmp(e->key, u, len)) { *out = rv_json(e); return 1; }
        return 0;
    }
    case PR_LIST: {
        const jnode *items = rh_json_get(param, "items");
        rv it = rv_json(items);
        const jnode *ip = items && rv_truthy(&it) ? items : NULL;
        static const jnode EMPTY_OBJ = {J_OBJ};
        rv *vals = rh_arena_alloc(A, (p->nitems + 1) * sizeof(rv));
        for (int k = 0; k < p->nitems; k++)
            if (!rh_run(&p->items[k], text, tn, ip ? ip : &EMPTY_OBJ, ctx, A, &vals[k])) return 0;
        rv l = {0};
        l.t = RV_LIST;
        l.items = vals;
        l.n = (size_t)p->nitems;
        *out = l;
        return 1;
    }
    }
    return 0;
}
