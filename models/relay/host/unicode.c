/* Unicode support for the native Relay host: general-category flags, canonical combining classes, NFD / NFC
   (canonical decomposition, canonical ordering, canonical composition incl. Hangul), all from unicode_data.h. */
#include "relay_host.h"
#include "unicode_data.h"

static int range_find(uint32_t cp) {
    int lo = 0, hi = UC_CAT_RANGES_N - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cp < uc_cat_ranges[mid][0]) hi = mid - 1;
        else if (cp > uc_cat_ranges[mid][1]) lo = mid + 1;
        else return mid;
    }
    return -1;
}

uint32_t uc_flags(uint32_t cp) {
    const int i = range_find(cp);
    return i < 0 ? 0u : uc_cat_ranges[i][2];
}

int uc_ccc(uint32_t cp) {
    int lo = 0, hi = UC_CCC_TAB_N - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cp < uc_ccc_tab[mid][0]) hi = mid - 1;
        else if (cp > uc_ccc_tab[mid][0]) lo = mid + 1;
        else return (int)uc_ccc_tab[mid][1];
    }
    return 0;
}

/* Full canonical decomposition of one code point into out (at most 18 code points); returns the count. */
static int decompose(uint32_t cp, uint32_t *out) {
    const uint32_t S = 0xAC00, L = 0x1100, V = 0x1161, T = 0x11A7, N = 588, TC = 28;
    if (cp >= S && cp < S + 11172) {
        const uint32_t s = cp - S;
        out[0] = L + s / N;
        out[1] = V + (s % N) / TC;
        if (s % TC) {
            out[2] = T + s % TC;
            return 3;
        }
        return 2;
    }
    int lo = 0, hi = UC_DECOMP_N - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cp < uc_decomp[mid][0]) hi = mid - 1;
        else if (cp > uc_decomp[mid][0]) lo = mid + 1;
        else {
            const uint32_t n = uc_decomp[mid][2];
            for (uint32_t k = 0; k < n; k++) out[k] = uc_decomp_pool[uc_decomp[mid][1] + k];
            return (int)n;
        }
    }
    out[0] = cp;
    return 1;
}

static uint32_t compose_pair(uint32_t a, uint32_t b) {
    const uint32_t S = 0xAC00, L = 0x1100, V = 0x1161, T = 0x11A7, TC = 28;
    if (a >= L && a < L + 19 && b >= V && b < V + 21) return S + ((a - L) * 21 + (b - V)) * TC;
    if (a >= S && a < S + 11172 && (a - S) % TC == 0 && b > T && b < T + TC) return a + (b - T);
    int lo = 0, hi = UC_COMP_N - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint32_t x = uc_comp[mid][0], y = uc_comp[mid][1];
        if (a < x || (a == x && b < y)) hi = mid - 1;
        else if (a > x || (a == x && b > y)) lo = mid + 1;
        else return uc_comp[mid][2];
    }
    return 0;
}

/* NFD of in[0..n) into a new buffer (*out, returned length). */
size_t uc_nfd(const uint32_t *in, size_t n, uint32_t **out) {
    uint32_t *o = rh_malloc((n * 18 + 1) * sizeof(uint32_t));
    size_t m = 0;
    for (size_t i = 0; i < n; i++) m += (size_t)decompose(in[i], o + m);
    /* canonical ordering: stable insertion sort of each run of nonzero combining classes */
    for (size_t i = 1; i < m; i++) {
        const int c = uc_ccc(o[i]);
        if (!c) continue;
        size_t j = i;
        while (j > 0) {
            const int p = uc_ccc(o[j - 1]);
            if (!p || p <= c) break;
            const uint32_t t = o[j - 1];
            o[j - 1] = o[j];
            o[j] = t;
            j--;
        }
    }
    *out = o;
    return m;
}

/* NFC of in[0..n) into a new buffer (*out, returned length). */
size_t uc_nfc(const uint32_t *in, size_t n, uint32_t **out) {
    uint32_t *d;
    size_t m = uc_nfd(in, n, &d);
    if (!m) {
        *out = d;
        return 0;
    }
    /* UAX #15 reference composition: a character composes with the last starter unless blocked */
    size_t starter = 0, w = 1;
    int last_class = uc_ccc(d[0]);
    if (last_class) last_class = 256; /* the text starts with a non-starter: nothing composes onto it */
    for (size_t i = 1; i < m; i++) {
        const uint32_t ch = d[i];
        const int c = uc_ccc(ch);
        const uint32_t comp = compose_pair(d[starter], ch);
        if (comp && (last_class < c || last_class == 0)) {
            d[starter] = comp;
            continue;
        }
        if (c == 0) starter = w;
        last_class = c;
        d[w++] = ch;
    }
    *out = d;
    return w;
}

/* ---- case mapping ---- */
static int case_find(const uint32_t (*tab)[3], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cp < tab[mid][0]) hi = mid - 1;
        else if (cp > tab[mid][0]) lo = mid + 1;
        else return mid;
    }
    return -1;
}
static int cased(uint32_t cp) {
    return case_find(uc_lower, UC_LOWER_N, cp) >= 0 || case_find(uc_upper, UC_UPPER_N, cp) >= 0;
}
static int case_ignorable(uint32_t cp) {
    return (uc_flags(cp) & RH_UC_M) || cp == 0x27 || cp == 0x2e || cp == 0x3a || cp == 0x5e || cp == 0x60 ||
           cp == 0xa8 || cp == 0xad || cp == 0xaf || cp == 0xb4 || cp == 0xb7 || cp == 0xb8 || cp == 0x2019;
}

/* str.lower() / toLowerCase() of s[0..n) (full mappings, final sigma) into a new buffer. */
size_t uc_lower_str(const uint32_t *s, size_t n, uint32_t **out) {
    uint32_t *o = rh_malloc((n * 3 + 1) * sizeof(uint32_t));
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t cp = s[i];
        if (cp == 0x3a3) { /* final sigma: preceded by a cased letter, not followed by one */
            int before = 0, after = 0;
            for (size_t j = i; j-- > 0;) { if (case_ignorable(s[j])) continue; before = cased(s[j]); break; }
            for (size_t j = i + 1; j < n; j++) { if (case_ignorable(s[j])) continue; after = cased(s[j]); break; }
            o[m++] = before && !after ? 0x3c2 : 0x3c3;
            continue;
        }
        const int k = case_find(uc_lower, UC_LOWER_N, cp);
        if (k < 0) { o[m++] = cp; continue; }
        for (uint32_t q = 0; q < uc_lower[k][2]; q++) o[m++] = uc_case_pool[uc_lower[k][1] + q];
    }
    *out = o;
    return m;
}
size_t uc_upper_str(const uint32_t *s, size_t n, uint32_t **out) {
    uint32_t *o = rh_malloc((n * 3 + 1) * sizeof(uint32_t));
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        const int k = case_find(uc_upper, UC_UPPER_N, s[i]);
        if (k < 0) { o[m++] = s[i]; continue; }
        for (uint32_t q = 0; q < uc_upper[k][2]; q++) o[m++] = uc_case_pool[uc_upper[k][1] + q];
    }
    *out = o;
    return m;
}
uint32_t uc_fold_cp(uint32_t cp) {
    int lo = 0, hi = UC_FOLD_N - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cp < uc_fold[mid][0]) hi = mid - 1;
        else if (cp > uc_fold[mid][0]) lo = mid + 1;
        else return uc_fold[mid][1];
    }
    return cp;
}
