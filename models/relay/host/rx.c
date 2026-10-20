/* Small backtracking regex engine for the native Relay host (patterns are fixed and short; inputs are requests and
   descriptions). Syntax: JavaScript regex syntax as the JS host writes it, in two dialects:
     RX_PY  Python `re` str semantics (what py.mjs's pyRe produces): \w = [\p{L}\p{N}_], \d = \p{Nd},
            \s = Python whitespace, \b = Unicode word boundary
     RX_JS  JavaScript without the u flag: \w = [A-Za-z0-9_], \d = [0-9], \s = JS whitespace, \b ASCII
   Supported: literals and escapes, . and [^], classes (ranges, negation, \p{L} \p{N} \p{Nd} \p{Z} \p{M} \p{C}),
   ^ $ \b \B, groups (capturing, (?:)), lookahead (?= (?!, lookbehind (?<= (?<!, alternation, * + ? {m} {m,} {m,n}
   greedy and lazy, RX_I (simple case folding). Leftmost-first matching like both engines. */
#include "relay_host.h"
#include <stdlib.h>
#include <string.h>

enum { I_CHAR, I_ANY, I_ANYNL, I_CLASS, I_SPLIT, I_JMP, I_SAVE, I_MATCH, I_BOL, I_EOL, I_WB, I_NWB, I_LOOK };

typedef struct { uint32_t lo, hi; } crange;
typedef struct {
    crange *r;
    int nr;
    uint32_t props; /* UC_* flags that match */
    int neg, word, digit, space, nword, ndigit, nspace;
} cclass;

typedef struct {
    int op;
    uint32_t c;   /* I_CHAR */
    int x, y;     /* I_SPLIT targets / I_JMP / I_SAVE slot / I_CLASS index / I_LOOK: start, end of sub-program */
    int behind, neg;
} inst;

struct rh_rx {
    inst *code;
    int n, cap;
    cclass *cls;
    int ncls;
    int ngroup;
    int flags;
};

typedef struct {
    const char *p;
    uint32_t *pat; /* pattern code points */
    size_t n, i;
    rh_rx *rx;
    int err;
} parser;

static int emit(rh_rx *rx, inst in) {
    if (rx->n == rx->cap) { rx->cap = rx->cap ? rx->cap * 2 : 64; rx->code = rh_realloc(rx->code, (size_t)rx->cap * sizeof(inst)); }
    rx->code[rx->n] = in;
    return rx->n++;
}

/* ---- character tests ---- */
static int py_space(uint32_t c) { return rh_py_isspace(c); }
static int js_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
           c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000 || c == 0xfeff;
}
static int is_word(const rh_rx *rx, uint32_t c) {
    if (rx->flags & RX_PY) return c == '_' || (uc_flags(c) & (RH_UC_L | RH_UC_N));
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}
static int is_digit(const rh_rx *rx, uint32_t c) {
    if (rx->flags & RX_PY) return (uc_flags(c) & RH_UC_ND) != 0;
    return c >= '0' && c <= '9';
}
static int is_space(const rh_rx *rx, uint32_t c) { return (rx->flags & RX_PY) ? py_space(c) : js_space(c); }

static int class_has(const rh_rx *rx, const cclass *k, uint32_t c) {
    for (int i = 0; i < k->nr; i++) if (c >= k->r[i].lo && c <= k->r[i].hi) return 1;
    if (k->props && (uc_flags(c) & k->props)) return 1;
    if (k->word && is_word(rx, c)) return 1;
    if (k->digit && is_digit(rx, c)) return 1;
    if (k->space && is_space(rx, c)) return 1;
    if (k->nword && !is_word(rx, c)) return 1;
    if (k->ndigit && !is_digit(rx, c)) return 1;
    if (k->nspace && !is_space(rx, c)) return 1;
    return 0;
}
static int class_match(const rh_rx *rx, const cclass *k, uint32_t c) {
    int hit = class_has(rx, k, c);
    if (!hit && (rx->flags & RX_I)) {
        const uint32_t f = uc_fold_cp(c);
        hit = class_has(rx, k, f);
        if (!hit) { /* an upper-case range member folding to c */
            uint32_t *u;
            const size_t m = uc_upper_str(&c, 1, &u);
            if (m == 1) hit = class_has(rx, k, u[0]);
            free(u);
        }
    }
    return k->neg ? !hit : hit;
}

/* ---- parser ---- */
static int peek(parser *ps) { return ps->i < ps->n ? (int)ps->pat[ps->i] : -1; }

static uint32_t parse_escape_cp(parser *ps, uint32_t e) {
    switch (e) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'f': return '\f';
    case 'v': return '\v';
    case '0': return 0;
    case 'x': {
        uint32_t v = 0;
        for (int k = 0; k < 2 && ps->i < ps->n; k++) v = v * 16 + (uint32_t)strtol((char[]){(char)ps->pat[ps->i++], 0}, NULL, 16);
        return v;
    }
    case 'u': {
        uint32_t v = 0;
        if (peek(ps) == '{') {
            ps->i++;
            while (ps->i < ps->n && ps->pat[ps->i] != '}') v = v * 16 + (uint32_t)strtol((char[]){(char)ps->pat[ps->i++], 0}, NULL, 16);
            ps->i++;
            return v;
        }
        for (int k = 0; k < 4 && ps->i < ps->n; k++) v = v * 16 + (uint32_t)strtol((char[]){(char)ps->pat[ps->i++], 0}, NULL, 16);
        return v;
    }
    default: return e;
    }
}

static uint32_t parse_prop(parser *ps) { /* after \p : {Name} */
    char name[16] = {0};
    int k = 0;
    if (peek(ps) == '{') {
        ps->i++;
        while (ps->i < ps->n && ps->pat[ps->i] != '}' && k < 15) name[k++] = (char)ps->pat[ps->i++];
        ps->i++;
    }
    if (!strcmp(name, "L")) return RH_UC_L;
    if (!strcmp(name, "N")) return RH_UC_N;
    if (!strcmp(name, "Nd")) return RH_UC_ND;
    if (!strcmp(name, "Z")) return RH_UC_Z;
    if (!strcmp(name, "M")) return RH_UC_M;
    if (!strcmp(name, "C")) return RH_UC_C;
    ps->err = 1;
    return 0;
}

static int new_class(rh_rx *rx) {
    rx->cls = rh_realloc(rx->cls, (size_t)(rx->ncls + 1) * sizeof(cclass));
    memset(&rx->cls[rx->ncls], 0, sizeof(cclass));
    return rx->ncls++;
}
static void add_range(cclass *k, uint32_t lo, uint32_t hi) {
    k->r = rh_realloc(k->r, (size_t)(k->nr + 1) * sizeof(crange));
    k->r[k->nr].lo = lo;
    k->r[k->nr].hi = hi;
    k->nr++;
}

/* Class item escapes (\w \d \s \W \D \S \p{..}) in or outside brackets; returns 1 if handled. */
static int class_escape(parser *ps, cclass *k, uint32_t e) {
    switch (e) {
    case 'w': k->word = 1; return 1;
    case 'd': k->digit = 1; return 1;
    case 's': k->space = 1; return 1;
    case 'W': k->nword = 1; return 1;
    case 'D': k->ndigit = 1; return 1;
    case 'S': k->nspace = 1; return 1;
    case 'p': k->props |= parse_prop(ps); return 1;
    default: return 0;
    }
}

static int parse_bracket(parser *ps) { /* after '[' */
    const int ki = new_class(ps->rx);
    if (peek(ps) == '^') { ps->rx->cls[ki].neg = 1; ps->i++; }
    if (peek(ps) == ']' && ps->i + 1 < ps->n) { /* JS: [] matches nothing, [^] anything; [^]...] handled by caller */
        ps->i++;
        if (ps->rx->cls[ki].neg) { ps->rx->cls[ki].neg = 0; add_range(&ps->rx->cls[ki], 0, 0x10ffff); }
        return ki;
    }
    while (ps->i < ps->n && ps->pat[ps->i] != ']') {
        uint32_t lo = ps->pat[ps->i++];
        if (lo == '\\') {
            const uint32_t e = ps->pat[ps->i++];
            if (class_escape(ps, &ps->rx->cls[ki], e)) continue;
            lo = e == 'b' ? 8 : parse_escape_cp(ps, e);
        }
        uint32_t hi = lo;
        if (peek(ps) == '-' && ps->i + 1 < ps->n && ps->pat[ps->i + 1] != ']') {
            ps->i++;
            hi = ps->pat[ps->i++];
            if (hi == '\\') hi = parse_escape_cp(ps, ps->pat[ps->i++]);
        }
        add_range(&ps->rx->cls[ki], lo, hi);
    }
    ps->i++; /* ']' */
    return ki;
}

static void parse_alt(parser *ps);

/* one atom; returns 1 if something was emitted (a quantifiable atom spans [start, rx->n)) */
static int parse_atom(parser *ps, int *quantifiable) {
    rh_rx *rx = ps->rx;
    const int c = peek(ps);
    *quantifiable = 1;
    if (c < 0 || c == '|' || c == ')') return 0;
    ps->i++;
    if (c == '^') { emit(rx, (inst){I_BOL}); *quantifiable = 0; return 1; }
    if (c == '$') { emit(rx, (inst){I_EOL}); *quantifiable = 0; return 1; }
    if (c == '.') { emit(rx, (inst){I_ANY}); return 1; }
    if (c == '[') {
        if (peek(ps) == '^' && ps->i + 1 < ps->n && ps->pat[ps->i + 1] == ']') { ps->i += 2; emit(rx, (inst){I_ANYNL}); return 1; }
        const int k = parse_bracket(ps);
        emit(rx, (inst){I_CLASS, 0, k});
        return 1;
    }
    if (c == '(') {
        if (peek(ps) == '?') {
            ps->i++;
            const int k = peek(ps);
            ps->i++;
            if (k == ':') { parse_alt(ps); ps->i++; return 1; }
            int behind = 0;
            int look = k;
            if (k == '<') { behind = 1; look = peek(ps); ps->i++; }
            /* lookaround: LOOK x=start y=end of sub-program, sub-program ends with MATCH */
            const int at = emit(rx, (inst){I_LOOK});
            parse_alt(ps);
            emit(rx, (inst){I_MATCH});
            ps->i++;
            rx->code[at].x = at + 1;
            rx->code[at].y = rx->n;
            rx->code[at].behind = behind;
            rx->code[at].neg = look == '!';
            *quantifiable = 0;
            return 1;
        }
        const int g = ++rx->ngroup;
        emit(rx, (inst){I_SAVE, 0, 2 * g});
        parse_alt(ps);
        emit(rx, (inst){I_SAVE, 0, 2 * g + 1});
        ps->i++;
        return 1;
    }
    if (c == '\\') {
        const uint32_t e = ps->pat[ps->i++];
        if (e == 'b') { emit(rx, (inst){I_WB}); *quantifiable = 0; return 1; }
        if (e == 'B') { emit(rx, (inst){I_NWB}); *quantifiable = 0; return 1; }
        const int k = new_class(rx);
        if (class_escape(ps, &rx->cls[k], e)) { emit(rx, (inst){I_CLASS, 0, k}); return 1; }
        rx->ncls--;
        emit(rx, (inst){I_CHAR, parse_escape_cp(ps, e)});
        return 1;
    }
    emit(rx, (inst){I_CHAR, (uint32_t)c});
    return 1;
}

/* Copy instructions [a, b) to the end, relocating jumps inside the range. */
static void copy_range(rh_rx *rx, int a, int b) {
    const int base = rx->n;
    for (int i = a; i < b; i++) {
        inst in = rx->code[i];
        if (in.op == I_SPLIT || in.op == I_JMP) { in.x += base - a; if (in.op == I_SPLIT) in.y += base - a; }
        if (in.op == I_LOOK) { in.x += base - a; in.y += base - a; }
        emit(rx, in);
    }
}

static void parse_seq(parser *ps) {
    rh_rx *rx = ps->rx;
    for (;;) {
        const int start = rx->n;
        int q;
        if (!parse_atom(ps, &q)) return;
        const int c = peek(ps);
        int lo = -1, hi = -1;
        if (c == '*') { lo = 0; hi = -2; }
        else if (c == '+') { lo = 1; hi = -2; }
        else if (c == '?') { lo = 0; hi = 1; }
        else if (c == '{') {
            size_t j = ps->i + 1;
            int a = 0, b = -1, digits = 0;
            while (j < ps->n && ps->pat[j] >= '0' && ps->pat[j] <= '9') { a = a * 10 + (int)(ps->pat[j] - '0'); j++; digits++; }
            if (digits && j < ps->n && ps->pat[j] == '}') { b = a; ps->i = j; lo = a; hi = b; }
            else if (digits && j < ps->n && ps->pat[j] == ',') {
                j++;
                int d2 = 0;
                b = 0;
                while (j < ps->n && ps->pat[j] >= '0' && ps->pat[j] <= '9') { b = b * 10 + (int)(ps->pat[j] - '0'); j++; d2++; }
                if (j < ps->n && ps->pat[j] == '}') { ps->i = j; lo = a; hi = d2 ? b : -2; }
            }
        }
        if (lo < 0 || !q) continue;
        ps->i++;
        int lazy = 0;
        if (peek(ps) == '?') { lazy = 1; ps->i++; }
        /* move the atom out, then build: lo mandatory copies, then optional copies (or a loop) */
        const int alen = rx->n - start;
        inst *atom = rh_malloc((size_t)alen * sizeof(inst));
        memcpy(atom, rx->code + start, (size_t)alen * sizeof(inst));
        rx->n = start;
        const int tmpl = rx->n;
        /* stash the atom at the end temporarily as the template, then re-emit relocated copies */
        for (int k = 0; k < alen; k++) emit(rx, atom[k]);
        free(atom);
        const int tend = rx->n;
        /* we re-emit after the template and finally shift everything down over it */
        const int out0 = rx->n;
        for (int k = 0; k < lo; k++) {
            /* relocate copy: jumps inside the atom were relative to `start`; copy_range uses [tmpl, tend) */
            copy_range(rx, tmpl, tend);
        }
        if (hi == -2) { /* loop: L: SPLIT body, out ; body ; JMP L */
            const int L = emit(rx, (inst){I_SPLIT});
            copy_range(rx, tmpl, tend);
            emit(rx, (inst){I_JMP, 0, L});
            rx->code[L].x = lazy ? rx->n : L + 1;
            rx->code[L].y = lazy ? L + 1 : rx->n;
        } else {
            int *splits = rh_malloc((size_t)(hi - lo + 1) * sizeof(int));
            const int m = hi - lo;
            for (int k = 0; k < m; k++) {
                splits[k] = emit(rx, (inst){I_SPLIT});
                copy_range(rx, tmpl, tend);
            }
            for (int k = 0; k < m; k++) {
                rx->code[splits[k]].x = lazy ? rx->n : splits[k] + 1;
                rx->code[splits[k]].y = lazy ? splits[k] + 1 : rx->n;
            }
            free(splits);
        }
        /* shift [out0, n) down to tmpl (removing the template) and relocate absolute targets */
        const int shift = out0 - tmpl;
        for (int k = out0; k < rx->n; k++) {
            inst in = rx->code[k];
            if (in.op == I_SPLIT || in.op == I_JMP) { in.x -= shift; if (in.op == I_SPLIT) in.y -= shift; }
            if (in.op == I_LOOK) { in.x -= shift; in.y -= shift; }
            rx->code[k - shift] = in;
        }
        rx->n -= shift;
        (void)start;
    }
}

static void parse_alt(parser *ps) {
    rh_rx *rx = ps->rx;
    const int split_at = rx->n;
    parse_seq(ps);
    if (peek(ps) != '|') return;
    /* insert SPLIT before the first alternative: shift [split_at, n) up by one */
    emit(rx, (inst){I_MATCH});
    for (int k = rx->n - 1; k > split_at; k--) {
        inst in = rx->code[k - 1];
        if (in.op == I_SPLIT || in.op == I_JMP) { in.x += 1; if (in.op == I_SPLIT) in.y += 1; }
        if (in.op == I_LOOK) { in.x += 1; in.y += 1; }
        rx->code[k] = in;
    }
    rx->code[split_at] = (inst){I_SPLIT, 0, split_at + 1, 0};
    const int jmp = emit(rx, (inst){I_JMP});
    rx->code[split_at].y = rx->n;
    ps->i++; /* '|' */
    parse_alt(ps);
    rx->code[jmp].x = rx->n;
}

rh_rx *rh_rx_compile(const char *pattern, int flags) {
    rh_rx *rx = rh_malloc(sizeof *rx);
    memset(rx, 0, sizeof *rx);
    rx->flags = flags;
    parser ps = {0};
    ps.n = rh_utf8_decode(pattern, strlen(pattern), &ps.pat);
    ps.rx = rx;
    emit(rx, (inst){I_SAVE, 0, 0});
    parse_alt(&ps);
    emit(rx, (inst){I_SAVE, 0, 1});
    emit(rx, (inst){I_MATCH});
    free(ps.pat);
    if (ps.err || ps.i < ps.n) { rh_rx_free(rx); return NULL; }
    return rx;
}

void rh_rx_free(rh_rx *rx) {
    if (!rx) return;
    for (int i = 0; i < rx->ncls; i++) free(rx->cls[i].r);
    free(rx->cls);
    free(rx->code);
    free(rx);
}

int rh_rx_groups(const rh_rx *rx) { return rx->ngroup; }

/* ---- matcher: recursive backtracking over the instruction list ---- */
typedef struct {
    const rh_rx *rx;
    const uint32_t *s;
    size_t n;
    long *caps;
    long steps;
} mstate;

static int wb_at(const mstate *m, size_t i) {
    const int a = i > 0 && is_word(m->rx, m->s[i - 1]);
    const int b = i < m->n && is_word(m->rx, m->s[i]);
    return a != b;
}
static int ceq(const rh_rx *rx, uint32_t a, uint32_t b) {
    if (a == b) return 1;
    return (rx->flags & RX_I) && uc_fold_cp(a) == uc_fold_cp(b);
}

static int run(mstate *m, int pc, size_t i, size_t *end) {
    for (;;) {
        if (++m->steps > 2000000) return 0;
        const inst *in = &m->rx->code[pc];
        switch (in->op) {
        case I_MATCH: *end = i; return 1;
        case I_CHAR:
            if (i >= m->n || !ceq(m->rx, m->s[i], in->c)) return 0;
            i++; pc++; break;
        case I_ANY:
            if (i >= m->n || m->s[i] == '\n' || m->s[i] == '\r' || m->s[i] == 0x2028 || m->s[i] == 0x2029) return 0;
            i++; pc++; break;
        case I_ANYNL:
            if (i >= m->n) return 0;
            i++; pc++; break;
        case I_CLASS:
            if (i >= m->n || !class_match(m->rx, &m->rx->cls[in->x], m->s[i])) return 0;
            i++; pc++; break;
        case I_BOL: if (i != 0) return 0; pc++; break;
        case I_EOL: if (i != m->n) return 0; pc++; break;
        case I_WB: if (!wb_at(m, i)) return 0; pc++; break;
        case I_NWB: if (wb_at(m, i)) return 0; pc++; break;
        case I_JMP: pc = in->x; break;
        case I_SAVE: {
            const long old = m->caps[in->x];
            m->caps[in->x] = (long)i;
            if (run(m, pc + 1, i, end)) return 1;
            m->caps[in->x] = old;
            return 0;
        }
        case I_SPLIT:
            if (run(m, in->x, i, end)) return 1;
            pc = in->y;
            break;
        case I_LOOK: {
            int ok = 0;
            size_t e;
            if (!in->behind) ok = run(m, in->x, i, &e);
            else
                for (size_t k = i + 1; k-- > 0;) { /* sub-match starting at k ending exactly at i */
                    mstate sub = *m;
                    sub.n = i;
                    if (run(&sub, in->x, k, &e) && e == i) { ok = 1; m->steps = sub.steps; break; }
                    m->steps = sub.steps;
                }
            if (ok == in->neg) return 0;
            pc = in->y;
            break;
        }
        default: return 0;
        }
    }
}

/* Leftmost match at or after `from`: caps[0..2*(groups+1)) get [start, end) pairs (-1 if unset). */
int rh_rx_search(const rh_rx *rx, const uint32_t *s, size_t n, size_t from, long *caps) {
    const int nc = 2 * (rx->ngroup + 1);
    for (size_t st = from; st <= n; st++) {
        for (int k = 0; k < nc; k++) caps[k] = -1;
        mstate m = {rx, s, n, caps, 0};
        size_t end;
        if (run(&m, 0, st, &end)) return 1;
    }
    return 0;
}
