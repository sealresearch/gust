/* Minimal JSON DOM for the native Relay host: objects keep key order, numbers keep their source text (Relay's
   output must spell 1.0 and 1 as the input did), strings are decoded to UTF-8. Nodes live in one arena. */
#include "relay_host.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p, *end;
    rh_arena *a;
    int err;
} jparser;

static void ws(jparser *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static jnode *node(jparser *j, jtype t) {
    jnode *n = rh_arena_alloc(j->a, sizeof(jnode));
    memset(n, 0, sizeof *n);
    n->t = t;
    return n;
}

static int hex4(const char *p, uint32_t *v) {
    *v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = p[i];
        *v <<= 4;
        if (c >= '0' && c <= '9') *v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') *v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') *v |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    return 1;
}

size_t rh_utf8_put(char *o, uint32_t cp) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xc0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 63)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xe0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63));
        return 3;
    }
    o[0] = (char)(0xf0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 63));
    o[2] = (char)(0x80 | ((cp >> 6) & 63)); o[3] = (char)(0x80 | (cp & 63));
    return 4;
}

static char *string(jparser *j, size_t *len) {
    /* j->p at the opening quote; output never exceeds the input length */
    const char *s = ++j->p;
    const char *e = s;
    while (e < j->end && *e != '"') e += (*e == '\\') ? 2 : 1;
    char *o = rh_arena_alloc(j->a, (size_t)(e - s) + 1);
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c != '\\') { o[n++] = c; continue; }
        c = *j->p++;
        switch (c) {
        case 'n': o[n++] = '\n'; break;
        case 't': o[n++] = '\t'; break;
        case 'r': o[n++] = '\r'; break;
        case 'b': o[n++] = '\b'; break;
        case 'f': o[n++] = '\f'; break;
        case 'u': {
            uint32_t v;
            if (j->end - j->p < 4 || !hex4(j->p, &v)) { j->err = 1; return o; }
            j->p += 4;
            if (v >= 0xd800 && v < 0xdc00 && j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                uint32_t lo;
                if (hex4(j->p + 2, &lo) && lo >= 0xdc00 && lo < 0xe000) {
                    v = 0x10000 + ((v - 0xd800) << 10) + (lo - 0xdc00);
                    j->p += 6;
                }
            }
            n += rh_utf8_put(o + n, v); /* a lone surrogate is kept as its 3-byte form */
            break;
        }
        default: o[n++] = c;
        }
    }
    j->p++; /* closing quote */
    o[n] = 0;
    *len = n;
    return o;
}

static jnode *value(jparser *j) {
    ws(j);
    if (j->p >= j->end) { j->err = 1; return NULL; }
    const char c = *j->p;
    if (c == '{' || c == '[') {
        jnode *n = node(j, c == '{' ? J_OBJ : J_ARR), *last = NULL;
        j->p++;
        ws(j);
        if (j->p < j->end && *j->p == (c == '{' ? '}' : ']')) { j->p++; return n; }
        for (;;) {
            char *key = NULL;
            size_t klen = 0;
            if (c == '{') {
                ws(j);
                if (j->p >= j->end || *j->p != '"') { j->err = 1; return n; }
                key = string(j, &klen);
                ws(j);
                if (j->p >= j->end || *j->p != ':') { j->err = 1; return n; }
                j->p++;
            }
            jnode *v = value(j);
            if (!v || j->err) { j->err = 1; return n; }
            v->key = key;
            v->klen = klen;
            if (last) last->next = v; else n->first = v;
            last = v;
            n->count++;
            ws(j);
            if (j->p < j->end && *j->p == ',') { j->p++; continue; }
            if (j->p < j->end && *j->p == (c == '{' ? '}' : ']')) { j->p++; return n; }
            j->err = 1;
            return n;
        }
    }
    if (c == '"') {
        jnode *n = node(j, J_STR);
        n->s = string(j, &n->len);
        return n;
    }
    if (!strncmp(j->p, "true", 4)) { j->p += 4; return node(j, J_TRUE); }
    if (!strncmp(j->p, "false", 5)) { j->p += 5; return node(j, J_FALSE); }
    if (!strncmp(j->p, "null", 4)) { j->p += 4; return node(j, J_NULL); }
    /* number (also NaN / Infinity, which Python's json writes) */
    const char *s = j->p;
    while (j->p < j->end && strchr("+-0123456789.eEInfityNa", *j->p)) j->p++;
    if (j->p == s) { j->err = 1; return NULL; }
    jnode *n = node(j, J_NUM);
    n->len = (size_t)(j->p - s);
    char *t = rh_arena_alloc(j->a, n->len + 1);
    memcpy(t, s, n->len);
    t[n->len] = 0;
    n->s = t;
    n->num = strtod(t, NULL);
    return n;
}

jnode *rh_json_parse(rh_arena *a, const char *text, size_t len) {
    jparser j = {text, text + len, a, 0};
    jnode *v = value(&j);
    return j.err ? NULL : v;
}

jnode *rh_json_get(const jnode *obj, const char *key) {
    if (!obj || obj->t != J_OBJ) return NULL;
    const size_t kl = strlen(key);
    jnode *hit = NULL; /* JSON.parse: the last duplicate key wins */
    for (jnode *c = obj->first; c; c = c->next)
        if (c->klen == kl && !memcmp(c->key, key, kl)) hit = c;
    return hit;
}
