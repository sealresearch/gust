/* Relay as a Gust model (sdk/include/gust_model.h): the plugin around the Relay host, nothing more. The request goes in
   as JSON and the calls come back as JSON; everything else (tokenizing, value programs, decoding) is the host's.
   Built into the model library with -DGUST_MODEL_RELAY. */
#include "gust_model.h"
#include "../host/relay.h"
#include "../host/relay_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char INFO[] =
    "{\"id\":\"relay-1\",\"name\":\"Relay 1\",\"kind\":\"tool-calling\","
    "\"summary\":\"Turns a request and a list of tools into exact function calls, or into no call when nothing fits.\","
    "\"inputs\":[{\"type\":\"json\",\"description\":\"{text, tools: [{name, description, params: [{name, type, description, "
    "required, enum}]}], kind?, context?: {now}}; tools written as JSON Schema (OpenAI parameters, Anthropic input_schema, MCP "
    "inputSchema) are converted\"}],"
    "\"outputs\":[{\"type\":\"json\",\"description\":\"[{tool, args}], or [] when no tool fits\"}],"
    "\"options\":{\"open\":{\"threads\":\"1 to 16, default 1\",\"prefilter\":\"rank long tool lists first, default true\","
    "\"k\":\"tools the prefilter keeps, default 10\",\"budget\":\"tool tokens the prefilter keeps, default 600\","
    "\"fixes\":\"host decoding fixes, default true\",\"require\":\"drop calls that leave a required argument empty, default true\"},"
    "\"run\":{\"mode\":\"calls (default), explain, trace or trace_ops\"}},"
    "\"files\":{\"required\":[\"relay.gust\",\"relay-route.gust\",\"relay-anchor.gust\",\"relay-args.gust\","
    "\"relay-tokenizer.json\"],\"optional\":[\"relay-prefilter.bin\"]},"
    "\"streaming\":false,\"license\":{\"code\":\"MIT\",\"weights\":\"CC BY 4.0\"}}";

/* An integer or boolean member of an options object; `dflt` when it is absent. */
static int opt_int(const jnode *o, const char *key, int dflt, int lo, int hi) {
    const jnode *v = o ? rh_json_get(o, key) : NULL;
    if (!v) return dflt;
    if (v->t == J_TRUE) return 1;
    if (v->t == J_FALSE) return 0;
    if (v->t != J_NUM) return dflt;
    int n = (int)v->num;
    return n < lo ? lo : n > hi ? hi : n;
}

/* ---- tool definitions ----
   Relay's own format is {name, description, params: [{name, type, description, required, enum, ...}]}. Tools written for
   OpenAI ({type: "function", function: {name, description, parameters}}), Anthropic (input_schema) or MCP (inputSchema)
   carry a JSON Schema instead, and the host would read them as tools without parameters. A request with any such tool is
   rewritten here into Relay's format before the host sees it; a request that is all Relay format passes through untouched. */

static void put_text(sb *b, const char *s, size_t n) {
    sb_str(b, "\"");
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { sb_put(b, "\\", 1); sb_put(b, s + i, 1); }
        else if (c == '\n') sb_str(b, "\\n");
        else if (c == '\r') sb_str(b, "\\r");
        else if (c == '\t') sb_str(b, "\\t");
        else if (c < 0x20) { char u[8]; snprintf(u, sizeof u, "\\u%04x", c); sb_str(b, u); }
        else sb_put(b, s + i, 1);
    }
    sb_str(b, "\"");
}

/* Any JSON value, as parsed. */
static void put_node(sb *b, const jnode *j) {
    switch (j->t) {
    case J_NULL: sb_str(b, "null"); break;
    case J_TRUE: sb_str(b, "true"); break;
    case J_FALSE: sb_str(b, "false"); break;
    case J_NUM: sb_put(b, j->s, j->len); break;
    case J_STR: put_text(b, j->s, j->len); break;
    case J_ARR:
        sb_str(b, "[");
        for (const jnode *c = j->first; c; c = c->next) { if (c != j->first) sb_str(b, ","); put_node(b, c); }
        sb_str(b, "]");
        break;
    case J_OBJ:
        sb_str(b, "{");
        for (const jnode *c = j->first; c; c = c->next) {
            if (c != j->first) sb_str(b, ",");
            put_text(b, c->key, c->klen);
            sb_str(b, ":");
            put_node(b, c);
        }
        sb_str(b, "}");
        break;
    }
}

static int is_text(const jnode *j, const char *s) { return j && j->t == J_STR && j->len == strlen(s) && !memcmp(j->s, s, j->len); }

static int lists(const jnode *arr, const char *key, size_t klen) {
    if (!arr || arr->t != J_ARR) return 0;
    for (const jnode *c = arr->first; c; c = c->next)
        if (c->t == J_STR && c->len == klen && !memcmp(c->s, key, klen)) return 1;
    return 0;
}

/* One JSON Schema property as a Relay parameter. */
static void put_param(sb *b, const char *name, size_t nlen, const jnode *s, int required) {
    const jnode *ty = rh_json_get(s, "type"), *desc = rh_json_get(s, "description"), *en = rh_json_get(s, "enum"),
                *df = rh_json_get(s, "default"), *items = rh_json_get(s, "items"), *props = rh_json_get(s, "properties"),
                *lo = rh_json_get(s, "minimum"), *hi = rh_json_get(s, "maximum");
    if (ty && ty->t == J_ARR) { /* ["string", "null"]: the first type that is not null */
        const jnode *pick = NULL;
        for (const jnode *c = ty->first; c && !pick; c = c->next) if (c->t == J_STR && !is_text(c, "null")) pick = c;
        ty = pick;
    }
    sb_str(b, "{\"name\":");
    put_text(b, name, nlen);
    sb_str(b, ",\"type\":");
    if (ty && ty->t == J_STR) put_node(b, ty);
    else sb_str(b, props && props->t == J_OBJ ? "\"object\"" : items && items->t == J_OBJ ? "\"array\"" : "\"string\"");
    /* The description, with the range appended the way the demo and the training data wrote it: "text (1-10)". */
    sb d = {0};
    if (desc && desc->t == J_STR) sb_put(&d, desc->s, desc->len);
    if ((lo && lo->t == J_NUM) || (hi && hi->t == J_NUM)) {
        if (d.n) sb_str(&d, " ");
        sb_str(&d, "(");
        if (lo && lo->t == J_NUM) sb_put(&d, lo->s, lo->len);
        sb_str(&d, "-");
        if (hi && hi->t == J_NUM) sb_put(&d, hi->s, hi->len);
        sb_str(&d, ")");
    }
    sb_str(b, ",\"description\":");
    put_text(b, d.s ? d.s : "", d.n);
    sb_free(&d);
    sb_str(b, required ? ",\"required\":true" : ",\"required\":false");
    if (en && en->t == J_ARR) { sb_str(b, ",\"enum\":"); put_node(b, en); }
    if (df) { sb_str(b, ",\"default\":"); put_node(b, df); }
    if (items && items->t == J_OBJ) { sb_str(b, ",\"items\":"); put_param(b, "item", 4, items, 0); }
    if (props && props->t == J_OBJ) {
        const jnode *req = rh_json_get(s, "required");
        sb_str(b, ",\"properties\":[");
        for (const jnode *c = props->first; c; c = c->next) {
            if (c != props->first) sb_str(b, ",");
            put_param(b, c->key, c->klen, c, lists(req, c->key, c->klen));
        }
        sb_str(b, "]");
    }
    sb_str(b, "}");
}

/* A tool written as a JSON Schema (and not already in Relay's format) needs converting. */
static const jnode *unwrap(const jnode *t) { /* OpenAI wraps the tool as {type: "function", function: {...}} */
    const jnode *f = rh_json_get(t, "function");
    return f && f->t == J_OBJ ? f : t;
}

static const jnode *schema_of(const jnode *t) { /* OpenAI parameters, Anthropic input_schema, MCP inputSchema */
    static const char *const KEYS[] = {"parameters", "input_schema", "inputSchema"};
    for (int i = 0; i < 3; i++) {
        const jnode *s = rh_json_get(t, KEYS[i]);
        if (s && s->t == J_OBJ) return s;
    }
    return NULL;
}

static int needs_conversion(const jnode *t) {
    if (!t || t->t != J_OBJ || rh_json_get(t, "params")) return 0;
    return unwrap(t) != t || schema_of(t) != NULL;
}

static void put_tool(sb *b, const jnode *t) {
    t = unwrap(t);
    const jnode *schema = schema_of(t);
    const jnode *name = rh_json_get(t, "name"), *desc = rh_json_get(t, "description"), *params = rh_json_get(t, "params");
    sb_str(b, "{\"name\":");
    if (name && name->t == J_STR) put_node(b, name); else sb_str(b, "\"\"");
    sb_str(b, ",\"description\":");
    if (desc && desc->t == J_STR) put_node(b, desc); else sb_str(b, "\"\"");
    sb_str(b, ",\"params\":");
    if (params && params->t == J_ARR) put_node(b, params);
    else {
        const jnode *props = schema ? rh_json_get(schema, "properties") : NULL, *req = schema ? rh_json_get(schema, "required") : NULL;
        sb_str(b, "[");
        if (props && props->t == J_OBJ)
            for (const jnode *c = props->first; c; c = c->next) {
                if (c != props->first) sb_str(b, ",");
                put_param(b, c->key, c->klen, c, lists(req, c->key, c->klen));
            }
        sb_str(b, "]");
    }
    sb_str(b, "}");
}

/* The request with every JSON Schema tool converted, or NULL (and `b` untouched) when no tool needs it. */
static int convert_request(sb *b, const jnode *req) {
    const jnode *tools = rh_json_get(req, "tools");
    int any = 0;
    for (const jnode *t = tools->first; t && !any; t = t->next) any = needs_conversion(t);
    if (!any) return 0;
    sb_str(b, "{");
    for (const jnode *c = req->first; c; c = c->next) {
        if (c != req->first) sb_str(b, ",");
        put_text(b, c->key, c->klen);
        sb_str(b, ":");
        if (c == tools) {
            sb_str(b, "[");
            for (const jnode *t = tools->first; t; t = t->next) {
                if (t != tools->first) sb_str(b, ",");
                if (needs_conversion(t)) put_tool(b, t); else put_node(b, t);
            }
            sb_str(b, "]");
        } else put_node(b, c);
    }
    sb_str(b, "}");
    return 1;
}

static const void *find(const gust_model_file *f, size_t n, const char *name, size_t *len) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(f[i].name, name)) { *len = f[i].len; return f[i].data; }
    *len = 0;
    return NULL;
}

static int relay_open_model(const gust_model_file *files, size_t n_files, const char *options_json, void **state, char *err,
                            size_t errlen) {
    relay_files rf;
    memset(&rf, 0, sizeof rf);
    rf.encoder = find(files, n_files, "relay.gust", &rf.encoder_len);
    rf.route = find(files, n_files, "relay-route.gust", &rf.route_len);
    rf.anchor = find(files, n_files, "relay-anchor.gust", &rf.anchor_len);
    rf.args = find(files, n_files, "relay-args.gust", &rf.args_len);
    rf.tokenizer = (const char *)find(files, n_files, "relay-tokenizer.json", &rf.tokenizer_len);
    rf.prefilter = find(files, n_files, "relay-prefilter.bin", &rf.prefilter_len);
    if (!rf.encoder || !rf.route || !rf.anchor || !rf.args || !rf.tokenizer) {
        snprintf(err, errlen, "relay needs relay.gust, relay-route.gust, relay-anchor.gust, relay-args.gust and relay-tokenizer.json");
        return GUST_MODEL_E_FILES;
    }
    relay_options o;
    relay_default_options(&o);
    rh_arena a = {0};
    if (options_json && options_json[0]) {
        const jnode *j = rh_json_parse(&a, options_json, strlen(options_json));
        if (!j || j->t != J_OBJ) {
            rh_arena_free(&a);
            snprintf(err, errlen, "the options are not a JSON object");
            return GUST_MODEL_E_ARGUMENT;
        }
        o.threads = opt_int(j, "threads", o.threads, 1, 16);
        o.prefilter = opt_int(j, "prefilter", o.prefilter, 0, 1);
        o.prefilter_k = opt_int(j, "k", o.prefilter_k, 1, 64);
        o.prefilter_budget = opt_int(j, "budget", o.prefilter_budget, 0, 100000);
        o.fixes = opt_int(j, "fixes", o.fixes, 0, 1);
        o.require = opt_int(j, "require", o.require, 0, 1);
    }
    rh_arena_free(&a);
    relay *r = relay_open_mem(&rf, &o, err, errlen);
    if (!r) return GUST_MODEL_E_OPEN;
    *state = r;
    return GUST_MODEL_OK;
}

static int relay_run_model(void *state, const gust_buffer *in, size_t n_in, const char *options_json, gust_result *out,
                           char *err, size_t errlen) {
    if (n_in != 1 || (in[0].type != GUST_BUF_JSON && in[0].type != GUST_BUF_TEXT) || !in[0].data) {
        snprintf(err, errlen, "relay takes one JSON request: {text, tools, kind?, context?}");
        return GUST_MODEL_E_INPUT;
    }
    const char *request = (const char *)in[0].data;
    size_t request_len = in[0].len;
    sb converted = {0};
    {   /* a request is an object with a text and a list of tools; anything else is the caller's mistake, said clearly */
        rh_arena a = {0};
        const jnode *req = rh_json_parse(&a, (const char *)in[0].data, in[0].len);
        const jnode *text = req && req->t == J_OBJ ? rh_json_get(req, "text") : NULL;
        const jnode *tools = req && req->t == J_OBJ ? rh_json_get(req, "tools") : NULL;
        const int ok = text && text->t == J_STR && tools && tools->t == J_ARR;
        if (ok && convert_request(&converted, req)) { request = converted.s; request_len = converted.n; }
        rh_arena_free(&a);
        if (!ok) {
            sb_free(&converted);
            snprintf(err, errlen, "the request must be a JSON object with a text (string) and tools (array)");
            return GUST_MODEL_E_INPUT;
        }
    }
    int mode = 0; /* 0 calls, 1 explain, 2 trace, 3 trace with the operators */
    if (options_json && options_json[0]) {
        rh_arena a = {0};
        const jnode *j = rh_json_parse(&a, options_json, strlen(options_json));
        const jnode *m = j && j->t == J_OBJ ? rh_json_get(j, "mode") : NULL;
        if (m && m->t == J_STR) {
            if (m->len == 7 && !memcmp(m->s, "explain", 7)) mode = 1;
            else if (m->len == 5 && !memcmp(m->s, "trace", 5)) mode = 2;
            else if (m->len == 9 && !memcmp(m->s, "trace_ops", 9)) mode = 3;
        }
        rh_arena_free(&a);
    }
    relay *r = (relay *)state;
    char *text = mode == 1 ? relay_explain(r, request, request_len)
                 : mode == 2 ? relay_trace(r, request, request_len)
                 : mode == 3 ? relay_trace_ops(r, request, request_len)
                             : relay_call(r, request, request_len);
    sb_free(&converted);
    if (!text) {
        snprintf(err, errlen, "the request is not valid JSON");
        return GUST_MODEL_E_INPUT;
    }
    int s = gust_result_add_json(out, text);
    relay_free(text);
    return s;
}

static void relay_close_model(void *state) { relay_close((relay *)state); }
static size_t relay_memory(void *state) { return relay_model_bytes((const relay *)state); }

static const gust_model_def DEF = {sizeof(gust_model_def), "relay-1", INFO, relay_open_model, relay_run_model,
                                   NULL, NULL, NULL, NULL, NULL, relay_close_model, relay_memory};

const gust_model_def *gust_model_relay_def(void) { return &DEF; }
