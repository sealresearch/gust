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
    "required, enum}]}], kind?, context?: {now}}\"}],"
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
    {   /* a request is an object with a text and a list of tools; anything else is the caller's mistake, said clearly */
        rh_arena a = {0};
        const jnode *req = rh_json_parse(&a, (const char *)in[0].data, in[0].len);
        const jnode *text = req && req->t == J_OBJ ? rh_json_get(req, "text") : NULL;
        const jnode *tools = req && req->t == J_OBJ ? rh_json_get(req, "tools") : NULL;
        const int ok = text && text->t == J_STR && tools && tools->t == J_ARR;
        rh_arena_free(&a);
        if (!ok) {
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
    char *text = mode == 1 ? relay_explain(r, (const char *)in[0].data, in[0].len)
                 : mode == 2 ? relay_trace(r, (const char *)in[0].data, in[0].len)
                 : mode == 3 ? relay_trace_ops(r, (const char *)in[0].data, in[0].len)
                             : relay_call(r, (const char *)in[0].data, in[0].len);
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
