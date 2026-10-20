/* Relay: on-device tool calling on the Gust runtime. C11, no dependencies beyond the C library.

     relay *r = relay_open("models/relay1", NULL, err, sizeof err);
     char *calls = relay_call(r, "{\"text\": \"...\", \"tools\": [...]}", (size_t)-1);  // [{"tool": ..., "args": {...}}]
     relay_free(calls);
     relay_close(r);

   A record is {text, tools: [{name, description, params: [{name, type, description, required, enum, default,
   items}]}], kind?, context?: {now}}. A model directory holds relay.gust, relay-route.gust, relay-anchor.gust, relay-args.gust, relay-tokenizer.json and optionally relay-prefilter.bin. */
#ifndef RELAY_H
#define RELAY_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct relay relay;

typedef struct {
    double gate, gate2;   /* route thresholds: 0 and 2 */
    int require;          /* drop calls that leave a required parameter empty: 1 */
    int fixes;            /* host decoding fixes (2026-10-17 audit): 1; 0 = Python-parity decoder */
    int prefilter;        /* rank tools with relay-prefilter.bin when present: 1 */
    int prefilter_k;      /* tools kept: 10 */
    int prefilter_budget; /* tool tokens kept (best first, at least 3 tools): 600; 0 = no budget */
    int threads;          /* worker threads (builds with threads): 1 */
    const char *tokenizer; /* tokenizer path when not in the model directory */
} relay_options;

void relay_default_options(relay_options *o);
/* the model files in memory (absent ones NULL); the buffers may be freed after relay_open_mem */
typedef struct {
    const void *encoder; size_t encoder_len; /* relay.gust */
    const void *route; size_t route_len;
    const void *anchor; size_t anchor_len;
    const void *args; size_t args_len;
    const char *tokenizer; size_t tokenizer_len;
    const void *prefilter; size_t prefilter_len;
} relay_files;
/* NULL on error (message in err) */
relay *relay_open_mem(const relay_files *files, const relay_options *opt, char *err, size_t errlen);
/* the files of a model directory (builds without RELAY_NO_FILES) */
relay *relay_open(const char *dir, const relay_options *opt, char *err, size_t errlen);
/* the calls of one record as JSON (caller frees with relay_free); NULL when the record is not valid JSON. len may be
   (size_t)-1 for a NUL-terminated record. */
char *relay_call(relay *r, const char *record, size_t len);
/* for demos and debugging, as relay.explain() of the JavaScript host: {"calls", "dropped" (held by require),
   "chunks": [{"margin", "margins": [{"tool", "margin", "count"}]}]}; each call also carries "why" (per argument: the
   value program with UTF-16 spans of the request, its confidence and option probability), "anchor" and "margin" */
char *relay_explain(relay *r, const char *record, size_t len);
/* Opt-in diagnostic call: {events: [...], result: <relay_explain output>}. Events follow execution order,
   including retrieval, token limits, chunks and graph status. Extra work/memory only while tracing.
   Caller frees with relay_free. A handle must not be called concurrently or re-entered. */
char *relay_trace(relay *r, const char *record, size_t len);
/* Same trace including Gust operator events; can be verbose for cold tool catalogs. */
char *relay_trace_ops(relay *r, const char *record, size_t len);
void relay_free(void *p);
void relay_close(relay *r);
/* memory held by the loaded graphs (bytes) */
size_t relay_model_bytes(const relay *r);

#ifdef __cplusplus
}
#endif
#endif
