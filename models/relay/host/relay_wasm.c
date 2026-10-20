/* WebAssembly entry points of the Relay host (wasm32-wasi reactor, no file system: the JavaScript wrapper passes the
   model files as memory buffers). See web/relay-native.mjs. */
#include "relay.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT(name) __attribute__((export_name(#name)))

static char last_error[512];

EXPORT(relay_wasm_alloc) void *relay_wasm_alloc(size_t n) { return malloc(n ? n : 1); }
EXPORT(relay_wasm_free) void relay_wasm_free(void *p) { free(p); }
EXPORT(relay_wasm_error) const char *relay_wasm_error(void) { return last_error; }

/* files: 6 (pointer, length) pairs in relay_files order; flags: bit0 fixes, bit1 prefilter */
EXPORT(relay_wasm_open)
relay *relay_wasm_open(const uint32_t *files, uint32_t flags, int k, int budget, double gate, double gate2, int require) {
    relay_files f;
    const void **p[6] = {&f.encoder, &f.route, &f.anchor, &f.args, (const void **)&f.tokenizer, &f.prefilter};
    size_t *l[6] = {&f.encoder_len, &f.route_len, &f.anchor_len, &f.args_len, &f.tokenizer_len, &f.prefilter_len};
    for (int i = 0; i < 6; i++) {
        *p[i] = (const void *)(uintptr_t)files[2 * i];
        *l[i] = files[2 * i + 1];
    }
    relay_options o;
    relay_default_options(&o);
    o.fixes = flags & 1u;
    o.prefilter = (flags >> 1) & 1u;
    o.prefilter_k = k;
    o.prefilter_budget = budget;
    o.gate = gate;
    o.gate2 = gate2;
    o.require = require;
    last_error[0] = 0;
    return relay_open_mem(&f, &o, last_error, sizeof last_error);
}
EXPORT(relay_wasm_call) char *relay_wasm_call(relay *r, const char *record, size_t len) { return relay_call(r, record, len); }
EXPORT(relay_wasm_explain) char *relay_wasm_explain(relay *r, const char *record, size_t len) { return relay_explain(r, record, len); }
EXPORT(relay_wasm_trace) char *relay_wasm_trace(relay *r, const char *record, size_t len, int operators) {
    return operators ? relay_trace_ops(r, record, len) : relay_trace(r, record, len);
}
EXPORT(relay_wasm_close) void relay_wasm_close(relay *r) { relay_close(r); }
EXPORT(relay_wasm_model_bytes) size_t relay_wasm_model_bytes(relay *r) { return relay_model_bytes(r); }
