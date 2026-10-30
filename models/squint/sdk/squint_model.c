/* Squint as a Gust model (sdk/include/gust_model.h): the plugin around the Squint host, nothing more. A frame goes in as a
   gray image and the read comes back as JSON; tracking, acquisition and the evidence check are the host's.
   Built into the model library with -DGUST_MODEL_SQUINT.

   Squint has no model files. Its hosts keep their state in static memory, so one reader can be open at a time. */
#include "gust_model.h"
#include "../host/squint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char INFO[] =
    "{\"id\":\"squint-1\",\"name\":\"Squint 1\",\"kind\":\"qr-reader\","
    "\"summary\":\"Reads QR codes from shaky, blurred and badly focused camera frames, and only returns a payload the current "
    "frame supports.\","
    "\"inputs\":[{\"type\":\"image/gray8\",\"description\":\"one camera frame, width * height bytes, at most 1280 x 720, in "
    "order; or image/rgba8, which is converted to gray with the luminance Squint was tuned on\"}],"
    "\"outputs\":[{\"type\":\"json\",\"description\":\"{accepted, reason, path, version, payload (hex), text (UTF-8 or null), "
    "corners: [[x, y] x 4]}\"}],"
    "\"options\":{\"open\":{},"
    "\"run\":{\"time\":\"frame time in milliseconds, never going backwards (default: the clock)\","
    "\"reset\":\"true forgets the tracked code first (a new camera stream or scene)\","
    "\"budget_ms\":\"time budget of the tracker, default none\","
    "\"acquire_budget_ms\":\"time budget of the full-frame reader, default none\","
    "\"tracked_acquire_budget_ms\":\"its budget while a code is tracked, default none\","
    "\"acquire_every\":\"while a code is tracked, read the full frame on every n-th frame, default 3\","
    "\"enhance\":\"alternate 1280 x 720 frames use a lighting-normalized view, default true\"}},"
    "\"files\":{\"required\":[]},\"streaming\":true,\"license\":{\"code\":\"MIT\",\"weights\":\"none\"}}";

/* The text of a member of a small JSON object; this only needs numbers and booleans. */
static int find_number(const char *json, const char *key, double *out) {
    if (!json) return 0;
    char pat[48];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ') p++;
    if (!strncmp(p, "true", 4)) { *out = 1; return 1; }
    if (!strncmp(p, "false", 5)) { *out = 0; return 1; }
    if (!strncmp(p, "null", 4)) return 0;
    char *end;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

extern double sq_clock_seconds(void);
static double now_ms(void) { return sq_clock_seconds() * 1000.0; }

typedef struct {
    int stream_open;
    int pending; /* a result waiting to be pulled */
    int finished;
    char result[16384];
    unsigned char *gray; /* scratch for RGBA input */
    size_t gray_cap;
} squint_state;

static squint_state *g_state;

static int squint_open_model(const gust_model_file *files, size_t n_files, const char *options_json, void **state,
                             char *err, size_t errlen) {
    (void)files;
    (void)n_files;
    (void)options_json;
    squint_state *s = (squint_state *)calloc(1, sizeof *s);
    if (!s) {
        snprintf(err, errlen, "out of memory");
        return GUST_MODEL_E_MEMORY;
    }
    if (sq_open(err, errlen)) {
        free(s);
        return GUST_MODEL_E_OPEN;
    }
    g_state = s;
    *state = s;
    return GUST_MODEL_OK;
}

static void squint_close_model(void *state) {
    squint_state *s = (squint_state *)state;
    sq_close();
    free(s->gray);
    free(s);
    g_state = NULL;
}

/* Whether `p` holds one UTF-8 sequence, and its length. */
static int utf8_len(const unsigned char *p, size_t left) {
    if (p[0] < 0x80) return 1;
    if (p[0] >= 0xC2 && p[0] <= 0xDF && left >= 2 && (p[1] & 0xC0) == 0x80) return 2;
    if (p[0] >= 0xE0 && p[0] <= 0xEF && left >= 3 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
        unsigned cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) return 3;
    }
    if (p[0] >= 0xF0 && p[0] <= 0xF4 && left >= 4 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 &&
        (p[3] & 0xC0) == 0x80) {
        unsigned cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        if (cp >= 0x10000 && cp <= 0x10FFFF) return 4;
    }
    return 0;
}

static void json_string(char **w, const unsigned char *p, size_t n) {
    char *o = *w;
    *o++ = '"';
    for (size_t i = 0; i < n; i++) {
        unsigned char c = p[i];
        if (c == '"' || c == '\\') { *o++ = '\\'; *o++ = (char)c; }
        else if (c == '\n') { *o++ = '\\'; *o++ = 'n'; }
        else if (c == '\r') { *o++ = '\\'; *o++ = 'r'; }
        else if (c == '\t') { *o++ = '\\'; *o++ = 't'; }
        else if (c < 0x20 || c == 0x7F) o += sprintf(o, "\\u%04x", c);
        else *o++ = (char)c;
    }
    *o++ = '"';
    *w = o;
}

/* {accepted, reason, path, version, payload, text, corners} */
static void format_result(char *buf, size_t cap, const sq_read_result *r) {
    char *w = buf;
    w += sprintf(w, "{\"accepted\":%s", r->accepted ? "true" : "false");
    if (r->accepted) {
        w += sprintf(w, ",\"reason\":null,\"path\":\"%s\",\"version\":%d,\"payload\":\"", r->path ? r->path : "", r->version);
        for (int i = 0; i < r->payload_len; i++) w += sprintf(w, "%02x", r->payload[i]);
        w += sprintf(w, "\",\"text\":");
        int valid = 1;
        for (size_t i = 0; i < (size_t)r->payload_len && valid;) {
            int l = utf8_len(r->payload + i, (size_t)r->payload_len - i);
            if (!l) valid = 0;
            i += (size_t)l;
        }
        if (valid && (size_t)(w - buf) + (size_t)r->payload_len * 6 + 400 < cap) json_string(&w, r->payload, (size_t)r->payload_len);
        else w += sprintf(w, "null");
        w += sprintf(w, ",\"corners\":[");
        for (int k = 0; k < 4; k++) w += sprintf(w, "%s[%.17g,%.17g]", k ? "," : "", r->corners[2 * k], r->corners[2 * k + 1]);
        w += sprintf(w, "]");
    } else {
        w += sprintf(w, ",\"reason\":\"%s\",\"path\":null,\"version\":null,\"payload\":null,\"text\":null,\"corners\":null",
                     r->reason ? r->reason : "");
    }
    w += sprintf(w, "}");
    (void)cap;
}

/* RGBA to gray with the luminance Squint was tuned on (the sr_luminance of the resampling host: 306, 601, 117 over 1024). */
static const unsigned char *to_gray(squint_state *s, const gust_buffer *b, int *w, int *h, char *err, size_t errlen) {
    *w = (int)b->width;
    *h = (int)b->height;
    size_t px = (size_t)*w * (size_t)*h;
    if (b->type == GUST_BUF_IMAGE_GRAY8) {
        if (b->len != px) {
            snprintf(err, errlen, "the gray frame is %zu bytes, but %d x %d needs %zu", b->len, *w, *h, px);
            return NULL;
        }
        return (const unsigned char *)b->data;
    }
    if (b->len != px * 4) {
        snprintf(err, errlen, "the RGBA frame is %zu bytes, but %d x %d needs %zu", b->len, *w, *h, px * 4);
        return NULL;
    }
    if (s->gray_cap < px) {
        unsigned char *g = (unsigned char *)realloc(s->gray, px);
        if (!g) {
            snprintf(err, errlen, "out of memory");
            return NULL;
        }
        s->gray = g;
        s->gray_cap = px;
    }
    const unsigned char *rgba = (const unsigned char *)b->data;
    for (size_t i = 0; i < px; i++)
        s->gray[i] = (unsigned char)((306u * rgba[4 * i] + 601u * rgba[4 * i + 1] + 117u * rgba[4 * i + 2] + 512u) >> 10);
    return s->gray;
}

static int process(squint_state *s, const gust_buffer *in, size_t n_in, const char *options_json, char *err, size_t errlen) {
    if (n_in < 1 || (in[0].type != GUST_BUF_IMAGE_GRAY8 && in[0].type != GUST_BUF_IMAGE_RGBA8) || !in[0].data) {
        snprintf(err, errlen, "squint takes one gray8 or rgba8 image");
        return GUST_MODEL_E_INPUT;
    }
    int w, h;
    const unsigned char *gray = to_gray(s, &in[0], &w, &h, err, errlen);
    if (!gray) return GUST_MODEL_E_INPUT;
    sq_read_options o;
    sq_default_options(&o);
    o.budget_ms = o.acquire_budget_ms = o.tracked_acquire_budget_ms = INFINITY;
    const char *opts = options_json;
    if (n_in > 1 && in[1].type == GUST_BUF_JSON && in[1].data) opts = (const char *)in[1].data; /* streams: options ride along */
    double v;
    o.time_ms = find_number(opts, "time", &v) ? v : now_ms();
    if (find_number(opts, "budget_ms", &v)) o.budget_ms = v;
    if (find_number(opts, "acquire_budget_ms", &v)) o.acquire_budget_ms = v;
    if (find_number(opts, "tracked_acquire_budget_ms", &v)) o.tracked_acquire_budget_ms = v;
    if (find_number(opts, "acquire_every", &v)) o.tracked_acquire_every = v < 1 ? 1 : (int)v;
    if (find_number(opts, "enhance", &v)) o.enhance = v != 0;
    if (!isfinite(o.time_ms)) {
        snprintf(err, errlen, "the time must be a finite number of milliseconds");
        return GUST_MODEL_E_INPUT;
    }
    if (find_number(opts, "reset", &v) && v != 0) sq_reset();
    static sq_read_result r;
    if (sq_read(gray, w, h, &o, &r, err, errlen)) return GUST_MODEL_E_RUN;
    format_result(s->result, sizeof s->result, &r);
    return GUST_MODEL_OK;
}

static int squint_run(void *state, const gust_buffer *in, size_t n_in, const char *options_json, gust_result *out,
                      char *err, size_t errlen) {
    squint_state *s = (squint_state *)state;
    int st = process(s, in, n_in, options_json, err, errlen);
    if (st != GUST_MODEL_OK) return st;
    return gust_result_add_json(out, s->result);
}

static int squint_stream_open(void *state, const char *options_json, void **stream, char *err, size_t errlen) {
    squint_state *s = (squint_state *)state;
    (void)options_json;
    (void)err;
    (void)errlen;
    if (s->stream_open) {
        snprintf(err, errlen, "a stream is already open on this reader");
        return GUST_MODEL_E_STATE;
    }
    s->stream_open = 1;
    s->pending = 0;
    s->finished = 0;
    sq_reset(); /* a stream is a camera stream: start from nothing */
    *stream = s;
    return GUST_MODEL_OK;
}

static int squint_stream_push(void *stream, const gust_buffer *in, size_t n_in, char *err, size_t errlen) {
    squint_state *s = (squint_state *)stream;
    if (s->finished) {
        snprintf(err, errlen, "the stream is finished");
        return GUST_MODEL_E_STATE;
    }
    int st = process(s, in, n_in, NULL, err, errlen);
    if (st == GUST_MODEL_OK) s->pending = 1;
    return st;
}

static int squint_stream_pull(void *stream, gust_result *out, int *done, char *err, size_t errlen) {
    squint_state *s = (squint_state *)stream;
    (void)err;
    (void)errlen;
    if (s->pending) {
        s->pending = 0;
        int st = gust_result_add_json(out, s->result);
        if (st != GUST_MODEL_OK) return st;
    }
    *done = s->finished && !s->pending;
    return GUST_MODEL_OK;
}

static int squint_stream_finish(void *stream, char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    ((squint_state *)stream)->finished = 1;
    return GUST_MODEL_OK;
}

static void squint_stream_close(void *stream) { ((squint_state *)stream)->stream_open = 0; }
static size_t squint_memory(void *state) {
    (void)state;
    return sq_memory_bytes();
}

static const gust_model_def DEF = {sizeof(gust_model_def), "squint-1", INFO, squint_open_model, squint_run,
                                   squint_stream_open, squint_stream_push, squint_stream_pull, squint_stream_finish,
                                   squint_stream_close, squint_close_model, squint_memory};

const gust_model_def *gust_model_squint_def(void) { return &DEF; }
