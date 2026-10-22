/* Two toy models for testing the model API, with nothing to do with Relay: toy-upper (text in, text out, one call) and
 toy-gain (float32 PCM in and out, streaming). abi_test.c includes this file; a library built with GUST_TOY=1 builds it in so
 that bindings can test streams and audio too. */
#include "gust_model.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Does the options JSON have "key": value (any spaces)? Enough for toys; a real model parses the JSON. */
static int opt_is(const char *o, const char *key, const char *value) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = o ? strstr(o, pat) : NULL;
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == ':') p++;
    return !strncmp(p, value, strlen(value));
}

/* ---- toy model 1: toy-upper, text in, text out, one call ---- */
static int upper_open(const gust_model_file *f, size_t n, const char *opts, void **state, char *err, size_t errlen) {
    (void)f; (void)n;
    char *suffix = (char *)calloc(1, 16);
    if (opt_is(opts, "suffix", "\"!\"")) strcpy(suffix, "!");
    if (opt_is(opts, "fail", "true")) { free(suffix); snprintf(err, errlen, "asked to fail"); return GUST_MODEL_E_OPEN; }
    *state = suffix;
    return GUST_MODEL_OK;
}
static int upper_run(void *state, const gust_buffer *in, size_t n, const char *opts, gust_result *out, char *err, size_t errlen) {
    (void)opts;
    if (n != 1 || in[0].type != GUST_BUF_TEXT) { snprintf(err, errlen, "toy-upper takes one text buffer"); return GUST_MODEL_E_INPUT; }
    char *s = (char *)malloc(in[0].len + 17);
    for (size_t i = 0; i < in[0].len; i++) s[i] = (char)toupper(((const unsigned char *)in[0].data)[i]);
    s[in[0].len] = 0;
    strcat(s, (const char *)state);
    int st = gust_result_add_text(out, s);
    free(s);
    return st;
}
static void upper_close(void *state) { free(state); }
static const gust_model_def UPPER = {sizeof(gust_model_def), "toy-upper", "{\"id\":\"toy-upper\",\"kind\":\"text\"}", upper_open,
                                     upper_run, NULL, NULL, NULL, NULL, NULL, upper_close};

/* ---- toy model 2: toy-gain, float32 PCM in and out, streaming. Pull returns what was pushed since the last pull, times the gain. ---- */
typedef struct { float gain; } gain_model;
typedef struct { float gain; float *queue; size_t n, cap; uint32_t rate; int finished; } gain_stream;
static int gain_open(const gust_model_file *f, size_t n, const char *opts, void **state, char *err, size_t errlen) {
    (void)f; (void)n; (void)err; (void)errlen;
    gain_model *m = (gain_model *)calloc(1, sizeof *m);
    m->gain = 2.0f;
    if (opt_is(opts, "gain", "3")) m->gain = 3.0f;
    *state = m;
    return GUST_MODEL_OK;
}
static int gain_stream_open(void *state, const char *opts, void **stream, char *err, size_t errlen) {
    (void)opts; (void)err; (void)errlen;
    gain_stream *s = (gain_stream *)calloc(1, sizeof *s);
    s->gain = ((gain_model *)state)->gain;
    *stream = s;
    return GUST_MODEL_OK;
}
static int gain_push(void *stream, const gust_buffer *in, size_t n, char *err, size_t errlen) {
    gain_stream *s = (gain_stream *)stream;
    for (size_t b = 0; b < n; b++) {
        if (in[b].type != GUST_BUF_PCM_F32) { snprintf(err, errlen, "toy-gain takes float32 PCM"); return GUST_MODEL_E_INPUT; }
        size_t count = in[b].len / sizeof(float);
        if (s->n + count > s->cap) { s->cap = (s->n + count) * 2; s->queue = (float *)realloc(s->queue, s->cap * sizeof(float)); }
        const float *x = (const float *)in[b].data;
        for (size_t i = 0; i < count; i++) s->queue[s->n + i] = x[i] * s->gain;
        s->n += count;
        s->rate = in[b].rate;
    }
    return GUST_MODEL_OK;
}
static int gain_pull(void *stream, gust_result *out, int *done, char *err, size_t errlen) {
    (void)err; (void)errlen;
    gain_stream *s = (gain_stream *)stream;
    if (s->n) { gust_result_add_pcm_f32(out, s->queue, s->n, s->rate, 1); s->n = 0; }
    *done = s->finished;
    return GUST_MODEL_OK;
}
static int gain_finish(void *stream, char *err, size_t errlen) { (void)err; (void)errlen; ((gain_stream *)stream)->finished = 1; return GUST_MODEL_OK; }
static void gain_stream_close(void *stream) { gain_stream *s = (gain_stream *)stream; free(s->queue); free(s); }
static void gain_close(void *state) { free(state); }
static const gust_model_def GAIN = {sizeof(gust_model_def), "toy-gain", "{\"id\":\"toy-gain\",\"kind\":\"audio\",\"streaming\":true}", gain_open,
                                    NULL, gain_stream_open, gain_push, gain_pull, gain_finish, gain_stream_close, gain_close};


const gust_model_def *gust_model_toy_upper_def(void) { return &UPPER; }
const gust_model_def *gust_model_toy_gain_def(void) { return &GAIN; }
