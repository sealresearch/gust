/* Tests of the Gust model API: the registry, results, errors, streams and lifetimes, with two toy models that have nothing
   to do with Relay (so the API cannot depend on it), then Relay through the API against the Relay host directly.
     abi_test                      the toy models only
     abi_test MODEL_DIR RECORDS    also Relay: MODEL_DIR holds its files, RECORDS is one JSON request per line */
#include "gust_model.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef GUST_MODEL_RELAY
#include "../../models/relay/host/relay.h"
#endif

static int failures;
#define CHECK(cond, ...)                                                                                              \
    do {                                                                                                              \
        if (!(cond)) {                                                                                                \
            failures++;                                                                                               \
            printf("FAIL %s:%d  ", __FILE__, __LINE__);                                                               \
            printf(__VA_ARGS__);                                                                                      \
            printf("\n");                                                                                             \
        }                                                                                                             \
    } while (0)

/* ---- toy model 1: toy-upper, text in, text out, one call ---- */
static int upper_open(const gust_model_file *f, size_t n, const char *opts, void **state, char *err, size_t errlen) {
    (void)f; (void)n;
    char *suffix = (char *)calloc(1, 16);
    if (opts && strstr(opts, "\"suffix\":\"!\"")) strcpy(suffix, "!");
    if (opts && strstr(opts, "\"fail\":true")) { free(suffix); snprintf(err, errlen, "asked to fail"); return GUST_MODEL_E_OPEN; }
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
    if (opts && strstr(opts, "\"gain\":3")) m->gain = 3.0f;
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

static int has_id(const char *id) {
    for (const char *const *p = gust_model_ids(); *p; p++)
        if (!strcmp(*p, id)) return 1;
    return 0;
}

static void test_registry_and_errors(void) {
    CHECK(gust_model_abi_version() == GUST_MODEL_ABI_VERSION, "abi version");
    CHECK(gust_model_register(&UPPER) == GUST_MODEL_OK, "register upper");
    CHECK(gust_model_register(&GAIN) == GUST_MODEL_OK, "register gain");
    CHECK(has_id("toy-upper") && has_id("toy-gain"), "ids list the registered models");
    gust_model_def bad = UPPER;
    bad.id = "";
    CHECK(gust_model_register(&bad) == GUST_MODEL_E_ARGUMENT, "a model without an id is refused");
    CHECK(gust_model_register(NULL) == GUST_MODEL_E_ARGUMENT, "a NULL definition is refused");
    CHECK(strstr(gust_model_describe("toy-upper"), "toy-upper") != NULL, "describe");
    CHECK(gust_model_describe("nope") == NULL && strstr(gust_model_last_error(), "nope"), "describe of an unknown model: %s", gust_model_last_error());

    gust_model *m = NULL;
    CHECK(gust_model_open("nope", NULL, 0, NULL, &m) == GUST_MODEL_E_UNKNOWN_MODEL && !m, "open of an unknown model");
    CHECK(gust_model_open(NULL, NULL, 0, NULL, &m) == GUST_MODEL_E_ARGUMENT, "open without an id");
    CHECK(gust_model_open("toy-upper", NULL, 0, "not json", &m) == GUST_MODEL_E_ARGUMENT, "options that are not JSON");
    CHECK(gust_model_open("toy-upper", NULL, 0, "{\"fail\":true}", &m) == GUST_MODEL_E_OPEN && !m &&
              !strcmp(gust_model_last_error(), "asked to fail"), "a plugin's open error and message: %s", gust_model_last_error());
    gust_model_file nameless = {NULL, "x", 1};
    CHECK(gust_model_open("toy-upper", &nameless, 1, NULL, &m) == GUST_MODEL_E_ARGUMENT, "a file without a name");
}

static void test_upper(void) {
    gust_model *m = NULL;
    CHECK(gust_model_open("toy-upper", NULL, 0, "{\"suffix\":\"!\"}", &m) == GUST_MODEL_OK && m, "open upper");
    CHECK(strstr(gust_model_info(m), "toy-upper"), "info");
    gust_buffer in = {sizeof in, GUST_BUF_TEXT, "hello", 5, 0, 0, 0, 0};
    gust_result *r = NULL;
    CHECK(gust_model_run(m, &in, 1, NULL, &r) == GUST_MODEL_OK && r, "run");
    CHECK(gust_result_count(r) == 1, "one output");
    const gust_buffer *b = gust_result_get(r, 0);
    CHECK(b && b->type == GUST_BUF_TEXT && b->len == 6 && !strcmp((const char *)b->data, "HELLO!"), "the text");
    CHECK(gust_result_get(r, 1) == NULL, "no second output");
    gust_result_free(r);

    gust_buffer wrong = {sizeof wrong, GUST_BUF_JSON, "{}", 2, 0, 0, 0, 0};
    CHECK(gust_model_run(m, &wrong, 1, NULL, &r) == GUST_MODEL_E_INPUT && !r && strstr(gust_model_last_error(), "text buffer"), "wrong input type: %s", gust_model_last_error());
    gust_buffer small = {4, GUST_BUF_TEXT, "x", 1, 0, 0, 0, 0};
    CHECK(gust_model_run(m, &small, 1, NULL, &r) == GUST_MODEL_E_ARGUMENT, "a buffer without its size");
    gust_stream *s = NULL;
    CHECK(gust_stream_open(m, NULL, &s) == GUST_MODEL_E_UNSUPPORTED && !s, "no streaming in toy-upper: %s", gust_model_last_error());
    CHECK(gust_model_run(NULL, &in, 1, NULL, &r) == GUST_MODEL_E_ARGUMENT, "run without a model");
    gust_model_close(m);
    gust_model_close(NULL);
    gust_result_free(NULL);
}

static void test_gain_stream(void) {
    gust_model *m = NULL;
    CHECK(gust_model_open("toy-gain", NULL, 0, "{\"gain\":3}", &m) == GUST_MODEL_OK, "open gain");
    gust_buffer dummy = {sizeof dummy, GUST_BUF_TEXT, "x", 1, 0, 0, 0, 0};
    gust_result *r = NULL;
    CHECK(gust_model_run(m, &dummy, 1, NULL, &r) == GUST_MODEL_E_UNSUPPORTED, "toy-gain has no one-call run");
    gust_stream *s = NULL;
    CHECK(gust_stream_open(m, NULL, &s) == GUST_MODEL_OK && s, "open a stream");

    float a[4] = {1, 2, 3, 4}, b[2] = {-1, 0.5f};
    gust_buffer pa = {sizeof pa, GUST_BUF_PCM_F32, a, sizeof a, 16000, 1, 0, 0};
    gust_buffer pb = {sizeof pb, GUST_BUF_PCM_F32, b, sizeof b, 16000, 1, 0, 0};
    int done = 7;
    CHECK(gust_stream_pull(s, &r, &done) == GUST_MODEL_OK && gust_result_count(r) == 0 && done == 0, "nothing ready yet");
    gust_result_free(r);
    CHECK(gust_stream_push(s, &pa, 1) == GUST_MODEL_OK, "push a");
    CHECK(gust_stream_pull(s, &r, &done) == GUST_MODEL_OK && gust_result_count(r) == 1 && !done, "pull a");
    const gust_buffer *o = gust_result_get(r, 0);
    CHECK(o && o->type == GUST_BUF_PCM_F32 && o->len == sizeof a && o->rate == 16000 && o->channels == 1, "the audio buffer");
    const float *x = o ? (const float *)o->data : NULL;
    CHECK(x && x[0] == 3 && x[3] == 12, "gain applied");
    gust_result_free(r);
    CHECK(gust_stream_push(s, &pb, 1) == GUST_MODEL_OK && gust_stream_finish(s) == GUST_MODEL_OK, "push b and finish");
    CHECK(gust_stream_push(s, &pb, 1) == GUST_MODEL_E_STATE, "a push after finish");
    CHECK(gust_stream_pull(s, &r, &done) == GUST_MODEL_OK && gust_result_count(r) == 1 && done == 1, "the last pull is done");
    x = (const float *)gust_result_get(r, 0)->data;
    CHECK(x[0] == -3 && x[1] == 1.5f, "gain on the last chunk");
    gust_result_free(r);
    CHECK(gust_stream_finish(s) == GUST_MODEL_OK, "finishing twice is fine");

    gust_buffer text = {sizeof text, GUST_BUF_TEXT, "x", 1, 0, 0, 0, 0};
    gust_stream *s2 = NULL;
    CHECK(gust_stream_open(m, NULL, &s2) == GUST_MODEL_OK, "second stream");
    CHECK(gust_stream_push(s2, &text, 1) == GUST_MODEL_E_INPUT, "wrong input in a stream");
    gust_model_close(m);                                   /* closed while two streams are open: they keep working */
    CHECK(gust_stream_push(s2, &pa, 1) == GUST_MODEL_OK && gust_stream_finish(s2) == GUST_MODEL_OK, "a stream outlives its model handle");
    CHECK(gust_stream_pull(s2, &r, &done) == GUST_MODEL_OK && done == 1, "pull from it");
    gust_result_free(r);
    gust_stream_close(s);
    gust_stream_close(s2);                                 /* the last stream frees the model */
    gust_stream_close(NULL);
}

#ifdef GUST_MODEL_RELAY
static unsigned char *slurp(const char *dir, const char *name, size_t *len) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc(*len + 1);
    if (fread(p, 1, *len, f) != *len) { fclose(f); free(p); return NULL; }
    p[*len] = 0;
    fclose(f);
    return p;
}

static void test_relay(const char *dir, const char *records) {
    static const char *names[6] = {"relay.gust", "relay-route.gust", "relay-anchor.gust", "relay-args.gust", "relay-tokenizer.json", "relay-prefilter.bin"};
    gust_model_file files[6];
    for (int i = 0; i < 6; i++) {
        size_t len = 0;
        files[i].name = names[i];
        files[i].data = slurp(dir, names[i], &len);
        files[i].len = len;
        CHECK(files[i].data != NULL, "read %s", names[i]);
    }
    CHECK(has_id("relay-1"), "relay-1 is built in");
    CHECK(strstr(gust_model_describe("relay-1"), "tool-calling") != NULL, "describe relay");

    gust_model *missing = NULL;
    CHECK(gust_model_open("relay-1", files, 3, NULL, &missing) == GUST_MODEL_E_FILES && !missing, "relay without all its files: %s", gust_model_last_error());
    gust_model *m1 = NULL, *m4 = NULL;
    CHECK(gust_model_open("relay-1", files, 6, NULL, &m1) == GUST_MODEL_OK, "open relay: %s", gust_model_last_error());
    CHECK(gust_model_open("relay-1", files, 6, "{\"threads\":4}", &m4) == GUST_MODEL_OK, "open relay on 4 threads: %s", gust_model_last_error());

    relay_options o;
    relay_default_options(&o);
    relay_files rf = {files[0].data, files[0].len, files[1].data, files[1].len, files[2].data, files[2].len, files[3].data, files[3].len,
                      (const char *)files[4].data, files[4].len, files[5].data, files[5].len};
    char err[256];
    relay *direct = relay_open_mem(&rf, &o, err, sizeof err);
    CHECK(direct != NULL, "open the host directly: %s", err);

    FILE *f = fopen(records, "rb");
    CHECK(f != NULL, "open %s", records);
    char *line = NULL;
    size_t cap = 0, rows = 0, same = 0, same4 = 0;
    if (f && direct && m1 && m4) {
        int c;
        size_t n = 0;
        while ((c = fgetc(f)) != EOF || n) {
            if (c == '\n' || c == EOF) {
                if (n) {
                    line[n] = 0;
                    char *want = relay_call(direct, line, n);
                    gust_buffer in = {sizeof in, GUST_BUF_JSON, line, n, 0, 0, 0, 0};
                    gust_result *r1 = NULL, *r4 = NULL;
                    int s1 = gust_model_run(m1, &in, 1, NULL, &r1), s4 = gust_model_run(m4, &in, 1, NULL, &r4);
                    rows++;
                    if (s1 == GUST_MODEL_OK && want && !strcmp((const char *)gust_result_get(r1, 0)->data, want)) same++;
                    if (s4 == GUST_MODEL_OK && want && !strcmp((const char *)gust_result_get(r4, 0)->data, want)) same4++;
                    gust_result_free(r1);
                    gust_result_free(r4);
                    relay_free(want);
                    n = 0;
                }
                if (c == EOF) break;
            } else {
                if (n + 2 > cap) { cap = (n + 2) * 2; line = (char *)realloc(line, cap); }
                line[n++] = (char)c;
            }
        }
        fclose(f);
    }
    CHECK(rows > 0 && same == rows, "relay through the API equals the host on %zu of %zu requests", same, rows);
    CHECK(rows > 0 && same4 == rows, "and on 4 threads, %zu of %zu", same4, rows);
    printf("relay: %zu requests through the model API, %zu identical to the host, %zu identical on 4 threads\n", rows, same, same4);

    gust_buffer bad = {sizeof bad, GUST_BUF_JSON, "{not json", 9, 0, 0, 0, 0};
    gust_result *r = NULL;
    CHECK(gust_model_run(m1, &bad, 1, NULL, &r) == GUST_MODEL_E_INPUT && !r, "an invalid request: %s", gust_model_last_error());
    gust_buffer audio = {sizeof audio, GUST_BUF_PCM_F32, "xxxx", 4, 16000, 1, 0, 0};
    CHECK(gust_model_run(m1, &audio, 1, NULL, &r) == GUST_MODEL_E_INPUT, "audio is not a Relay input");
    const char *req = "{\"text\":\"Show me the Eiffel Tower on the map\",\"tools\":[{\"name\":\"show_map\",\"description\":\"Shows a place.\",\"params\":[{\"name\":\"query\",\"type\":\"string\",\"description\":\"The place.\",\"required\":true}]}]}";
    gust_buffer rq = {sizeof rq, GUST_BUF_JSON, req, strlen(req), 0, 0, 0, 0};
    CHECK(gust_model_run(m1, &rq, 1, "{\"mode\":\"explain\"}", &r) == GUST_MODEL_OK && strstr((const char *)gust_result_get(r, 0)->data, "\"why\""), "mode explain");
    gust_result_free(r);
    gust_stream *s = NULL;
    CHECK(gust_stream_open(m1, NULL, &s) == GUST_MODEL_E_UNSUPPORTED, "relay has no streaming");

    gust_model_close(m1);
    gust_model_close(m4);
    relay_close(direct);
    free(line);
    for (int i = 0; i < 6; i++) free((void *)files[i].data);
}
#endif

int main(int argc, char **argv) {
    test_registry_and_errors();
    test_upper();
    test_gain_stream();
#ifdef GUST_MODEL_RELAY
    if (argc == 3) test_relay(argv[1], argv[2]);
    else printf("relay: skipped (pass MODEL_DIR and RECORDS)\n");
#else
    (void)argc; (void)argv;
#endif
    printf(failures ? "%d FAILURES\n" : "all model API tests passed\n", failures);
    return failures ? 1 : 0;
}
