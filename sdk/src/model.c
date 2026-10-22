/* The Gust model API (sdk/include/gust_model.h): the registry of models, the handles, the results and the errors.
   It knows nothing about any one model; a model is a gust_model_def that registers itself. */
#include "gust_model.h"
#include "gust.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static SRWLOCK g_lock = SRWLOCK_INIT;
static void lock(void) { AcquireSRWLockExclusive(&g_lock); }
static void unlock(void) { ReleaseSRWLockExclusive(&g_lock); }
static uint32_t cores(void) { SYSTEM_INFO s; GetSystemInfo(&s); return s.dwNumberOfProcessors; }
#elif !defined(__wasm__)
#include <pthread.h>
#include <unistd.h>
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void lock(void) { pthread_mutex_lock(&g_lock); }
static void unlock(void) { pthread_mutex_unlock(&g_lock); }
static uint32_t cores(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (uint32_t)n : 1; }
#else
static void lock(void) {}
static void unlock(void) {}
static uint32_t cores(void) { return 1; }
#endif

#define MAX_MODELS 32
#define ERR_BYTES 512

struct gust_result {
    gust_buffer *items;
    size_t count, cap;
};
struct gust_model {
    const gust_model_def *def;
    void *state;
    size_t streams; /* open streams */
    int closed;     /* gust_model_close was called while streams were still open */
};
struct gust_stream {
    gust_model *model;
    void *state;
    int finished;
};

static _Thread_local char g_error[ERR_BYTES];
static const gust_model_def *g_models[MAX_MODELS];
static size_t g_count;
static const char *g_ids[MAX_MODELS + 1];
static int g_builtin_done;

/* The models built into this library (compile with -DGUST_MODEL_RELAY and so on). Each returns its definition. */
#ifdef GUST_MODEL_RELAY
const gust_model_def *gust_model_relay_def(void);
#endif
#ifdef GUST_MODEL_TOY /* test models for bindings: build with GUST_TOY=1 */
const gust_model_def *gust_model_toy_upper_def(void);
const gust_model_def *gust_model_toy_gain_def(void);
#endif

static gust_model_status fail(gust_model_status s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);
    return s;
}
/* A plugin's status, with its message (or a default one) kept for gust_model_last_error. */
static gust_model_status plugin(int status, const char *err, const char *what) {
    if (status == GUST_MODEL_OK) return GUST_MODEL_OK;
    if (err[0]) snprintf(g_error, sizeof g_error, "%s", err);
    else snprintf(g_error, sizeof g_error, "%s failed (%d)", what, status);
    return (gust_model_status)status;
}

static gust_model_status register_locked(const gust_model_def *def) {
    if (!def || def->size < sizeof(gust_model_def) || !def->id || !def->id[0] || !def->info_json || !def->open || !def->close)
        return fail(GUST_MODEL_E_ARGUMENT, "a model definition needs its size, an id, a description, open and close");
    for (size_t i = 0; i < g_count; i++)
        if (!strcmp(g_models[i]->id, def->id)) { g_models[i] = def; return GUST_MODEL_OK; }
    if (g_count == MAX_MODELS) return fail(GUST_MODEL_E_MEMORY, "too many models are registered");
    g_models[g_count] = def;
    g_ids[g_count] = def->id;
    g_count++;
    g_ids[g_count] = NULL;
    return GUST_MODEL_OK;
}
static void builtin_locked(void) {
    if (g_builtin_done) return;
    g_builtin_done = 1;
#ifdef GUST_MODEL_RELAY
    register_locked(gust_model_relay_def());
#endif
#ifdef GUST_MODEL_TOY
    register_locked(gust_model_toy_upper_def());
    register_locked(gust_model_toy_gain_def());
#endif
}
static const gust_model_def *find_locked(const char *id) {
    builtin_locked();
    for (size_t i = 0; i < g_count; i++)
        if (!strcmp(g_models[i]->id, id)) return g_models[i];
    return NULL;
}

uint32_t gust_model_abi_version(void) { return GUST_MODEL_ABI_VERSION; }
const char *gust_model_last_error(void) { return g_error; }

gust_model_status gust_model_register(const gust_model_def *def) {
    lock();
    builtin_locked();
    gust_model_status s = register_locked(def);
    unlock();
    return s;
}
const char *const *gust_model_ids(void) {
    lock();
    builtin_locked();
    unlock();
    return g_ids;
}
const char *gust_model_describe(const char *id) {
    if (!id) { fail(GUST_MODEL_E_ARGUMENT, "the model id is NULL"); return NULL; }
    lock();
    const gust_model_def *d = find_locked(id);
    unlock();
    if (!d) fail(GUST_MODEL_E_UNKNOWN_MODEL, "no model called \"%s\" is built in", id);
    return d ? d->info_json : NULL;
}

uint32_t gust_model_set_threads(uint32_t threads) {
    if (threads == 0) { threads = cores(); if (threads > 8) threads = 8; }
    if (threads > 16) threads = 16;
    if (threads < 1) threads = 1;
#if defined(GUST_THREADS_POSIX) || defined(GUST_THREADS_WIN32)
    lock();
    threads = gust_threads_start(threads);
    unlock();
    return threads;
#else
    return 1;
#endif
}

/* ---- results ---- */
static gust_result *result_new(void) { return (gust_result *)calloc(1, sizeof(gust_result)); }

gust_model_status gust_result_add(gust_result *r, const gust_buffer *in) {
    if (!r || !in || in->size < sizeof(gust_buffer) || (!in->data && in->len))
        return fail(GUST_MODEL_E_ARGUMENT, "a result buffer needs its size and its data");
    if (r->count == r->cap) {
        size_t cap = r->cap ? r->cap * 2 : 4;
        gust_buffer *grown = (gust_buffer *)realloc(r->items, cap * sizeof(gust_buffer));
        if (!grown) return fail(GUST_MODEL_E_MEMORY, "out of memory");
        r->items = grown;
        r->cap = cap;
    }
    void *copy = malloc(in->len + 1);
    if (!copy) return fail(GUST_MODEL_E_MEMORY, "out of memory");
    if (in->len) memcpy(copy, in->data, in->len);
    ((unsigned char *)copy)[in->len] = 0;
    gust_buffer *b = &r->items[r->count++];
    *b = *in;
    b->size = sizeof(gust_buffer);
    b->data = copy;
    return GUST_MODEL_OK;
}
gust_model_status gust_result_add_json(gust_result *r, const char *json) {
    if (!json) return fail(GUST_MODEL_E_ARGUMENT, "the JSON is NULL");
    gust_buffer b = {sizeof b, GUST_BUF_JSON, json, strlen(json), 0, 0, 0, 0};
    return gust_result_add(r, &b);
}
gust_model_status gust_result_add_text(gust_result *r, const char *text) {
    if (!text) return fail(GUST_MODEL_E_ARGUMENT, "the text is NULL");
    gust_buffer b = {sizeof b, GUST_BUF_TEXT, text, strlen(text), 0, 0, 0, 0};
    return gust_result_add(r, &b);
}
gust_model_status gust_result_add_pcm_f32(gust_result *r, const float *samples, size_t count, uint32_t rate, uint32_t channels) {
    gust_buffer b = {sizeof b, GUST_BUF_PCM_F32, samples, count * sizeof(float), rate, channels, 0, 0};
    return gust_result_add(r, &b);
}
size_t gust_result_count(const gust_result *r) { return r ? r->count : 0; }
const gust_buffer *gust_result_get(const gust_result *r, size_t i) { return r && i < r->count ? &r->items[i] : NULL; }
void gust_result_free(gust_result *r) {
    if (!r) return;
    for (size_t i = 0; i < r->count; i++) free((void *)r->items[i].data);
    free(r->items);
    free(r);
}

/* ---- models ---- */
static int options_ok(const char *o) { return !o || o[0] == 0 || strchr(o, '{'); }

gust_model_status gust_model_open(const char *id, const gust_model_file *files, size_t n_files, const char *options_json,
                                  gust_model **out) {
    if (!out) return fail(GUST_MODEL_E_ARGUMENT, "out is NULL");
    *out = NULL;
    if (!id || (n_files && !files)) return fail(GUST_MODEL_E_ARGUMENT, "the model id and the file list are needed");
    if (!options_ok(options_json)) return fail(GUST_MODEL_E_ARGUMENT, "the options must be a JSON object");
    for (size_t i = 0; i < n_files; i++)
        if (!files[i].name || (!files[i].data && files[i].len))
            return fail(GUST_MODEL_E_ARGUMENT, "file %zu has no name or no data", i);
    lock();
    const gust_model_def *d = find_locked(id);
    if (!d) {
        unlock();
        return fail(GUST_MODEL_E_UNKNOWN_MODEL, "no model called \"%s\" is built in", id);
    }
    gust_model *m = (gust_model *)calloc(1, sizeof *m);
    if (!m) {
        unlock();
        return fail(GUST_MODEL_E_MEMORY, "out of memory");
    }
    char err[ERR_BYTES] = "";
    int s = d->open(files, n_files, options_json, &m->state, err, sizeof err);
    unlock();
    if (s != GUST_MODEL_OK) {
        free(m);
        return plugin(s, err, "opening the model");
    }
    m->def = d;
    *out = m;
    return GUST_MODEL_OK;
}
const char *gust_model_info(const gust_model *m) { return m ? m->def->info_json : NULL; }

gust_model_status gust_model_run(gust_model *m, const gust_buffer *in, size_t n_in, const char *options_json,
                                 gust_result **out) {
    if (!out) return fail(GUST_MODEL_E_ARGUMENT, "out is NULL");
    *out = NULL;
    if (!m || (n_in && !in)) return fail(GUST_MODEL_E_ARGUMENT, "the model and its input are needed");
    if (!options_ok(options_json)) return fail(GUST_MODEL_E_ARGUMENT, "the options must be a JSON object");
    for (size_t i = 0; i < n_in; i++)
        if (in[i].size < sizeof(gust_buffer) || (!in[i].data && in[i].len))
            return fail(GUST_MODEL_E_ARGUMENT, "input %zu has no size or no data", i);
    if (!m->def->run) return fail(GUST_MODEL_E_UNSUPPORTED, "%s cannot run in one call", m->def->id);
    gust_result *r = result_new();
    if (!r) return fail(GUST_MODEL_E_MEMORY, "out of memory");
    char err[ERR_BYTES] = "";
    lock();
    int s = m->def->run(m->state, in, n_in, options_json, r, err, sizeof err);
    unlock();
    if (s != GUST_MODEL_OK) {
        gust_result_free(r);
        return plugin(s, err, "running the model");
    }
    *out = r;
    return GUST_MODEL_OK;
}

static void model_free(gust_model *m) {
    lock();
    m->def->close(m->state);
    unlock();
    free(m);
}
void gust_model_close(gust_model *m) {
    if (!m) return;
    if (m->streams) { m->closed = 1; return; } /* freed when the last stream closes */
    model_free(m);
}

/* ---- streams ---- */
gust_model_status gust_stream_open(gust_model *m, const char *options_json, gust_stream **out) {
    if (!out) return fail(GUST_MODEL_E_ARGUMENT, "out is NULL");
    *out = NULL;
    if (!m || m->closed) return fail(GUST_MODEL_E_ARGUMENT, "the model is needed and must be open");
    if (!options_ok(options_json)) return fail(GUST_MODEL_E_ARGUMENT, "the options must be a JSON object");
    const gust_model_def *d = m->def;
    if (!d->stream_open || !d->stream_push || !d->stream_pull || !d->stream_finish || !d->stream_close)
        return fail(GUST_MODEL_E_UNSUPPORTED, "%s has no streaming", d->id);
    gust_stream *s = (gust_stream *)calloc(1, sizeof *s);
    if (!s) return fail(GUST_MODEL_E_MEMORY, "out of memory");
    char err[ERR_BYTES] = "";
    lock();
    int st = d->stream_open(m->state, options_json, &s->state, err, sizeof err);
    unlock();
    if (st != GUST_MODEL_OK) {
        free(s);
        return plugin(st, err, "opening the stream");
    }
    s->model = m;
    m->streams++;
    *out = s;
    return GUST_MODEL_OK;
}
gust_model_status gust_stream_push(gust_stream *s, const gust_buffer *in, size_t n_in) {
    if (!s || (n_in && !in)) return fail(GUST_MODEL_E_ARGUMENT, "the stream and its input are needed");
    if (s->finished) return fail(GUST_MODEL_E_STATE, "the stream is finished: it takes no more input");
    for (size_t i = 0; i < n_in; i++)
        if (in[i].size < sizeof(gust_buffer) || (!in[i].data && in[i].len))
            return fail(GUST_MODEL_E_ARGUMENT, "input %zu has no size or no data", i);
    char err[ERR_BYTES] = "";
    lock();
    int st = s->model->def->stream_push(s->state, in, n_in, err, sizeof err);
    unlock();
    return plugin(st, err, "pushing to the stream");
}
gust_model_status gust_stream_pull(gust_stream *s, gust_result **out, int *done) {
    if (!out || !done) return fail(GUST_MODEL_E_ARGUMENT, "out and done are needed");
    *out = NULL;
    *done = 0;
    if (!s) return fail(GUST_MODEL_E_ARGUMENT, "the stream is NULL");
    gust_result *r = result_new();
    if (!r) return fail(GUST_MODEL_E_MEMORY, "out of memory");
    char err[ERR_BYTES] = "";
    lock();
    int st = s->model->def->stream_pull(s->state, r, done, err, sizeof err);
    unlock();
    if (st != GUST_MODEL_OK) {
        gust_result_free(r);
        *done = 0;
        return plugin(st, err, "pulling from the stream");
    }
    *out = r;
    return GUST_MODEL_OK;
}
gust_model_status gust_stream_finish(gust_stream *s) {
    if (!s) return fail(GUST_MODEL_E_ARGUMENT, "the stream is NULL");
    if (s->finished) return GUST_MODEL_OK;
    char err[ERR_BYTES] = "";
    lock();
    int st = s->model->def->stream_finish(s->state, err, sizeof err);
    unlock();
    if (st == GUST_MODEL_OK) s->finished = 1;
    return plugin(st, err, "finishing the stream");
}
void gust_stream_close(gust_stream *s) {
    if (!s) return;
    gust_model *m = s->model;
    lock();
    m->def->stream_close(s->state);
    unlock();
    free(s);
    if (--m->streams == 0 && m->closed) model_free(m);
}
