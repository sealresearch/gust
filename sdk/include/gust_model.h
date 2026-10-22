/* The Gust model API: one small C interface for every model, whatever it takes in and gives back.

   A model is a plugin. It registers a gust_model_def (an id, a description, and the functions that open, run and close it);
   a program, or a binding in another language, only ever calls the functions below. Nothing here knows about any one
   model: JSON for a tool-calling model, PCM for speech, text for a rewriter and frames for a scanner all travel as typed
   buffers.

     gust_model *m;
     gust_model_open("relay-1", files, n_files, "{\"threads\": 4}", &m);
     gust_buffer in = {sizeof in, GUST_BUF_JSON, request_json, strlen(request_json)};
     gust_result *out;
     gust_model_run(m, &in, 1, NULL, &out);
     puts(gust_result_get(out, 0)->data);                      // [{"tool": ..., "args": {...}}]
     gust_result_free(out);
     gust_model_close(m);

   A model that works on a stream (live audio in, text out; text in, audio out) also provides gust_stream_*:
   push what you have, pull what is ready, finish when the input ends.

   Rules:
   - The library never keeps a pointer you gave it after the call returns. Model files are copied while a model opens.
   - Results belong to the library until gust_result_free.
   - Calls are serialised across the whole process: the runtime shares its scratch memory, so two models never run at
     once. A handle must not be used from two threads at the same moment.
   - Every call that can fail returns a gust_model_status, and gust_model_last_error() says why, for the calling thread.
   - The answers do not depend on the thread count, the CPU or the platform (see docs/FORMAT.md and the runtime notes).
   - ABI version 1: handles are opaque, no struct is passed by value, and every struct starts with its own size so that
     fields can be added at the end. A binding checks gust_model_abi_version() when it loads. */
#ifndef GUST_MODEL_H
#define GUST_MODEL_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define GUST_MODEL_API __declspec(dllexport)
#else
#define GUST_MODEL_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define GUST_MODEL_ABI_VERSION 1u

typedef enum {
    GUST_MODEL_OK = 0,
    GUST_MODEL_E_ARGUMENT = 1,      /* a null pointer, a size that does not match, options that are not JSON */
    GUST_MODEL_E_UNKNOWN_MODEL = 2, /* no model with that id is built in */
    GUST_MODEL_E_FILES = 3,         /* a model file is missing or damaged */
    GUST_MODEL_E_OPEN = 4,          /* the model could not be opened */
    GUST_MODEL_E_INPUT = 5,         /* this model does not take that input */
    GUST_MODEL_E_RUN = 6,           /* the model failed while running */
    GUST_MODEL_E_UNSUPPORTED = 7,   /* the model has no such operation (for example no streaming) */
    GUST_MODEL_E_MEMORY = 8,
    GUST_MODEL_E_STATE = 9          /* a call in the wrong order, such as a push after finish */
} gust_model_status;

typedef enum {
    GUST_BUF_BYTES = 0,
    GUST_BUF_TEXT = 1,        /* UTF-8 */
    GUST_BUF_JSON = 2,        /* UTF-8 JSON */
    GUST_BUF_PCM_F32 = 3,     /* interleaved float32 samples, `rate` and `channels` set */
    GUST_BUF_PCM_I16 = 4,     /* interleaved int16 samples, `rate` and `channels` set */
    GUST_BUF_IMAGE_GRAY8 = 5, /* `width` x `height` bytes */
    GUST_BUF_IMAGE_RGBA8 = 6  /* `width` x `height` x 4 bytes */
} gust_buffer_type;

/* A typed view of memory. `len` is in bytes. Buffers you pass in stay yours; buffers in a result stay the library's. Text
   and JSON buffers in a result are also followed by a NUL byte that `len` does not count. */
typedef struct {
    size_t size;           /* sizeof(gust_buffer) */
    gust_buffer_type type;
    const void *data;
    size_t len;
    uint32_t rate, channels;  /* audio */
    uint32_t width, height;   /* images */
} gust_buffer;

/* One model file, in memory. */
typedef struct {
    const char *name;
    const void *data;
    size_t len;
} gust_model_file;

typedef struct gust_model gust_model;
typedef struct gust_stream gust_stream;
typedef struct gust_result gust_result;

GUST_MODEL_API uint32_t gust_model_abi_version(void);
/* The message of the last failure on this thread; empty when there was none. */
GUST_MODEL_API const char *gust_model_last_error(void);

/* The ids of the models in this build, ending with NULL. */
GUST_MODEL_API const char *const *gust_model_ids(void);
/* A JSON description of a model, before it is opened: its id, what it does, its inputs, outputs, options and the files it
   needs. NULL when there is no such model. The string lives as long as the library. */
GUST_MODEL_API const char *gust_model_describe(const char *id);

/* The instruction set this build runs, such as "x86-64 AVX-512 VNNI" or "AArch64 NEON dotprod". */
GUST_MODEL_API const char *gust_model_runtime(void);

/* Threads for the operators, for the whole process (1 to 16; 0 asks for the number of cores, at most 8). Returns the
   count in use. The answers are the same for every count. */
GUST_MODEL_API uint32_t gust_model_set_threads(uint32_t threads);

/* Opens a model from its files. `options_json` may be NULL; what it can hold is in the model's description. */
GUST_MODEL_API gust_model_status gust_model_open(const char *id, const gust_model_file *files, size_t n_files,
                                                  const char *options_json, gust_model **out);
GUST_MODEL_API const char *gust_model_info(const gust_model *);   /* the same JSON as gust_model_describe */
/* Bytes the loaded model holds (its graphs and buffers), or 0 when the model does not say. */
GUST_MODEL_API size_t gust_model_memory(const gust_model *);
/* One input set in, one result out. Which and how many buffers it takes is in the description. */
GUST_MODEL_API gust_model_status gust_model_run(gust_model *, const gust_buffer *in, size_t n_in, const char *options_json,
                                                gust_result **out);
GUST_MODEL_API void gust_model_close(gust_model *);

/* Streaming, for models that have it (GUST_MODEL_E_UNSUPPORTED otherwise). */
GUST_MODEL_API gust_model_status gust_stream_open(gust_model *, const char *options_json, gust_stream **out);
/* Gives the stream more input: audio as it arrives, a chunk of text, a frame. */
GUST_MODEL_API gust_model_status gust_stream_push(gust_stream *, const gust_buffer *in, size_t n_in);
/* Takes what is ready. The result may be empty. After gust_stream_finish, pulling continues until `*done` is 1. */
GUST_MODEL_API gust_model_status gust_stream_pull(gust_stream *, gust_result **out, int *done);
/* No more input. */
GUST_MODEL_API gust_model_status gust_stream_finish(gust_stream *);
GUST_MODEL_API void gust_stream_close(gust_stream *);

GUST_MODEL_API size_t gust_result_count(const gust_result *);
GUST_MODEL_API const gust_buffer *gust_result_get(const gust_result *, size_t index);
GUST_MODEL_API void gust_result_free(gust_result *);

/* ---- For models: what a plugin provides and uses. A program that only runs models does not need anything below. ---- */

/* Each function returns a gust_model_status and writes a message into err (errlen bytes) when it fails. Any of the
   functions after `open` may be NULL when the model does not have it. */
typedef struct {
    size_t size;               /* sizeof(gust_model_def) */
    const char *id;            /* "relay-1" */
    const char *info_json;     /* static; see gust_model_describe */
    int (*open)(const gust_model_file *files, size_t n_files, const char *options_json, void **state, char *err,
                size_t errlen);
    int (*run)(void *state, const gust_buffer *in, size_t n_in, const char *options_json, gust_result *out, char *err,
               size_t errlen);
    int (*stream_open)(void *state, const char *options_json, void **stream, char *err, size_t errlen);
    int (*stream_push)(void *stream, const gust_buffer *in, size_t n_in, char *err, size_t errlen);
    int (*stream_pull)(void *stream, gust_result *out, int *done, char *err, size_t errlen);
    int (*stream_finish)(void *stream, char *err, size_t errlen);
    void (*stream_close)(void *stream);
    void (*close)(void *state);
    size_t (*memory)(void *state);   /* optional: bytes the model holds */
} gust_model_def;

/* Adds a model to this process. The definition must outlive the process (a static is right). A second model with the
   same id replaces the first. The models built into the library are registered for you. */
GUST_MODEL_API gust_model_status gust_model_register(const gust_model_def *def);

/* Appends a copy of a buffer to a result. `in->size` and the type must be set. */
GUST_MODEL_API gust_model_status gust_result_add(gust_result *, const gust_buffer *in);
/* Shorthands: a NUL-terminated JSON or text string, and a block of float32 PCM. */
GUST_MODEL_API gust_model_status gust_result_add_json(gust_result *, const char *json);
GUST_MODEL_API gust_model_status gust_result_add_text(gust_result *, const char *text);
GUST_MODEL_API gust_model_status gust_result_add_pcm_f32(gust_result *, const float *samples, size_t count, uint32_t rate,
                                                          uint32_t channels);

#ifdef __cplusplus
}
#endif
#endif
