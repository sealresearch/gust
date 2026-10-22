# The Gust model API

One small C interface for every model. A program, or a binding in another language, opens a model by id, gives it typed
buffers and gets typed buffers back. It never needs to know which model it is talking to: JSON for a tool-calling model, PCM
audio for speech, text for a rewriter and frames for a scanner all travel the same way.

```c
#include "gust_model.h"

gust_model *m;
gust_model_open("relay-1", files, n_files, "{\"threads\": 4}", &m);    // files: name + bytes, from wherever you keep them

gust_buffer in = {sizeof in, GUST_BUF_JSON, request, strlen(request)};
gust_result *out;
gust_model_run(m, &in, 1, NULL, &out);
puts(gust_result_get(out, 0)->data);                                    // [{"tool": "show_map", "args": {...}}]

gust_result_free(out);
gust_model_close(m);
```

The whole interface is in [`include/gust_model.h`](include/gust_model.h), about two dozen functions:

| | |
|---|---|
| `gust_model_ids`, `gust_model_describe` | which models are built in, and what each one takes and gives (JSON) |
| `gust_model_open`, `gust_model_run`, `gust_model_close` | open from files, one call, close |
| `gust_stream_open`, `_push`, `_pull`, `_finish`, `_close` | live input (speech in) or long output (speech out): push what you have, pull what is ready |
| `gust_result_count`, `_get`, `_free` | read the outputs, then free them |
| `gust_model_set_threads`, `gust_model_runtime`, `gust_model_memory` | threads for the operators (for the whole process), the instruction set in use, bytes a model holds |
| `gust_model_last_error` | why the last call on this thread failed |

Rules that make it safe to bind: handles are opaque; the library never keeps a pointer you gave it after a call returns (model
files are copied while a model opens); results stay the library's until `gust_result_free`; calls are serialised across the
process because the runtime shares its scratch memory; every call returns a status and an error message; and the answers do
not depend on the thread count, the CPU or the platform.

## Bindings

| Language | | |
|---|---|---|
| Python | [`packages/python`](../packages/python) | `pip install sealresearch`: generic over every model, wheels for five platforms |
| JavaScript | [`packages/relay`](../packages/relay) | `npm install @sealresearch/relay`: Relay in Node and in the browser, on the WebAssembly build |
| Rust | [`cli`](../cli) | the `sealpup` command line loads `libgust` and runs models through it (`cli/src/engine.rs`, about 200 lines) |
| C, C++ | this folder | link `libgust` and include `gust_model.h` |
| Rust, Go and the rest | | bind the functions above; the Python package is a few hundred lines |

## The library

```sh
ZIG=zig bash build.sh lib         # out/lib/<platform>/gust.dll, libgust.so or libgust.dylib, for five platforms
ZIG=zig bash build.sh test-sdk-x64 && out/sdk-test-x64.exe [MODEL_DIR RECORDS]
```

The library holds the runtime, the model API and the models built into it. Linux builds target glibc 2.17, so one file runs on
every distribution a Python wheel targets.

## Adding a model

A model is a `gust_model_def`: an id, a JSON description, and the functions that open, run and close it. Streaming functions
are optional. [`models/relay/sdk/relay_model.c`](../models/relay/sdk/relay_model.c) is the Relay plugin (about 120 lines around the
Relay host); [`tests/abi_test.c`](tests/abi_test.c) has two toy models that have nothing to do with Relay, one text to text and one
streaming audio, which is how the tests show the API does not depend on any model.

```c
static const gust_model_def DEF = {
    sizeof(gust_model_def), "my-model", "{\"id\":\"my-model\",\"kind\":\"...\",\"inputs\":[...],\"outputs\":[...],"
    "\"files\":{\"required\":[\"my.gust\"]}}",
    open, run, /* stream_open ... */ NULL, NULL, NULL, NULL, NULL, close};
const gust_model_def *gust_model_my_def(void) { return &DEF; }
```

Then register it in `sdk/src/model.c` behind a `-DGUST_MODEL_MY` flag, add its sources to `sharedlib` in `build.sh`, and put its
files and checksums in `cli/registry.json`. Bindings need no change: they read the description.

What a plugin must do: copy what it keeps from the files during `open`; return `GUST_MODEL_E_INPUT` with a clear message for an
input it does not take; and give the same output bits whatever the thread count.

## Versioning

This is ABI version 1. Structs start with their own size so that fields can be added at the end; functions are only added, never
changed. A binding checks `gust_model_abi_version()` when it loads the library.
