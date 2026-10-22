# Relay host

One C11 codebase for every target: the Relay host (tokenizer, sequence building, decoding, value programs, host
fixes, prefilter) on top of the Gust runtime. No dependencies beyond the C library. The same code runs as a native
library and CLI (Windows, Linux, macOS, Raspberry Pi, Android) and as WebAssembly in browsers and Node; the
JavaScript side (`web/relay-native.mjs`) only moves bytes.

Outputs are byte-identical to the JavaScript reference host (`web/relay.mjs`), with the host fixes on and off and with
the prefilter, for any thread count.

## API (relay.h)

```c
relay_options o; relay_default_options(&o);           // fixes on, prefilter k=10 budget=600, gate 0 / 2, 1 thread
relay *r = relay_open("relay-model", &o, err, sizeof err);     // or relay_open_mem(&files, ...) without a file system
char *calls = relay_call(r, record_json, (size_t)-1);  // [{"tool": ..., "args": {...}}]
relay_free(calls);
char *why = relay_explain(r, record_json, (size_t)-1); // {calls, dropped, chunks}: as relay.explain() in JavaScript
relay_free(why);
relay_close(r);
```

A record is `{text, tools: [{name, description, params: [...]}], kind?, context?: {now}}`. A model directory holds
`relay.gust`, `relay-route.gust`, `relay-anchor.gust`, `relay-args.gust`, `relay-tokenizer.json` and optionally
`relay-prefilter.bin`. All of them are in the Hugging Face repository `sealresearch/relay`.

`relay_explain` (CLI: `--record JSON --explain`, JavaScript: `RelayNative.explain`) adds what the demo shows: per
argument the value program with UTF-16 spans of the request, its confidence and option probability, the calls
`requireArgs` held back, and every tool's route margin. It matches the JavaScript host's `explain()` exactly.

## Diagnostic trace

Use `relay_trace(r, record, len)` or `RelayNative.trace(record)` for an ordered execution trace plus the existing
explanation: `{events: [...], result: {calls, dropped, chunks}}`. The CLI accepts a JSON file so large tool catalogs
do not need shell escaping:

```sh
out/relay-x64.exe relay-model --record-file request.json --prefilter --trace > trace.json
```

`--prefilter` enables retrieval in the CLI, as it does for ordinary calls. The C and JavaScript APIs enable it by
default when the prefilter artifact is available. `--explain` remains the smaller decision-only view.

Events show the selected ISA and configuration, retrieval scores, ranks and retained tools, vector-cache hits,
tool-vector lengths, actual request tokens, schema-field token caps, joint-sequence chunks and tool names, graph
begin/end names, valid rows and status codes (zero means success), and required-argument filtering. `result` supplies
tool margins, instance counts, argument programs, confidence, spans and withheld calls. No tensors are dumped.

Graph events come from Gust's model-independent `gust_run_traced` / `gust_run_valid_inputs_traced` observer. Relay
adds graph labels and serializes the events alongside its own decisions; the runtime knows no Relay tool names or
decoding rules. For individual operators, use CLI `--trace-ops`, C `relay_trace_ops`, or
`relay.trace(record, {operators: true})`. This adds begin/end events with operator indices and opcodes, output tensor
indices, active rows, full shapes and status. The compact default omits those per-operator events.

Tracing runs the same inference with extra formatting, token counting and explanation storage. It is for diagnosis,
not for latency or memory measurements. Ordinary calls allocate no trace buffers. Output can contain argument values
and request spans. A handle must not be called concurrently or re-entered. Generic kernel profiling is described in
`runtime/README.md`.

Real-model regression (expected JSON maps record IDs to exact reference call strings):

```sh
node models/relay/tests/trace.mjs out/relay.wasm relay-model requests.jsonl expected.json out/trace-check
```

## Files

| file | port of |
|---|---|
| unicode.c, unicode_data.h (gen_unicode.py) | Unicode 16 categories, NFC/NFD, case mapping |
| json.c | JSON DOM keeping key order and number source text |
| bpe.c | web/tokenizer.mjs (byte-level BPE, NFC alignment, GPT-2 pre-tokenizer) |
| rx.c | the regular expressions of programs and decoding (Python `re` and JavaScript dialects) |
| val.c | web/py.mjs: values, Python str/repr, JavaScript number formatting, Math.exp (fdlibm) |
| programs.c, tables_data.h (gen_tables.mjs) | web/programs.mjs (value programs) |
| features.c | web/features.mjs |
| net.c | GustNet of web/relay.mjs (graphs, head dot products) |
| decode.c | web/decode.mjs, requireArgs |
| relay.c, relay.h | web/relay.mjs + web/prefilter.mjs, public API |
| relay_cli.c | command line: one record, or evaluation suites |
| relay_wasm.c | WebAssembly exports for web/relay-native.mjs |

## Builds (zig cc)

From the repository root, `ZIG=/path/to/zig ./build.sh all` builds x64, ARM64, WebAssembly and threaded WebAssembly
hosts, including plain SIMD128 for Safari, in `out/`. The x64 executable selects SSE2, AVX2 or VNNI at runtime; the
ARM64 build requires only base NEON. Common flags: `-std=c11 -O2 -ffp-contract=off -DGUST_ATTENTION -DGUST_CONFORMER
-DGUST_PACKED_WEIGHTS -DGUST_FLOAT32 -DGUST_FLOAT32_DECODER -DGUST_SHARED_SCRATCH -Iruntime/include`, sources
`models/relay/host/*.c` (minus relay_wasm.c or relay_cli.c) + `runtime/src/gust.c`. `-ffp-contract=off` is required:
float expressions must round exactly as written on every compiler.

| target | extra flags |
|---|---|
| x64 (AVX-512 VNNI, threads) | `-target x86_64-windows-gnu -mavx2 -mavx512f -mavx512vl -mavx512vnni -DGUST_THREADS -DGUST_THREADS_WIN32` |
| x64 (AVX2) | `-mavx2` |
| ARM64 Linux / Pi (threads) | `-target aarch64-linux-musl -static -DGUST_THREADS -DGUST_THREADS_POSIX` |
| WebAssembly | `-target wasm32-wasi -mexec-model=reactor -msimd128 -mrelaxed-simd -DRELAY_NO_FILES` + relay_wasm.c |
| WebAssembly, threads | the above + `-matomics -mbulk-memory -DGUST_THREADS -Wl,--import-memory -Wl,--export-memory -Wl,--shared-memory -Wl,--max-memory=1073741824 -Wl,--export=__stack_pointer` |

`GUST_SHARED_SCRATCH` keeps one 5 MiB scratch per process instead of one per loaded graph (Relay 1: 45.5 -> 24.5 MB
of graph memory).

The host also shares one activation arena across serialized stateless encoder and head runs. Every live output is
copied before another graph reuses that arena. Admission requires float32 graphs with no state tensors and only
operators 32..41; full artifact validation still runs in `gust_init`. Ineligible graph sets keep independent arenas.
Each graph owns its blob and context, and the host frees the shared arena once, after destroying its graphs.
`relay_model_bytes()` counts shared storage once: Relay 1 falls from 24,543,040 to 21,248,064 bytes.

Run the admission and ownership regression from the repository root:

```sh
ZIG=/path/to/zig bash models/relay/tests/shared_arena.sh relay-model
```

It checks all four model artifacts, private loading, external ownership and refusal of truncated graphs, state and
cache graphs and undersized shared arenas. Assertions remain enabled in optimized test builds.

## Speed (Relay 1, one request of 600 tokens, one thread, Ryzen 7800X3D)

| | encoder |
|---|---|
| first runtime (2026-10-18) | 337 ms |
| exact kernels (256/512-bit tiles, VNNI, vectorized softmax max) | 132 ms |

All kernels are exact (integer sums, unchanged float operation order): every build gives the same bits.
