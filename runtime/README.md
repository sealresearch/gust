# Gust runtime

Dependency-free C11 inference runtime: one translation unit (`src/gust.c`), no allocation on the hot path, and the
same output bits on every target (x86, ARM, WebAssembly) and for any thread count.

```
include/gust.h            public API (gust_init, gust_run_valid, gust_tensor, gust_threads_start, gust_variant, ...)
src/gust.c                the translation unit (includes everything below)
src/rename.h              per-variant entry point names (x86-64 multi-variant builds)
src/dispatch_x86.c        x86-64: picks the SSE2 / AVX2 / AVX-512 VNNI (+BW) build of the runtime from CPUID at first call
src/int8/                 INT8 graphs: loader (format.c, also marks non-negative tensors and small-weight convolutions),
                          executor, operators 1-16 (ops/)
                          kernels.c     scalar dot / clamp / requantization: the definitions every vector kernel matches
                          tiles.c       four-channel dots, WASM SIMD / AVX2 register tiles, AVX-512 VNNI tiles, vector
                                        requantization, clamped copies, SSE2-only 4x4 pointwise tiles
                          ops/gemm.c    conv_pixel, packed-weight GEMM tiles (VNNI, AVX2 u8 x s8 and int16) and the
                                        convolution driver conv_rows
                          ops/depthwise.c  row-interleaved and grouped depthwise kernels (VNNI, AVX2, WASM SIMD)
                          ops/packed.c  3/4-bit code decoding; ops/conv.c runs packed convolutions through the same
                                        kernels (codes decoded per GEMM channel block or per channel block / filter)
src/float32/              float32 graphs: loader, executor, operators 32-37 and 46, threads, decoder state (DECODER.md),
                          T5 operator modes (T5.md), temporal convolution operators 47-49 (TEMPORAL.md)
src/float32/kernels/      common.c      vector types, profiling hooks, weight decoding
                          conv32.c      float32 / float64-accurate convolution
                          attention.c   reference attention loops
                          fast.c        wide float32 panels, attention tiles, softmax, LayerNorm
                          conv16.c      int16-activation convolution (F_CONV p0 = 4)
                          conv16_neon.c short-row NEON tiles retaining packed 4-bit weights
                          conv8.c       int8-activation convolution  (F_CONV p0 = 12)
tools/gust_bench.c        time one graph at several lengths, print an output hash
tests/conv16.c            packed-weight / row-remainder kernel regression
tests/arena.c             compare every arena byte with eager execution at varied lengths
tests/input_rows.c        per-input row counts against dense head execution and refusal guards
tests/workshare.c         changing task counts and generation-wrap regression
tests/bitexact.sh         every build (x86 variants, 1 / 8 threads, WebAssembly, ARM64 on a device) against references
tests/relay_wasm.mjs      Relay through a WebAssembly build (used by bitexact.sh)
tests/t5.c                float32-t5-v1 operator modes against their contracts, and refusals
tests/temporal.c          float32-temporal-v1 operators and modes against their contracts, and refusals
```

Build with `../build.sh [x64 | linux-x64 | arm64 | wasm | wasm-mt | wasm-simd | wasm-mt-simd | all]` (zig cc). Required flags: `-std=c11
-ffp-contract=off` (float expressions must round exactly as written). Kernels choose instructions per target (AVX2,
AVX-512 VNNI, NEON, NEON dotprod, WASM SIMD / relaxed SIMD). Integer sums are exact and float operation order is fixed,
so every build agrees bit for bit; `tests/bitexact.sh` checks it.
`wasm` / `wasm-mt` enable relaxed SIMD; `wasm-simd` / `wasm-mt-simd` use plain SIMD128 for browsers such as
Safari. The `-mt` targets use workers and shared memory. `all` builds both WASM variants alongside native targets.
`runtime-wasm` (relaxed SIMD) and `runtime-wasm-simd` (plain SIMD128) build the runtime alone for JavaScript hosts:
`out/gust.wasm` and `out/gust-simd.wasm`, fixed 128 MiB memory, the public entry points; used by `@gust/core` and the
model packages.

Threads: `gust_threads_start(n)` (POSIX, Win32) or, in WebAssembly, the host's workers parked in `gust_worker`
(models/relay/web/relay-native.mjs). Results do not depend on the thread count.
Default Relay builds use `GUST_SHARED_SCRATCH`: the 4 MiB activation area and sixteen 64 KiB slabs are shared
process-wide, so runs of different graph contexts must also be serialized. Query `gust_context_bytes()` for the
configured Context size; graph allocations and process/guest memory are reported separately by the tools below.

The normal Relay host additionally shares one activation arena between its serialized, stateless encoder and
heads. It copies live outputs before running the next graph; stateful or unsupported graphs retain
private arenas. This is host scheduling in `models/relay/host/`, with no Relay-specific runtime dispatch or new
runtime mode. Original Relay 1 graph storage falls from 24,543,040 to 21,248,064 bytes. See the host README for
the ownership regression and [Relay baseline](../models/relay/BASELINE.md) for measured process memory and parity evidence.

Whole Relay calls: `../build.sh bench-calls-x64` and `../build.sh bench-calls-arm64` build
`out/relay-calls-x64.exe` and `out/relay-calls-arm64`. Arguments are `MODEL_DIR RECORDS.jsonl REPEATS THREADS PREFILTER`.
They measure tokenization, prefilter, inference and decoding together, retain exact call strings, and report
peak process memory. `models/relay/tools/bench_calls.mjs` measures the same calls in WASM. See
[Relay's baseline and reproduction commands](../models/relay/BASELINE.md) before comparing changes.

INT8 kernels: inputs proven non-negative at load (their producer clamps at zero) take u8 x s8 forms (vpdpbusd,
vpmaddubsw, relaxed i8 x i7 dots); signed inputs go through int16 pairs or the x ^ 0x80 correction, and every sum is an
exact int32. Requantization in vector form uses float64 steps only where each value is exact (comments in tiles.c), so
all builds return the scalar bits. Packed 3/4-bit convolutions run the byte kernels on decoded channel blocks (bounded
stack scratch, nothing expanded wholesale), as fast as byte weights at half the graph memory. A 2048-frame Conformer
window (617 operators, one thread, Ryzen 7 7800X3D): AVX-512 VNNI 80 ms, AVX2 147 ms, WebAssembly SIMD 565 ms,
SSE2 675 ms.

Engine measurements: `../build.sh bench-x64` and `../build.sh bench-arm64` produce `out/gust-bench-x64.exe` and
`out/gust-bench-arm64`. Run `gust-bench MODEL.gust 35,350,600 11 8`: lengths, repetitions (1..64), threads.
It warms up each length, reports median and best-of-N time, output and complete-arena hashes, and process peak
RSS. The arena starts with a fixed sentinel, including allocation padding, so the arena hash also checks bytes
not covered by public outputs. Windows and POSIX both honor the requested thread count. Compare the same model,
ISA and harness before/after; profiler builds have overhead. `GUST_F32_PROFILE` adds per-operator counters, with
slot 4 measuring tail clearing separately; native hosts provide `double gust_now(void)` in milliseconds.
For end-to-end Relay decisions, the opt-in [host trace](../models/relay/host/README.md#diagnostic-trace)
extends Gust's execution observer with retrieval, token limits and decoding. Model-specific events stay in
`models/relay/`.
`../build.sh bench-wasm` builds the same single-thread engine benchmark for WASI. Run
`node runtime/tools/bench_wasm.mjs out/gust-bench.wasm MODEL.gust 35,350,600 11`; it additionally reports the Node
process's peak RSS and guest linear-memory size. This isolates the encoder from host tokenization/decoding.

## Execution tracing

`gust_run_traced(ctx, valid_rows, observer, opaque)` mirrors `gust_run_valid`; zero rows means the full graph.
`gust_run_valid_inputs_traced(ctx, rows, count, observer, opaque)` mirrors the per-input row API. Both support
the same graph formats and validation as their ordinary counterparts. A NULL observer runs normally.

The callback receives borrowed `gust_trace_event` values in execution order: run begin, operator begin/end
pairs, then run end with the returned status. Preflight failures produce only run begin/end. Operator
events include the artifact index/opcode, output tensor index, full height/width/channels and active rows.
An operator failure reports its error on operator end and run end. The observer chooses its clock, formatting
and storage; the core performs no logging, tensor copying or allocation and retains no callback state.

```c
static void observe(void *log, const gust_trace_event *event) {
    /* Inspect event->kind, operator_index, opcode, output_tensor, rows and status.
       Forward the useful fields to the application's logger. */
}
int status = gust_run_traced(ctx, 0, observe, log);
```

Observers must not mutate or re-enter the context, arena or input row counts. At operator end, only active
output rows are meaningful: intermediate storage can alias and padding can be cleared later. Copy values
inside the callback if they need to survive later operators. Shared-scratch builds retain their existing
serialization requirement. Context and graph memory sizes are unchanged; diagnostic callbacks add execution
overhead and whatever storage the application requests. Normal calls allocate no trace buffers.

Generic contract tests cover INT8 and float32, arena parity, event boundaries and errors:

```sh
ZIG=/path/to/zig bash runtime/tests/trace.sh x64  # arm64 or wasm for those targets
out/gust-trace-test-x64.exe /path/to/model.gust
node runtime/tests/trace_wasm.mjs out/gust-trace-test.wasm /path/to/model.gust
```

## Kernel implementation notes

The exact int16 convolution uses half-height tiles for row remainders. AArch64 loads each activation pair once
and uses indexed widening MACs, with the same integer sum and scratch layout. Ordinary float32 graphs skip tail
clears that are wholly overwritten later; checked execution and dynamic cache graphs retain eager clearing.
Large remaining clears share the existing worker pool. No graph, Context or scratch allocation grows.

For AArch64 int16 convolutions with 4-bit weights, even K and at most 64 output rows, weights stay packed in
the existing thread slab and widen inside the tile. Longer rows and other targets retain the expanded slab.
This automatic kernel choice changes neither activation precision nor integer sums; it needs no model conversion.

`gust_run_valid_inputs(ctx, rows, count)` (capability bit 11, 2048) supplies a separate valid length for each
float32 graph input in tensor-index order. Every count must be 1..input height. The executor validates all row
relationships before writing the arena, uses the existing row map, and zeros produced tails. Legacy INT8 and
dynamic position/cache/count graphs reject this API; `gust_run_valid` remains available. Relay uses it for
independent-row request heads to omit unused instance and option rows. Heads containing GroupNorm or dynamic
operations retain their existing execution path. See the public header for status codes and aliasing rules.

The Relay host releases each tool's temporary prefilter encoder state after projecting its cached vector.
The cached vectors, ranking arithmetic and model outputs are unchanged; cold requests no longer retain all
tool hidden states together. End-to-end measurements with 20+ tools are recorded separately from engine timings.
Prefilter coefficients keep their original FP16 storage and widen exactly when read; this does not quantize
the model further. Ranking temporaries are freed before joint inference, and file loading releases each source
buffer after its independent copy is built. The in-memory API retains caller ownership of all input buffers.

The worker pool invalidates the previous task generation before replacing its arguments and task count. This
prevents a late worker from claiming an exhausted old task after the next round increases the task count.
Its task counter carries a 48-bit generation tag, also rejecting a worker stalled across 65,536 rounds.
The pool remains 144 bytes on x64/AArch64 and 104 bytes on WASM: unused task pointer slots cover the wider
counters, and the task kind travels with immutable arguments. No Context or scratch capacity grows.

Private regression programs include `src/gust.c`; compile each with the runtime's feature, ISA and thread flags
instead of linking a second copy of it. `conv16` needs no arguments; `arena MODEL.gust THREADS` checks varied
lengths against eager clearing byte by byte using a temporary file and a 4 KiB buffer; `workshare` checks 100,000
rounds with changing task counts. Run on the device as well as the PC. `relay_wasm.mjs` additionally reports suite
latency, process peak RSS and guest memory size; those are different from graph allocation bytes.
`../build.sh workshare-wasm` and `node runtime/tests/workshare_wasm.mjs out/workshare.wasm` run the pool
regression with eight real WASM workers across the 32-bit wait-word boundary. The native test also simulates
workers stalled across 16-bit and 32-bit generation gaps. The WASM harness enforces a 180-second timeout.
