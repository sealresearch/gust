# float32-graph-v1

Kernel timings use `GUST_F32_PROFILE` (see `runtime/README.md`). The generic `gust_run_traced` and
`gust_run_valid_inputs_traced` APIs emit per-run/operator metadata through a caller-provided observer, using
the same executor, valid-row checks and tail optimization as ordinary runs. No Context/arena growth or
retained observer state is required. The [Relay host trace](../../../models/relay/host/README.md#diagnostic-trace)
extends these events with retrieval, token caps and decoding. Trace formatting/storage are opt-in.

For Relay, end-to-end latency includes tokenization, the retrieval prefilter and decoding as well as these
kernels. The host keeps the existing projected tool-vector cache but releases temporary prefilter hidden
states after each tool. See the [Relay baseline](../../../models/relay/BASELINE.md) for original Relay 1 real-call measurements with 20+ tools.
The prefilter also retains its original FP16 coefficient storage, widening values exactly for the unchanged
double sums. Its ranking temporaries are freed before joint inference. File-backed loading releases copied
source buffers promptly; caller-owned in-memory buffers retain their original lifetime.
The normal Relay host now also lends the same activation arena to its serialized stateless graphs, copying
live encoder/head outputs before reuse. This reduces original Relay 1 graph storage to 21,248,064 bytes without
changing any arithmetic. Stateful or unsupported graphs keep private arenas. The host owns and
frees the borrowed allocation once; `gust_init` still validates every artifact and arena independently.
This adds no runtime mode or model-specific kernel; other users retain their existing arena ownership.

`build.sh wasm` / `wasm-mt` use relaxed SIMD; `wasm-simd` / `wasm-mt-simd` use plain SIMD128 for Safari.
The latter retains exact integer sums by widening int8 operands before signed int16 dot products. Both
single-thread and worker builds use the same model artifacts and float operation order; `build.sh all` builds all four.

Whole-request host benchmarks are built with `build.sh bench-calls-x64` / `bench-calls-arm64` and run as
`relay-calls MODEL_DIR RECORDS.jsonl REPEATS THREADS PREFILTER`. They use the original artifacts, measure the
complete call and print output strings plus process peaks. `models/relay/tools/bench_calls.mjs` provides WASM
measurements. The [Relay baseline](../../../models/relay/BASELINE.md) separates graph storage, process memory,
real-call timing, and the full parity/quality gates.

Opt-in float32 activation graphs with packed low-bit weights, executed by the same single-translation-unit GUST
runtime. They exist because the INT8-activation kernels cannot carry models that are only accurate with float
activations (the first user was the public Moonshine-base encoder, a research probe not in this repository). Everything
here is additive: builds without `GUST_FLOAT32` omit these operators and retain legacy graph execution.

|              |                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| Build define | `GUST_FLOAT32` (the float builds also define `GUST_ATTENTION`, `GUST_CONFORMER`, `GUST_PACKED_WEIGHTS`) |
| Capability   | `gust_capabilities()` bit 9 (512), manifest `required_capabilities: ["float32-graph-v1"]`              |
| Artifact     | `Header.flags = 1`, `r0 = r1 = 0`, version 1; every tensor has flag 4                                 |
| Operators    | 32 `F_CONV`, 33 `F_ADD`, 34 `F_NORM`, 35 `F_UNARY`, 36 `F_ROPE`, 37 `F_MHA`; decoder builds additionally support 38..46 (see `DECODER.md`), the float32-t5-v1 modes (capability bit 12, see `T5.md`) and the float32-temporal-v1 operators 47..49 and modes (capability bit 13, see `TEMPORAL.md`) |
| Entry points | `gust_init`, `gust_run`, `gust_run_valid`, `gust_run_valid_inputs` (capability bit 11), `gust_run_checked`, `gust_tensor(_bytes)` |
| Files        | `abi.h` constants, `format.c` loader, `executor.c` valid rows + dispatch, `ops.c` operators, `kernels/` tiles and attention loops, `threads.c` pool, `math.c` expf/expm1f/tanhf/erff |

`gust_init` hands an artifact with `flags == 1` to `float_init` right after the magic check; every other artifact
takes the unchanged legacy path. A legacy build rejects `flags == 1` as `GUST_FORMAT` before touching the arena
(its sentinel stays intact). A float Context uses `ready == 2`, the legacy 16 KiB `packed_offsets` array as the
per-tensor valid-row map and the 64 KiB `packed_weights`
storage as scratch. Float32 builds additionally have 4 MiB of activation scratch and sixteen 64 KiB thread slabs.
`GUST_SHARED_SCRATCH` (the default Relay builds) stores those 5 MiB once per process and puts pointers in each
Context; without it they live in each Context. Use `gust_context_bytes()` for the configured size. Shared-scratch
runs of different contexts must be serialized. No heap, no libc, no host math on the runtime's hot path.

**What a refused load promises.** `gust_init` clears `ready` first, and the float loader sets it only after every
check passed. A refused artifact therefore never writes the arena or the blob and leaves the context not runnable
(`gust_run` reports `GUST_GRAPH`). It does _not_ leave the whole Context untouched: the loader stages the header,
tensor, operator, blob and arena pointers in it while it validates the rest, so those fields can hold values of the
refused artifact (nothing reads them while `ready` is 0).

## Artifact contract

Header, tensor table, operator table, payload, JSON trailer and CRC32 are laid out as in `docs/FORMAT.md`.

**Tensors.** `w == 1` (`[h, 1, c]` sequences), `h` and `c` in `1 .. 2^24`, `bytes == 4*h*c` (64-bit arithmetic),
`scale == 1.0f` exactly, `reserved == 0`, `flags = 4 | (1 input) | (2 output)` (nothing else; a tensor is not both
input and output; tensor 0 is an input), `offset % 16 == 0`, inside the arena (at most 64 MiB), and the arena base
passed to `gust_init` is 16-byte aligned. Inputs and outputs are live for the whole run, every other tensor has
exactly one producer, operands exist before they are read, and tensors live at the same time never share arena
bytes (so no destination aliases an operand). The blob is 4-byte aligned like the legacy one.

**Operators.** Every `Op80` field not listed below must be zero. `weights`, `bias`, `mult` are absolute blob offsets
(at or after the payload start, 4-aligned, inside the artifact) unless stated.

| Code | Name | Fields used | Meaning |
| ---- | ---- | ----------- | ------- |
| 32 | `F_CONV` | `a, out, weights, bias, mult, kh, kw=1, sy, sx=1, py=0, px=0, dy=1, dx=1, p0, p1` | VALID convolution over time of `[Hin,1,C]` to `[Hout,1,N]`, `Hout = floor((Hin-kh)/sy)+1`, `K = kh*C`. `weights`: `N*K` signed two's-complement codes of `p1` bits (2 to 8) as a little-endian bit stream in row-major `[N][K]` order (taps in time order, channel fastest), trailing unused bits zero. `bias`: `F32[N]` or 0. `mult`: scales, `F32[N]` (per row) or, with `p0` bit 0, `FP16[N*ceil(K/64)]` (per 64 weights of a row, `kh == 1` only). Scales are finite and positive; biases finite. `p0` bit 1: accurate float64 sum. `p0` bit 4: int16 activations (bit 0 and 1 clear, `p1 <= 4`, `K` and output rows at most 2048; see Numerics). No other `p0` bits. Decoder builds: `p0 = 17` grouped partial sums (`T5.md`). |
| 33 | `F_ADD` | `a, b, out` | float32 sum of equal-shape tensors. |
| 34 | `F_NORM` | `a, out, weights, bias, mult, p0` | `weights`: gamma `F32[C]`, `bias`: beta `F32[C]` or 0, `mult`: epsilon (one positive `F32`). `p0 = 0` LayerNorm of every row over its `C` channels; `p0 = 1` one group (GroupNorm, 1 group) over all valid rows times `C`, per-channel affine. Decoder builds: `p0 = 4` RMS norm, no beta (`T5.md`). |
| 35 | `F_UNARY` | `a, out, p0` | `p0 = 0` tanh, `p0 = 1` GELU (erf form). Decoder builds: `p0 = 3` ReLU (`T5.md`). |
| 36 | `F_ROPE` | `a, out, weights, bias, kh, kw` | `weights`: cos table, `bias`: sin table, each `F32[Hin * kw/2]`; `kh` heads (`C % kh == 0`), `kw` even rotary width, `2 <= kw <= C/kh`. Not in place. |
| 37 | `F_MHA` | `a, b, out, weights, mult, kh, p0=0` | Fused non-causal multi-head attention: `a` queries, `b` keys, **`weights` = tensor index of the values** (not a blob offset), `mult` = scale (positive `F32`), `kh` heads, head width `C/kh <= 64`; keys at most 16384; `a`, `out` same shape, `b` and values same shape and same `C`. Decoder builds: `p0 = 4` adds a score bias from the tensor at `bias` (`T5.md`). |

**Validation.** `float_init` checks, before returning success and without writing the arena, blob or any context
field the run depends on: header and table geometry, CRC32 (a nibble-table implementation, identical value to the
legacy bitwise one), arena placement and capacity, every tensor field, every operator's opcode, unused fields,
indices, shapes and geometry, every blob range and alignment, trailing code bits, and every scale, bias, gamma,
beta, epsilon, table and attention-scale value (finite, and positive where required; FP16 scales must be positive
finite). Status codes: `GUST_FORMAT` header/table/field/offset/trailing-bit violations; `GUST_CHECKSUM`;
`GUST_SHAPE` dimensions and operator geometry; `GUST_MEMORY` arena/context placement, alignment and overlap;
`GUST_OPERATOR` opcode outside 32..37; `GUST_QUANTIZATION` non-finite or non-positive scale, bias or constant, or a
tensor scale other than exactly 1; `GUST_GRAPH` producers, operands, input/output roles.

## Numerics (deterministic: every build returns the same bits)

All builds compile with `-ffp-contract=off` and each source carries `#pragma clang fp contract(off)`: a multiply and
an add are never fused. No relaxed SIMD, no fast-math, no host `libm`. Each kernel fixes its reduction order and the
SIMD and scalar variants spell out the same order.

- **Weights.** Coefficient `= float32(code) * float32(scale)`, one float32 multiply. FP16 scales are decoded with
  integer bit operations.
- **`F_CONV`, accurate (`p0` bit 1, the exporter default).** Seed `acc = float64(bias)` (or 0); for `k = 0 .. K-1` in
  increasing order `acc = acc + float64(x[k]) * float64(coefficient[k])`; the output is `float32(acc)`, the only
  rounding to float32. A product of two float32 values is exact in float64. SIMD lanes are output channels, never the
  reduction axis, so tile shape and width cannot change a result.
- **`F_CONV`, float32 mode (`p0` bit 1 clear).** `acc = float32(acc + float32(x[k] * coefficient[k]))` in the same
  order, seeded with the bias. Faster arithmetic, larger error; kept for the stage-0 probe contract.
- **Large `K`.** Weights are restored in panels of at most 4 outputs x `GUST_F32_KBLOCK` (4096) values, so `K` has no
  4096 cap. For `K > 4096` the partial sums are carried unrounded (float64) or in float32 across blocks and the
  panel is restored per 4-row time tile; results are bit-identical for every block size (checked with blocks of 64
  and 7, `K = 4097, 4101, 9216`).
- **`F_NORM`.** Means and variances are float64 sums in increasing (row, channel) order (variance = mean of squared
  deviations from the mean); output `float32(((x - mean) * (1 / sqrt(var + eps))) * gamma + beta)` evaluated in
  float64. `sqrt` is the IEEE operation.
- **`F_UNARY`.** `tanh` is musl/FreeBSD `tanhf`; GELU is `(x * 0.5f) * (1.0f + erff(x * 0.70710678f))`.
- **`F_ROPE`.** Per head, pair `i < kw/2`: `out[2i] = a*cos + (-b)*sin`, `out[2i+1] = b*cos + a*sin`, each product
  rounded to float32, then the sum; the other dimensions are copied. Row `t` uses table row `t`.
- **`F_MHA`.** Per query row and head: `score[j] = float32(dot64(q, k_j) * scale)` where the float64 dot keeps an
  even-dimension and an odd-dimension partial sum (a trailing odd dimension joins the even one), combined as
  even + odd; `m = max score`; `e[j] = expf(score[j] - m)` in float32; `sum` = float64 sum of `e[j]` in increasing
  `j`; `out[d] = float32((float64 sum over j of e[j] * v[j][d]) / sum)`. The score row is the only intermediate (at
  most 64 KiB of Context scratch, so at most 16384 keys); no score matrix exists.
- **`F_CONV`, int16 activations (`p0` bit 4, the Relay encoder).** Each input row gets its own scale
  `s = max|x| / 32767` and is quantized once to `q = round-half-even(x * (32767 / max|x|))` (via `+/- 1.5 * 2^23`);
  `sum = codes . q` is exact in int32 (`|code| <= 8`, `K <= 2048`); the output is `float32(sum) * (s * scale)`
  (+ bias). Integer sums have no order, so tile shapes (4 rows in WASM, 6 in NEON/SSE2, 12 in AVX2/VNNI)
  cannot change a result. The
  quantized rows live in `Context.f32_act` (4 MiB, float32 builds only).
  Partial row groups use a half-height tile before falling back to single rows. NEON loads an activation pair
  once and uses its two lanes for widening MACs. These change instruction scheduling only; quantization,
  integer sums, epilogues and scratch capacity stay the same.
  AArch64 additionally retains packed 4-bit codes in the existing thread slab for even K and at most 64 rows,
  widening them inside the tile. Other shapes use the expanded int16 slab. The choice is automatic, requires
  no extra allocation or model format change, and preserves the same signed codes and exact int32 sums.
- **`F_MHA`, fast (`p0` bit 1).** When the head width is a multiple of the tile width and the per-head panels fit
  `Context.f32_act`, attention runs on the convolution tile: `score[r][j] = float32(sum over d increasing of
  q_r[d] * k_j[d]) * scale`, then `e = expf(score - m)` and `sum` per query exactly as the 8-lane softmax, then
  `out[r][d] = (float32 sum over keys j increasing of e[r][j] * v[j][d]) / sum`; keys of a query tile outside one
  query's band carry `e = 0`. Otherwise the per-query fast path (8-lane dot products) runs.
- **Math sources.** `math.c` ports musl 1.2.5 `expf`, `expm1f`, `tanhf`, `erff` (see `NOTICE`); accuracy against
  float64 on dense grids: `expf` 0.502 ulp, `expm1f` 0.75, `erff` 0.78, `tanhf` 1.89 ulp.

## Threads (`GUST_THREADS`)

`build.sh wasm-mt` at the repository root builds `out/relay-mt.wasm` with atomics, bulk memory and an imported (and exported)
shared memory. The host instantiates it once per thread over the same memory, gives each worker its own stack
(`__stack_pointer`), parks workers in `gust_worker(id)` and enables them with `gust_set_threads(n)`. The executor
still runs operators in order on one thread; the int16 convolution (row quantization chunks, 8-column slabs), the
float32-mode convolution (column blocks), the tiled fast attention (per-head packing, then head x query-chunk tasks)
and the row-wise fast operators (add, unary, rope, fast norm) split into independent tasks (`threads.c`). Every
output value is computed by one task with the same code and query chunks start at multiples of the tile height, so
results are bit-identical for every thread count (checked for 1 to 12 threads). Scratch per thread:
`Context.f32_slab[16]` (64 KiB each); attention panels live in `Context.f32_act`.

Native builds also use this pool. Before publishing a new round's arguments, the controller invalidates the
old `next` generation, then publishes `gen` last. Workers finishing an old drain cannot claim tasks using a new,
larger task count. A 48-bit generation tag prevents an old worker stalled across a full 16-bit wrap from
matching a round being published. Wider counters reuse unused task pointer slots; the global pool remains
144 bytes on x64/AArch64 and 104 bytes on WASM. WASM parks on the low 32 bits and rechecks the full generation
after waking. `runtime/tests/workshare.c` exercises increasing/decreasing counts, stale-worker publication
snapshots and low-word wrap. `build.sh workshare-wasm` builds its eight-worker WASM regression; run it with
`node runtime/tests/workshare_wasm.mjs out/workshare.wasm` from the repository root.

## Valid-length execution

`gust_run_valid(ctx, n)`: `n` counts the real rows of tensor 0 (samples for a waveform); 0 or the full height runs
everything; more than the full height is `GUST_SHAPE`. The legacy `ceil(valid*h/H)` rule is not used. Before any
write, the executor derives every tensor's valid rows in operator order: convolution `V < kh` is invalid, otherwise
`floor((V - kh)/sy) + 1`; `F_ADD` needs equal valid rows on both operands; `F_MHA` takes the query rows, and keys and
values need equal valid rows; every other operator keeps its input's rows. A zero anywhere (for Moonshine 894
samples) returns `GUST_SHAPE`, writes nothing and leaves the context usable. Norm statistics, attention keys and
convolution windows read valid rows only, so the valid rows of every output equal those of a graph compiled for
exactly that many rows, bit for bit, whatever the padding contains (NaN, huge values). Rows past the valid rows of
every produced tensor are written as zero; input tensors and their padding are never written. Input values must be
finite (NaN inputs are not claimed to be portable).

`gust_run_valid_inputs(ctx, rows, count)` advertises capability bit 11 (2048). Supply one count per graph input,
in increasing tensor-index order, each between 1 and that input's height. The same propagation rules apply,
including equal-length joins and attention keys/values; all counts and joins are checked before arena writes.
Counts are consumed into the existing Context row map before execution and may alias arena storage, but must
not overlap Context storage. Null/unready contexts return `GUST_GRAPH`; null arrays, incorrect input counts,
invalid lengths or inconsistent joins return `GUST_SHAPE`; Context overlap returns `GUST_MEMORY`. Legacy INT8
and graphs with dynamic position/cache/count operators return `GUST_OPERATOR`. Refusals leave the arena
untouched and the Context usable. `gust_run_valid` and checked execution retain their existing semantics.

Changing a GroupNorm input's length changes its normalization domain. Callers seeking dense-output parity
must retain that domain. Relay restricts independent input lengths to request heads without GroupNorm or
dynamic operators, then passes the fed request, instance and option lengths. `tests/input_rows.c` compares
valid head outputs with dense execution, checks NaN padding and verifies refused calls and legacy fallback.

For ordinary runs, a tail clear is skipped when a later operator's complete destination interval contains that
tail. Every operator reads valid rows only, and the loader rejects overlapping live tensors. The later operator
therefore replaces every skipped byte before the run returns. If its own tail clear is skipped, the same argument
applies inductively to a still later writer. This preserves the entire final arena, including dead intermediate
storage, with no additional liveness map or allocation. Scanning later destinations uses the validated graph.
The optimization is disabled for checked runs with callbacks and for graphs containing operators above SiLU
(dynamic position/cache/count operations). Remaining tails of at least 256 KiB are cleared in disjoint 64 KiB
tasks when several threads are active. `runtime/tests/arena.c` compares this execution with eager clearing at
varied lengths, byte by byte; both paths compute identical valid rows.

## Resource bounds

| Resource | Bound |
| -------- | ----- |
| Artifact, arena | 64 MiB each; offsets and arena base 16-byte aligned |
| Context | configuration-dependent (`gust_context_bytes()`); `packed_weights` (64 KiB) is the float scratch, `packed_offsets` (16 KiB) the valid-row map; activation/slab scratch is 5 MiB per Context or once per process with `GUST_SHARED_SCRATCH` |
| Weight panel | 4 outputs x min(K, 4096) float32 = at most 64 KiB, aliasing-safe through `may_alias` types |
| Attention score row | at most 16384 float32 = 64 KiB (same scratch; never live together with a weight panel) |
| Attention accumulators | two `double[64]` on the stack (1 KiB) |
| Stack | depends on features and compiler; loader liveness arrays alone use 32 KiB. Current Relay WASM builds reserve 1 MiB for the main stack; the threaded SDK allocates 1 MiB per worker. Native thread stacks are host-managed |
| Heap | none |
| Guest memory | build/host-dependent; measure `memory.buffer.byteLength` separately from process RSS and graph allocations. `build.sh` Relay modules have a 1 MiB main stack; the threaded SDK starts shared memory at 64 MiB with a 1 GiB maximum. The SDK also keeps artifact bytes outside the guest. `relay_wasm.mjs` and `bench_wasm.mjs` report the measured guest size |

## Limits and non-goals

- No causal attention, no state or KV cache, no decoder, no reshape/permute, no broadcast, no padding or dilation in
  `F_CONV`, `w` is always 1.
- A refused `gust_run_valid` call writes nothing; a `gust_run_checked` stop callback may stop between operators
  (`GUST_BUDGET`) and the partial output must not be consumed.
- Speed: weights are restored on every run (about 19 ms for the 20M weights of the Moonshine encoder), so very short
  inputs are dominated by that fixed cost.
