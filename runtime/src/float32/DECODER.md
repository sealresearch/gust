# float32-decoder-v1: persistent state and the operators 38 to 45

An opt-in extension of `float32-graph-v1` (see `README.md`) that lets a float32 graph keep
state between runs, which is what a cached (KV) transformer decoder needs: the keys and values of the tokens already
decoded are written once, a step attends over them and never recomputes the prefix. Offline decoding only: there is no
sliding audio window, no streaming model and no live-audio claim in this contract.

Compile with `-DGUST_FLOAT32 -DGUST_FLOAT32_DECODER` (plus the features `float32-graph-v1` needs). Every other build (legacy, plain
float32) is byte for byte what it was: it does not report capability bit 10, does not export `gust_reset`, and
refuses a state tensor (`GUST_FORMAT`) or an operator above 37 (`GUST_OPERATOR`) before it writes the arena.
`GUST_FLOAT32_DECODER` without `GUST_FLOAT32` is a compile error.

- Capability bit 10 (1024), manifest `required_capabilities` `float32-graph-v1` plus `float32-decoder-v1` (the second
  entry exactly when the graph has a state tensor or an operator above 37). Header, Tensor, Op and Context layouts are
  unchanged.
- Fixed-function operators 32 to 37 keep every rule of `float32-graph-v1`, and may not read or write state.

## State tensors

Tensor flag 8, always together with flag 4 (float32): `12`. Never an input (1) or an output (2). Height `1 .. 256`.
A state tensor is live for the whole model, is **zero after every successful load** (the only arena write of a load,
after every check passed) and after `gust_reset`, and its bytes never overlap any other tensor. The host does not copy
state in or out. The `GdLoad` walk of `float_init` additionally requires, for a graph with state:

- exactly one `[1, 1]` position **counter** state, written by exactly one `F_COMMIT`, which is the last operator;
- every other state is a cache with exactly one `F_CACHE_APPEND` per run, no state nobody writes, and a cache read by
  `F_MHA_PREFIX` (mode 1) only after its append;
- every append, the commit and each mode 1 attention name the **same** position input;
- ordinary operators never write a state tensor.

## Operators (all unused fields zero)

| Code | Name            | Fields                                                                                                   |
| ---- | --------------- | -------------------------------------------------------------------------------------------------------- |
| 38   | `F_MUL`         | `a, b, out` equal shapes, IEEE float32 multiply                                                          |
| 39   | `F_SLICE`       | `a, out, p0` first channel; equal height, `p0 + Cout <= Cin`                                             |
| 40   | `F_GATHER`      | `a` token `[1,1]` input, `out [1,C]`, `weights` packed codes `[p0 rows][C]`, `mult` float32 scale per row, `p1` bits 2 to 8 |
| 41   | `F_SILU`        | `a, out` equal shape; `x/(1+exp(-x))` for `x >= 0`, `x*exp(x)/(1+exp(x))` for `x < 0`                      |
| 42   | `F_ROPE_AT`     | `a [1,C], b` position input, `out`, `weights` cos, `bias` sin float32 `[p0, kw/2]`, `kh` heads, `kw` rotary width, `p0` table positions |
| 43   | `F_CACHE_APPEND`| `a [1,C], b` position input, `out` cache state `[cap, C]`, `weights` the counter tensor index              |
| 44   | `F_COMMIT`      | `a` position input, `out` the counter state                                                              |
| 45   | `F_MHA_PREFIX`  | `a` query `[1,C], b` keys, `weights` values tensor index, `out`, `mult` scale, `kh` heads, `p0` scalar input tensor index, `p1` mode |

`F_GATHER` decodes the one requested row as `float32(code) * scale` (no full-matrix decode); `F_CONV` over the **same**
blob and scales is the tied output projection, so a tied embedding is stored once. `F_ROPE_AT` rotates interleaved
pairs of the first `kw` channels of every head with the table row `position` and copies the rest; operations are the
`F_ROPE` ones of `float32-graph-v1`. `F_MHA_PREFIX` reuses the deterministic `F_MHA` inner loops over the first `n` key rows:
mode 0 `n` = the value of the scalar input (cross attention over the encoder rows, keys and values ordinary inputs),
mode 1 `n` = that value + 1 (self attention: the scalar is the position, keys and values are the caches the same run
appended to). No row past `n` is read. Head width at most 64, at most 16384 keys.

## Dynamic values and refusal

Token, position, counts and the counter are float32 values the host writes. They are checked for the **whole graph** by
`gd_preflight` before the first operator runs: they must be finite exact integers inside their bounds, an append and
the commit need `position == counter < capacity`, an attention count must be `1 ..` rows. NaN, infinities, negatives
(also -0), fractions and out-of-range values give `GUST_SHAPE`, and a refused call changes no arena, state, input or
output byte. A replayed or skipped position is refused the same way.

A run that completes advances the counter exactly once (the last operator). `gust_run_checked` may stop between
operators (`GUST_BUDGET`): the cache row of the current position and intermediates may have been written, the counter has
**not** advanced, rows of earlier positions are untouched, and a retry of the same position rewrites the same row and
computes identically. Logits of a stopped run must not be used.

`gust_run_valid` keeps its meaning: `0` or the height of tensor 0 runs everything, larger is refused; the explicit
scalars never decide a row count. Cross-attention padding is ignored through the count of `F_MHA_PREFIX`.

## `gust_reset(context)`

Exported by decoder builds only. Clears every state tensor (cache rows and the counter); static inputs, outputs and the
context stay usable; no allocation. `NULL` or a context that is not loaded: `GUST_GRAPH`; a loaded INT8 graph:
`GUST_OPERATOR` (nothing is written in either case); a float32 graph without state: `GUST_OK`. After a reset, the next
utterance is bitwise what a freshly loaded model computes.

## Numerics

`expf` is the deterministic musl port of `float32-graph-v1`; `F_SILU`, `F_MUL`, RoPE and the attention loops use IEEE float32
add/multiply/divide only, no host libm, `fp-contract off`, no fused operations. Native and WebAssembly (SIMD and
scalar) produce identical bits. No part of this extension is derived from Whistle sources; the `NOTICE` is unchanged.

Its first user was a Moonshine pipeline of three artifacts (encoder, cross-prefill, cached step), a research probe
that is not part of this repository.
