# float32-temporal-v1: temporal convolution networks

A temporal convolution network is built from ConvNeXt blocks over time conditioned on a vector: a depthwise convolution
along the frames, LayerNorm, a FiLM modulation computed from the conditioning row, a two-layer MLP and a residual. A
prosody model pools a sentence vector, and a decoder ends in an unquantised projection. `float32-graph-v1`,
`float32-decoder-v1` and `float32-t5-v1` lack the depthwise convolution, the conditioning, the row mean and float32
weights; this extension adds them, plus 8-bit codes for the int16-activation convolution. Every decoder build
(`GUST_FLOAT32_DECODER`) has it; there is no separate define.

- Capability bit 13 (8192); manifest `required_capabilities` entry `float32-temporal-v1`, present exactly when a graph
  uses an operator or mode below. Operators 47 to 49 are part of decoder builds, so such a graph also declares
  `float32-decoder-v1` (as the JavaScript binding requires for every operator above 37).
- Header, tensor, operator and context layouts do not change. Each operator or mode uses an operator code or field
  value that a runtime without it refuses (`GUST_OPERATOR` or `GUST_FORMAT`) before it writes the arena, so no
  existing graph changes behaviour; the Relay bit-exactness suite and `runtime/tests/t5.c` pass unchanged.

| Operator | Fields | Meaning |
| --- | --- | --- |
| `F_DWCONV` (47) | `a, out` equal shape; `weights` float32 `[kh][C]`; `bias` float32 `[C]` or 0; `kh` odd, 1..31; `dy` dilation >= 1 | `y = bias[j]` (or 0); for taps `k = 0 .. kh-1` in order, `s = t + (k - (kh-1)/2) * dy`, and when `0 <= s < V`: `y = y + x[s][j] * w[k][j]` |
| `F_FILM` (48) | `a, out` equal shape `[T, C]`; `b` a tensor with `2C` channels, row 0 read; `p0` 0 or 1; `C <= 16384` | `z[t][j] = x[t][j] * g[j] + v[j]` with `[u | v] = b[0]`; `p0` 0: `g = 1 + u`; `p0` 1: `g = 1 + 0.5 * tanh(u)` |
| `F_MEAN` (49) | `a [T, C]`, `out [1, C]` | `z[0][j] = float32((sum over valid rows t, increasing, of double(x[t][j])) / V)`; the output has one valid row |

| Mode | Fields | Meaning |
| --- | --- | --- |
| `F_CONV` (32) float32 weights | `p1 = 32`, `p0` 0 or 2; `weights` float32 `[N][K]`, `mult` float32 scale per row | the weight is `float32(value * scale)`; sums as in float32 (`p0` 0) or float64 (`p0` 2) mode |
| `F_CONV` (32) int16 activations, 5- to 8-bit codes | `p0 = 4`, `p1` 5..8 (4 and below as before) | activation codes are bounded by `L = min(32767, floor((2^31 - 1) / (2^(p1-1) * K)))` instead of 32767, so no int32 partial sum can overflow: `s = max|x| / L`, `q = round-half-even(x * (L / max|x|))`, `y = float32(sum code * q) * (s * scale) (+ bias)` |

`V` is the executor's valid row count of the input (`gust_run_valid` / `gust_run_valid_inputs`): the depthwise
convolution treats rows past it as zero, like the edge of the sequence, so a window of a longer sequence computes what a
graph compiled for exactly that many rows computes. `F_FILM`'s conditioning tensor is normally a one-row input or the
output of an `F_CONV` over a one-row conditioning input; its gains are computed once per call. The temporal operators
may be combined with `gust_run_valid_inputs` (per-input rows).

Int16 activations with 8-bit codes: integer sums are exact, so every build returns the same bits; the bound costs at
most one bit of activation precision at K = 2048 (L = 8191 for 8-bit codes) and none for K <= 512. Decoder MLPs use this
mode, where it changes the output less than the difference between a fused and an unfused
multiply-add build.

Numerics: separate float32 multiplies and adds (contraction off), `tanh` of math.c, no host libm. Native and
WebAssembly builds, SIMD or not, produce identical bits. Tests: `runtime/tests/temporal.c` (`build.sh test-temporal`:
every operator and mode against its contract, refusals).

## Host work: `gust_parallel`

`gust_parallel(fn, opaque, items)` (gust.h) runs host tasks on the runtime's worker pool, so a model host's signal
processing between graph runs (for example STFT features) uses the same threads as
the operators. One task per item; the thread number (below 16) selects per-thread scratch. Items must be independent
and write disjoint memory, which keeps results independent of the thread count.
