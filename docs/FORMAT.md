# The `.gust` file format

A `.gust` file is one model graph: a header, a table of tensors, a table of operators, a payload of weights and
constants, and a JSON trailer. The runtime reads it in place. `gust_init` validates every byte it will use, once, and
`gust_run` then runs without bounds checks. Everything is little-endian.

This page describes the container. What each operator computes is in the source: the comments in
`runtime/src/int8/ops/` for the INT8 operators and [`runtime/src/float32/README.md`](../runtime/src/float32/README.md),
[`DECODER.md`](../runtime/src/float32/DECODER.md), [`T5.md`](../runtime/src/float32/T5.md) and
[`TEMPORAL.md`](../runtime/src/float32/TEMPORAL.md) for the float32 operators.

## Layout

```
offset 0            Header           64 bytes
offset 64           tensor table     32 bytes x nt
oo = 64 + 32*nt     operator table   80 bytes x no
data = oo + 80*no   payload          weights, scales, biases, tables (every blob 4-byte aligned, most 16)
...                 JSON trailer     UTF-8 text, then its length as a uint32
end                                  total bytes
```

A file is at most 64 MiB. The blob must be 4-byte aligned in memory.

## Header (64 bytes)

| Field | Type | |
|---|---|---|
| `magic` | `u8[8]` | `47 55 53 54 00 00 01 00`: the letters `GUST`, then `00 00 01 00` |
| `version` | `u32` | `1` |
| `total` | `u32` | the size of the whole file in bytes |
| `crc` | `u32` | CRC32 of the whole file, see below |
| `nt`, `no` | `u32` | the number of tensors and of operators, each 1 to 4096 |
| `to` | `u32` | start of the tensor table, always `64` |
| `oo` | `u32` | start of the operator table, `64 + 32*nt` |
| `data` | `u32` | start of the payload, `oo + 80*no` |
| `arena` | `u32` | the bytes of activation memory the graph needs, at most 64 MiB |
| `ni`, `nout` | `u32` | the number of input and of output tensors |
| `flags` | `u32` | `0`: an INT8 graph. `1`: a float32 graph (`float32-graph-v1`) |
| `r0`, `r1` | `u32` | reserved, zero |

**Checksum.** CRC32 with the reflected polynomial `0xEDB88320`, initial value `0xFFFFFFFF` and a final inversion, over
every byte of the file, with bytes 16 to 19 (the `crc` field itself) counted as zero.

## Tensor (32 bytes)

| Field | Type | |
|---|---|---|
| `h`, `w`, `c` | `u32` | height, width and channels |
| `offset` | `u32` | where the tensor lives in the arena (16-byte aligned in a float32 graph) |
| `bytes` | `u32` | its size. INT8: `h*w*c`, with each dimension up to 65,536. Float32: `4*h*c`, with `h` and `c` up to 2^24 |
| `scale` | `f32` | the INT8 quantisation scale, above 0 and at most 1e20. Exactly `1.0` in a float32 graph |
| `flags` | `u32` | `1` input, `2` output. A float32 graph also sets `4` on every tensor, and `8` marks state (a cache or a counter, never an input or output) |
| `reserved` | `u32` | zero |

Tensors that are alive at the same time never share arena bytes, and the loader checks it. Every tensor except the inputs
has exactly one producer, and an operator reads only tensors that exist before it. In a float32 graph `w` is always `1`
(a tensor is a sequence of rows of `c` floats) and tensor 0 is an input.

## Operator (80 bytes)

Twenty 32-bit fields:

```
code, a, b, out, weights, bias, mult, shift,
kh, kw, sy, sx, py, px, dy, dx, p0, p1, lo, hi
```

`code` is the operator, `a` and `b` are input tensor indices, `out` the output tensor. `weights`, `bias`, `mult` and
`shift` are absolute offsets into the file (at or after `data`, 4-byte aligned, inside the file), except where an
operator says a field is a tensor index. `kh` to `dx` are convolution geometry, `p0` and `p1` are mode and parameter
words, and `lo` and `hi` are the clamp of an INT8 result (within -128 to 127, `lo` not above `hi`). In a float32 graph a
field an operator does not use must be zero, and the loader refuses the file otherwise; the INT8 loader checks the same
for `p0` and `p1`.

| Codes | Family | Flags |
|---|---|---|
| 1 to 11 | INT8: convolution, depthwise, add, pool, resize, concat, requantise, slice, gather, top-k pool, weighted pool | 0 |
| 12 to 13 | INT8 attention: matmul, attend | 0 |
| 14 to 16 | INT8 conformer: norm, lookup table, mul | 0 |
| 32 to 37 | float32: conv, add, norm, unary, rope, attention | 1 |
| 38 to 46 | float32 decoder: mul, slice, gather, SiLU, rope-at, cache append, commit, cached attention | 1 |
| 47 to 49 | float32 temporal: depthwise convolution, FiLM, row mean | 1 |

## Payload

Weights are stored packed: signed two's-complement codes of 2 to 8 bits as a little-endian bit stream, in row-major
`[output channel][input]` order, with trailing unused bits zero. Scales, biases, gammas, betas, epsilons and tables are
`f32` (some scales are `f16`, one per 64 weights). Every scale, bias and constant must be finite, and a scale positive.
Relay 1's weights are 4-bit codes with one `f32` scale per output channel.

## JSON trailer

After the payload comes a UTF-8 JSON object with tensor names and shapes and the model's constants (Relay's encoder
carries `vocab`, `max_len`, `hidden` and `pad`). It is for tools and hosts: the runtime never reads it. Its length in
bytes is stored in the last four bytes of the file, so a reader finds it from the end. The checksum covers it.

## What `gust_init` checks

Before it returns success, and without writing the arena, it checks: the size, the alignment, the magic and the
version; every table offset and range; the checksum; that the arena is large enough, 16-byte aligned and does not overlap
the file or the context; every tensor field; every operator's code, unused fields, indices, shapes and geometry; every
blob offset and range; and every scale, bias and constant. A refused file leaves the context not runnable, and the
status says what kind of failure it was:

| Status | |
|---|---|
| `GUST_FORMAT` | header, table, field, offset or alignment |
| `GUST_CHECKSUM` | the CRC |
| `GUST_SHAPE` | dimensions and operator geometry |
| `GUST_MEMORY` | arena or context placement, alignment, overlap |
| `GUST_OPERATOR` | an operator this build does not have |
| `GUST_QUANTIZATION` | a scale, bias or constant that is not finite or not positive |
| `GUST_GRAPH` | producers, operands, input and output roles |

## Capabilities

`gust_capabilities()` says what a build can run, and a build refuses a file that needs more, with `GUST_OPERATOR` or
`GUST_FORMAT`, before it writes anything.

| Bit | |
|---|---|
| 0 | SIMD |
| 1 to 4 | the gather and pooling operators (9 to 11) |
| 5 | attention (12, 13) |
| 6 | valid-length execution (`gust_run_valid`) |
| 7 | normalisation, lookup table, mul (14 to 16) |
| 8 | packed convolution weights |
| 9 | `float32-graph-v1` (32 to 37) |
| 10 | `float32-decoder-v1` (38 to 46, state tensors, `gust_reset`) |
| 11 | per-input row counts (`gust_run_valid_inputs`) |
| 12 | `float32-t5-v1` modes |
| 13 | `float32-temporal-v1` (47 to 49) |
| 14 | `gust_run_valid_outputs` |

## An example

`relay-route.gust`, the 80 KB router head of Relay 1:

```
magic   47 55 53 54 00 00 01 00
version 1        total 79844     nt 6         no 4
to 64            oo 256          data 576     arena 591360
ni 2             nout 2          flags 1
```

Six tensors (192 bytes), four operators (320 bytes), a payload from byte 576 to 78,976, and an 864-byte JSON trailer
that ends with its own length. The four operators are three `F_CONV` layers (4-bit weights) and one `F_UNARY`.
