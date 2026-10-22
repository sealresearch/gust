# Relay 1 baseline

Measurements of Relay 1 to compare a change against: memory, speed, and the evidence that every build returns the same
bits. Take them again on your own machine before you compare, because only the hashes are machine independent.

Measured on 2026-10-22 on one desktop PC: AMD Ryzen 7 7800X3D (8 cores, 16 threads), Windows 11, Zig 0.14.1 for the
builds, Node 24.16 for WebAssembly. The model files are the ones in
[sealresearch/relay](https://huggingface.co/sealresearch/relay).

## Memory

| | |
|---|---|
| Graph storage, all four graphs | 21,248,064 bytes. The host lends one activation arena to its serialised graphs and copies live outputs before reuse; it was 24,543,040 bytes before that |
| Encoder alone, peak working set | about 25 MB (`gust-bench`) |
| Whole host, peak working set, with the prefilter | 47.3 MB on one thread, 33.0 MB on eight (`relay-calls`, the first 100 Mobile Actions requests, three repeats) |
| Shared scratch | 5 MiB per process: a 4 MiB area for quantised activations and sixteen 64 KiB thread slabs |

Peak working set is a Windows measure and varies by a few MB from run to run. It is not the same as the graph storage
above or the WebAssembly guest memory (`relay_wasm.mjs` and `bench_wasm.mjs` print that one).

## Speed

**A whole request**, with the prefilter on, native (`relay-calls`), the first 100 Mobile Actions requests (about 350 tokens
each), three repeats each:

| Threads | Median | 90th percentile |
|---:|---:|---:|
| 1 | 81.1 ms | 90.8 ms |
| 8 | 18.4 ms | 23.1 ms |

**A whole request in WebAssembly**, the C host in Node (`relay-mt.wasm`, `relay.wasm` for one thread), all 961 Mobile Actions
requests (median 350 tokens):

| Threads | Median | 90th percentile |
|---:|---:|---:|
| 1 | 156.4 ms | 176.6 ms |
| 4 | 68.3 ms | 76.5 ms |
| 8 | 38.5 ms | 43.2 ms |
| 11 | 29.9 ms | 34.0 ms |
| 16 | 26.8 ms | 30.0 ms |

**The encoder alone** (`gust-bench`), median of three runs of 21, 350 tokens:

| Build | 1 thread | 8 threads |
|---|---:|---:|
| x86-64 SSE2 | 137.2 ms | 23.6 ms |
| x86-64 AVX2 | 78.8 ms | 14.1 ms |
| x86-64 AVX-512 VNNI | 74.3 ms | 13.8 ms |
| WebAssembly, relaxed SIMD | 152.2 ms | |
| WebAssembly, plain SIMD128 | 152.7 ms | |

On the AVX-512 VNNI build with one thread the encoder takes 9.95 ms for 35 tokens, 74.31 ms for 350 and 149.0 ms for 650.

## Same bits

`gust-bench` prints an FNV-1a hash of the output tensor and of the whole activation arena (filled with `0xa5` before the
run, so padding and intermediates count). Every build above, at 1 and 8 threads where it has threads, prints:

| Tokens | Output hash | Arena hash |
|---:|---|---|
| 35 | `26377ab50d98a40e` | `fe93361c0608a3eb` |
| 350 | `f0e885d1b5158127` | `0f57a2b1f3348caa` |
| 650 | `4171737002ffe514` | `dcf62d5ca3f1a7e6` |

The hashes depend on the input the benchmark generates, not on the machine. A change that moves them has changed the
arithmetic. (Allowing the compiler to fuse multiply-adds moves the 350-token output hash to `72f133a23a180119`.)

Above the engine, on 1,800 requests (150 from each of the 12 evaluations), the C host in WebAssembly and the JavaScript
reference host (`web/relay.mjs`) return the same calls on 1,799. On the other, a sealed-set request, the calls and values
are equal and only the order of one call's argument keys differs. With the prefilter on, the C host reads at most 10
tools or 600 tokens of tool text, and the JavaScript host reads them all, so they differ on some requests with long
tool lists (30 of the 1,800).

## Reproduce

```sh
ZIG=zig bash build.sh bench-x64 bench-calls-x64 wasm wasm-mt     # bench-arm64 and bench-calls-arm64 for ARM64

# the encoder alone, forcing the x86-64 build, at 35, 350 and 650 tokens: lengths, repetitions, threads
GUST_X86=avx2 out/gust-bench-x64.exe relay.gust 35,350,650 21 8

# whole requests, native: MODEL_DIR RECORDS.jsonl REPEATS THREADS PREFILTER
out/relay-calls-x64.exe relay-model records.jsonl 3 8 1

# whole requests in WebAssembly
node models/relay/tools/bench_calls.mjs out/relay-mt.wasm relay-model records.jsonl 3 8
```

`runtime/README.md` describes `gust-bench` and `relay-calls`, and `runtime/tests/bitexact.sh` runs the bit-exactness check
across every build.
