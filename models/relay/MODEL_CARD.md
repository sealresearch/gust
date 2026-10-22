---
license: cc-by-4.0
language:
  - en
base_model: jhu-clsp/ettin-encoder-32m
library_name: gust
tags:
  - tool-calling
  - function-calling
  - on-device
  - edge
  - encoder
  - quantized
  - gust
---

# Relay 1

Relay 1 turns what a person said and the tools an app offers into the calls to make, or into no call when nothing fits.
It is 22 million parameters in 11.5 MB, and it runs on the device: natively on x86-64 and ARM64, and in a browser tab
through WebAssembly. There is no GPU, no server and no Python in the path.

Most tool-calling models are language models, so they _write_ the call one token at a time. Relay writes nothing. It
reads the request and the tool schemas together, then points at the answer. That is why it is small, why it cannot
invent a value that is not in the request, and why it is unusually good at filling slots. It is also where its limits
come from, and they are listed below.

```
User   Set an alarm for 6:45 tomorrow and text Sam I'm running late
Tools  set_alarm(time, label?), send_message(recipient, body), get_weather(city), ...

Relay  set_alarm(time="2026-10-18T06:45:00")
       send_message(recipient="Sam", body="I'm running late")
```

Relay is built by [Seal Research](https://seal.computer). The write-up has the method, the full benchmark tables and the
failure cases: **[Relay 1: a tool-calling model that reads instead of writes](https://seal.computer/research/relay-1)**.
You can also try it in your browser on that page.

## Model details

| | |
|---|---|
| Developer | Seal Research |
| Model type | Encoder with routing, anchor and argument heads; outputs are value programs over the request text |
| Parameters | 21.8M (12.5M encoder, 6.1M embeddings, 3.1M heads) |
| Size | 11.47 MB of model files with 4-bit weights; 14.0 MB with the tokenizer and the retrieval prefilter |
| Vocabulary | 16k tokens, pruned from 50k |
| Language | English (evaluated on English benchmarks only) |
| Runtime | [Gust](https://github.com/sealresearch/gust), Seal's dependency-free C runtime |
| Licence | CC BY 4.0 |
| Version | 1 |

## Files

| File | What it is | Size |
|---|---|---|
| `relay.gust` | the encoder | 9.9 MB |
| `relay-route.gust` | the router head: which tools to call | 0.08 MB |
| `relay-anchor.gust` | the anchor head: where each call's values come from | 0.05 MB |
| `relay-args.gust` | the argument head | 1.5 MB |
| `relay-tokenizer.json` | the tokenizer | 0.5 MB |
| `relay-prefilter.bin` | optional retrieval prefilter for long tool lists | 2.1 MB |
| `SHA256SUMS` | checksums of the files above | |

The `.gust` files are quantized graphs for the Gust runtime. They are not PyTorch or safetensors files, and they do not
load in `transformers`.

## How to use

Download the files and build the host from the [Gust repository](https://github.com/sealresearch/gust). You need
[zig](https://ziglang.org) to build it.

```sh
git clone https://huggingface.co/sealresearch/relay relay-model
git clone https://github.com/sealresearch/gust && cd gust
ZIG=/path/to/zig ./build.sh x64

out/relay-x64.exe ../relay-model --record '{
  "text": "Show me the Eiffel Tower on the map",
  "tools": [
    {"name": "show_map", "description": "Shows a location on the map.",
     "params": [{"name": "query", "type": "string", "description": "The place to show.", "required": true}]},
    {"name": "turn_on_flashlight", "description": "Turns the flashlight on.", "params": []}
  ]
}'
# [{"tool": "show_map", "args": {"query": "Eiffel Tower"}}]
```

In a browser or in Node, build `relay.wasm` with `./build.sh wasm` and use `models/relay/web/relay-native.mjs`; the
repository README has a short example. The native and WebAssembly builds return the same calls, byte for byte, for any
thread count.

An app gives Relay a record: `{text, tools: [{name, description, params: [...]}], kind?, context?: {now}}`. Relay returns
`[{tool, args}]`, or `[]` when no tool fits.

## Intended use

Deciding which of an app's tools to call, and with which arguments, from a short English request, on the device and
without sending the request anywhere. It also fills typed forms: dialogue state tracking and slot filling are the same
problem. It is meant to sit behind an interface where a person can see and confirm what it is about to do.

Out of scope: composing new text, general reasoning, languages other than English (not evaluated), and anything where a wrong call is costly and
there is no confirmation step.

## Evaluation

We ran Relay 1 and 12 other models, including Cactus Needle 3, Google FunctionGemma 270M, Liquid LFM2.5 and
Qwen3.5, on 12 evaluations in one harness, and report all of them, including the ones Relay loses. The full tables,
the harness details and the per-set scores are in the [write-up](https://seal.computer/research/relay-1). In short:

- Mean over the 12 evaluations: **67.2**. The best other on-device model scores 48.7.
- Tied for first of all 13 models on Mobile Actions, and first on both SNIPS slot-filling sets, against models of up to
  748B parameters.
- About 39 ms per request on 8 threads in WebAssembly, and under 100 ms on every benchmark with the prefilter.
- It loses on general tool calling over arbitrary schemas: on BFCL, ToolACE and NexusRaven, language models from 2B
  parameters up are well ahead.

## Training

**Data.** Everything Relay trained on is data we can license cleanly: about 410k verified records we generated
ourselves (the generator is Qwen3.6-35B-A3B, Apache-2.0), procedural templated data, the CC0 SNIPS release, and CC BY
sets (listed under Attribution). We did not train on DroidCall, SGD, xLAM, ToolACE, Glaive or Hermes data. One caveat:
the token frequencies that chose the pruned 16k vocabulary were also counted on some third-party corpora (among them
SGD, Hermes, xLAM and DroidCall). No text or label from them was trained on, but they shaped which existing tokens
were kept.

**Method.** A 150M teacher (ettin-encoder-150M, MIT licence) was trained on all our data. A 32M student
(ettin-encoder-32M) was distilled from it on its outputs and its value-program distributions over three stages. The
vocabulary was pruned from 50k to 16k tokens, the linear layers and embeddings were trained with 4-bit quantization,
and the result was exported to Gust graphs. The exported graphs reproduce the PyTorch model exactly on our golden
records.

## Limitations

- **General tool calling still trails 2B+ models.** BFCL, ToolACE and NexusRaven reward reasoning over long,
  unfamiliar API documentation, and computing or rephrasing values.
- **Some values cannot be expressed.** Relay cannot write text that is not in the request: a composed email body, a
  reworded search query, or a value reformatted in an unusual way.
- **Several calls in one request.** Handling multi-call requests better is one of the gaps Relay 2 is aimed at.
- **Many tools.** Without the prefilter, accuracy degrades with long tool lists and the time grows with the tool text.
  At most 32 parameters per tool are read.
- **English only.** It was evaluated on public English benchmarks, not on your data.
- **Labelling conventions.** Some benchmarks (DroidCall in particular) reward their own labelling conventions, which
  Relay did not train on.

## Licence

The weights are released under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
licence (CC BY 4.0): you may use, share and adapt them, including commercially, as long as you give appropriate credit
to **Seal Research** (https://seal.computer), link to the licence, and say if you changed them. The code that runs the
model is MIT, in the [Gust repository](https://github.com/sealresearch/gust).

## Attribution

Relay was distilled into the ettin-encoder-32M architecture, from a teacher based on ettin-encoder-150M. Both are from
the [Ettin suite](https://huggingface.co/jhu-clsp) by Johns Hopkins University's CLSP (MIT licence).

Training data from third parties:

| Dataset | By | Licence |
|---|---|---|
| [Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) (train split) | Google | CC BY 4.0 |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | NVIDIA | CC BY 4.0 |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | Amazon Science | CC BY 4.0 |
| [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) | Larson et al. | CC BY 3.0 |
| SNIPS NLU (CC0 release) | Snips | CC0 |

Relay is evaluated on public benchmarks, including Mobile Actions, SNIPS, DSTC8, BFCL, ToolACE, NexusRaven and
DroidCall; they are named, with their licences and our scoring, in the write-up.

## Contact

[support@seal.computer](mailto:support@seal.computer) · [seal.computer](https://seal.computer)
