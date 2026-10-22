# Relay

Relay 1 turns what a person said and the tools an app offers into the calls to make, or into no call when nothing
fits. It reads instead of writes: it looks at the request and the tool schemas together and points at the answer, so
every value it returns comes from the request or the schema.

|  |  |
|---|---|
| Parameters | 21.8M (12.5M encoder, 6.1M embeddings, 3.1M heads) |
| Size | 11.5 MB of model files, with 4-bit weights |
| Runs on | x86-64, ARM64 and WebAssembly, with no GPU and no Python |
| Weights | [`sealresearch/relay`](https://huggingface.co/sealresearch/relay) on Hugging Face, CC BY 4.0 |
| Write-up | [seal.computer/research/relay-1](https://seal.computer/research/relay-1) |
| Model card | [MODEL_CARD.md](MODEL_CARD.md): training data, evaluation, limits, attributions |

## Use it

Get the weights, build, and run:

```sh
git clone https://huggingface.co/sealresearch/relay relay-model
ZIG=/path/to/zig ./build.sh x64 wasm
out/relay-x64.exe relay-model --record-file request.json --prefilter
```

A request is a JSON record: `{text, tools: [{name, description, params: [...]}], kind?, context?: {now}}`. `context.now`
is the date and time Relay resolves "tomorrow" against. From C, call `relay_open` and `relay_call` (see
[host/README.md](host/README.md)); from JavaScript, use `web/relay-native.mjs` with the WebAssembly build:

```js
import {readFileSync} from 'node:fs';
import {RelayNative} from './models/relay/web/relay-native.mjs';

const read = (name) => readFileSync(`relay-model/${name}`);
const relay = await RelayNative.create({
  wasm: readFileSync('out/relay.wasm'),
  files: {
    encoder: read('relay.gust'), route: read('relay-route.gust'), anchor: read('relay-anchor.gust'),
    args: read('relay-args.gust'), tokenizer: read('relay-tokenizer.json'), prefilter: read('relay-prefilter.bin'),
  },
});
relay.predict({text: 'Set an alarm for 6:45', tools: [/* ... */]});  // [{tool, args}]
```

The native and WebAssembly builds return the same calls, byte for byte, for any thread count.

## What is here

```
host/    the C host: tokenizer, decoding, value programs, prefilter; builds for every target (host/README.md)
web/     the JavaScript reference host (relay.mjs) and the bindings for the WebAssembly build (relay-native.mjs)
demo/    a browser sandbox: pick a simulated device, type a request, watch the calls (demo/README.md)
tests/   shared-arena and trace regressions
tools/   benchmark tools for whole-call latency
```

## Limits

Relay 1 does one job well and is weak at others. It trails 2B+ parameter language models on general tool calling over
long, unfamiliar API documentation (BFCL, ToolACE, NexusRaven). It cannot write text that is not in the request, such as
a composed email body, and it reads at most 32 parameters per tool. The research post covers all of this, with numbers.
