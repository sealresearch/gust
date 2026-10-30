# Models

Every model has its own folder here. The runtime, the command line and the build are shared; a model brings its own code,
tests and demo.

| Model | Folder | What it does | Weights | Licence of the weights |
|---|---|---|---|---|
| Relay 1 | [`relay/`](relay/README.md) | Turns a request and a list of tools into exact function calls | [sealresearch/relay](https://huggingface.co/sealresearch/relay) | CC BY 4.0 |
| Squint 1 (research) | [`squint/`](squint/README.md) | Reads QR codes from shaky, blurred camera frames | none: hand-built C, no weights | MIT (code) |

## What a model folder holds

Relay is the first model, so its folder shows the shape. A model has what it needs and no more:

```
models/<name>/
  README.md     what it does, how to use it, its limits
  MODEL_CARD.md the model card: the file that is the README of the weights repository on Hugging Face
  assets/       the images of the model card (a banner, charts)
  host/         the C host: the model's own pipeline on top of the Gust runtime, for every target
  web/          the JavaScript host and the bindings for the WebAssembly build
  demo/         a browser demo
  tests/        regression tests
  tools/        benchmarks and other tools
```

The weights are never in this repository. They live on Hugging Face in `sealresearch/<name>`, with a model card that
states what the model was trained on and its licence. Code here is MIT; each model's weights carry their own licence.

## Adding a model

1. **The code**, in `models/<name>/`, with a README, tests that check its output against a reference byte for byte, and
   a host that builds with `zig cc` for every target. The runtime's rule applies: outputs are bit-exact on every target
   and for any thread count (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
2. **A build target** in [`build.sh`](../build.sh), including the program in `build.sh release` so it is attached to
   releases for every platform. Today the unprefixed targets (`x64`, `arm64`, `wasm`, ...) build Relay; the next model
   gets targets of its own, prefixed with its name.
3. **The weights and a model card** on Hugging Face, in `sealresearch/<name>`, with the licence, the training data and
   its sources, the evaluation, and the limits. The card is written here, in `MODEL_CARD.md` with its `assets/`, and copied to the weights
   repository as its `README.md` and `assets/`, so the two do not drift apart.
4. **An entry in the command line's registry**, [`cli/registry.json`](../cli/registry.json): the model's files with their
   SHA-256 checksums, and the program that runs it. After that `sealpup install <name>` works.
5. **A row in the table above** and in the [README](../README.md).
