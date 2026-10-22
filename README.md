# Gust

Tiny specialist models that run on the device, in the browser and in native hosts, and the runtime that runs them.

[Seal](https://seal.computer) builds small models, each for one job, and Gust is what they run on: a dependency-free
runtime written in C. It gives the same output bits on x86-64, ARM64 and WebAssembly, and for any number of threads, so
a model behaves the same wherever it runs. Models are released here as they are ready, each with its code in this
repository and its weights on [Hugging Face](https://huggingface.co/sealresearch).

**[Try Relay in your browser](https://seal.computer/research/relay-1#try-it)** • [Write-up](https://seal.computer/research/relay-1) • [Weights](https://huggingface.co/sealresearch/relay)

## Models

| Model | What it does | Size | Weights |
|---|---|---|---|
| [Relay 1](models/relay/README.md) | Turns a request and a list of tools into exact function calls | 11.5 MB | [sealresearch/relay](https://huggingface.co/sealresearch/relay), CC BY 4.0 |

More will be added here as they are released. Each model lives in its own folder under [`models/`](models/README.md).

## Try a model

The quickest way is the `sealpup` command line ([cli/](cli/README.md)). It downloads a model and the program that runs it,
checks both, and runs them:

```sh
curl -fsSL https://raw.githubusercontent.com/sealresearch/gust/main/cli/install.sh | sh
sealpup install relay-1
sealpup run relay-1 "Show me the Eiffel Tower on the map"
```

`sealpup list` shows every model that can be installed. To build a model yourself, see its README; for Relay, that is
[models/relay/README.md](models/relay/README.md).

## From Python

```sh
pip install sealresearch
```

```python
import sealresearch as sr

relay = sr.Model.load("relay-1")        # downloads the files the first time, checks them, keeps them
relay.run({"text": "Show me the Eiffel Tower on the map", "tools": tools})   # [{'tool': 'show_map', 'args': {'query': 'Eiffel Tower'}}]
```

The package is generic: every model is loaded and run the same way, whatever it takes in and gives back
([packages/python/README.md](packages/python/README.md)). It sits on the [model API](sdk/README.md), one small C interface that
other languages can bind too.

## From JavaScript

```sh
npm install @sealresearch/relay
```

```js
import { Relay } from '@sealresearch/relay';

const relay = await Relay.load();   // downloads Relay 1 the first time (14 MB), checks it, and keeps it
relay.predict({ text: 'Show me the Eiffel Tower on the map', tools });   // [{ tool: 'show_map', args: { query: 'Eiffel Tower' } }]
```

It runs in Node and in the browser on the same WebAssembly runtime, and returns the same calls as `sealpup`
([packages/relay/README.md](packages/relay/README.md)).

## Repository

```
runtime/    the Gust runtime: C, one translation unit, native and WebAssembly (see runtime/README.md)
models/     one folder per model, with its code, tests and demo (see models/README.md)
cli/        the sealpup command line: install models, run them, measure them (Rust, see cli/README.md)
sdk/        the model API: one C interface for every model (see sdk/README.md)
packages/   the Python package (sealresearch) and the npm package for Relay (@sealresearch/relay)
docs/       the .gust file format (docs/FORMAT.md)
build.sh    builds the runtime, the model programs and the runtime tests with zig cc; outputs go to out/
```

## Licence

The code is MIT, see [LICENSE](LICENSE). Model weights are licensed separately: each model's licence is in its model card
on Hugging Face and in [models/README.md](models/README.md). Third-party notices are in [NOTICE.md](NOTICE.md).

## Contact

[support@seal.computer](mailto:support@seal.computer) · [seal.computer](https://seal.computer)
