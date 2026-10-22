# Gust

Gust is the inference runtime behind [Seal](https://seal.computer)'s models: a small, dependency-free runtime written
in C that runs tiny models on the device, in the browser and in native hosts. It gives the same output bits on x86-64,
ARM64 and WebAssembly, and for any number of threads.

This repository holds the runtime and the code for **Relay**, Seal's first released model. Relay turns a request and a
list of tools into exact function calls. It is 22 million parameters in 11.5 MB, and it needs no GPU, no server and no
Python.

```sh
# Relay's weights are on Hugging Face
git clone https://huggingface.co/sealresearch/relay relay-model

# build the native command line (needs zig: https://ziglang.org)
ZIG=/path/to/zig ./build.sh x64

# ask it to call a tool
out/relay-x64.exe relay-model --record '{
  "text": "Show me the Eiffel Tower on the map",
  "tools": [
    {"name": "show_map", "description": "Shows a location on the map.",
     "params": [{"name": "query", "type": "string", "description": "The place to show.", "required": true}]},
    {"name": "turn_on_flashlight", "description": "Turns the flashlight on.", "params": []}
  ]
}'
# [{"tool": "show_map", "args": {"query": "Eiffel Tower"}}]
```

Relay does not write its calls one token at a time. It reads the request and the tool schemas together and points at
the answer, so it cannot invent a value that is not in the request. How it works, how it was tested against twelve other
models, and where it loses are in the [research post](https://seal.computer/research/relay-1).

## Repository

```
runtime/        the Gust runtime: C, one translation unit, native and WebAssembly (see runtime/README.md)
models/relay/   Relay: the C host (host/), the JavaScript host and bindings (web/), a browser demo (demo/),
                tests (tests/) and benchmark tools (tools/)
build.sh        builds the runtime, the Relay host and the runtime tests with zig cc; outputs go to out/
```

Start with [models/relay/README.md](models/relay/README.md) to use Relay, or [runtime/README.md](runtime/README.md) for
the runtime.

## Licence

The code is MIT, see [LICENSE](LICENSE). The model weights are licensed separately, under CC BY 4.0, in
[sealresearch/relay](https://huggingface.co/sealresearch/relay). Third-party notices are in [NOTICE.md](NOTICE.md).

## Contact

[support@seal.computer](mailto:support@seal.computer) · [seal.computer](https://seal.computer)
