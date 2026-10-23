# sealresearch

[Seal Research](https://seal.computer)'s on-device models from Python. One interface for every model, on the
[Gust](https://github.com/sealresearch/gust) runtime: no GPU, no server, nothing leaves the machine.

```sh
pip install sealresearch
```

```python
import sealresearch as sr

relay = sr.Model.load("relay-1")        # downloads the files the first time (14 MB), checks them, keeps them

tools = [
    {"name": "show_map", "description": "Shows a location on the map.",
     "params": [{"name": "query", "type": "string", "description": "The place to show.", "required": True}]},
    {"name": "turn_on_flashlight", "description": "Turns the flashlight on.", "params": []},
]

relay.run({"text": "Show me the Eiffel Tower on the map", "tools": tools})
# [{'tool': 'show_map', 'args': {'query': 'Eiffel Tower'}}]

relay.run({"text": "Tell me a joke", "tools": tools})
# []   no tool fits, so no call
```

Relay's `tools` are `{"name", "description", "params": [{"name", "type", "description", "required", "enum"}]}`;
[every field](https://github.com/sealresearch/gust/blob/main/models/relay/TOOLS.md) is described, with a converter for the JSON
Schema tools of OpenAI, Anthropic and MCP. [Relay 1](https://seal.computer/research/relay-1) is the first model. Every model is used the same way: what it takes in and gives
back is in `model.info`, so new models need no new code here.

## The interface

```python
model = sr.Model.load("relay-1", options={...}, threads=4, directory=None, cache=True, optional=True)
model.info                      # a dict: id, kind, inputs, outputs, options, files, license
model.memory_bytes              # bytes the loaded model holds
model.run(*inputs, options={})  # one call; one output comes back as itself, several as a list
model.stream(options={})        # for models that work on live input
model.close()                   # or use `with`

sr.models()                     # the models in this build
sr.describe("relay-1")          # a model's description, before it is opened
sr.set_threads(4)               # threads for the operators, for the whole process
sr.runtime()                    # the instruction set in use, e.g. "x86-64 AVX-512 VNNI"
```

What goes in, by Python type:

| You pass | The model receives |
|---|---|
| a `dict`, `list`, number, `bool` or `None` | JSON |
| a `str` | text |
| `bytes` | bytes |
| `sr.Pcm(samples, rate, channels=1)` | audio, float32 or int16 (numpy arrays, `array.array` and lists work) |
| `sr.Image(pixels, width, height, format="gray8")` | a frame, `"gray8"` or `"rgba8"` |

Outputs come back the same way: JSON as Python objects, text as `str`, audio as `sr.Pcm` with a numpy array when numpy is
installed, images as `sr.Image`.

### Streams

For models that take live input (speech in) or give a long output (speech out):

```python
with model.stream() as s:
    for chunk in audio_chunks:
        s.push(sr.Pcm(chunk, 16000))
        for out in s.pull():            # whatever is ready; may be empty
            ...
    s.finish()
    for out in s.drain():               # everything that is left
        ...
```

A model that has no streaming raises `sr.GustError` with `status == "unsupported"`.

## Files, cache and threads

`Model.load` finds a model's files in this order: the `directory` you give, the cache, a download from Hugging Face. Every file is
checked against a SHA-256 checksum from the registry (the same one the `sealpup` command line uses) before it is used, so a
damaged or changed file is refused. The cache is `$GUST_CACHE`, or the platform's cache folder, or the path you pass as
`cache=`; `cache=False` downloads every time. `optional=False` skips the files a model lists as optional (Relay's 2.1 MB
prefilter, which ranks long tool lists).

`threads` is for the whole process. The answers are identical for every thread count, and on every platform: that is what Gust
guarantees.

Calls are serialised across the process, so Python threads can share a model safely; to use more cores, use `threads`.

## Errors

Anything the library reports is a `sr.GustError`: `e.status` is one of `"argument"`, `"unknown_model"`, `"files"`, `"open"`,
`"input"`, `"run"`, `"unsupported"`, `"memory"` or `"state"`, and the message says why.

## Platforms

Wheels carry the library for Windows (x86-64), Linux (x86-64 and ARM64, glibc 2.17 or newer) and macOS (Intel and Apple silicon).
To use a library you built yourself, set `GUST_LIB` to its path.

## Licences

The code is MIT. Each model's weights have their own licence, in its model card and in `model.info`.
