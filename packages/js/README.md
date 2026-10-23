# @sealresearch/gust

The [Gust](https://github.com/sealresearch/gust) model API in Node and in the browser. One interface for every model: you
open a model by id, give it typed values, and get typed values back. It is the same C code that runs in the native library, in
the Python package (`pip install sealresearch`) and behind the `sealpup` command line, compiled to WebAssembly, so the
answers are the same bits on every platform and for any number of threads.

```sh
npm install @sealresearch/gust
```

```js
import { Model } from '@sealresearch/gust';

const model = await Model.load('relay-1');     // downloads the model once (11.5 MB), checks it, keeps it

model.run({
  text: 'Show me the Eiffel Tower on the map',
  tools: [{ name: 'show_map', description: 'Shows a location on the map.',
            params: [{ name: 'query', type: 'string', description: 'The place to show.', required: true }] }],
});
// [{ tool: 'show_map', args: { query: 'Eiffel Tower' } }]

model.close();
```

Relay's `tools` are `{ name, description, params: [{ name, type, description, required, enum }] }`, or the JSON Schema tools of
OpenAI, Anthropic or MCP as they are; [every field](https://github.com/sealresearch/gust/blob/main/models/relay/TOOLS.md) is
described. The calls are synchronous and take as long as the model takes. In a browser, load and run the model in a
[Worker](https://developer.mozilla.org/docs/Web/API/Worker) so the page stays responsive.

## What goes in and out

A model says what it takes and gives in its description (`model.info`, or `await describe(id)` before opening it). The package
only moves typed buffers:

| In JavaScript | Reaches the model as |
|---|---|
| an object, array, number, boolean or `null` | JSON |
| a string | text |
| `Uint8Array`, `ArrayBuffer` | bytes |
| `new Pcm(samples, rate, channels)` | audio (float32 or int16) |
| `new Image(pixels, width, height, 'rgba8')` | an image (`'gray8'` or `'rgba8'`) |

Outputs come back the same way: JSON is parsed, text is a string, audio is a `Pcm`, an image is an `Image`. Pass `{ raw: true }`
to keep a JSON output as the text the model wrote.

```js
model.run(input, { options: { mode: 'explain' } });     // options for this call
model.runAll([a, b]);                                   // a model that takes several inputs at once
```

## Streams

For models that work on live input, such as speech in and text out, or text in and audio out:

```js
const stream = model.stream();
for (const chunk of chunks) {
  stream.push(new Pcm(chunk, 16000));
  for (const out of stream.pull()) use(out);            // may be empty
}
stream.finish();
for (const out of stream.drain()) use(out);             // everything that is left
stream.close();
```

## Loading

`Model.load(id, options)` finds the files, checks them against the SHA-256 sums in the registry, and opens the model.

| Option | Default | |
|---|---|---|
| `options` | | What the model's description lists under `options.open` (Relay: `prefilter`, `k`, `budget`, `fixes`, `require`). |
| `threads` | `1` | `1` to `16`, or `'auto'`. More threads is faster, and the answers are identical for every count. More than one needs `SharedArrayBuffer`: always in Node, and in a browser only on a [cross-origin isolated](https://developer.mozilla.org/docs/Web/API/Window/crossOriginIsolated) page (otherwise it runs on one). |
| `directory` | | Node only. Read the files from this folder instead of downloading them. |
| `cache` | `true` | Keep downloads. In Node that is `$GUST_CACHE`, the platform's cache folder, or a path you pass; in a browser it is the Cache API. `false` downloads every time. |
| `baseUrl` | `https://huggingface.co` | Where to download from, for a mirror. |
| `optional` | `true` | Also fetch the files the model lists as optional (Relay's 2.1 MB prefilter). |
| `onProgress` | | Called with `{ file, loaded, total, cached? }`. |

Have the files already? `Model.open(id, { 'relay.gust': bytes, ... })` takes them from memory: no registry, no download.

Close models you are done with: `model.close()` frees the memory and stops the worker threads when it was the last model using
them.

## Pages that serve the WebAssembly themselves

The package finds its two modules (`wasm/gust-models.wasm`, and `wasm/gust-models-mt.wasm` for more than one thread) next to the
code. If your bundler or server puts them elsewhere:

```js
import { configure } from '@sealresearch/gust';
configure({ wasm: { st: '/assets/gust-models.wasm', mt: '/assets/gust-models-mt.wasm' } });
```

## Relay only?

[`@sealresearch/relay`](https://www.npmjs.com/package/@sealresearch/relay) is Relay 1 with a smaller surface
(`relay.predict(request)`), built on this package.

## Licence

MIT. The weights each model downloads have their own licence, listed in its model card.
