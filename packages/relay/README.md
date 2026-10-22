# @gust/relay

[Relay 1](https://seal.computer/research/relay-1) in Node and in the browser. Relay turns what a person said and the
tools an app offers into the calls to make, or into no call when nothing fits. It is 22 million parameters in 11.5 MB,
it runs on the CPU through the [Gust](https://github.com/sealresearch/gust) WebAssembly runtime, and nothing leaves the
device.

```sh
npm install @gust/relay
```

```js
import { Relay } from '@gust/relay';

const relay = await Relay.load();   // downloads the model the first time (11.5 MB, 14 MB with the prefilter), then keeps it

const tools = [
  { name: 'show_map', description: 'Shows a location on the map.',
    params: [{ name: 'query', type: 'string', description: 'The place to show.', required: true }] },
  { name: 'turn_on_flashlight', description: 'Turns the flashlight on.', params: [] },
];

relay.predict({ text: 'Show me the Eiffel Tower on the map', tools });
// [{ tool: 'show_map', args: { query: 'Eiffel Tower' } }]

relay.predict({ text: 'Tell me a joke', tools });
// []   no tool fits, so no call

relay.close();
```

Relay reads the request and the tool schemas together and points at the answer, so every value it returns comes from the
request or the schema. It cannot write text that is not in the request, such as a composed email body. The
[write-up](https://seal.computer/research/relay-1) has the method, the benchmarks and the limits.

## A request

```ts
{
  text: string,                       // what the person said
  tools: [{                           // what the app offers
    name: string,
    description?: string,
    params: [{ name, type?, description?, required?, enum? }],
  }],
  context?: { now?: string },         // "2026-10-18T09:00:00": what "tomorrow" means
}
```

`predict` returns `[{ tool, args }]`. `explain` also returns, for each argument, how Relay got it: the value program, the
words of the request it came from and its confidence. Types are included.

## Options

`Relay.load(options)`:

| Option | Default | |
|---|---|---|
| `threads` | `1` | `1` to `16`, or `'auto'`. More threads is faster (about 39 ms a request on 8 threads, in WebAssembly). The calls are identical for every count. In a browser, more than one thread needs a [cross-origin isolated](https://developer.mozilla.org/docs/Web/API/Window/crossOriginIsolated) page, otherwise it runs on one. |
| `prefilter` | `true` | Rank a long tool list first and read only the best. Downloads 2.1 MB more. |
| `dir` | | Node only. Read the model files from this folder instead of downloading them. |
| `baseUrl` | `https://huggingface.co` | Where to download from, for a mirror. |
| `cache` | `true` | Keep downloads. In Node that is `$GUST_CACHE`, or the platform's cache folder, or a path you pass. In a browser it is the Cache API. `false` downloads every time. |
| `onProgress` | | Called with `{ file, loaded, total, cached? }`. |

Every file is checked against a SHA-256 checksum built into the package before it is used, so a damaged or changed download
is refused. The package pins one revision of [`sealresearch/relay`](https://huggingface.co/sealresearch/relay).

## Same answers everywhere

The runtime is the same C code as the native builds, and Gust gives the same output bits on x86-64, ARM64 and WebAssembly,
for any number of threads. The calls this package returns are the ones the `sealpup` command line returns.

## Licences

The code is MIT. The Relay weights are CC BY 4.0 and are downloaded separately; their licence and the attributions of the
data they were trained on are in the
[model card](https://huggingface.co/sealresearch/relay).
