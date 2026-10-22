// @sealresearch/gust: the Gust model API in Node and in the browser, on WebAssembly. One interface for every model.
//
//   import { Model } from '@sealresearch/gust';
//   const model = await Model.load('relay-1');                 // downloads the model once, checks it, keeps it
//   model.run({ text: 'Show me the Eiffel Tower on the map', tools });
//   model.close();
//
// It is the same C code as the native libraries, the Python package and the sealpup command line, so the answers are bit for bit
// the same on every platform and for any thread count.
export { Model, Stream, Pcm, Image, models, describe, runtime } from './model.mjs';
export { GustError, configure } from './runtime.mjs';
export { registry } from './fetch.mjs';
