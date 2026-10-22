export type Status = 'ok' | 'argument' | 'unknown_model' | 'files' | 'open' | 'input' | 'run' | 'unsupported' | 'memory' | 'state';

/** A failure reported by the module. `status` names it. */
export class GustError extends Error {
  readonly code: number;
  readonly status: Status;
}

/** Audio: interleaved samples with their rate and channel count. */
export class Pcm {
  /** `samples` is a Float32Array, an Int16Array, an array of numbers (float32), or bytes together with `dtype`. */
  constructor(samples: Float32Array | Int16Array | number[] | ArrayBuffer | Uint8Array, rate: number, channels?: number, dtype?: 'float32' | 'int16');
  readonly samples: Float32Array | Int16Array;
  readonly rate: number;
  readonly channels: number;
  readonly dtype: 'float32' | 'int16';
  readonly length: number;
}

/** A frame. `pixels` can be `ImageData.data`. */
export class Image {
  constructor(pixels: Uint8Array | Uint8ClampedArray | ArrayBuffer, width: number, height: number, format?: 'gray8' | 'rgba8');
  readonly pixels: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  readonly format: 'gray8' | 'rgba8';
}

/** What goes into a model: text, bytes, audio, an image, or anything JSON can hold. */
export type Input = string | Uint8Array | ArrayBuffer | Pcm | Image | object | number | boolean | null;

export interface Progress {
  file: string;
  loaded: number;
  total: number;
  /** True when the file came from the cache. */
  cached?: boolean;
}

export interface LoadOptions {
  /** What the model's description lists under options.open. */
  options?: Record<string, unknown>;
  /** Operator threads, 1 to 16 or 'auto'. More than one needs SharedArrayBuffer (Node, or a cross-origin isolated page). Default 1. */
  threads?: number | 'auto';
  /** Read the files from this folder (Node only) instead of downloading them. */
  directory?: string;
  /** Keep downloads: true for the default place, a folder path in Node, false for none. Default true. */
  cache?: boolean | string;
  /** Default "https://huggingface.co". */
  baseUrl?: string;
  /** Also fetch the files the model lists as optional. Default true. */
  optional?: boolean;
  /** Check files read from `directory` against the registry. Default true. */
  verify?: boolean;
  /** The Hugging Face revision to download, instead of the registry's. */
  revision?: string;
  onProgress?: (progress: Progress) => void;
}

export interface RunOptions {
  /** Options for this call; what a model accepts is in `model.info.options.run`. */
  options?: Record<string, unknown>;
  /** Keep JSON outputs as the text the model wrote instead of parsing them. */
  raw?: boolean;
}

type Bytes = Uint8Array | ArrayBuffer;

export class Model {
  readonly id: string;
  /** What the model says about itself: what it takes, gives and accepts. */
  readonly info: Record<string, any>;
  /** The number of threads the operators run on. */
  readonly threads: number;
  /** Bytes the loaded model holds (0 when it does not say). */
  readonly memoryBytes: number;

  /** Finds the model's files (directory, cache, or download checked against the registry), then opens it. */
  static load(id: string, options?: LoadOptions): Promise<Model>;
  /** Opens a model from files you already have: an object of name to bytes, a Map, or a list of [name, bytes]. */
  static open(id: string, files?: Record<string, Bytes> | Map<string, Bytes> | [string, Bytes][], options?: { options?: Record<string, unknown>; threads?: number | 'auto' }): Promise<Model>;

  /** One call: one input in, the output out (a list when the model gives several). */
  run(input: Input, options?: RunOptions): any;
  /** For a model that takes several inputs at once. Always returns a list. */
  runAll(inputs: Input[], options?: RunOptions): any[];
  /** For models that work on live input. */
  stream(options?: Record<string, unknown>): Stream;
  setThreads(threads: number | 'auto'): Promise<number>;
  /** Frees the model and, when it was the last one, the worker threads. */
  close(): void;
  [Symbol.dispose](): void;
}

export class Stream {
  readonly model: Model;
  /** True once the stream is finished and drained. */
  readonly done: boolean;
  push(...inputs: Input[]): void;
  /** The outputs that are ready now (possibly none). */
  pull(options?: { raw?: boolean }): any[];
  finish(): void;
  /** After finish(): every remaining output until the stream is done. */
  drain(options?: { raw?: boolean }): Generator<any>;
  close(): void;
  [Symbol.dispose](): void;
}

/** The ids of the models in this build. */
export function models(): Promise<string[]>;
/** What a model is, takes and gives, before it is opened. */
export function describe(id: string): Promise<Record<string, any>>;
/** The instruction set this build runs ("wasm32 SIMD"). */
export function runtime(): Promise<string>;
/** Points the package at its WebAssembly modules, for a page that serves them elsewhere. Each is a URL, a path or the bytes. */
export function configure(options: { wasm?: { st?: string | URL | Uint8Array; mt?: string | URL | Uint8Array } }): void;

export interface RegistryFile { name: string; size: number; sha256: string }
export interface RegistryModel { name: string; task: string; repo: string; revision: string; files: RegistryFile[]; [key: string]: unknown }
export const registry: { models: Record<string, RegistryModel>; aliases?: Record<string, string> };
