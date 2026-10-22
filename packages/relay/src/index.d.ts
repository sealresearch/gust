export interface Param {
  name: string;
  /** "string", "integer", "number", "boolean", "array" or "object". */
  type?: string;
  description?: string;
  required?: boolean;
  enum?: unknown[];
  default?: unknown;
}

export interface Tool {
  name: string;
  description?: string;
  params: Param[];
}

export interface Request {
  /** What the person said. */
  text: string;
  /** The tools the app offers. */
  tools: Tool[];
  kind?: string;
  /** `now` is the date and time Relay resolves "tomorrow" against, as "2026-10-18T09:00:00". */
  context?: { now?: string; [key: string]: unknown };
}

export interface Call {
  tool: string;
  args: Record<string, unknown>;
}

export interface ModelFile {
  readonly name: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface Model {
  readonly id: string;
  readonly repo: string;
  readonly revision: string;
  readonly files: {
    readonly encoder: ModelFile;
    readonly route: ModelFile;
    readonly anchor: ModelFile;
    readonly args: ModelFile;
    readonly tokenizer: ModelFile;
    readonly prefilter: ModelFile;
  };
}

export interface Progress {
  file: string;
  loaded: number;
  total: number;
  /** True when the file came from the cache. */
  cached?: boolean;
}

export interface LoadOptions {
  /** More than one needs worker threads (Node) or a cross-origin isolated page (browsers). Default 1. */
  threads?: number | 'auto';
  /** Rank long tool lists first and read only the best. Default true; downloads 2.1 MB more. */
  prefilter?: boolean;
  /** Read the model files from this folder (Node only) instead of downloading them. */
  dir?: string;
  /** Where to download from. Default "https://huggingface.co". */
  baseUrl?: string;
  /** Keep downloads: true for the default place, a folder path in Node, false for none. Default true. */
  cache?: boolean | string;
  onProgress?: (progress: Progress) => void;
  /** Host decoding fixes. Default true. */
  fixes?: boolean;
  /** How many tools the prefilter keeps. Default 10. */
  k?: number;
  /** The most tool tokens the prefilter keeps. Default 600. */
  budget?: number;
}

export const MODEL: Model;

export class Relay {
  readonly threads: number;
  readonly model: Model;
  readonly modelBytes: number;
  static load(options?: LoadOptions): Promise<Relay>;
  predict(request: Request): Call[];
  predictJSON(request: Request): string;
  explain(request: Request): unknown;
  trace(request: Request, options?: { operators?: boolean }): unknown;
  close(): void;
}
