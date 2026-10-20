// The Gust WebAssembly binding (MIT, no dependencies).
/** Small browser/Node binding around the Gust WebAssembly runtime. No dependencies. */
const ERRORS = [
    'ok',
    'invalid format',
    'checksum mismatch',
    'invalid shape',
    'invalid memory plan',
    'unsupported operator',
    'invalid quantization',
    'invalid graph',
    'execution budget exceeded',
];
const MAX_ARTIFACT_BYTES = 64 * 1024 * 1024;
const align = (n) => Math.ceil(n / 16) * 16;
/** The last fixed float32-graph-v1 operator; 38 to 45 belong to float32-decoder-v1. */
const LAST_FLOAT32_OPERATOR = 37;
const TENSOR_STATE = 8;
/**
 * True when a float32 artifact's executable tables name a state tensor (flag 8) or an operator above 37, i.e. it
 * needs float32-decoder-v1. A table that does not fit inside the artifact counts as false: gust_init reports it.
 */
function usesDecoder(data, total) {
    const tensors = data.getUint32(20, true);
    const operators = data.getUint32(24, true);
    const tensorTable = data.getUint32(28, true);
    const operatorTable = data.getUint32(32, true);
    if (tensorTable + 32 * tensors > total || operatorTable + 80 * operators > total)
        return false;
    for (let i = 0; i < tensors; i++)
        if (data.getUint32(tensorTable + 32 * i + 24, true) & TENSOR_STATE)
            return true;
    for (let i = 0; i < operators; i++)
        if (data.getUint32(operatorTable + 80 * i, true) > LAST_FLOAT32_OPERATOR)
            return true;
    return false;
}
/** A WebAssembly instance of the runtime with a first-fit allocator over its fixed memory. */
export class GustEngine {
    /** Instantiates the runtime from a URL (fetched) or from its bytes. */
    static async create(wasm) {
        const bytes = wasm instanceof URL || typeof wasm === 'string'
            ? await (async () => {
                const r = await fetch(wasm);
                if (!r.ok)
                    throw new Error(`WASM fetch: ${r.status}`);
                return r.arrayBuffer();
            })()
            : wasm;
        const { instance } = await WebAssembly.instantiate(bytes, {});
        return new GustEngine(instance);
    }
    constructor(instance) {
        this.api = instance.exports;
        const capabilities = this.api.gust_capabilities();
        this.capabilities = {
            simd: !!(capabilities & 1),
            gatherU16: !!(capabilities & 2),
            weightedGatherU16: !!(capabilities & 4),
            attention: !!(capabilities & 32),
            validFrames: !!(capabilities & 64),
            normLutMul: !!(capabilities & 128),
            packedConvWeights: !!(capabilities & 256),
            float32: !!(capabilities & 512),
            float32Decoder: !!(capabilities & 1024) && typeof this.api.gust_reset === 'function',
        };
        this.memory = this.api.memory;
        const heapBase = align(this.api.__heap_base.value);
        this.free = [{ offset: heapBase, size: this.memory.buffer.byteLength - heapBase }];
    }
    /** Reserves `bytes` (rounded up to 16) of runtime memory. */
    allocate(bytes) {
        const size = align(bytes);
        const i = this.free.findIndex(b => b.size >= size);
        if (i < 0)
            throw new Error(`Gust's fixed ${this.memory.buffer.byteLength / 1048576} MiB WASM memory budget is exhausted`);
        const block = this.free[i];
        const offset = block.offset;
        block.offset += size;
        block.size -= size;
        if (!block.size)
            this.free.splice(i, 1);
        return { offset, size };
    }
    /** Returns a region to the allocator and merges adjacent free blocks. */
    release(region) {
        this.free.push(region);
        this.free.sort((a, b) => a.offset - b.offset);
        for (let i = 0; i < this.free.length - 1;) {
            const a = this.free[i];
            const b = this.free[i + 1];
            if (a.offset + a.size === b.offset) {
                a.size += b.size;
                this.free.splice(i + 1, 1);
            }
            else
                i++;
        }
    }
    /** Validates an artifact and allocates its context and activation arena. */
    load(artifact) {
        return new GustModel(this, artifact);
    }
}
/** A loaded artifact. Calls must be serialized; outputs are borrowed views unless copied. */
export class GustModel {
    constructor(engine, artifact) {
        this.regions = [];
        this.closed = false;
        this.blob = 0;
        this.ctx = 0;
        this.arena = 0;
        this.tensors = [];
        this.inputs = [];
        this.outputs = [];
        /** True for a float32 graph (header flags 1): inputs and outputs are Float32Arrays, see {@link runFloat}. */
        this.float = false;
        this.engine = engine;
        const bytes = artifact instanceof Uint8Array ? artifact : new Uint8Array(artifact);
        if (bytes.byteLength < 68 || bytes.byteLength > MAX_ARTIFACT_BYTES)
            throw new Error('invalid artifact size');
        const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const arenaBytes = data.getUint32(40, true);
        if (!arenaBytes || arenaBytes > MAX_ARTIFACT_BYTES)
            throw new Error('invalid arena size');
        // Header flags 1 marks a float32 graph; only a runtime with capability float32-graph-v1 may execute it.
        const flags = data.getUint32(52, true);
        if (flags === 1 && !engine.capabilities.float32)
            throw new Error('artifact requires capability float32-graph-v1, which this Gust WebAssembly build lacks');
        this.float = flags === 1;
        // State tensors (flag 8) and operators above 37 need float32-decoder-v1; refuse before allocating anything.
        const needsDecoder = this.float && usesDecoder(data, bytes.byteLength);
        if (needsDecoder && !engine.capabilities.float32Decoder)
            throw new Error('artifact requires capability float32-decoder-v1, which this Gust WebAssembly build lacks');
        const allocate = (n) => {
            const r = engine.allocate(n);
            this.regions.push(r);
            return r.offset;
        };
        try {
            this.blob = allocate(bytes.byteLength);
            this.ctx = allocate(engine.api.gust_context_bytes());
            this.arena = allocate(arenaBytes);
            new Uint8Array(engine.memory.buffer, this.blob, bytes.byteLength).set(bytes);
            const result = engine.api.gust_init(this.ctx, this.blob, bytes.byteLength, this.arena, arenaBytes);
            if (result)
                throw new Error(ERRORS[result] ?? `engine error ${result}`);
            const length = data.getUint32(bytes.byteLength - 4, true);
            if (length > bytes.byteLength - data.getUint32(36, true) - 4)
                throw new Error('invalid manifest length');
            this.manifest = JSON.parse(new TextDecoder().decode(bytes.subarray(bytes.byteLength - 4 - length, bytes.byteLength - 4)));
            const count = data.getUint32(20, true);
            if (!Array.isArray(this.manifest.tensors) || this.manifest.tensors.length !== count)
                throw new Error('manifest tensor count mismatch');
            if (this.float !== (this.manifest.dtype === 'f32'))
                throw new Error(this.float
                    ? 'float32 artifact whose manifest does not declare dtype "f32"'
                    : 'manifest declares dtype "f32" but the artifact is not a float32 graph');
            if (this.float) {
                const declared = this.manifest.required_capabilities;
                if ((Array.isArray(declared) && declared.includes('float32-decoder-v1')) !== needsDecoder)
                    throw new Error('manifest float32-decoder-v1 declaration disagrees with the executable tables');
            }
            this.tensors = this.manifest.tensors.map((t, i) => {
                const base = 64 + i * 32;
                const shape = [0, 4, 8].map(off => data.getUint32(base + off, true));
                const size = data.getUint32(base + 16, true);
                const flags = data.getUint32(base + 24, true);
                const scale = data.getFloat32(base + 20, true);
                if (typeof t.name !== 'string' ||
                    !t.name ||
                    !Array.isArray(t.shape) ||
                    shape.some((n, j) => n !== t.shape[j]) ||
                    size !== t.bytes ||
                    flags !== t.flags ||
                    scale !== t.scale ||
                    (this.float && t.dtype !== 'f32'))
                    throw new Error('manifest disagrees with executable tensor table');
                const pointer = engine.api.gust_tensor(this.ctx, i);
                return {
                    name: t.name,
                    shape,
                    size,
                    flags,
                    scale,
                    index: i,
                    view: new Int8Array(engine.memory.buffer, pointer, size),
                    dtype: this.float ? 'f32' : 'int8',
                    ...(this.float ? { f32: new Float32Array(engine.memory.buffer, pointer, size / 4) } : {}),
                };
            });
            if (new Set(this.tensors.map(t => t.name)).size !== count)
                throw new Error('duplicate tensor names');
            this.inputs = this.tensors.filter(t => t.flags & 1);
            this.outputs = this.tensors.filter(t => t.flags & 2);
        }
        catch (error) {
            this.close();
            throw error;
        }
    }
    /** Runs on INT8 inputs. Borrowed outputs are overwritten by the next run or close. */
    run(inputs, { copyOutputs = false, validFrames = 0 } = {}) {
        if (this.closed)
            throw new Error('model is closed');
        if (this.float)
            throw new Error('this is a float32 graph: call runFloat() with Float32Array inputs');
        if (Object.keys(inputs).length !== this.inputs.length)
            throw new Error('input keys mismatch');
        for (const t of this.inputs) {
            const input = inputs[t.name];
            if (!(input instanceof Int8Array) || input.length !== t.size)
                throw new Error(`${t.name} requires Int8Array(${t.size})`);
            t.view.set(input);
        }
        return this.runPrepared({ copyOutputs, validFrames });
    }
    /** Runs when the inputs have already been written by a native adapter or into input views. */
    runPrepared({ copyOutputs = false, validFrames = 0 } = {}) {
        if (this.closed)
            throw new Error('model is closed');
        if (this.float)
            throw new Error('this is a float32 graph: use runFloat() (outputs are Float32Arrays)');
        this.execute(validFrames);
        return Object.fromEntries(this.outputs.map(t => [t.name, copyOutputs ? t.view.slice() : t.view]));
    }
    /**
     * Runs a float32 graph (capability `float32`). Every input is a Float32Array with exactly the tensor's element
     * count (height times channels); `validFrames` counts the real rows of the first input (samples for a waveform),
     * 0 or the full height runs everything. The valid rows of every output equal those of a graph compiled for exactly
     * that many rows, bit for bit, the rest are zero, and the input and its padding are not modified. Inputs must be
     * finite. Borrowed outputs are overwritten by the next run or close.
     */
    runFloat(inputs, { copyOutputs = false, validFrames = 0 } = {}) {
        if (this.closed)
            throw new Error('model is closed');
        if (!this.float)
            throw new Error('this is an INT8 graph: call run() with Int8Array inputs');
        if (Object.keys(inputs).length !== this.inputs.length)
            throw new Error('input keys mismatch');
        for (const t of this.inputs) {
            const input = inputs[t.name];
            if (!(input instanceof Float32Array) || input.length * 4 !== t.size)
                throw new Error(`${t.name} requires Float32Array(${t.size / 4})`);
        }
        for (const t of this.inputs)
            t.f32.set(inputs[t.name]);
        this.execute(validFrames);
        return Object.fromEntries(this.outputs.map(t => [t.name, copyOutputs ? t.f32.slice() : t.f32]));
    }
    /**
     * Runs a float32 graph on the inputs that are already in its input views and returns the outputs. Nothing is copied
     * in, so a decoder step rewrites only its scalars (token, position, count) while the static cross-attention arrays
     * stay where they were written once per utterance. Every bound is still checked by the runtime before it writes
     * anything: an out-of-range token, position or count, or a position that is not the state counter, is refused with
     * `invalid shape` and changes no byte. Borrowed outputs are overwritten by the next run or close.
     */
    runPreparedFloat({ copyOutputs = false, validFrames = 0 } = {}) {
        if (this.closed)
            throw new Error('model is closed');
        if (!this.float)
            throw new Error('this is an INT8 graph: use runPrepared() (outputs are Int8Arrays)');
        this.execute(validFrames);
        return Object.fromEntries(this.outputs.map(t => [t.name, copyOutputs ? t.f32.slice() : t.f32]));
    }
    /** The tensor called `name` (inputs, outputs, state and temporaries alike); its views are borrowed. */
    tensor(name) {
        const found = this.tensors.find(t => t.name === name);
        if (!found)
            throw new Error(`no tensor named ${name}`);
        return found;
    }
    /** State tensors (flag 8: key/value caches, the position counter); live borrowed views, never inputs or outputs. */
    get state() {
        return this.tensors.filter(t => t.flags & TENSOR_STATE);
    }
    /**
     * Clears every state tensor (`gust_reset`): caches and the position counter become zero; the static inputs and the
     * model stay usable. Needs a float32 graph and a build with capability `float32Decoder`.
     */
    reset() {
        if (this.closed)
            throw new Error('model is closed');
        if (!this.float)
            throw new Error('this is an INT8 graph: it has no state to reset');
        const reset = this.engine.api.gust_reset;
        if (!this.engine.capabilities.float32Decoder || typeof reset !== 'function')
            throw new Error('this Gust WebAssembly build lacks float32-decoder-v1 (gust_reset)');
        const result = reset.call(this.engine.api, this.ctx);
        if (result)
            throw new Error(ERRORS[result] ?? 'reset failed');
    }
    execute(validFrames) {
        const rows = this.tensors[0]?.shape[0] ?? 0;
        if (!Number.isInteger(validFrames) || validFrames < 0 || validFrames > rows)
            throw new RangeError(`validFrames ${validFrames} is outside 0..${rows}`);
        const partial = validFrames > 0 && validFrames < rows;
        const runValid = this.engine.api.gust_run_valid;
        if (partial && !runValid)
            throw new Error('this Gust WebAssembly build has no gust_run_valid (use gust-asr.wasm)');
        const start = performance.now();
        const result = partial
            ? runValid.call(this.engine.api, this.ctx, validFrames)
            : this.engine.api.gust_run(this.ctx);
        this.lastKernelMs = performance.now() - start;
        if (result)
            throw new Error(ERRORS[result] ?? 'inference failed');
    }
    /** Releases the model's memory. Safe to call more than once. */
    close() {
        if (this.closed)
            return;
        this.closed = true;
        for (const r of this.regions)
            this.engine.release(r);
        this.regions = [];
    }
}
