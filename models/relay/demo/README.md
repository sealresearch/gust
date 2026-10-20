# Relay sandbox

A browser playground for Relay 1: pick a simulated device, type (or say) a request, and watch Relay's calls drive it.
Everything runs in the page: the Relay C host and the Gust runtime compiled to WebAssembly (`host/`, `runtime/`),
the same code and output bytes as the native builds. Nothing is sent to a server.

```
./build.sh wasm && ./build.sh wasm-mt && ./build.sh wasm-simd && ./build.sh wasm-mt-simd   # out/relay*.wasm
node models/relay/demo/serve.mjs            # http://localhost:8787/  (--port N, --model DIR, --wasm DIR)
node models/relay/demo/test_env.mjs phone   # run an environment's examples through Relay in Node
```

The model folder defaults to `relay-model/`, the Relay 1 files from Hugging Face (`sealresearch/relay`). The page loads `relay.wasm` (relaxed SIMD) or `relay-simd.wasm` where relaxed SIMD is missing
(Safari), and the `-mt` builds for more than one thread. The server sets cross-origin isolation headers so the
threaded builds can use SharedArrayBuffer; without them the page falls back to one thread.

## What is in it

- **Environments** (`envs/*.mjs`): phone, smart home, robot vacuum, watch, AR glasses, car, laptop, many tools, plus
  **Extract** (messy text into typed fields, schema chosen by Relay) and **Benchmark samples** (60 Mobile Actions eval
  rows, CC BY 4.0, scored live against the gold calls).
- **Editable tools**: JSON-schema tools; edits apply on the next run. Optional `triggers` (regexes) restrict the
  request to the matched tools and force a call.
- **Act / ask / hold**: calls run when the router margin is at least 2 (or a trigger matched); below that the page
  asks first; a call that would leave a required value empty is held, and the page says which value is missing.
- **Why**: every argument is shown on the request text, with the value program that produced it (copy, number,
  datetime, enum…). Relay never writes values, so each one can be traced to your words or the schema.
- **Tool ranking**: the router score of every tool (tool logit minus the learned "none" logit).
- **Distractor tools**: add up to 40 tools from other environments to see accuracy and latency with long tool lists.
- **Threads**: 1-8 worker threads, bit-identical results; latency is shown per run.
- **Prefilter**: long tool lists are ranked first and the best 10 within 600 tool tokens are kept.
- **Mic**: the browser's own speech recognition (Chrome / Edge / Safari), not an on-device model.
- **Real side effects** (laptop, phone): speech synthesis, theme colour, new tabs, mail drafts, clipboard, timers.

## Environment module contract

`export default {key, label, subtitle, tagline, placeholder, tools, init(), render(state) -> SVG string (560x460),
apply(call, state, ctx) -> note, examples: [{title, sub, q}], help, mode?: 'extract', custom?: 'bench'}`;
`ctx` = `{render, sleep, speak, toast, setAccent, openUrl, mailto, clipboard}`.
