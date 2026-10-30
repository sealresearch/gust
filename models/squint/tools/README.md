# Squint tools

Build tools for the hosts, and the tools behind the regression in [../tests](../tests): test sequences, and the reads the
JavaScript release of Squint returned for them.

| Tool | What it does |
| --- | --- |
| `build_hosts.sh TARGET [OUT]` | Compiles every host instance for a zig target (`build.sh` calls it for each platform) |
| `instances.py` | The host instances and the source files of each |
| `gen_names.py` | Writes `host/names_<instance>.h`, the symbol prefix of each instance |
| `gen_api.py` | Writes `host/squint_api.h`, the prototypes of the hosts' exported functions, from their sources |
| `gen_kernels.mjs` | Writes `host/squint_kernels.h`, the Gaussian kernels of the template blur, computed as V8 does |
| `make_sequences.mjs` | Renders the 48 test sequences (2,154 frames) into `out/squint-seq/`, deterministically |
| `seq2cmd.mjs` | Writes the `.cmd` file of each sequence for the C drivers |
| `reference_run.mjs` | Runs the JavaScript release over the sequences and writes `<name>.ref.jsonl` (what `tests/golden` holds) |
| `compare.mjs`, `compare_api.mjs` | Compare the C reader's output (driver, model API, WebAssembly) with the references, bit for bit |
| `wasm_run.mjs` | Runs a sequence through `@sealresearch/gust` on the WebAssembly module |
| `device_run.sh` | Runs the regression on an Android or Linux ARM64 device over adb: the same check, plus open time, time per frame and peak memory |
| `gen_switch_codes.py`, `switch_codes.json`, `render.mjs` | The QR grids and the renderer behind the sequences |

The headers are generated and committed. Regenerate them after changing a host: `python tools/gen_names.py ZIG`,
`python tools/gen_api.py`.

## The sequences

`node tools/make_sequences.mjs [--out DIR] [--only a,b] [--list]`. About 1 GB of `.gray` files (1280 x 720 frames are 0.92 MB
each). Generation is deterministic: seeded xoshiro128** (`render.mjs`), no `Math.random`, no `Date`; running it twice gives
byte-identical files. The module grids are in `tests/fixtures` (24 codes for each version 1 to 10 and level L, M, Q and H) plus
a few made with Segno (`switch_codes.json`).

Families: static sharp, straight and curved shake at several blur strengths, entering from the edge, distant and close codes,
lighting gradients, glare, noise and gamma ramps, perspective tilt, versions 1 to 10 (7 to 10 are not tracked), all four error
correction levels, payload switches (one digit, numeric, another version, toggling), vanish and reappear (up to a 3.3 s gap),
junk frames (blank, noise, stripes, gradient, checker, clutter, a forged code), module damage, occlusion, blur beyond
readability, resets, seed twins, irregular and restarting clocks, and other sizes (800 x 600, 320 x 240, 1000 x 562, 64 x 64,
640 x 479, and the illegal 641 x 480, 1920 x 1080 and 62 x 64).

`<name>.gray` holds `frames` frames of `width * height` bytes. `<name>.json` has `width`, `height`, `frames`, `times` (the frame
time in ms), `resets` (frame indices before which the reader is reset), `truth` (per frame: the payload physically shown, in
hex, or `null`), `visible`, `kinds`, `codes`, `seed` and the `sha256` of the `.gray`.

## The reference

`<name>.ref.jsonl` has one line per frame:
`{"i":0,"accepted":true,"reason":null,"path":"...","version":3,"payload":"hex","corners":[8 strings]}`. Corners are
`x0,y0,x1,y1,x2,y2,x3,y3` (top left, top right, bottom right, bottom left), each the 16 hex digit big-endian bit pattern of the
float64. A rejected frame has `accepted` false, a `reason`, and null for the rest.

`tests/golden` holds the reads the JavaScript release returned, so the regression does not need the release. To record them
again, `reference_run.mjs` needs that release (its `js/dist` and `package/`): the archive `squint-js-release-2026-10-23.tar.gz` in
the sandbox's `archive/` folder has it. Unpack it somewhere and set `SQUINT_REF` to that folder. The reference is called as
`Squint.load()` once per sequence (its warm-up included), then `squint.read(gray, width, height, {time, budgetMs: Infinity,
acquireBudgetMs: Infinity, trackedAcquireBudgetMs: Infinity})` for every frame, with `squint.reset()` where `resets` says.
With infinite budgets no clock value reaches a decision.

## What the JavaScript release does that a port has to copy

Found while building the set; each one cost a mismatch until the C did the same.

1. The warm-up that runs when the release loads (24 scans of a synthetic code, then 16 verifications) leaves state behind: the
   verifier's last contradiction measurement (`reset()` does not clear it) and the parity of the full-resolution stage inside
   the tracker's verifier host. With that state, a tracked code and a frame whose height is not a multiple of 16 make the
   scanner read the whole frame first. Without it, four sequences differ.
2. The tracker and the acquirer are separate instances with separate state; `reset()` clears the tracker and the tick, but not
   the acquirer's alternating lighting-normalized view.
3. Tracked state expires 3000 ms after the last fresh decode, and a time that goes backwards restarts the tracker.
4. `Math.hypot` is V8's (scaled, Kahan summed), not `sqrt(x * x + y * y)`; `Math.exp` is fdlibm's (the kernels are embedded
   instead); sorts are stable, so ties keep their order.
5. With the library's default time budgets results change on 10 of 48 sequences, so those budgets cannot be used for
   byte-exact checks.
