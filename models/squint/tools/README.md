# Squint port: test sequences and reference results

Tools that produce frame sequences and record what the reference JavaScript Squint
(`squint-release/models/squint/js/dist/index.js`) returns for each frame, so that the C plugin can be compared
byte for byte. Nothing here touches the reference or the C code.

## Commands

```
cd gust/models/squint/tools
node make_sequences.mjs            # writes out/squint-seq/<name>.gray, <name>.json, manifest.json   (about 40 s)
node reference_run.mjs             # writes out/squint-seq/<name>.ref.jsonl, summary.json, timing.json (about 17 s)
node view_frame.mjs <name> <frame> [out.png]   # dev helper: look at one frame
C:/Python314/python.exe gen_switch_codes.py    # only to rebuild switch_codes.json (needs `pip install segno`)
```

Both scripts lower their own process priority (`os.setPriority`, idle class on Windows). Options:
`make_sequences.mjs [--out DIR] [--only a,b] [--list]`;
`reference_run.mjs [--dir DIR] [--only a,b] [--suffix S] [--scalar] [--nowarm] [--default-budgets]`.
The reference location (only `reference_run.mjs` needs it: the JavaScript release, which wrote `tests/golden/`) can be changed with the `SQUINT_REF` environment variable.

Module grids come from `tests/fixtures/` (`clean-corpus.json`: 24 codes for each version 1 to 10 and level L/M/Q/H, byte,
numeric and alphanumeric; `demo-codes.json`) plus a few Segno codes in
`switch_codes.json` (pairs that differ by one digit, same version, level and mask).

Generation is deterministic: seeded xoshiro128** (`render.mjs`), no `Math.random`, no `Date`. Re-running gives
byte-identical `.gray` files (checked). `<name>.json` has a `sha256` of the `.gray`.

## Formats

`<name>.gray`: `frames` frames of `width * height` bytes, row-major, one byte per pixel, concatenated.

`<name>.json`:

| field | meaning |
| --- | --- |
| `name`, `width`, `height`, `frames` | as named |
| `times` | frame time in ms per frame, passed as `time`. Mostly `i * 33.3333`; `timing_irregular` and `clock_restart` differ on purpose |
| `resets` | frame indices BEFORE which `squint.reset()` is called |
| `truth` | per frame: lowercase hex of the payload bytes physically shown, or `null` (no code, a forged code, or less than 25 percent of the code inside the frame) |
| `visible` | per frame: fraction of the code area inside the frame (0 to 1) |
| `kinds` | per frame: `code`, `empty`, `forged`, or `junk:<kind>` |
| `codes` | the QR grids used: id, version, level, payloadHex |
| `seed`, `description`, `sha256` | |

`<name>.ref.jsonl`, one line per frame:
`{"i":0,"accepted":true,"reason":null,"path":"...","version":3,"payload":"hex","corners":[8 strings]}`.
Corners are `x0,y0,x1,y1,x2,y2,x3,y3` (top-left, top-right, bottom-right, bottom-left as returned), each the 16 hex
digit big-endian bit pattern of the float64. For a rejected frame `accepted` is false, `reason` is the string the JS
returned (`decode-failed`, `unsupported-size`, ...), and `path`, `version`, `payload`, `corners` are null. An exception
inside `read()` would be recorded as `reason: "exception: ..."` (none occurred).

`summary.json`: per sequence `frames`, `accepted`, `wrong` (accepted payload differs from a non-null truth),
`phantom` (accepted while truth is null), `exceptions`, `paths` (accepted frames per `path` label), `reasons`
(rejected frames per reason), `wrongFrames`, `phantomFrames`; plus `totals`. `timing.json` is kept apart because
it is not deterministic.

## How the reference is called

`Squint.load()` once per sequence (default warm-up included), then for every frame
`await squint.read(gray, width, height, {time, budgetMs: Infinity, acquireBudgetMs: Infinity, trackedAcquireBudgetMs: Infinity})`,
with `squint.reset()` where `resets` says. With infinite budgets no wall-clock value reaches a decision (see below).

## Sequences

48 sequences, 2154 frames, about 1.0 GB of `.gray` (1280x720 frames are 0.92 MB each). `--list` prints them.
Families: static sharp, straight and curved shake at several blur strengths, entering from the edge, distant and
close-up codes, lighting gradient and vignette, glare, noise and gamma ramps, perspective tilt (flat to 65 degrees),
phone-screen palette, versions 1 to 10 (7 to 10 are not tracked), error correction L/M/Q/H, payload switches
(one digit, numeric, different version, toggling, static), vanish and reappear (including a 3.3 s gap), junk frames
(blank, noise, stripes, gradient, checker, clutter, a forged code), module damage ramp, occlusion, blur beyond
readability, defocus ramp, reset in the middle, seed twins, irregular and restarting clocks, and sizes: 800x600,
320x240, 1000x562, 64x64, 640x479, plus illegal 641x480, 1920x1080 and 62x64.

## Things in the JavaScript that are state or order dependent

Found while building the set (details in the hand-off report):

1. `VerifiedPayloadReader.contradiction` is not cleared by `reset()`, and the warm-up leaves it at
   `{noSample: true}`. `CPUScanPipeline.scan` then takes the "global reader first" branch whenever a tracked state exists
   but the quick read did not run (for example height not a multiple of 16), even with `noGlobal: true`. That global
   read runs in the tracker's own verify-host instance, with the same label `Squint acquisition`.
   Running with `warm: false` changes results in 4 sequences (`--nowarm`).
2. There are two verify-host wasm instances (tracker and `Acquirer`), each with its own `sa_acquire` statics
   (`fullTick` parity, which decides whether the full-resolution stage runs). `Squint.read` only warms the tracker's.
3. `Acquirer.frameCounter` (the alternating lighting-normalized view at 1280x720) and `Squint.tick`
   (every third frame while tracked) are state that `reset()` does not all clear: `frameCounter` survives `reset()`.
4. Tracker state expires 3000 ms after the last fresh decode, and a time that goes backward restarts the tracker.
5. With the library's default budgets (`--default-budgets`) results change on 10 of 48 sequences on this machine, so
   those budgets must not be used for byte-exact checks.
6. SIMD and scalar wasm builds gave identical output (`--scalar`).
7. `Math.exp` is used (gaussian targets in `verified-payload-reader.ts`), and sorts rely on JavaScript's stable
   sort; ties must keep input order in C.
