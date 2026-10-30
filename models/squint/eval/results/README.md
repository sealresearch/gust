# Squint results

All on a Ryzen 7 7800X3D (Windows 11, no GPU). The webcam recordings are five clips of one code on a phone screen, from one webcam
(3,918 frames at 1280 x 720); all five were used while developing Squint, so these are development results, not
held-out ones. The expected payload is read by the scorer only after each scan returns; it never reaches the scanner.

| File | What it measures |
| --- | --- |
| `port-parity.json` | This package against the previous Gust package, every frame, unlimited budgets: 3,918 of 3,918 frames identical, 2,744 read, 0 wrong |
| `safety-suites.json` | Verify, blurred, revealed and motion switch suites: 0 wrong, 0 stale; identical to the previous package |
| `live-demo.json` | The browser demo with a fake camera (Chromium, 20 s runs), against the previous demo on the same video, and the generated test codes |
| `competitor-bench.json` | Every frame read on its own (no tracking) by Squint's acquisition (native, 9 ms budget), zxing-cpp, ZBar and OpenCV WeChat QR |
| `tracking-bench-summary.json` | The 3,918 frames in an older Chromium harness, in order, with three scanner setups (see below) |
| `native-resources.json` | CPU time and memory of each reader in its own process on 300 motion frames |

`competitor-bench.json`, `tracking-bench-summary.json` and `native-resources.json` were measured on 2026-10-09 with the
previous build of the same scanner; `port-parity.json` shows this package reads exactly the same frames. The tracking
benchmark is an older Chromium harness: `og` is the scanner without the evidence preset, `track` one worker with
acquisition inside the tracker's 12.5 ms budget, and `full` a simulated second worker (both instances run in turn and
the slower time counts), so its absolute numbers are not the demo's.

On 2026-10-21 the unused INT8 verifier (`squint-verify.gust`, which the shipped settings never ran) was removed from the
package and the hosts. Re-checked afterwards: all 3,918 frames give the same read, payload, path and corners, and every
safety-suite count, including the per-path counts, is unchanged.
