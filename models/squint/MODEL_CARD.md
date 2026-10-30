# Squint

A QR code reader for shaky hands, motion blur and poor focus. It runs on the CPU, locally, with its own decoder: no
third-party reader, no GPU, no cloud. It has no trained weights: every part is a hand-built algorithm in C, built into the
Gust model library as the model `squint-1`.

**Status: research.** Its release gates are not met: there is no untouched test set and no real-photo study across phones and
webcams yet.

## How it works

| Part | What it does |
| --- | --- |
| Acquisition | Finds the finder patterns in the full frame (inferring a third from two), fits the grid, tries the versions the finder spacing allows and decodes with Squint's own Reed-Solomon decoder; while no code is tracked, alternate 1280 x 720 frames use a lighting-normalized view |
| Tracking | Follows a read code from frame to frame (optical flow and registration), re-samples its module grid in the current frame and first tries to decode it there afresh. Codes up to version 6 are tracked |
| Evidence certificate | When the tracked code does not decode afresh: fits a blur, defocus and camera-response model to the current frame and accepts the tracked payload only if every alternative codeword (one symbol, several symbols or a random codeword different) is at least e^20 times less likely. A frame that fails returns no read; the evidence for one code never carries over to another |
| Decoder | QR versions 1-10, error correction L/M/Q/H, numeric, alphanumeric and byte data; every Reed-Solomon block must correct with at least two parity symbols to spare, and the re-encoded symbol must agree with at least 88% of the sampled modules |

## Intended use

Reading QR codes from a live camera or still images, on a device or in a browser, when the code is moving or slightly out of
focus, with everything kept on the device. Treat it as a research component, not a drop-in production scanner.

## Performance

The numbers below were measured on the JavaScript and WebAssembly release of Squint. The Gust model in this repository is a
port of that release to C and returns the same reads for the same frames, bit for bit (see "The port" below).

**Five webcam recordings of one code shown on a phone screen** (3,918 frames at 1280 x 720, one webcam, all used while
developing Squint). Ryzen 7 7800X3D, one CPU thread per worker, no GPU.

Single frames, against other readers (each frame on its own, no tracking; Squint's acquisition as a native C build, zxing-cpp
and ZBar set to QR only, wall-clock time per frame):

| Reader | Frames read | Wrong payloads | Median time per frame | Slowest 5% of frames |
| --- | --- | --- | --- | --- |
| Squint acquisition (9 ms budget) | 931 | 0 | 3.0-3.8 ms | about 9.1 ms |
| Squint acquisition (no budget) | 994 | 0 | 2.8-3.1 ms | 10-20 ms |
| WeChat QR (OpenCV, CNN + super-resolution) | 909 | 0 | 7-19 ms | 34-48 ms |
| zxing-cpp | 454 | 0 | 2.2-2.4 ms | 2.4-3.9 ms |
| ZBar | 287 | 0 | 11-12 ms | 14-17 ms |

With tracking, 2,744 of the 3,918 frames are read (70%) with zero wrong payloads, against 994 when the acquisition reads every
frame on its own, both with no time limit. Zero wrong payloads is a finite count on development material, not a guarantee.

**On a tablet** (Lenovo TB373FU, in an Android app, the WebAssembly release): Squint needs about 3 frames of 12-16 ms each;
the first good read usually lands on the first or second frame, and the app waits for a second frame to agree before it
accepts. Tapping scan until the first read takes 0.2-0.5 s, almost all of it the camera turning on. Keeping Squint loaded
holds about 40 MB.

## The port

The Gust model is not a new algorithm. It is the same C hosts the WebAssembly release ran, with the JavaScript that drove
them (about 2,000 lines) rewritten in C, so that Squint runs on the Gust runtime like every other model: one library, one
model API, the same code on x86-64, ARM64 and WebAssembly.

Checked by [tests/run.sh](tests/run.sh): 48 generated sequences, 2,154 frames (blur from light to extreme, curved and
straight shake, entering and leaving the frame, distant and close codes, versions 1-10 at all four error-correction levels,
payloads that change mid-motion, vanishing codes, junk frames, damage, glare and lighting gradients, frame sizes from 64 x 64
to 1920 x 1080, clock restarts and resets). The JavaScript release read 1,721 of them; the C port returns the same read for
every frame, down to the last bit of every corner coordinate, through the C driver, through the Gust model API (`run` and
streams), through the shared library and the Python package, and in WebAssembly through the JavaScript package. The golden outputs are in [tests/golden](tests/golden); the
generator is deterministic. The port reproduces a warm-up the JavaScript ran when it loaded, which leaves state in the hosts that
later reads depend on.

What the port cannot show: it matches the release on generated frames, not on the real recordings, which no longer exist, so
the figures above are the release's and have not been re-measured with the port. Native C reads a 1280 x 720 frame in 2-9 ms on
average on the desktop above with no budgets (up to 29 ms for the slowest frame in these sequences).

## Limitations

- Not evaluated on an untouched test set, and not validated with real photos from a range of phones and webcams. All
  recordings come from one webcam and were used during development. By hand on a tablet, with printed codes, in low light and
  on glossy surfaces, Squint read well, but we saved no frames and counted nothing, so that is an impression, not a result.
- Not compared with commercial SDKs (Dynamsoft, Scandit) or the readers built into iOS and Android. Timings are desktop CPU
  and one tablet.
- One code per frame. ECI and kanji modes, empty payloads, Micro QR and versions above 10 are not supported (a code using ECI
  or kanji is not read at all); tracking covers versions 1-6. Frames up to 1280 x 720.
- After a code changes, only about a fifth of the frames are read until the new code decodes afresh. That is deliberate.
- One reader per process, 33 MB of fixed memory.
- Decoded payloads are untrusted input.

## Data and licence

Squint has no trained weights. The material it was developed and measured on is described in
[DATASET_CARD.md](DATASET_CARD.md). Code is MIT; the QR tables come from Segno (BSD-3-Clause), see [NOTICE.md](NOTICE.md).

## History

An earlier Squint accepted tracked codes with a 27 KB INT8 verifier network, trained on rendered codes only. On the
motion-switch suite it returned the previous code's payload 301 times after the code had changed, so it was replaced by
the evidence certificate, which returns none. The network is no longer part of Squint.
