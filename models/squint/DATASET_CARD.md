# Squint data

Squint has no trained weights, so it has no training data. Its parts are hand-built algorithms; their settings (time
budgets, gates, the evidence threshold) were tuned on the webcam recordings below.

## Webcam recordings

Five recordings of one QR code shown on a phone screen (version 5, ECC level Q, a 30-digit payload), filmed with one
webcam at 1280 x 720 and 30 frames per second: 3,918 frames. They include fast shaking, a code moving into view, a
distant code and a close-up. They were used to develop and tune the scanner and to produce the benchmarks in
[`eval/results/`](eval/results/README.md), so none of them is a held-out test. They show a person and are not
published; the result files hold per-frame numbers only (read or not, payload correct, time).

## Generated safety fixtures

The safety suites run on generated codes: version and payload switches with blank, noise and stripe frames; sharp codes
followed by blurred codes with one digit changed; replays of synthetic switches that fooled an earlier verifier; and
160 moving, motion-blurred, over-exposed sequences in which the code switches or disappears. Exact code grids come from
Segno (BSD-3-Clause, see [NOTICE.md](NOTICE.md)). The generators live in the private training repository
(`models/squint/eval/`) and write the fixtures that `eval/safety-suites.mjs` and `eval/motion-switch-safety.mjs` read
from `out/squint/fixtures/`.

## Test sequences of the Gust port

The port to C is checked on 48 generated sequences (2,154 frames, deterministic, rendered by `tools/make_sequences.mjs` from
Segno-made module grids in `tests/fixtures/`): motion blur from light to extreme, curved and straight shake, a code entering
and leaving the frame, distant and close-up codes, versions 1 to 10 at all four error-correction levels, payloads that change
mid-motion, vanishing codes, junk frames, damage, glare and lighting gradients, other frame sizes (64 x 64 to 1920 x 1080),
clock restarts and resets. What the JavaScript release returned for them is in `tests/golden/`. They contain no photographs
and no people.

## Gaps

No phones as cameras, no other webcams, no printed codes, glossy surfaces, low light or real autofocus. A real-photo
test set of at least 500 untouched frames from three or more phones and a webcam is required before any release
([`eval/gates.json`](eval/gates.json)).

## Retired verifier

Earlier versions used a 27 KB INT8 verifier network, trained only on codes rendered on the fly (Segno grids with
perspective, rotation, gamma, noise, Gaussian defocus and straight or curved motion blur; negatives were the same image
paired with a payload one character different). The webcam recordings were never used to train it. It was retired in
favour of the evidence certificate (see [MODEL_CARD.md](MODEL_CARD.md)); its training code stays in the private training
repository.
