# Third-party notices

The code in this repository is MIT, copyright Seal (see LICENSE). It includes or is derived from the following.

## musl libc, FreeBSD msun, Arm optimized routines

`runtime/src/float32/math.c` contains ports of single-precision math routines from musl libc 1.2.5 (MIT), and through
musl from FreeBSD msun (Sun Microsystems notice) and Arm's optimized routines (MIT). The full notices, with the files
ported and the changes made, are in [runtime/src/float32/NOTICE](runtime/src/float32/NOTICE).

## Unicode data

`models/relay/host/unicode_data.h` is generated from Python's `unicodedata` module (Unicode 16.0) by
`gen_unicode.py`. Unicode data files are distributed under the
[Unicode License v3](https://www.unicode.org/license.txt).

## Demo data

`models/relay/demo/data/mobile_actions_sample.jsonl` holds 60 rows sampled from the evaluation split of Google's
[Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) dataset, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), converted to Relay's record format.
