#!/usr/bin/env bash
# The Squint regression: regenerates the test sequences (deterministic, 2,154 frames, about 1 GB in out/squint-seq), runs the
# C reader on every one in a fresh process, and compares each read with tests/golden, bit for bit. The golden files are what
# the JavaScript release returned for the same frames (tools/reference_run.mjs wrote them), so a pass means the C port still
# returns what the release did.
#
#   ZIG=/path/to/zig bash models/squint/tests/run.sh [TARGET]      (TARGET: a zig target that this machine can run)
set -euo pipefail
cd "$(dirname "$0")/../../.."
ZIG=${ZIG:-zig}
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) T=x86_64-windows-gnu; EXE=.exe ;; Darwin) T=$(uname -m | sed 's/arm64/aarch64/')-macos; EXE= ;; *) T=$(uname -m)-linux-gnu; EXE= ;; esac
T=${1:-$T}
out=out/squint-seq
root=$(pwd -W 2>/dev/null || pwd) # a Windows path for node when this is Git Bash
node models/squint/tools/make_sequences.mjs --out "$root/$out" > /dev/null
cp models/squint/tests/golden/*.ref.jsonl "$out"/
node models/squint/tools/seq2cmd.mjs "$out"
objs=$(ZIG=$ZIG bash models/squint/tools/build_hosts.sh "$T")
$ZIG cc -std=c11 -O2 -Wall -Imodels/squint/host -target "$T" models/squint/tests/parity.c "$objs"/*.o -o out/squint/parity$EXE
for ref in "$out"/*.ref.jsonl; do
  n=${ref%.ref.jsonl}
  out/squint/parity$EXE "$n"
done
node models/squint/tools/compare.mjs "$out"
