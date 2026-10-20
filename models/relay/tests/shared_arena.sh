#!/usr/bin/env bash
# Build and check shared-arena admission/ownership against the original Relay artifacts.
set -euo pipefail
cd "$(dirname "$0")/../../.."
model=${1:?usage: shared_arena.sh MODEL_DIRECTORY}
mkdir -p out
host=()
for source in models/relay/host/*.c; do
  case "$source" in */relay_cli.c|*/relay_wasm.c) continue ;; esac
  host+=("$source")
done
"${ZIG:-zig}" cc -std=c11 -O2 -Wall -Wextra -Wno-missing-field-initializers -ffp-contract=off -fno-math-errno \
  -DGUST_ATTENTION -DGUST_CONFORMER -DGUST_PACKED_WEIGHTS -DGUST_FLOAT32 \
  -DGUST_FLOAT32_DECODER -DGUST_SHARED_SCRATCH -Iruntime/include -Imodels/relay/host \
  runtime/src/gust.c "${host[@]}" models/relay/tests/shared_arena.c -o out/relay-shared-arena-test
for graph in relay relay-route relay-anchor relay-args; do
  ./out/relay-shared-arena-test "$model/$graph.gust"
done
