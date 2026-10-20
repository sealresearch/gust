#!/usr/bin/env bash
# Build the generic observer contract test; run the resulting binary with an optional float32 graph path.
set -euo pipefail
cd "$(dirname "$0")/../.."
target=${1:?usage: trace.sh x64|arm64|wasm}
case $target in
  x64) flags=(-target x86_64-windows-gnu); output=out/gust-trace-test-x64.exe ;;
  arm64) flags=(-target aarch64-linux-musl -static); output=out/gust-trace-test-arm64 ;;
  wasm) flags=(-target wasm32-wasi -msimd128 -Wl,-z,stack-size=1048576); output=out/gust-trace-test.wasm ;;
  *) echo "unsupported test target: $target" >&2; exit 2 ;;
esac
mkdir -p out
"${ZIG:-zig}" cc -std=c11 -O2 -Wall -Wextra -ffp-contract=off -fno-math-errno \
  -DGUST_ATTENTION -DGUST_CONFORMER -DGUST_PACKED_WEIGHTS -DGUST_FLOAT32 \
  -DGUST_FLOAT32_DECODER -DGUST_SHARED_SCRATCH "${flags[@]}" runtime/tests/trace.c -o "$output"
echo "$output"
