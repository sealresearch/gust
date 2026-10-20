#!/usr/bin/env bash
# Bit-exactness: Relay through every build must reproduce the reference outputs byte for byte.
#   runtime/tests/bitexact.sh MODEL_DIR REF_DIR [SUITES] [EVAL_DIR]
# REF_DIR/<suite>.jsonl are the reference predictions (e.g. the JavaScript host's). Checks: x86-64 variants (base, avx2,
# vnni where the CPU has them) at 1 and 8 threads, and WebAssembly single- and multi-threaded. For a device such as a
# Raspberry Pi, run the ARM64 build (out/relay-arm64) there with the same arguments and compare the output files.
set -uo pipefail
cd "$(dirname "$0")/../.."
M=$1; REF=$2; SUITES=${3:-droidcall,mobile_actions}; EV=${4:-eval}
T=$(mktemp -d); fail=0
check() { # name dir
  for s in ${SUITES//,/ }; do
    if cmp -s "$2/$s.jsonl" "$REF/$s.jsonl"; then echo "  ok    $1 $s"; else echo "  FAIL  $1 $s"; fail=1; fi
  done
}
for v in base avx2 vnni; do for th in 1 8; do
  GUST_X86=$v out/relay-x64.exe "$M" "$T/$v-$th" --eval "$EV" --suites "$SUITES" --threads $th > /dev/null 2> "$T/log" || mkdir -p "$T/$v-$th"
  check "x64 $(grep -o 'x86-64[^,]*' "$T/log" | head -1) t$th" "$T/$v-$th"
done; done
node runtime/tests/relay_wasm.mjs out/relay.wasm "$M" "$EV" "$T/wasm" "$SUITES" 1 2>/dev/null; check "wasm t1" "$T/wasm"
node runtime/tests/relay_wasm.mjs out/relay-mt.wasm "$M" "$EV" "$T/wasm-mt" "$SUITES" 8 2>/dev/null; check "wasm-mt t8" "$T/wasm-mt"
rm -rf "$T"
[[ $fail == 0 ]] && echo "bit-exact: all builds match" || { echo "bit-exact: FAILURES"; exit 1; }
