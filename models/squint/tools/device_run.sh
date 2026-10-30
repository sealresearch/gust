#!/usr/bin/env bash
# Runs Squint on a connected Android (or any Linux ARM64) device over adb and checks the reads against the golden files:
# the same bit-exactness test as tests/run.sh, on the device's CPU, plus the time to open the reader, the time per frame and the
# peak memory. A static Linux binary runs on Android's kernel, so no NDK is needed.
#
#   ZIG=/path/to/zig bash models/squint/tools/device_run.sh [sequence ...]     (run tests/run.sh once first: it makes the frames)
set -euo pipefail
cd "$(dirname "$0")/../../.."
ZIG=${ZIG:-zig}
T=aarch64-linux-musl
seqs=("$@")
[ ${#seqs[@]} -eq 0 ] && seqs=(static_sharp_v3 straight_medium straight_heavy curved_hard distant_small glare_sweep tilt_ramp_1024x576 switch_toggle_corpus_640x480 vanish_reappear_640x480 ecc_levels_v5_640x480 versions_05_06_640x480 timing_irregular_640x480)
adb get-state > /dev/null
objs=$(ZIG=$ZIG bash models/squint/tools/build_hosts.sh $T)
$ZIG cc -std=c11 -O2 -static -Imodels/squint/host -target $T models/squint/tests/parity.c "$objs"/*.o -o out/squint/parity-arm64
dir=/data/local/tmp/squint
adb shell "mkdir -p $dir"
adb push out/squint/parity-arm64 $dir/parity > /dev/null
adb shell "chmod 755 $dir/parity"
echo "device: $(adb shell getprop ro.product.model | tr -d '\r') ($(adb shell getprop ro.product.cpu.abi | tr -d '\r')), $(adb shell getconf _NPROCESSORS_ONLN | tr -d '\r') cores"
for n in "${seqs[@]}"; do
  adb push out/squint-seq/$n.gray out/squint-seq/$n.cmd $dir/ > /dev/null
  adb shell "cd $dir && SQ_TIMING=1 ./parity $n" 2>&1 | tr -d '\r'
  adb pull $dir/$n.c.jsonl out/squint-seq/$n.c.jsonl > /dev/null
  adb shell "rm -f $dir/$n.gray $dir/$n.c.jsonl"
done
node models/squint/tools/compare.mjs out/squint-seq "${seqs[@]}"
