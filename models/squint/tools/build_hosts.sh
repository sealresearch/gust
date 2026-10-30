#!/usr/bin/env bash
# Compiles the Squint hosts, one object per host file per instance, each instance with its own symbol prefix
# (host/names_<instance>.h), into OUT (default out/squint/obj-<target>). Usage: tools/build_hosts.sh TARGET [OUT]
# The Gust build.sh calls this for every target. Needs ZIG (the zig executable) and a writable zig cache.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
tgt=${1:?target}
out=${2:-$here/../../out/squint/obj-$tgt}
ZIG=${ZIG:-zig}
FLAGS="-std=c11 -O2 -Wall -Wno-unused-function -ffp-contract=off -fno-math-errno -fvisibility=hidden -I$here/host -target $tgt ${SQUINT_CFLAGS:-}"
case $tgt in wasm*) ;; *) FLAGS="$FLAGS -fPIC" ;; esac
mkdir -p "$out"
inst() { # instance name, extra flags, files
  local name=$1 extra=$2; shift 2
  for f in "$@"; do
    $ZIG cc $FLAGS $extra -include "names_$name.h" -c "$here/host/$f" -o "$out/${name}_${f%.c}.o"
  done
}
inst geometry "-DSQUINT_FRAME_CAPACITY=921600" geometry.c
inst qr "" qr.c
inst quads "" white_quads.c
inst flow "" flow.c
V="geometry.c white_quads.c registration.c phase.c qr.c evidence.c acquire.c verify_host.c"
inst verifier "-DSQUINT_FRAME_CAPACITY=921600" $V
inst acquirer "-DSQUINT_FRAME_CAPACITY=921600" geometry.c qr.c acquire.c
$ZIG cc $FLAGS -c "$here/host/squint_clock.c" -o "$out/clock.o"
$ZIG cc $FLAGS -c "$here/host/squint.c" -o "$out/squint.o"
echo "$out"
