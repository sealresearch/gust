#!/usr/bin/env bash
# Builds the Gust runtime, the Relay host, the standalone WebAssembly runtime and the runtime tests. Outputs: out/.
# Every build has every feature; outputs are bit-identical across targets and thread counts.
set -euo pipefail
cd "$(dirname "$0")"
ZIG=${ZIG:-zig}
mkdir -p out
FEAT="-DGUST_ATTENTION -DGUST_CONFORMER -DGUST_PACKED_WEIGHTS -DGUST_FLOAT32 -DGUST_FLOAT32_DECODER -DGUST_SHARED_SCRATCH"
CFLAGS="-std=c11 -O2 -Wall -Wextra -Wno-missing-field-initializers -fno-math-errno -ffp-contract=off $FEAT -Iruntime/include"
HOST=$(ls models/relay/host/*.c | grep -v -E 'relay_cli.c|relay_wasm.c')
CLI="runtime/src/gust.c $HOST models/relay/host/relay_cli.c"
WASM="runtime/src/gust.c $HOST models/relay/host/relay_wasm.c -DRELAY_NO_FILES -mexec-model=reactor -Wl,--strip-debug -Wl,-z,stack-size=1048576"
MT="-matomics -mbulk-memory -DGUST_THREADS -Wl,--import-memory -Wl,--export-memory -Wl,--shared-memory
  -Wl,--max-memory=1073741824 -Wl,--export=__stack_pointer"
# The runtime alone for JavaScript hosts (models/*/js): no libc, fixed 128 MiB memory, the public entry points.
RUNTIME_WASM="runtime/src/gust.c -nostdlib -Wl,--no-entry -Wl,--export-memory -Wl,--export=__heap_base
  -Wl,--initial-memory=134217728 -Wl,--max-memory=134217728 -Wl,-z,stack-size=262144 -Wl,--strip-debug
  -Wl,--export=gust_context_bytes -Wl,--export=gust_capabilities -Wl,--export=gust_init -Wl,--export=gust_run
  -Wl,--export=gust_run_valid -Wl,--export=gust_run_valid_outputs -Wl,--export=gust_prefer_relaxed_dot -Wl,--export=gust_tensor -Wl,--export=gust_tensor_bytes -Wl,--export=gust_reset"
# x86-64: one binary, the runtime compiled for SSE2, AVX2 and AVX-512 VNNI, picked from CPUID (dispatch_x86.c)
x86() {
  local tgt=$1 extra=$2 out=$3 obj=out/obj-$1 main=${4:-"$HOST models/relay/host/relay_cli.c"}
  mkdir -p $obj
  $ZIG cc $CFLAGS -target $tgt $extra -DGUST_THREADS -DGUST_VARIANT=base -c runtime/src/gust.c -o $obj/base.o
  $ZIG cc $CFLAGS -target $tgt $extra -DGUST_THREADS -DGUST_VARIANT=avx2 -mavx2 -c runtime/src/gust.c -o $obj/avx2.o
  $ZIG cc $CFLAGS -target $tgt $extra -DGUST_THREADS -DGUST_VARIANT=vnni -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512vnni -mevex512 -c runtime/src/gust.c -o $obj/vnni.o
  $ZIG cc $CFLAGS -target $tgt $extra -DGUST_THREADS $obj/base.o $obj/avx2.o $obj/vnni.o runtime/src/dispatch_x86.c $main -o $out
}
# The model library for Python, Rust and other languages: the runtime, the Relay host and the model API in one shared
# library, out/lib/<platform>/gust.dll, libgust.so or libgust.dylib. Linux is built against glibc 2.17 so that one file
# runs on every distribution a Python wheel targets (manylinux2014).
sharedlib() { # platform, zig target, file name, thread define, "x86" for the three-variant build
  local plat=$1 tgt=$2 file=$3 thr=$4 arch=${5:-} out=out/lib/$1 obj=out/obj-lib-$1
  local srcs="$HOST sdk/src/model.c models/relay/sdk/relay_model.c"
  local toy=""
  if [ -n "${GUST_TOY:-}" ]; then srcs="$srcs sdk/tests/toy_models.c"; toy="-DGUST_MODEL_TOY"; fi  # test models for bindings: GUST_TOY=1
  local flags="$CFLAGS -target $tgt -fPIC -DGUST_THREADS -D$thr -Isdk/include -Imodels/relay/host -DGUST_MODEL_RELAY $toy -fvisibility=hidden"
  mkdir -p $out $obj
  if [ "$arch" = x86 ]; then
    $ZIG cc $flags -DGUST_VARIANT=base -c runtime/src/gust.c -o $obj/base.o
    $ZIG cc $flags -DGUST_VARIANT=avx2 -mavx2 -c runtime/src/gust.c -o $obj/avx2.o
    $ZIG cc $flags -DGUST_VARIANT=vnni -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512vnni -mevex512 -c runtime/src/gust.c -o $obj/vnni.o
    $ZIG cc $flags -shared $obj/base.o $obj/avx2.o $obj/vnni.o runtime/src/dispatch_x86.c $srcs -o $out/$file
  else
    $ZIG cc $flags -shared runtime/src/gust.c $srcs -o $out/$file
  fi
}
build() {
  case $1 in
  x64)       x86 x86_64-windows-gnu -DGUST_THREADS_WIN32 out/relay-x64.exe ;;
  bench-x64) x86 x86_64-windows-gnu -DGUST_THREADS_WIN32 out/gust-bench-x64.exe runtime/tools/gust_bench.c ;;
  lib-windows-x64) sharedlib windows-x64 x86_64-windows-gnu gust.dll GUST_THREADS_WIN32 x86 ;;
  lib-linux-x64)   sharedlib linux-x64 x86_64-linux-gnu.2.17 libgust.so GUST_THREADS_POSIX x86 ;;
  lib-linux-arm64) sharedlib linux-arm64 aarch64-linux-gnu.2.17 libgust.so GUST_THREADS_POSIX ;;
  lib-macos-x64)   sharedlib macos-x64 x86_64-macos libgust.dylib GUST_THREADS_POSIX x86 ;;
  lib-macos-arm64) sharedlib macos-arm64 aarch64-macos libgust.dylib GUST_THREADS_POSIX ;;
  lib)       for t in lib-windows-x64 lib-linux-x64 lib-linux-arm64 lib-macos-x64 lib-macos-arm64; do build "$t"; done ;;
  # The model API (sdk/include/gust_model.h) with Relay built in, and its tests: out/sdk-test-x64.exe [MODEL_DIR RECORDS]
  test-sdk-x64) x86 x86_64-windows-gnu -DGUST_THREADS_WIN32 out/sdk-test-x64.exe "-Isdk/include -DGUST_MODEL_RELAY $HOST sdk/src/model.c models/relay/sdk/relay_model.c sdk/tests/abi_test.c" ;;
  bench-calls-x64) x86 x86_64-windows-gnu -DGUST_THREADS_WIN32 out/relay-calls-x64.exe "-Imodels/relay/host $HOST models/relay/tools/bench_calls.c -lpsapi" ;;
  bench-calls-arm64) $ZIG cc $CFLAGS -target aarch64-linux-musl -static -DGUST_THREADS -DGUST_THREADS_POSIX \
               -Imodels/relay/host runtime/src/gust.c $HOST models/relay/tools/bench_calls.c -o out/relay-calls-arm64 ;;
  bench-arm64) $ZIG cc $CFLAGS -target aarch64-linux-musl -static -DGUST_THREADS -DGUST_THREADS_POSIX \
               runtime/src/gust.c runtime/tools/gust_bench.c -o out/gust-bench-arm64 ;;
  bench-wasm) $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 -mrelaxed-simd -Wl,-z,stack-size=1048576 \
              runtime/src/gust.c runtime/tools/gust_bench.c -o out/gust-bench.wasm
              $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 -Wl,-z,stack-size=1048576 \
              runtime/src/gust.c runtime/tools/gust_bench.c -o out/gust-bench-simd.wasm ;;
  workshare-wasm) $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 -mrelaxed-simd -matomics -mbulk-memory \
              -DGUST_THREADS -mexec-model=reactor -Wl,-z,stack-size=1048576 -Wl,--import-memory \
              -Wl,--export-memory -Wl,--shared-memory -Wl,--max-memory=1073741824 -Wl,--export=__stack_pointer \
              runtime/tests/workshare.c -o out/workshare.wasm ;;
  linux-x64) x86 x86_64-linux-musl "-static -DGUST_THREADS_POSIX" out/relay-linux-x64 ;;
  arm64)    $ZIG cc $CFLAGS -target aarch64-linux-musl -static -DGUST_THREADS -DGUST_THREADS_POSIX $CLI -o out/relay-arm64 ;;
  macos-arm64) $ZIG cc $CFLAGS -target aarch64-macos -DGUST_THREADS -DGUST_THREADS_POSIX $CLI -o out/relay-macos-arm64 ;;
  macos-x64)   x86 x86_64-macos "-DGUST_THREADS_POSIX" out/relay-macos-x64 ;;
  # The Relay command line for every platform, named as the seal command line downloads it, with checksums: out/release/
  release)  mkdir -p out/release
            for t in x64 linux-x64 arm64 macos-arm64 macos-x64; do build $t; done
            cp out/relay-x64.exe out/release/relay-windows-x64.exe
            cp out/relay-linux-x64 out/release/relay-linux-x64
            cp out/relay-arm64 out/release/relay-linux-arm64
            cp out/relay-macos-arm64 out/release/relay-macos-arm64
            cp out/relay-macos-x64 out/release/relay-macos-x64
            (cd out/release && sha256sum relay-* > SHA256SUMS) ;;
  # WebAssembly: relaxed SIMD (Chrome, Edge, Firefox) and plain SIMD128 (Safari), each single and threaded
  wasm)         $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 -mrelaxed-simd $WASM -o out/relay.wasm ;;
  wasm-mt)      $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 -mrelaxed-simd $MT $WASM -o out/relay-mt.wasm ;;
  wasm-simd)    $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 $WASM -o out/relay-simd.wasm ;;
  wasm-mt-simd) $ZIG cc $CFLAGS -target wasm32-wasi -msimd128 $MT $WASM -o out/relay-mt-simd.wasm ;;
  runtime-wasm)      $ZIG cc $CFLAGS -target wasm32-freestanding -msimd128 -mrelaxed-simd $RUNTIME_WASM -o out/gust.wasm ;;
  runtime-wasm-simd) $ZIG cc $CFLAGS -target wasm32-freestanding -msimd128 $RUNTIME_WASM -o out/gust-simd.wasm ;;
  test-t5)  $ZIG cc $CFLAGS -target x86_64-windows-gnu runtime/tests/t5.c -o out/t5.exe ;;
  test-temporal) $ZIG cc $CFLAGS -target x86_64-windows-gnu runtime/tests/temporal.c -o out/temporal.exe ;;
  *) echo "unknown target $1"; exit 2 ;;
  esac
  echo "built $1"
}
for target in "${@:-all}"; do
  if [[ "$target" == all ]]; then
    for t in x64 arm64 wasm wasm-mt wasm-simd wasm-mt-simd; do build "$t"; done
  else
    build "$target"
  fi
done
