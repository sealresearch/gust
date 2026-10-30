/* One platform layer for every Squint host: the same math, memory and clock on every target (x86-64, ARM64, WebAssembly).
   The reference wasm builds used the compiler's built-in math, so every build does, and results do not depend on the
   C library. The clock is only read for deadlines; a deadline of infinity makes a run depend on nothing but its input. */
#ifndef SQUINT_PLATFORM_H
#define SQUINT_PLATFORM_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#define API
#define sqrt     __builtin_sqrt
#define fabs     __builtin_fabs
#define floor    __builtin_floor
#define fmax     __builtin_fmax
#define fmin     __builtin_fmin
#define isfinite __builtin_isfinite
/* Seconds from a monotonic clock (squint_clock.c). */
double sq_clock_seconds(void);
#define seconds sq_clock_seconds
#endif
