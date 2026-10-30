/* The clock the Squint hosts read for deadlines (squint_platform.h). */
#define _POSIX_C_SOURCE 200809L
#include "squint_platform.h"
#ifdef _WIN32
#include <windows.h>
double sq_clock_seconds(void) {
    LARGE_INTEGER frequency, tick;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&tick);
    return (double)tick.QuadPart / (double)frequency.QuadPart;
}
#else
#include <time.h>
double sq_clock_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
#endif
