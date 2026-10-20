/* Measurement-only harness: complete public Relay calls on unchanged real records. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "relay.h"
#include "relay_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#else
#include <windows.h>
#include <psapi.h>
#endif

double gust_now(void) {
#ifdef _WIN32
    LARGE_INTEGER value, frequency;
    QueryPerformanceCounter(&value);
    QueryPerformanceFrequency(&frequency);
    return (double)value.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec * 1e-6;
#endif
}

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    const int repetitions = atoi(argv[3]);
    if (repetitions < 1 || repetitions > 31) return 2;
    relay_options options;
    relay_default_options(&options);
    options.threads = atoi(argv[4]);
    options.prefilter = atoi(argv[5]);
    char error[512];
    relay *model = relay_open(argv[1], &options, error, sizeof error);
    if (!model) {fprintf(stderr, "%s\n", error); return 1;}
    printf("graph_bytes=%zu prefilter=%d threads=%d\n", relay_model_bytes(model), options.prefilter, options.threads);
    FILE *source = fopen(argv[2], "rb");
    if (!source) return 1;
    size_t capacity = 4096, size = 0, record = 0;
    char *line = malloc(capacity);
    for (;;) {
        int c;
        size = 0;
        while ((c = fgetc(source)) != EOF && c != '\n') {
            if (size + 1 >= capacity) line = realloc(line, capacity *= 2);
            line[size++] = (char)c;
        }
        if (!size && c == EOF) break;
        if (!size) continue;
        line[size] = 0;
        rh_arena arena = {0};
        const jnode *json = rh_json_parse(&arena, line, size);
        const jnode *tools = rh_json_get(json, "tools");
        if (!json || !tools) return 1;
        char *reference = NULL;
        for (int repeat = 0; repeat < repetitions; repeat++) {
            const double start = gust_now();
            char *calls = relay_call(model, line, size);
            const double elapsed = gust_now() - start;
            if (!calls || (reference && strcmp(calls, reference))) return 1;
            printf("record=%zu tools=%zu repeat=%d elapsed_ms=%.3f calls=%s\n", record, tools->count, repeat, elapsed, calls);
            fflush(stdout);
            if (!reference) reference = calls;
            else relay_free(calls);
        }
        relay_free(reference);
        rh_arena_free(&arena);
        record++;
        if (c == EOF) break;
    }
    free(line);
    fclose(source);
    relay_close(model);
#ifndef _WIN32
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("peak_rss_kib=%ld\n", usage.ru_maxrss);
#else
    PROCESS_MEMORY_COUNTERS usage;
    usage.cb = sizeof usage;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &usage, sizeof usage)) return 1;
    printf("peak_working_set_kib=%llu\n", (unsigned long long)usage.PeakWorkingSetSize / 1024u);
#endif
    return 0;
}
