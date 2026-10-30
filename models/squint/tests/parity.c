/* Replays a frame sequence through the C reader and writes one JSON line per frame, in the format of the JavaScript
   reference (tools/reference_run.mjs), so the two can be compared exactly.

   parity SEQUENCE_PREFIX      reads PREFIX.gray (frames, back to back) and PREFIX.cmd ("W H N", then "reset time" per
                               frame), writes PREFIX.c.jsonl. Budgets are infinite: a result depends on the frames alone. */
#include "squint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double sq_clock_seconds(void);

static void hex_double(char *dst, double v) {
    union { double d; uint64_t u; } x;
    x.d = v;
    sprintf(dst, "%016llx", (unsigned long long)x.u);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: parity SEQUENCE_PREFIX\n"); return 2; }
    char path[1024], err[256];
    snprintf(path, sizeof path, "%s.cmd", argv[1]);
    FILE *cmd = fopen(path, "r");
    if (!cmd) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    int w, h, n;
    if (fscanf(cmd, "%d %d %d", &w, &h, &n) != 3) return 2;
    snprintf(path, sizeof path, "%s.gray", argv[1]);
    FILE *gf = fopen(path, "rb");
    if (!gf) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    snprintf(path, sizeof path, "%s.c.jsonl", argv[1]);
    FILE *out = fopen(path, "wb");
    if (!out) return 2;
    if (sq_open(err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    uint8_t *gray = malloc((size_t)w * h);
    sq_read_options o;
    sq_default_options(&o);
    o.budget_ms = o.acquire_budget_ms = o.tracked_acquire_budget_ms = INFINITY;
    static sq_read_result r;
    double total = 0, worst = 0;
    for (int i = 0; i < n; i++) {
        int reset;
        double t;
        if (fscanf(cmd, "%d %lf", &reset, &t) != 2) { fprintf(stderr, "bad cmd file\n"); return 2; }
        if (fread(gray, 1, (size_t)w * h, gf) != (size_t)w * h) { fprintf(stderr, "short gray file\n"); return 2; }
        if (reset) sq_reset();
        o.time_ms = t;
        double t0 = sq_clock_seconds();
        int rc = sq_read(gray, w, h, &o, &r, err, sizeof err);
        double dt = (sq_clock_seconds() - t0) * 1000;
        total += dt;
        if (dt > worst) worst = dt;
        if (rc) {
            fprintf(out, "{\"i\":%d,\"error\":\"%s\"}\n", i, err);
            continue;
        }
        fprintf(out, "{\"i\":%d,\"accepted\":%s,", i, r.accepted ? "true" : "false");
        if (r.reason && !r.accepted) fprintf(out, "\"reason\":\"%s\",", r.reason); else fprintf(out, "\"reason\":null,");
        if (r.path && r.accepted) fprintf(out, "\"path\":\"%s\",", r.path); else fprintf(out, "\"path\":null,");
        if (r.accepted) {
            fprintf(out, "\"version\":%d,\"payload\":\"", r.version);
            for (int k = 0; k < r.payload_len; k++) fprintf(out, "%02x", r.payload[k]);
            fprintf(out, "\",\"corners\":[");
            for (int k = 0; k < 8; k++) {
                char hx[20];
                hex_double(hx, r.corners[k]);
                fprintf(out, "%s\"%s\"", k ? "," : "", hx);
            }
            fprintf(out, "]}\n");
        } else fprintf(out, "\"version\":null,\"payload\":null,\"corners\":null}\n");
    }
    fclose(out);
    if (getenv("SQ_TIMING")) fprintf(stderr, "%s: %d frames, %.2f ms per frame on average, %.2f ms at most\n", argv[1], n, total / n, worst);
    return 0;
}
