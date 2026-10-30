/* Squint through the Gust model API (sdk/include/gust_model.h): the same sequences as parity.c, but every frame goes through
   gust_model_open / gust_model_run, or through a stream, and the model's JSON is written out for comparison.

   api_test SEQUENCE_PREFIX [stream]   reads PREFIX.gray and PREFIX.cmd, writes PREFIX.api.jsonl (or PREFIX.stream.jsonl) */
#include "gust_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: api_test SEQUENCE_PREFIX [stream]\n"); return 2; }
    int stream = argc > 2 && !strcmp(argv[2], "stream");
    char path[1024];
    snprintf(path, sizeof path, "%s.cmd", argv[1]);
    FILE *cmd = fopen(path, "r");
    int w, h, n;
    if (!cmd || fscanf(cmd, "%d %d %d", &w, &h, &n) != 3) { fprintf(stderr, "no cmd file\n"); return 2; }
    snprintf(path, sizeof path, "%s.gray", argv[1]);
    FILE *gf = fopen(path, "rb");
    snprintf(path, sizeof path, "%s.%s.jsonl", argv[1], stream ? "stream" : "api");
    FILE *out = fopen(path, "wb");
    if (!gf || !out) return 2;

    const char *desc = gust_model_describe("squint-1");
    if (!desc) { fprintf(stderr, "squint-1 is not built in\n"); return 1; }
    gust_model *m;
    if (gust_model_open("squint-1", NULL, 0, NULL, &m) != GUST_MODEL_OK) { fprintf(stderr, "open: %s\n", gust_model_last_error()); return 1; }
    gust_stream *st = NULL;
    if (stream && gust_stream_open(m, NULL, &st) != GUST_MODEL_OK) { fprintf(stderr, "stream: %s\n", gust_model_last_error()); return 1; }
    unsigned char *gray = malloc((size_t)w * h);
    for (int i = 0; i < n; i++) {
        int reset;
        double t;
        if (fscanf(cmd, "%d %lf", &reset, &t) != 2 || fread(gray, 1, (size_t)w * h, gf) != (size_t)w * h) return 2;
        char opts[96];
        snprintf(opts, sizeof opts, "{\"time\":%.17g%s}", t, reset ? ",\"reset\":true" : "");
        gust_buffer frame = {sizeof frame, GUST_BUF_IMAGE_GRAY8, gray, (size_t)w * h, 0, 0, (uint32_t)w, (uint32_t)h};
        gust_result *res = NULL;
        gust_model_status s;
        if (!stream) {
            s = gust_model_run(m, &frame, 1, opts, &res);
        } else {
            gust_buffer o = {sizeof o, GUST_BUF_JSON, opts, strlen(opts), 0, 0, 0, 0};
            gust_buffer in[2] = {frame, o};
            s = gust_stream_push(st, in, 2);
            int done = 0;
            if (s == GUST_MODEL_OK) s = gust_stream_pull(st, &res, &done);
        }
        if (s != GUST_MODEL_OK) {
            fprintf(out, "{\"i\":%d,\"error\":\"%s\"}\n", i, gust_model_last_error());
            continue;
        }
        fprintf(out, "{\"i\":%d,\"r\":%s}\n", i, gust_result_count(res) ? (const char *)gust_result_get(res, 0)->data : "null");
        gust_result_free(res);
    }
    if (st) { gust_stream_finish(st); gust_stream_close(st); }
    gust_model_close(m);
    fclose(out);
    printf("size %zu MB\n", (size_t)0);
    return 0;
}
