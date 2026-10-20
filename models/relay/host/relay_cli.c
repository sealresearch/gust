/* Native counterpart of web/relay.mjs: runs evaluation suites through the C host and writes PRED_DIR/<suite>.jsonl
   ({id, calls}, the same bytes as the JavaScript host), or answers one record.
     relay <model dir> <pred dir> [--eval eval] [--suites a,b] [--tokenizer path] [--gate G --gate2 G2]
           [--no-require] [--no-fixes] [--prefilter] [--pk 10] [--budget 600] [--threads N]
     relay <model dir> --record '{"text": ..., "tools": [...]}' [--explain | --trace | --trace-ops]
     relay <model dir> --record-file request.json [--prefilter] [--explain | --trace | --trace-ops] */
#include "relay.h"
#include "relay_host.h"
#include "gust.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir(p, 0777)
#endif

int rh_relay_record(relay *r, const jnode *rec, sb *out);

static double now_ms(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model dir> <pred dir> [options] | %s <model dir> --record JSON\n", argv[0], argv[0]);
        return 2;
    }
    relay_options o;
    relay_default_options(&o);
    o.gate = atof(opt(argc, argv, "--gate", "0"));
    o.gate2 = atof(opt(argc, argv, "--gate2", "2"));
    o.require = !flag(argc, argv, "--no-require");
    o.fixes = !flag(argc, argv, "--no-fixes");
    o.prefilter = flag(argc, argv, "--prefilter");
    o.prefilter_k = atoi(opt(argc, argv, "--pk", "10"));
    o.prefilter_budget = atoi(opt(argc, argv, "--budget", "600"));
    o.threads = atoi(opt(argc, argv, "--threads", "1"));
    o.tokenizer = opt(argc, argv, "--tokenizer", NULL);
    char err[512];
    const double t0 = now_ms();
    relay *r = relay_open(argv[1], &o, err, sizeof err);
    if (!r) { fprintf(stderr, "relay: %s\n", err); return 1; }
    fprintf(stderr, "loaded in %.0f ms, graphs %.1f MB, %s\n", now_ms() - t0, relay_model_bytes(r) / 1e6, gust_variant());
    const char *one = opt(argc, argv, "--record", NULL);
    const char *record_file = opt(argc, argv, "--record-file", NULL);
    char *owned_record = NULL;
    if (record_file) {
        FILE *f = fopen(record_file, "rb");
        long size = -1;
        if (f && !fseek(f, 0, SEEK_END)) size = ftell(f);
        if (size >= 0 && !fseek(f, 0, SEEK_SET)) {
            owned_record = rh_malloc((size_t)size + 1);
            if (fread(owned_record, 1, (size_t)size, f) != (size_t)size) { free(owned_record); owned_record = NULL; }
            else owned_record[size] = 0;
        }
        if (f) fclose(f);
        if (!owned_record) { fprintf(stderr, "cannot read %s\n", record_file); relay_close(r); return 1; }
        one = owned_record;
    }
    if (one) {
        char *c = flag(argc, argv, "--trace-ops") ? relay_trace_ops(r, one, (size_t)-1) :
                  flag(argc, argv, "--trace") ? relay_trace(r, one, (size_t)-1) :
                  flag(argc, argv, "--explain") ? relay_explain(r, one, (size_t)-1) : relay_call(r, one, (size_t)-1);
        printf("%s\n", c ? c : "invalid record");
        const int status = c ? 0 : 1;
        relay_free(c);
        free(owned_record);
        relay_close(r);
        return status;
    }
    const char *evalDir = opt(argc, argv, "--eval", "eval");
    make_dir(argv[2]); /* like gauge_run: the prediction directory is created (one level) */
    char suites[512];
    snprintf(suites, sizeof suites, "%s", opt(argc, argv, "--suites", "mobile_actions,droidcall,bfcl,dstc8,snips_gold,snips_7way"));
    for (char *s = strtok(suites, ","); s; s = strtok(NULL, ",")) {
        char path[1024];
        snprintf(path, sizeof path, "%s/%s.jsonl", evalDir, s);
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", path); return 1; }
        snprintf(path, sizeof path, "%s/%s.jsonl", argv[2], s);
        FILE *w = fopen(path, "wb");
        if (!w) { fprintf(stderr, "cannot write %s (does the directory exist?)\n", path); return 1; }
        size_t cap = 1 << 20, rows = 0, mcap = 1024;
        char *line = malloc(cap);
        double *ms = malloc(mcap * sizeof(double));
        for (;;) {
            size_t n = 0;
            int c;
            while ((c = fgetc(f)) != EOF && c != '\n') {
                if (n + 1 >= cap) line = realloc(line, cap *= 2);
                line[n++] = (char)c;
            }
            if (n == 0 && c == EOF) break;
            if (n == 0) continue;
            line[n] = 0;
            rh_arena A = {0};
            const jnode *rec = rh_json_parse(&A, line, n);
            if (!rec) { fprintf(stderr, "%s: invalid JSON line %zu\n", s, rows + 1); return 1; }
            sb out = {0};
            sb_str(&out, "{\"id\": ");
            rv id = rv_json(rh_json_get(rec, "id"));
            rv_dumps(&out, &id);
            sb_str(&out, ", \"calls\": ");
            const double t = now_ms();
            rh_relay_record(r, rec, &out);
            if (rows == mcap) ms = realloc(ms, (mcap *= 2) * sizeof(double));
            ms[rows++] = now_ms() - t;
            sb_str(&out, "}\n");
            fwrite(out.s, 1, out.n, w);
            sb_free(&out);
            rh_arena_free(&A);
            if (c == EOF) break;
        }
        fclose(f);
        fclose(w);
        qsort(ms, rows, sizeof(double), cmpd);
        if (rows) printf("%s: %zu rows, median %.1f ms, p90 %.1f ms\n", s, rows, ms[rows >> 1], ms[(size_t)(rows * 0.9)]);
        fflush(stdout);
        free(line);
        free(ms);
    }
    relay_close(r);
    return 0;
}
