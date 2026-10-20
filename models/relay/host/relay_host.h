/* Native Relay host (C11, no dependencies beyond the C library): tokenizer, features, decoding on top of the
   Gust engine. Bit-exact port of the JavaScript host (the web folder modules), checked against its golden files. */
#ifndef RELAY_HOST_H
#define RELAY_HOST_H
#include <stddef.h>
#include <stdint.h>
/* Bit-exactness: float expressions are evaluated exactly as written (no fused multiply-add), on every compiler. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

/* ---- memory ---- */
void *rh_malloc(size_t n);  /* aborts on failure */
void *rh_realloc(void *p, size_t n);

typedef struct rh_block rh_block;
typedef struct {
    rh_block *head;
} rh_arena;
void *rh_arena_alloc(rh_arena *a, size_t n); /* 8-byte aligned, freed together */
void rh_arena_free(rh_arena *a);

/* growable arrays */
typedef struct { int32_t *v; size_t n, cap; } rh_ivec;
void rh_ivec_push(rh_ivec *a, int32_t x);
void rh_ivec_free(rh_ivec *a);

/* ---- JSON ---- */
typedef enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } jtype;
typedef struct jnode {
    jtype t;
    const char *s; /* J_STR: UTF-8 text; J_NUM: source text */
    size_t len;
    double num;
    const char *key; /* member key inside an object */
    size_t klen;
    struct jnode *first, *next; /* children in order */
    size_t count;
} jnode;
jnode *rh_json_parse(rh_arena *a, const char *text, size_t len);
jnode *rh_json_get(const jnode *obj, const char *key);
size_t rh_utf8_put(char *out, uint32_t cp);

/* ---- Unicode (unicode.c) ---- */
#define RH_UC_L 1u
#define RH_UC_N 2u
#define RH_UC_Z 4u
#define RH_UC_M 8u
#define RH_UC_ND 16u
#define RH_UC_C 32u
uint32_t uc_flags(uint32_t cp);
int uc_ccc(uint32_t cp);
size_t uc_nfd(const uint32_t *in, size_t n, uint32_t **out); /* caller frees */
size_t uc_nfc(const uint32_t *in, size_t n, uint32_t **out);
size_t uc_lower_str(const uint32_t *s, size_t n, uint32_t **out); /* str.lower / toLowerCase; caller frees */
size_t uc_upper_str(const uint32_t *s, size_t n, uint32_t **out);
uint32_t uc_fold_cp(uint32_t cp); /* simple case folding */
size_t rh_utf8_decode(const char *s, size_t n, uint32_t **out); /* code points; caller frees */
int rh_py_isspace(uint32_t cp);

/* ---- tokenizer (bpe.c) ---- */
typedef struct rh_tokenizer rh_tokenizer;
rh_tokenizer *rh_tok_load(const char *json, size_t len);
void rh_tok_free(rh_tokenizer *t);
int32_t rh_tok_id(const rh_tokenizer *t, const char *name); /* added token or vocab entry, -1 if absent */
/* Encoding without special tokens of the code points cps[0..n): ids and [start, end) code point offsets. */
void rh_tok_encode(rh_tokenizer *t, const uint32_t *cps, size_t n, rh_ivec *ids, rh_ivec *offs);

/* ---- regular expressions (rx.c) ---- */
#define RX_PY 1 /* Python re (Unicode \w \d \s \b); otherwise JavaScript without the u flag */
#define RX_I 2  /* case-insensitive */
typedef struct rh_rx rh_rx;
rh_rx *rh_rx_compile(const char *pattern, int flags); /* NULL on unsupported syntax */
void rh_rx_free(rh_rx *rx);
int rh_rx_groups(const rh_rx *rx);
/* leftmost match at or after `from`; caps[2k], caps[2k+1] = group k span (-1 if unset), k = 0 whole match */
int rh_rx_search(const rh_rx *rx, const uint32_t *s, size_t n, size_t from, long *caps);

/* ---- values (val.c) ---- */
typedef struct { char *s; size_t n, cap; } sb; /* string builder, NUL-terminated */
void sb_put(sb *b, const char *s, size_t n);
void sb_str(sb *b, const char *s);
void sb_cp(sb *b, uint32_t cp);
void sb_free(sb *b);
/* Optional per-call diagnostic events. NULL is a no-op; no tensor data is retained. */
void rh_tracef(sb *trace, const char *format, ...);
void rh_trace_name(sb *trace, const jnode *name);

enum { RV_UNDEF, RV_NULL, RV_BOOL, RV_NUM, RV_STR, RV_JSON, RV_LIST };
typedef struct rv {
    int t;
    double num;      /* RV_BOOL (0/1), RV_NUM */
    int isfloat;     /* RV_NUM: a Python float (PyFloat or non-integral) */
    const char *s;   /* RV_STR: UTF-8 */
    size_t len;
    const jnode *j;  /* RV_JSON: array or object from the request / tables */
    struct rv *items; /* RV_LIST */
    size_t n;
} rv;
rv rv_undef(void);
rv rv_null(void);
rv rv_bool(int b);
rv rv_int(double x);
rv rv_float(double x);   /* pyFloat */
rv rv_jsnum(double x);   /* plain JavaScript number */
rv rv_str(const char *s, size_t n);
rv rv_json(const jnode *j);
int rv_truthy(const rv *v);
int rv_is_pyfloat(const rv *v);
void py_str(sb *b, const rv *v);   /* Python str() */
void js_num(sb *b, double v);      /* Number.prototype.toString */
void py_float(sb *b, double v);    /* Python repr(float) */
void js_string(sb *b, const char *s, size_t n); /* JSON.stringify(string) */
void rv_dumps(sb *b, const rv *v); /* output serializer (Python json.dumps separators) */
size_t py_ascii_digits(const uint32_t *s, size_t n, uint32_t *out);
int py_float_parse(const uint32_t *s, size_t n, double *out);
int py_int_parse(const uint32_t *s, size_t n, double *out);
double py_round(double x);

/* ---- host fixes (hostfix.mjs) ---- */
typedef struct { int enumOnly, numReject, numSpan, perCallHold, orderTie, dedupe, chunks, caps, locale; } rh_fix;
extern rh_fix FIX;
void rh_set_fixes(int all); /* 1: every fix except orderTie (the default), 0: Python-parity decoder */

/* ---- JavaScript semantics (val.c) ---- */
double rh_js_exp(double x);           /* Math.exp (fdlibm, as V8) */
int rh_js_truthy(const jnode *j);     /* JavaScript truthiness of a JSON value (NULL = undefined) */
/* members of a JSON object in JavaScript property order (integer keys first, ascending; duplicate keys: first
   position, last value); returns the count, *out allocated in A */
size_t rh_js_members(const jnode *obj, rh_arena *A, const jnode ***out);

/* ---- value programs (programs.c) ---- */
enum { PR_DEFAULT, PR_BOOL, PR_ENUM, PR_OPTION, PR_CONST, PR_DIGITS, PR_AFFIX, PR_REF, PR_COPY, PR_NUMBER,
       PR_TIMEPART, PR_DATETIME, PR_LOOKUP, PR_LIST };
typedef struct rh_prog {
    int op;
    long span[2], date[2], time[2]; /* code point [start, end) into the request text */
    int has_span, has_date, has_time;
    int value, index, call, is_int, part_hour, table;
    double scale;
    const char *format;
    const char *cvalue; /* PR_CONST: UTF-8 */
    size_t clen;
    struct rh_prog *items;
    int nitems;
} rh_prog;
extern const char *const RH_DATE_FORMATS[];
#define RH_N_DATE_FORMATS 19
extern const double RH_NUM_SCALES[];
#define RH_N_NUM_SCALES 15
int rh_parse_number(const uint32_t *s, size_t n, double *out);
size_t rh_desc_options(const jnode *param, rh_arena *A, rv *out); /* out: room for 64 */
int rh_coerce(rv v, const jnode *typ, rh_arena *A, rv *out);      /* 0 where Python raises */
/* run(): 1 and the coerced value, or 0 where Python raises (the argument is dropped) */
int rh_run(const rh_prog *p, const uint32_t *text, size_t tn, const jnode *param, const jnode *ctx, rh_arena *A, rv *out);
const jnode *rh_tables(void);
const jnode *rh_affixes(void);
size_t rh_n_tables(void);

/* ---- features (features.c) ---- */
#define RH_K 6
enum { OP_ABSENT, OP_COPY, OP_NUMBER, OP_DATETIME, OP_TIMEPART, OP_ENUM, OP_BOOL_TRUE, OP_BOOL_FALSE, OP_LOOKUP,
       OP_DEFAULT, OP_REF, OP_LIST, OP_DIGITS, OP_AFFIX, OP_SELECT, RH_N_OPS };
#define RH_LIM_REQUEST 192
#define RH_LIM_TOOL 48
#define RH_LIM_PARAM 40
#define RH_LIM_OPTION 8
#define RH_LIM_OPTIONS 16
#define RH_LIM_PARAMS 32
#define RH_MAX_LEN 1024
typedef struct rh_tok rh_tok; /* the tokenizer, its marker ids and an encoding cache */
rh_tok *rh_tok_open(const char *json, size_t len);
void rh_tok_close(rh_tok *t);
int32_t rh_tok_marker(const rh_tok *t, int which); /* 0 [CLS], 1 [SEP], 2 [PAD], 3 T, 4 P, 5 O */
void rh_tok_set_trace(rh_tok *t, sb *trace);
/* ids of UTF-8 text without specials, truncated to n; *ids valid until the next call (count returned) */
size_t rh_tok_ids(rh_tok *t, const char *s, size_t len, size_t n, const int32_t **ids);
/* the request: ids truncated to n and code point offsets with leading whitespace skipped (arena-allocated) */
size_t rh_tok_request(rh_tok *t, const uint32_t *cps, size_t ncp, size_t n, rh_arena *A, int32_t **ids, long (**offs)[2]);

typedef struct { int pos, nopts; int *opts; const jnode *param; } rh_pslot;
typedef struct { const jnode *tool, *name; int pos, nparams; rh_pslot *params; } rh_tslot;
typedef struct rh_example {
    int32_t *ids;
    int n, q0, nq;
    long (*offs)[2];          /* request token code point offsets */
    const uint32_t *cps;      /* the request text */
    size_t ncp;
    rh_tslot *tools;
    int ntools, chunk;
    const jnode *rec;         /* kind, context */
    float *H;                 /* encoder states, filled by the net */
    int extract;
} rh_example;
/* options(p): the enum (Python str of each item) or the description's options; count, -1 where Python raises */
int rh_options(const jnode *p, rh_arena *A, rv **out, int *is_enum);
void rh_trim(const uint32_t *cps, long *a, long *b);
/* tool segment ids (arena) and, per parameter (at most 32), its marker and option marker offsets; -1 on error */
int rh_tool_segment(rh_tok *tk, const jnode *tool, rh_arena *A, int32_t **seg, int *nparams, int **ppos, int **nopts, int ***opos);
/* eval_examples: the tools split into as many sequences as needed; count of examples, -1 when unbuildable */
int rh_eval_examples(rh_tok *tk, const jnode *rec, const uint32_t *text, size_t tn, const jnode *const *tools,
                     int ntools, rh_arena *A, rh_example **out);

/* ---- Gust graphs and head arithmetic (net.c) ---- */
typedef struct { const char *name; uint32_t h, w, c, bytes, flags; float *p; } rh_gtensor;
typedef struct rh_gmodel rh_gmodel;
rh_gmodel *rh_gmodel_load(const void *bytes, size_t n, int *err); /* copies the artifact */
size_t rh_gmodel_share_bytes(const void *bytes, size_t n);
rh_gmodel *rh_gmodel_load_shared(const void *bytes, size_t n, int *err, void *arena, size_t arena_bytes);
void rh_gmodel_free(rh_gmodel *m);
rh_gtensor *rh_gm_tensor(rh_gmodel *m, const char *name);         /* NULL if absent */
size_t rh_gmodel_bytes(const rh_gmodel *m);                        /* owned blob + context + private arena; excludes borrowed arena */

typedef struct rh_net rh_net;
/* graphs: encoder, route, anchor, args. NULL on error */
rh_net *rh_net_open(rh_gmodel *encoder, rh_gmodel *route, rh_gmodel *anchor, rh_gmodel *args, char *err, size_t errlen);
void rh_net_close(rh_net *n);
int rh_net_d(const rh_net *n);
void rh_net_set_trace(rh_net *n, sb *trace, int operators);
sb *rh_net_trace(const rh_net *n);
/* encoder states of an arbitrary id sequence (ex->H when ex is given); NULL on error */
const float *rh_net_hidden(rh_net *n, rh_example *ex, const int32_t *ids, int len, rh_arena *A);
int rh_net_route(rh_net *n, rh_example *ex, float *tool, float **count, int *ncount, float *none, rh_arena *A);
/* anchor logits [ninst][nq] for instances (tool index, slot) */
int rh_net_anchors(rh_net *n, rh_example *ex, const int (*insts)[2], int ninst, float *out, rh_arena *A);
typedef struct { int ti, k, pj, anchor; } rh_rowreq;
typedef struct rh_args rh_args;
rh_args *rh_net_args(rh_net *n, rh_example *ex, const rh_rowreq *rows, int nrows, rh_arena *A);
const float *rh_args_op(rh_args *a, int i);
const float *rh_args_aux(rh_args *a, int i);
int rh_args_naux(rh_args *a);
const float *rh_args_spans(rh_args *a, int i);  /* [4][nq + 1], last = null slot */
const float *rh_args_tags(rh_args *a, int i);   /* [nq][3] */
const float *rh_args_enum(rh_args *a, int i, int *E); /* [E], -1e4 beyond the parameter's options */

/* ---- decoding (decode.c) ---- */
typedef struct { const char *key; size_t klen; rv v; } rh_kv; /* one argument; key = String(param.name) */
/* explain mode: why one argument has its value (the program, its confidence, the option probability) */
typedef struct { const char *key; size_t klen; const rh_prog *prog; double conf, opt; int has_opt; } rh_why;
typedef struct {
    const jnode *tool; /* the tool's name value */
    rh_kv *args;
    int nargs, ti, k, pos, ref;
    double first;
    rh_why *why;       /* explain mode: one per argument */
    int nwhy;
    float margin;      /* explain mode: the tool's route margin */
} rh_call;
typedef struct rh_pred {
    rh_call *calls;
    int n;
    float margin;
    /* explain mode: per tool of this sequence its route margin and instance count; per sequence of the record */
    const rh_tslot *tools;
    float *margins;
    int *counts, ntools;
    struct rh_pred *chunks;
    int nchunks;
} rh_pred;
/* predict_records for one record (tools = the record's tool list, text = its request as code points); explain also
   fills the why / margin fields */
int rh_predict_record(rh_net *net, rh_tok *tk, const jnode *rec, const uint32_t *text, size_t tn,
                      const jnode *const *tools, int ntools, double gate, double gate2, int explain, rh_arena *A,
                      rh_pred *out);
/* requireArgs: drop the calls of a record when one leaves a required parameter empty (FIX.perCallHold: only it) */
void rh_require_args(rh_pred *p, const jnode *const *tools, int ntools);
/* JSON.stringify(v) (dedupe keys) and the gauge_run output form of a call list */
void rh_js_stringify(sb *b, const rv *v);
void rh_dump_calls(sb *b, const rh_pred *p);
/* relay.explain() of the JavaScript host: {"calls", "dropped", "chunks": [{"margin", "margins": [{tool, margin, count}]}]};
   calls carry why (per argument {op, conf, optProb, prog} with UTF-16 spans of the request), anchor and margin */
void rh_dump_explain(sb *b, const rh_pred *kept, const rh_pred *all, const uint32_t *text, size_t tn);

#endif
