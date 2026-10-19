/* float32-decoder-v1: constants, loader state and forward declarations (builds with GUST_FLOAT32_DECODER only).
   Part of the runtime/src/gust.c translation unit; see runtime/src/float32/DECODER.md and docs/FORMAT.md. */
#ifndef GUST_FLOAT32_DECODER_H
#define GUST_FLOAT32_DECODER_H
#if defined(GUST_FLOAT32_DECODER) && !defined(GUST_FLOAT32)
#error "GUST_FLOAT32_DECODER needs GUST_FLOAT32 (and its ASR features)"
#endif

/* Tensor.flags bit 8: persistent state. A state tensor is 4 | 8 (float32 state), never an input or an output. */
#define GUST_F32_STATE 8u
/* Capability bit 10 (GUST_F32_DECODER_CAPABILITY) is defined next to bit 9 in abi.h. */
/* Largest state: rows of a key/value cache; the position counter holds an exact integer 0 .. this value. */
#define GUST_F32_STATE_ROWS 256u

/* Operators 38 to 45 (float32 graphs of a decoder build only). */
enum {
    GUST_F_MUL = 38,
    GUST_F_SLICE = 39,
    GUST_F_GATHER = 40,
    GUST_F_SILU = 41,
    GUST_F_ROPE_AT = 42,
    GUST_F_CACHE_APPEND = 43,
    GUST_F_COMMIT = 44,
    GUST_F_MHA_PREFIX = 45,
    GUST_F_MHA_CACHE = 46, /* one-way late fusion: own keys plus a counted prefix of cached keys, one softmax */
    /* float32-temporal-v1 (temporal.c, TEMPORAL.md) */
    GUST_F_DWCONV = 47,
    GUST_F_FILM = 48,
    GUST_F_MEAN = 49
};

/* What the loader learns about the state of a graph while it walks the operators. */
typedef struct {
    int32_t counter;  /* tensor index of the position counter, -1 until an operator names it */
    int32_t position; /* tensor index of the position input shared by appends, the commit and self attention */
    uint32_t commits;
    uint16_t append_at[4096]; /* operator index of the append that writes each state tensor, 0xffff = none yet */
} GdLoad;

static void gd_load_begin(GdLoad *g);
/* Validates one decoder operator (opcode 38 to 45) for float_init; 0 or a GUST_* status. */
static int gd_load_op(const Context *c, GdLoad *g, int32_t *first, int32_t *last, uint32_t index, const Op *o);
/* A fixed-function operator may not read or write a state tensor. */
static int gd_ordinary_operands_ok(const Context *c, const Op *o);
/* Whole-graph checks once every operator was seen. */
static int gd_load_end(const Context *c, const GdLoad *g);
static void gd_zero_state(const Context *c);
/* Run-time checks of every dynamic value, before any arena byte is written. */
static int gd_preflight(const Context *c);
static void gd_run_op(Context *c, const Op *o, uint32_t rows);
/* float32-temporal-v1 operators 47 to 49 (temporal.c): Op fields each may use (gf_used), loader checks, execution. */
#define TP_USED_DWCONV (GF_F(0) | GF_F(1) | GF_F(3) | GF_F(4) | GF_F(5) | GF_F(8) | GF_F(14))
#define TP_USED_FILM   (GF_F(0) | GF_F(1) | GF_F(2) | GF_F(3) | GF_F(16))
#define TP_USED_MEAN   (GF_F(0) | GF_F(1) | GF_F(3))
static int tp_load_op(const Context *c, int32_t *first, int32_t *last, uint32_t index, const Op *o);
static void tp_run_op(Context *c, const Op *o, uint32_t rows);
#endif
