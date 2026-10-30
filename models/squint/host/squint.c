/* Squint's pipeline in C: a port of the JavaScript release's Squint.read (the tracker, the verifier host, the acquirer and
   the partial-code reader), written to return the same results for the same frames. Where the JavaScript relies on a
   language rule (Math.hypot, Math.floor, a stable sort, Float32 storage) the C does the same on purpose.

   The hosts (the C files in host/) are the same C the JavaScript ran as WebAssembly. Each runs as its own instance (see
   tools/instances.py), exactly as the JavaScript instantiated one module per host, because they keep state in statics. */
#include "squint.h"
#include "squint_api.h"
#include "squint_kernels.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

SQUINT_API_GEOMETRY(sq_geometry_)
SQUINT_API_QR(sq_qr_)
SQUINT_API_QUADS(sq_quads_)
SQUINT_API_FLOW(sq_flow_)
SQUINT_API_VERIFIER(sq_verifier_)
SQUINT_API_VERIFIER(sq_acquirer_)
double sq_clock_seconds(void);

#define GEO(n) sq_geometry_##n /* the standalone geometry module: soft sampling for the tracker and the partial reader */
#define QRD(n) sq_qr_##n       /* the standalone decoder module */
#define QUA(n) sq_quads_##n    /* white-patch detection */
#define FLO(n) sq_flow_##n     /* sparse optical flow */
#define VER(n) sq_verifier_##n /* the tracker's verifier host: registration, fresh decode, evidence certificate, acquisition */
#define ACQ(n) sq_acquirer_##n /* the acquirer's own copy of the verifier host */

#define MAXPIX (SQ_MAX_WIDTH * SQ_MAX_HEIGHT)
#define MAXTRACK 196 /* the host's template side for versions 1 to 6: (41 + 8) * 4 */

/* ---- state -------------------------------------------------------------------------------------------------------- */

typedef struct { /* VerifiedState: the code being tracked */
    int version, size, payload_len;
    uint8_t payload[SQ_MAX_PAYLOAD];
    uint8_t encoded[SQ_MAX_GRID]; /* clean module grid, n x n, 1 = dark */
    float candidate[MAXTRACK * MAXTRACK];
    float targets[3][MAXTRACK * MAXTRACK];
    double corners[8];
    double fresh_time;
} vstate;

typedef struct { /* a soft sample of the module grid */
    int valid, has_vis;
    float prob[SQ_MAX_GRID];
    uint8_t vis[SQ_MAX_GRID];
    double cost;
} soft_t;

typedef struct { /* the decoder's answer */
    int accepted, reason, payload_len, residual_parity;
} qdec_t;

static struct {
    int opened, failed;
    char err[160];
    /* VerifiedPayloadReader */
    int has_state;
    vstate st;
    unsigned serial, cached_serial; /* identity of the state, and of the state the verifier host holds */
    int session;                    /* -1: none */
    double last_time;
    int width, height;
    int flow_valid;
    int contra_set, contra_no_sample;
    double contra_confident, contra_fraction;
    /* CPUScanPipeline */
    int fail_streak, lost_streak;
    /* PartialPatchReader */
    int p_version, p_session;
    double p_last_verified;
    /* geometry module */
    int gw, gh, g_soft_version;
    /* Squint */
    int sess, tick, tracked, acq_frames;
} S;

static soft_t s_quick;
static soft_t s_cand[8];
static double s_cand_corners[8][8];
static double s_cand_cost[8];

static double now_ms(void) { return sq_clock_seconds() * 1000.0; }

static int fail(const char *msg) {
    if (!S.failed) snprintf(S.err, sizeof S.err, "%s", msg);
    S.failed = 1;
    return -1;
}

/* ---- the JavaScript helpers --------------------------------------------------------------------------------------- */

/* squint-core.ts validQuad: four finite corners forming a strictly convex quadrilateral with an area of at least 2. */
static int valid_quad(const double *q) {
    for (int i = 0; i < 8; i++)
        if (!isfinite(q[i])) return 0;
    double sign = 0, area = 0;
    for (int i = 0; i < 4; i++) {
        const double *a = q + 2 * i, *b = q + 2 * ((i + 1) % 4), *c = q + 2 * ((i + 2) % 4);
        double cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if (fabs(cross) < 1e-9) return 0;
        double s = cross > 0 ? 1 : -1;
        if (sign != 0 && sign != s) return 0;
        sign = s;
        area += a[0] * b[1] - b[0] * a[1];
    }
    return fabs(area) >= 2;
}

/* squint-core.ts homography: destination pixel index to source pixel index for four clockwise corners. */
static int homography(const double *c, int size, double *h) {
    if (!valid_quad(c)) return fail("Invalid homography geometry");
    double dest[4][2] = {{0, 0}, {size - 1, 0}, {size - 1, size - 1}, {0, size - 1}};
    double m[8][9];
    for (int i = 0; i < 4; i++) {
        double x = dest[i][0], y = dest[i][1], u = c[2 * i], v = c[2 * i + 1];
        double r0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        double r1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        memcpy(m[2 * i], r0, sizeof r0);
        memcpy(m[2 * i + 1], r1, sizeof r1);
    }
    for (int col = 0; col < 8; col++) {
        int pivot = col;
        for (int row = col + 1; row < 8; row++)
            if (fabs(m[row][col]) > fabs(m[pivot][col])) pivot = row;
        if (fabs(m[pivot][col]) < 1e-12) return fail("Singular homography");
        if (pivot != col) {
            double t[9];
            memcpy(t, m[col], sizeof t);
            memcpy(m[col], m[pivot], sizeof t);
            memcpy(m[pivot], t, sizeof t);
        }
        double scale = m[col][col];
        for (int j = col; j < 9; j++) m[col][j] /= scale;
        for (int row = 0; row < 8; row++)
            if (row != col) {
                double factor = m[row][col];
                for (int j = col; j < 9; j++) m[row][j] -= factor * m[col][j];
            }
    }
    for (int i = 0; i < 8; i++) h[i] = m[i][8];
    h[8] = 1;
    return 0;
}

static void hmap(const double *h, double x, double y, double *ox, double *oy) {
    double d = h[6] * x + h[7] * y + 1;
    *ox = (h[0] * x + h[1] * y + h[2]) / d;
    *oy = (h[3] * x + h[4] * y + h[5]) / d;
}

/* V8's Math.hypot(a, b): scaled by the largest magnitude, summed with Kahan compensation. */
static double js_hypot(double a, double b) {
    double v[2] = {fabs(a), fabs(b)}, max = 0;
    for (int i = 0; i < 2; i++)
        if (v[i] > max) max = v[i];
    if (max == 0) return 0;
    double sum = 0, comp = 0;
    for (int i = 0; i < 2; i++) {
        double n = v[i] / max, summand = n * n - comp, prelim = sum + summand;
        comp = (prelim - sum) - summand;
        sum = prelim;
    }
    return sqrt(sum) * max;
}

/* ---- the geometry and decoder modules ----------------------------------------------------------------------------- */

static int geo_set_frame(const uint8_t *gray, int w, int h) {
    memcpy((void *)GEO(sg_frame)(), gray, (size_t)w * (size_t)h);
    S.gw = w;
    S.gh = h;
    return 0;
}

static int geo_set_quad(const double *c) {
    if (!valid_quad(c)) return fail("Invalid geometry");
    for (int i = 0; i < 8; i++)
        if (fabs(c[i]) > 1e9) return fail("Invalid geometry");
    memcpy((void *)GEO(sg_quad)(), c, 8 * sizeof(double));
    return 0;
}

/* SquintGeometryWasm.sampleSoft */
static int sample_soft(const double *corners, int version, double ox, double oy, int partial, soft_t *o) {
    GEO(sg_partial_sampling)(partial ? 1 : 0);
    if (geo_set_quad(corners)) return -1;
    int n = 17 + 4 * version;
    if (S.g_soft_version != version) {
        QRD(sq_known_layout)(version, (uint8_t *)GEO(sg_known_mask)(), (uint8_t *)GEO(sg_known_values)());
        S.g_soft_version = version;
    }
    int status = GEO(sg_sample_grid)(S.gw, S.gh, S.gw, version, ox, oy);
    if (status < 0) return fail("Soft sampling failed");
    o->valid = status != 0;
    if (!status) return 0;
    const double *grid = (const double *)GEO(sg_soft_grid)();
    for (int i = 0; i < n * n; i++) o->prob[i] = (float)grid[i];
    o->has_vis = partial;
    if (partial) memcpy(o->vis, (const void *)GEO(sg_visible)(), (size_t)n * n);
    o->cost = ((const double *)GEO(sg_result)())[1];
    return 0;
}

/* SquintQRWasm.decodeInto with the options the scanner uses; the rest are the defaults of the JavaScript. */
static int q_decode(const float *prob, int n, const uint8_t *vis, int adaptive, int ranked, double erasure_conf,
                    int max_erasures, int min_parity, int reliability, qdec_t *o) {
    QRD(sq_padding)(1);
    if (vis) {
        int visible = 0;
        for (int i = 0; i < n * n; i++) visible += vis[i];
        if (visible < n * n * 0.5) return fail("Invalid visible module support");
        memcpy((void *)QRD(sq_visible)(), vis, (size_t)n * n);
    }
    QRD(sq_visibility)(vis ? 1 : 0);
    QRD(sq_reliability)(reliability); /* 0 minimum, 1 product, 2 mean */
    QRD(sq_format_rank)(-1);
    QRD(sq_gmd)(adaptive ? 1 : 0);
    if (QRD(sq_reserve)(min_parity)) return fail("Invalid verification reserve");
    double *in = (double *)QRD(sq_input)();
    for (int i = 0; i < n * n; i++) in[i] = (double)prob[i];
    int status = QRD(sq_decode)(n, 1, erasure_conf, max_erasures, ranked ? 1 : 0, 0.12, 0.1, 1, 0.05);
    if (status < 0) return fail("Invalid probability values");
    const int32_t *r = (const int32_t *)QRD(sq_result)();
    o->accepted = status != 0;
    o->reason = r[1];
    o->payload_len = r[5];
    o->residual_parity = r[10];
    return 0;
}

static void clear_result(sq_read_result *r, const char *reason) {
    memset(r, 0, offsetof(sq_read_result, payload));
    r->reason = reason;
}

static void copy_corners(double *dst, const double *src) { memcpy(dst, src, 8 * sizeof(double)); }

/* A decoder answer as a read: payload and grid from the decoder module. */
static void result_from_qr(sq_read_result *r, const qdec_t *q, int version, const double *corners, const char *path) {
    int n = 17 + 4 * version;
    memset(r, 0, offsetof(sq_read_result, payload));
    r->accepted = 1;
    r->path = path;
    r->version = version;
    r->payload_len = q->payload_len;
    memcpy(r->payload, (const void *)QRD(sq_payload)(), (size_t)q->payload_len);
    r->has_grid = 1;
    r->grid_len = n * n;
    memcpy(r->grid, (const void *)QRD(sq_grid)(), (size_t)n * n);
    copy_corners(r->corners, corners);
}

/* ---- the partial-code reader's state (partial-patch-reader.ts) ---------------------------------------------------- */

static void partial_reset(void) {
    S.p_version = 0;
    S.p_last_verified = -INFINITY;
    S.p_session = -1;
}

static void partial_observe(const sq_read_result *r, double time, int session) {
    if (S.p_session != session) {
        partial_reset();
        S.p_session = session;
    }
    if (r->accepted && r->version >= 1 && r->version <= 10) {
        S.p_version = r->version;
        S.p_last_verified = time;
    }
}

/* GrayScanPipeline.observe: the only tracker that matters here is the partial reader's version hint. */
static void pipeline_observe(const sq_read_result *r, double time, int session) { partial_observe(r, time, session); }

/* ---- VerifiedPayloadReader ---------------------------------------------------------------------------------------- */

static void vr_reset(void) {
    S.has_state = 0;
    S.session = -1;
    S.last_time = -INFINITY;
    S.width = S.height = 0;
    FLO(sf_reset)();
}

static void vr_check(int width, int height, double time, int session) {
    if (S.session != session || width != S.width || height != S.height || time < S.last_time) {
        vr_reset();
        S.session = session;
        S.width = width;
        S.height = height;
    }
    S.last_time = time;
    if (S.has_state && time - S.st.fresh_time > 3000) S.has_state = 0;
}

/* gaussian(): separable blur of a size x size image, edge-clamped, Float32 in and out, double sums. */
static void gaussian(const float *in, int size, int sigma, float *out) {
    static float tmp[MAXTRACK * MAXTRACK];
    const uint32_t *bits = sigma == 1 ? sq_kernel1_bits : sigma == 2 ? sq_kernel2_bits : sq_kernel3_bits;
    int radius = sigma == 1 ? SQ_KERNEL1_RADIUS : sigma == 2 ? SQ_KERNEL2_RADIUS : SQ_KERNEL3_RADIUS;
    float k[19];
    for (int i = 0; i < 2 * radius + 1; i++) memcpy(&k[i], &bits[i], 4);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            double value = 0;
            for (int j = -radius; j <= radius; j++) {
                int xx = x + j < 0 ? 0 : x + j > size - 1 ? size - 1 : x + j;
                value += (double)in[y * size + xx] * (double)k[j + radius];
            }
            tmp[y * size + x] = (float)value;
        }
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            double value = 0;
            for (int j = -radius; j <= radius; j++) {
                int yy = y + j < 0 ? 0 : y + j > size - 1 ? size - 1 : y + j;
                value += (double)tmp[yy * size + x] * (double)k[j + radius];
            }
            out[y * size + x] = (float)value;
        }
}

static void flow_set(const uint8_t *gray, int w, int h) { memcpy((void *)FLO(sf_input)(), gray, (size_t)w * (size_t)h); }

/* VerifiedPayloadReader.anchor: adopts a fresh read as the tracked code and seeds the flow tracker. */
static int vr_anchor(const uint8_t *gray, int width, int height, double time, int session, const sq_read_result *r) {
    vr_check(width, height, time, session);
    if (!r->accepted || !valid_quad(r->corners) || r->version < 1) return 0;
    int version = r->version, n = 17 + 4 * version;
    if (n > 41) { /* tracking covers versions 1 to 6; larger codes are re-acquired each frame */
        S.has_state = 0;
        return 0;
    }
    geo_set_frame(gray, width, height);
    const uint8_t *grid = r->grid;
    uint8_t derived[SQ_MAX_GRID];
    if (!r->has_grid) { /* re-derive the clean grid from the current image; it must decode to the same payload */
        if (sample_soft(r->corners, version, 0, 0, 1, &s_quick) < 0) return -1;
        if (!s_quick.valid) return 0;
        qdec_t q;
        if (q_decode(s_quick.prob, n, s_quick.has_vis ? s_quick.vis : NULL, 1, 0, 0.18, 30, 2, 0, &q) < 0) return -1;
        if (!q.accepted || q.payload_len != r->payload_len ||
            memcmp((const void *)QRD(sq_payload)(), r->payload, (size_t)r->payload_len))
            return 0;
        memcpy(derived, (const void *)QRD(sq_grid)(), (size_t)n * n);
        grid = derived;
    }
    int same = S.has_state && S.st.version == version && S.st.payload_len == r->payload_len &&
               !memcmp(S.st.payload, r->payload, (size_t)r->payload_len) && !memcmp(S.st.encoded, grid, (size_t)n * n);
    if (same) {
        copy_corners(S.st.corners, r->corners);
        S.st.fresh_time = time;
    } else {
        vstate *s = &S.st;
        int size = (n + 8) * 4;
        s->version = version;
        s->size = size;
        s->payload_len = r->payload_len;
        memcpy(s->payload, r->payload, (size_t)r->payload_len);
        memcpy(s->encoded, grid, (size_t)n * n);
        for (int i = 0; i < size * size; i++) s->candidate[i] = 1.0f;
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        s->candidate[((y + 4) * 4 + dy) * size + (x + 4) * 4 + dx] = (float)(1 - grid[y * n + x]);
        for (int t = 0; t < 3; t++) gaussian(s->candidate, size, t + 1, s->targets[t]);
        copy_corners(s->corners, r->corners);
        s->fresh_time = time;
        S.has_state = 1;
        S.serial++;
    }
    flow_set(gray, width, height);
    memcpy((void *)FLO(sf_quad)(), r->corners, 8 * sizeof(double));
    FLO(sf_frame)(width, height, 1);
    return 0;
}

/* BrowserVerifierHost.cache: uploads the per-code data when the tracked state changed. */
static void host_cache(void) {
    vstate *s = &S.st;
    int n = 17 + 4 * s->version;
    memcpy((void *)VER(se_grid)(), s->encoded, (size_t)n * n);
    memcpy((void *)VER(sv_candidate)(), s->candidate, (size_t)s->size * s->size * sizeof(float));
    for (int t = 0; t < 3; t++)
        memcpy((float *)VER(sv_targets)() + (size_t)t * s->size * s->size, s->targets[t],
               (size_t)s->size * s->size * sizeof(float));
    VER(sq_known_layout)(s->version, (uint8_t *)VER(sg_known_mask)(), (uint8_t *)VER(sg_known_values)());
    VER(sv_motion_targets)(s->size, 0);
    VER(sv_phase_cache)(s->size);
    S.cached_serial = S.serial;
}

/* BrowserVerifierHost.verify: registration, then a fresh decode or the evidence certificate. */
static int host_verify(const uint8_t *gray, int width, int height, int flow_valid, double deadline, sq_read_result *r) {
    vstate *s = &S.st;
    if (S.cached_serial != S.serial || S.cached_serial == 0) host_cache();
    memcpy((void *)VER(sg_frame)(), gray, (size_t)width * (size_t)height);
    memcpy((void *)VER(sv_corners)(), s->corners, 8 * sizeof(double));
    VER(sv_geometry_gate)(-1);
    int result = VER(sv_browser)(width, height, s->version, flow_valid ? 1 : 0, deadline / 1000);
    if (result < 0) return fail("Browser verifier failed");
    const double *quad = (const double *)VER(sv_quad)();
    int n = 17 + 4 * s->version;
    if (result == 2) { /* a fresh decode of the registered current image */
        const int32_t *q = (const int32_t *)VER(sq_result)();
        clear_result(r, NULL);
        r->accepted = 1;
        r->version = s->version;
        r->path = "WASM registered fresh QR decode";
        r->payload_len = q[5];
        memcpy(r->payload, (const void *)VER(sq_payload)(), (size_t)q[5]);
        r->has_grid = 1;
        r->grid_len = n * n;
        memcpy(r->grid, (const void *)VER(sq_grid)(), (size_t)n * n);
        copy_corners(r->corners, quad);
        return 0;
    }
    if (!result) {
        if (VER(sv_registered)()) copy_corners(s->corners, quad); /* geometryFromRegistration */
        clear_result(r, "current-image-verification-failed");
        return 0;
    }
    copy_corners(s->corners, quad); /* 4: spectral, 5: evidence certificate */
    clear_result(r, NULL);
    r->accepted = 1;
    r->version = s->version;
    r->path = result == 5 ? "WASM evidence-certified current-image verification" : "WASM spectral current-image verification";
    r->payload_len = s->payload_len;
    memcpy(r->payload, s->payload, (size_t)s->payload_len);
    r->has_grid = 1;
    r->grid_len = n * n;
    memcpy(r->grid, s->encoded, (size_t)n * n);
    copy_corners(r->corners, s->corners);
    return 0;
}

static void measure_contradiction(const soft_t *sample, const vstate *s, int n) {
    int confident = 0, disagree = 0;
    for (int i = 0; i < n * n; i++) {
        if (sample->has_vis && !sample->vis[i]) continue;
        double c = fabs((double)sample->prob[i] - 0.5);
        if (c < 0.35) continue;
        confident++;
        if ((sample->prob[i] >= 0.5) != (s->encoded[i] != 0)) disagree++;
    }
    S.contra_set = 1;
    S.contra_no_sample = 0;
    S.contra_confident = confident;
    S.contra_fraction = confident ? (double)disagree / (double)confident : 0;
}

static void measure_flatness(const uint8_t *gray, int width, int height, const double *q) {
    double x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
    for (int i = 0; i < 4; i++) {
        double x = q[2 * i], y = q[2 * i + 1];
        x0 = fmin(x0, x);
        x1 = fmax(x1, x);
        y0 = fmin(y0, y);
        y1 = fmax(y1, y);
    }
    x0 = fmax(0, floor(x0));
    y0 = fmax(0, floor(y0));
    x1 = fmin(width - 1, ceil(x1));
    y1 = fmin(height - 1, ceil(y1));
    double sum = 0, sq = 0, count = 0;
    if (x1 > x0 && y1 > y0)
        for (int j = 0; j < 32; j++) {
            double y = floor(y0 + ((y1 - y0) * (j + 0.5)) / 32);
            for (int i = 0; i < 32; i++) {
                double v = gray[(int)y * width + (int)floor(x0 + ((x1 - x0) * (i + 0.5)) / 32)];
                sum += v;
                sq += v * v;
                count++;
            }
        }
    double mean = count ? sum / count : 0;
    double std = count ? sqrt(fmax(0, sq / count - mean * mean)) : 0;
    S.contra_set = 1;
    S.contra_confident = 0;
    S.contra_fraction = 0;
    S.contra_no_sample = std < 12;
}

/* VerifiedPayloadReader.read: a quick hinted decode at the flow-predicted position (robust: five sub-module offsets and
   three symbol-reliability rules), or the verifier host (registration, fresh decode, evidence certificate). Returns -1 on
   an exception of the JavaScript. */
static soft_t s_hint[5];
static int vr_read(const uint8_t *gray, int width, int height, double time, int session, int quick_only,
                   int prediction_ready, int robust, double deadline, sq_read_result *out) {
    vr_check(width, height, time, session);
    if (!S.has_state || width % 16 || height % 16 || width > SQ_MAX_WIDTH || height > SQ_MAX_HEIGHT) {
        clear_result(out, "no-verified-payload");
        return 0;
    }
    vstate *s = &S.st;
    int flow_valid = S.flow_valid;
    if (!prediction_ready) {
        flow_set(gray, width, height);
        memcpy((void *)FLO(sf_quad)(), s->corners, 8 * sizeof(double));
        flow_valid = FLO(sf_frame)(width, height, 0) == 1;
        if (flow_valid) {
            double q[8];
            memcpy(q, (const void *)FLO(sf_quad)(), sizeof q);
            int ok = valid_quad(q);
            for (int i = 0; ok && i < 8; i++) ok = fabs(q[i]) < 1e5;
            if (ok) copy_corners(s->corners, q);
            else flow_valid = 0;
        }
        S.flow_valid = flow_valid;
    }
    int n = 17 + 4 * s->version, rc = 0;
    geo_set_frame(gray, width, height);
    GEO(sg_normalized)(1);
    if (quick_only) {
        S.contra_set = 1;
        S.contra_confident = 0;
        S.contra_fraction = 0;
        S.contra_no_sample = 1;
        if (!flow_valid) {
            clear_result(out, "no-current-motion-hint");
            goto done;
        }
        const double radius = 0.15; /* hintRadius */
        const double offsets[5][2] = {{0, 0}, {-radius, 0}, {radius, 0}, {0, -radius}, {0, radius}};
        int count = 0;
        for (int k = 0; k < (robust ? 5 : 1); k++) {
            static soft_t tmp;
            if (sample_soft(s->corners, s->version, offsets[k][0], offsets[k][1], 1, &tmp) < 0) {
                rc = -1;
                goto done;
            }
            if (!tmp.valid) continue;
            int pos = count; /* stable ascending by cost */
            for (int i = 0; i < count; i++)
                if (s_hint[i].cost > tmp.cost) {
                    pos = i;
                    break;
                }
            for (int i = count; i > pos; i--) memcpy(&s_hint[i], &s_hint[i - 1], sizeof(soft_t));
            memcpy(&s_hint[pos], &tmp, sizeof(soft_t));
            count++;
        }
        int use = robust ? (count < 3 ? count : 3) : (count < 1 ? count : 1); /* hintSamples */
        for (int k = 0; k < use; k++)
            for (int rel = 0; rel < (robust ? 3 : 1); rel++) {
                if (now_ms() >= deadline) {
                    clear_result(out, "scan-budget");
                    goto done;
                }
                qdec_t q;
                if (q_decode(s_hint[k].prob, n, s_hint[k].has_vis ? s_hint[k].vis : NULL, 1, robust, robust ? 1.0 : 0.5,
                             30, 2, rel, &q) < 0) {
                    rc = -1;
                    goto done;
                }
                if (q.accepted) {
                    result_from_qr(out, &q, s->version, s->corners, "Current-image hinted native QR decoder");
                    rc = vr_anchor(gray, width, height, time, session, out);
                    goto done;
                }
            }
        if (count) measure_contradiction(&s_hint[0], s, n);
        else measure_flatness(gray, width, height, s->corners);
        clear_result(out, "hint-native-failed");
        goto done;
    }
    if (host_verify(gray, width, height, flow_valid, deadline, out) < 0) {
        rc = -1;
        goto done;
    }
    if (out->accepted && out->path && strstr(out->path, "fresh")) rc = vr_anchor(gray, width, height, time, session, out);
done:
    GEO(sg_normalized)(0);
    return rc;
}

/* ---- full-frame acquisition ---------------------------------------------------------------------------------------- */

/* The own reader on the tracker's verifier host (createOwnReader): finder detection, geometry, own Reed-Solomon. */
static int own_read(const uint8_t *gray, int width, int height, double deadline, sq_read_result *out) {
    if (width > SQ_MAX_WIDTH || height > SQ_MAX_HEIGHT || width < 64 || height < 64 || width % 2) return 0;
    memcpy((void *)VER(sg_frame)(), gray, (size_t)width * (size_t)height);
    if (VER(sa_acquire_wasm)(width, height, isfinite(deadline) ? deadline / 1000 : 1e300) != 1) return 0;
    const int32_t *r = (const int32_t *)VER(sq_result)();
    clear_result(out, NULL);
    out->accepted = 1;
    out->path = "Squint acquisition";
    out->payload_len = r[5];
    memcpy(out->payload, (const void *)VER(sq_payload)(), (size_t)r[5]);
    out->version = VER(sa_acquired_version)();
    memcpy(out->corners, (const void *)VER(sa_acquired_quad)(), 8 * sizeof(double));
    return 1;
}

/* PartialPatchReader.read: a recently verified code whose bottom is clipped by the frame edge. */
static int partial_read(const uint8_t *gray, int width, int height, double time, int session, double deadline,
                        sq_read_result *out) {
    if (S.p_session != session || time < S.p_last_verified) {
        partial_reset();
        S.p_session = session;
    }
    if (time - S.p_last_verified > 4500) S.p_version = 0;
    if (!S.p_version) {
        clear_result(out, "no-version-hint");
        return 0;
    }
    int bright = 0;
    for (int y = height - 9 > 0 ? height - 9 : 0; y < height && bright < 4; y += 4)
        for (int x = 0; x < width && bright < 4; x += 8) bright += gray[y * width + x] > 140;
    if (bright < 4) {
        clear_result(out, "no-clipped-white-patch");
        return 0;
    }
    int version = S.p_version, n = 17 + 4 * version, rc = 0, count = 0;
    geo_set_frame(gray, width, height);
    GEO(sg_normalized)(1);
    QUA(sqw_profiles)(1);
    QUA(sqw_refine)(0);
    memcpy((void *)QUA(sqw_input)(), gray, (size_t)width * (size_t)height);
    int patches = QUA(sqw_detect)(width, height);
    if (patches < 0 || patches > 16) {
        rc = fail("White-patch detection failed");
        goto done;
    }
    const int32_t *quads = (const int32_t *)QUA(sqw_quads)();
    for (int p = 0; p < patches; p++)
        for (int rotation = 0; rotation < 4; rotation++) {
            double q[4][2];
            for (int k = 0; k < 4; k++) {
                q[k][0] = quads[p * 8 + ((k + rotation) % 4) * 2];
                q[k][1] = quads[p * 8 + ((k + rotation) % 4) * 2 + 1];
            }
            if (!(q[2][1] >= height - 6 && q[3][1] >= height - 6) || q[0][1] >= height - 10 || q[1][1] >= height - 10)
                continue;
            double top = js_hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]);
            double left = js_hypot(q[3][0] - q[0][0], q[3][1] - q[0][1]);
            double right = js_hypot(q[2][0] - q[1][0], q[2][1] - q[1][1]);
            if (now_ms() >= deadline) {
                clear_result(out, "scan-budget");
                goto done;
            }
            static const double factors[3] = {1, 0.95, 1.05};
            static const int borders[3] = {3, 4, 5};
            for (int fi = 0; fi < 3; fi++)
                for (int bi = 0; bi < 3; bi++) {
                    double factor = factors[fi];
                    int border = borders[bi];
                    if (now_ms() >= deadline) {
                        clear_result(out, "scan-budget");
                        goto done;
                    }
                    double full[8] = {q[0][0], q[0][1], q[1][0], q[1][1], 0, 0, 0, 0};
                    for (int k = 0; k < 2; k++) {
                        full[4 + k] = q[1][k] + ((q[2][k] - q[1][k]) * top * factor) / right;
                        full[6 + k] = q[0][k] + ((q[3][k] - q[0][k]) * top * factor) / left;
                    }
                    if (!valid_quad(full)) continue;
                    double h[9], sh[9], ch[9];
                    if (homography(full, 128, h) < 0) {
                        rc = -1;
                        goto done;
                    }
                    double low = ((double)(border - 4) / (double)(n + 2 * border)) * 127;
                    double high = 127 - low;
                    double sq4[4][2] = {{low, low}, {high, low}, {high, high}, {low, high}};
                    double standard[8];
                    for (int k = 0; k < 4; k++) hmap(h, sq4[k][0], sq4[k][1], &standard[2 * k], &standard[2 * k + 1]);
                    int ok = valid_quad(standard);
                    for (int k = 0; ok && k < 8; k++) ok = !(fabs(standard[k]) > 1e6);
                    if (!ok) continue;
                    /* Align the quad's top finders with the two visible ones (similarity transform). */
                    if (geo_set_quad(standard)) {
                        rc = -1;
                        goto done;
                    }
                    if (GEO(sg_partial_fit)(width, height, width, version)) {
                        rc = fail("Partial geometry failed");
                        goto done;
                    }
                    const double *fr = (const double *)GEO(sg_result)();
                    if (!fr[0]) continue;
                    double c0x = fr[5], c0y = fr[6], c1x = fr[7], c1y = fr[8];
                    if (homography(standard, 128, sh) < 0) {
                        rc = -1;
                        goto done;
                    }
                    double ax, ay, bx, by;
                    hmap(sh, (7.5 * 127) / (n + 8), (7.5 * 127) / (n + 8), &ax, &ay);
                    hmap(sh, ((n + 0.5) * 127) / (n + 8), (7.5 * 127) / (n + 8), &bx, &by);
                    double u0 = bx - ax, u1 = by - ay, v0 = c1x - c0x, v1 = c1y - c0y;
                    double norm = u0 * u0 + u1 * u1;
                    double real = (u0 * v0 + u1 * v1) / norm, imag = (u0 * v1 - u1 * v0) / norm;
                    double corrected[8];
                    for (int k = 0; k < 4; k++) {
                        double px = standard[2 * k], py = standard[2 * k + 1];
                        corrected[2 * k] = c0x + real * (px - ax) - imag * (py - ay);
                        corrected[2 * k + 1] = c0y + imag * (px - ax) + real * (py - ay);
                    }
                    if (!valid_quad(corrected)) continue;
                    if (homography(corrected, 128, ch) < 0) {
                        rc = -1;
                        goto done;
                    }
                    double lo = (4.0 * 127) / (n + 8), hi = ((double)(n + 4) * 127) / (n + 8);
                    double sq5[4][2] = {{lo, lo}, {hi, lo}, {hi, hi}, {lo, hi}};
                    double corners[8];
                    for (int k = 0; k < 4; k++) hmap(ch, sq5[k][0], sq5[k][1], &corners[2 * k], &corners[2 * k + 1]);
                    static soft_t sample;
                    if (sample_soft(corners, version, 0, 0, 1, &sample) < 0) {
                        rc = -1;
                        goto done;
                    }
                    if (!sample.valid) continue;
                    /* Keep the eight lowest-cost samples; equal costs keep their order (the JavaScript sort is stable). */
                    int pos = count;
                    for (int i = 0; i < count; i++)
                        if (s_cand_cost[i] > sample.cost) {
                            pos = i;
                            break;
                        }
                    if (pos >= 8) continue;
                    int last = count < 8 ? count : 7;
                    for (int i = last; i > pos; i--) {
                        memcpy(&s_cand[i], &s_cand[i - 1], sizeof(soft_t));
                        memcpy(s_cand_corners[i], s_cand_corners[i - 1], sizeof s_cand_corners[0]);
                        s_cand_cost[i] = s_cand_cost[i - 1];
                    }
                    memcpy(&s_cand[pos], &sample, sizeof(soft_t));
                    copy_corners(s_cand_corners[pos], corners);
                    s_cand_cost[pos] = sample.cost;
                    if (count < 8) count++;
                }
        }
    for (int i = 0; i < count; i++) {
        if (now_ms() >= deadline) {
            clear_result(out, "scan-budget");
            goto done;
        }
        qdec_t q;
        if (q_decode(s_cand[i].prob, n, s_cand[i].has_vis ? s_cand[i].vis : NULL, 1, 1, 1.0, 30, 2, 0, &q) < 0) {
            rc = -1;
            goto done;
        }
        if (q.accepted) {
            result_from_qr(out, &q, version, s_cand_corners[i], "Experimental current-frame partial-code rescue");
            partial_observe(out, time, session);
            goto done;
        }
    }
    clear_result(out, "decode-failed");
done:
    GEO(sg_normalized)(0);
    return rc;
}

/* GrayScanPipeline.scan with the scanner's settings: no tracking, no extra patch reader. */
static int gray_scan(const uint8_t *gray, int width, int height, double time, int session, double deadline,
                     sq_read_result *out) {
    if (now_ms() + 2 >= deadline) {
        clear_result(out, "scan-budget");
        return 0;
    }
    if (own_read(gray, width, height, deadline, out)) {
        pipeline_observe(out, time, session);
        return 0;
    }
    if (now_ms() >= deadline) {
        clear_result(out, "scan-budget");
        return 0;
    }
    /* The module rescue of codes the reader located but could not decode has nothing to do: Squint's reader returns
       only codes it decoded. */
    if (now_ms() >= deadline) {
        clear_result(out, "scan-budget");
        return 0;
    }
    int patch_supported = width <= SQ_MAX_WIDTH && height <= SQ_MAX_HEIGHT && width % 2 == 0 && height % 2 == 0;
    if (patch_supported) {
        if (partial_read(gray, width, height, time, session, deadline, out) < 0) return -1;
        if (out->accepted) {
            pipeline_observe(out, time, session);
            return 0;
        }
    }
    clear_result(out, "decode-failed");
    return 0;
}

/* ---- CPUScanPipeline.scan (noGlobal) ------------------------------------------------------------------------------ */

static void cpu_accept(const sq_read_result *r, double time, int session) {
    S.fail_streak = 0;
    pipeline_observe(r, time, session);
}

/* `robust`: the warm-up's hint settings. `change_first`: the production policy that puts the global reader first on a
   measured loss. `no_global`: never fall back to the full-frame reader (what Squint.read asks of the tracker). */
static int cpu_scan(const uint8_t *gray, int width, int height, double time, int session, double budget_ms, int robust,
                    int change_first, int no_global, sq_read_result *out) {
    double start = now_ms(), deadline = isfinite(budget_ms) ? start + budget_ms : INFINITY;
    if (vr_read(gray, width, height, time, session, 1, 0, robust, deadline, out) < 0) return -1;
    if (out->accepted) {
        cpu_accept(out, time, session);
        return 0;
    }
    if (now_ms() >= deadline) {
        clear_result(out, "scan-budget");
        return 0;
    }
    /* Measured tracking loss: a confident disagreement with the tracked code, or nothing to sample, puts the global
       reader first. */
    int changed = change_first && S.contra_set &&
                  (S.contra_no_sample || (S.contra_confident >= 120 && S.contra_fraction >= 0.25));
    if (changed && S.has_state) {
        if (gray_scan(gray, width, height, time, session, deadline, out) < 0) return -1;
        if (out->accepted) {
            S.fail_streak = 0;
            S.lost_streak = 0;
            if (vr_anchor(gray, width, height, time, session, out) < 0) return -1;
            return 0;
        }
        S.fail_streak++;
        if (now_ms() + 6 < deadline) {
            if (vr_read(gray, width, height, time, session, 0, 1, robust, deadline, out) < 0) return -1;
            if (out->accepted) {
                S.lost_streak = 0;
                cpu_accept(out, time, session);
                return 0;
            }
        }
        clear_result(out, now_ms() >= deadline ? "scan-budget" : "decode-failed");
        return 0;
    }
    /* Bounded recovery (registration and verifier) before the global reader. */
    if (vr_read(gray, width, height, time, session, 0, 1, robust, deadline, out) < 0) return -1;
    if (out->accepted) {
        cpu_accept(out, time, session);
        return 0;
    }
    if (no_global) {
        clear_result(out, "decode-failed");
        return 0;
    }
    if (now_ms() + 2 >= deadline) {
        clear_result(out, "scan-budget");
        return 0;
    }
    /* The global reader is not interruptible: start it only if it fits under the hard ceiling. */
    if (isfinite(budget_ms) && now_ms() + 3 > start + budget_ms + 2) {
        if (S.has_state) S.fail_streak++;
        clear_result(out, "scan-budget");
        return 0;
    }
    if (gray_scan(gray, width, height, time, session, deadline, out) < 0) return -1;
    if (out->accepted) {
        S.fail_streak = 0;
        if (vr_anchor(gray, width, height, time, session, out) < 0) return -1;
        return 0;
    }
    if (S.has_state) S.fail_streak++;
    return 0;
}

/* ---- Acquirer ------------------------------------------------------------------------------------------------------ */

static int acquire(const uint8_t *gray, int width, int height, int tracked, double budget_ms, double tracked_budget_ms,
                   int enhance, sq_read_result *out) {
    double start = now_ms();
    if (width > SQ_MAX_WIDTH || height > SQ_MAX_HEIGHT || width < 64 || height < 64 || width % 2) {
        clear_result(out, "unsupported-size");
        return 0;
    }
    memcpy((void *)ACQ(sg_frame)(), gray, (size_t)width * (size_t)height);
    double budget = tracked ? tracked_budget_ms : budget_ms, deadline = start + budget;
    int enhanced = enhance && !tracked && width == 1280 && height == 720 && (++S.acq_frames & 1);
    if (enhanced) ACQ(sa_enhance_frame)(width, height);
    if (ACQ(sa_acquire_wasm)(width, height, isfinite(deadline) ? deadline / 1000 : 1e300) != 1) {
        clear_result(out, "acquisition-failed");
        return 0;
    }
    const int32_t *r = (const int32_t *)ACQ(sq_result)();
    int version = ACQ(sa_acquired_version)(), n = 17 + 4 * version;
    clear_result(out, NULL);
    out->accepted = 1;
    out->version = version;
    out->path = enhanced ? "Squint acquisition (lighting-normalized)" : "Squint acquisition";
    out->payload_len = r[5];
    memcpy(out->payload, (const void *)ACQ(sq_payload)(), (size_t)r[5]);
    out->has_grid = 1;
    out->grid_len = n * n;
    memcpy(out->grid, (const void *)ACQ(sq_grid)(), (size_t)n * n);
    memcpy(out->corners, (const void *)ACQ(sa_acquired_quad)(), 8 * sizeof(double));
    return 0;
}

/* The warm-up of the JavaScript release (warmup.ts), run once when the reader opens: 24 scans of a public synthetic code
   (version 5), then 16 verifications, then a reset. It is part of the behaviour, not a speed trick: it leaves state in the
   hosts and in the verifier (the last contradiction, the parity of the full-resolution stage), and later reads depend on it. */
static const char WARM_GRID[] = "1111111010100000010101101001101111111100000100111111010001110000000100000110111010010001100100011010101010111011011101000010001101110100000101011101101110101100000011110100101000101110110000010100111100100110000000010000011111111010101010101010101010101111111000000000010011011000110101000000000001111111000110010011110010000001100011100000100101100100010100011111000101001110111110100110100011101001110111000111001101110110010100100101000001001010011110101010010101111010000101010100000000000000010001001001111100100111111011011110011011101100000011010010111100100110011001010010010111000001011001100011011011011111100010100000111001000010001000001110001110111101111111001100100101100001011111110011110110111000111111000000101010100011010010001010001100001101110110110100110001000010001000100011110000100101011011110111110110010010000011111111001000111000000001111111000000101011000010111000011010110000011111110010110000001100100010011010100000100100100111101110000110001001010110110001011010100011000010111001101011001101011011111011100000110010000000101100101111111000100000000101100010000111111101000100101111111010100101011111011111101011101100000101100110101110111010110001110110111010101001101011010000111111110111011101010100011100010101000000001001101110101010110011010010000100011000110000010111110010011100010110100110001111111000000010111101100000100110001";

static int warm_up(void) {
    static uint8_t gray[SQ_MAX_WIDTH * SQ_MAX_HEIGHT];
    static sq_read_result r;
    const int w = 1280, h = 720, n = 37, scale = 6, x0 = 400, y0 = 180, session = -2;
    memset(gray, 255, sizeof gray);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            if (WARM_GRID[y * n + x] == '1')
                for (int dy = 0; dy < scale; dy++)
                    memset(gray + (y0 + y * scale + dy) * w + x0 + x * scale, 0, (size_t)scale);
    for (int i = 0; i < 24; i++)
        if (cpu_scan(gray, w, h, i * 33.0, session, INFINITY, 1, 0, 0, &r) < 0) return -1;
    if (S.has_state)
        for (int i = 0; i < 16; i++)
            if (host_verify(gray, w, h, 1, INFINITY, &r) < 0) return -1;
    partial_reset();
    vr_reset();
    S.fail_streak = S.lost_streak = 0;
    return 0;
}

/* ---- Squint -------------------------------------------------------------------------------------------------------- */

void sq_default_options(sq_read_options *o) {
    o->time_ms = 0;
    o->budget_ms = 12.5;
    o->acquire_budget_ms = 9;
    o->tracked_acquire_budget_ms = 5;
    o->tracked_acquire_every = 3;
    o->enhance = 1;
}

int sq_open(char *err, size_t errlen) {
    if (S.opened) {
        snprintf(err, errlen, "a Squint reader is already open: its hosts keep their state in static memory");
        return 1;
    }
    memset(&S, 0, sizeof S);
    S.opened = 1;
    /* createBrowserScanner with the motion-evidence preset, in the order of the JavaScript. */
    GEO(sg_fast_mapping)(1);
    VER(sg_fast_mapping)(1);
    VER(sg_search_radius)(2);
    VER(sg_normalized)(1);
    VER(sqw_profiles)(1);
    VER(sv_decode_registered)(1);
    VER(sv_fresh_resample)(1);
    VER(sv_policy)(12, 4, 0);
    if (VER(se_configure)(20, 4, 10, 15) || VER(se_symbol_cap)(6)) {
        snprintf(err, errlen, "bad evidence configuration");
        S.opened = 0;
        return 2;
    }
    VER(sv_evidence)(1);
    FLO(sf_budget)(80);
    FLO(sf_reverse)(0);
    partial_reset();
    vr_reset();
    if (VER(sv_proposals)(1) || VER(sqw_profiles)(1) || VER(sv_relocalize)(-1) || VER(sv_align_gate)(0.6) ||
        VER(sv_evidence_reserve)(2.5) || VER(sv_roi_detect)(0)) {
        snprintf(err, errlen, "bad proposal policy");
        S.opened = 0;
        return 2;
    }
    FLO(sf_scale)(4);
    FLO(sf_budget)(80);
    FLO(sf_window)(4);
    if (FLO(sf_model)(1)) {
        snprintf(err, errlen, "bad flow model");
        S.opened = 0;
        return 2;
    }
    S.failed = 0;
    if (warm_up() < 0) {
        snprintf(err, errlen, "warm-up failed: %s", S.err);
        S.opened = 0;
        return 2;
    }
    return 0;
}

void sq_close(void) { S.opened = 0; }

void sq_reset(void) {
    partial_reset();
    vr_reset();
    S.fail_streak = S.lost_streak = 0;
    S.sess++;
    S.tick = 0;
    S.tracked = 0;
}

int sq_read(const uint8_t *gray, int width, int height, const sq_read_options *o, sq_read_result *out, char *err,
            size_t errlen) {
    if (!S.opened) {
        snprintf(err, errlen, "the reader is not open");
        return 1;
    }
    if (!gray || width < 1 || height < 1 || width > 4096 || height > 4096 || !isfinite(o->time_ms)) {
        snprintf(err, errlen, "Squint needs width * height gray bytes and a finite time");
        return 1;
    }
    S.failed = 0;
    if (cpu_scan(gray, width, height, o->time_ms, S.sess, o->budget_ms, 0, 1, 1, out) < 0) goto failed;
    S.tick++;
    if (!S.tracked || S.tick % o->tracked_acquire_every == 0) {
        static sq_read_result a;
        acquire(gray, width, height, S.tracked, o->acquire_budget_ms, o->tracked_acquire_budget_ms, o->enhance, &a);
        if (!out->accepted && a.accepted) {
            /* Squint.anchor: hand the tracker a read made elsewhere on the same frame. */
            if (vr_anchor(gray, width, height, o->time_ms, S.sess, &a) < 0) goto failed;
            pipeline_observe(&a, o->time_ms, S.sess);
            memcpy(out, &a, sizeof a);
        }
    }
    S.tracked = out->accepted;
    return 0;
failed:
    snprintf(err, errlen, "%s", S.err);
    return 2;
}

size_t sq_memory_bytes(void) {
    /* The static memory of the hosts and the pipeline: the .data section of the x86-64 build, measured (45 MiB). */
    return (size_t)45 * 1024 * 1024;
}
