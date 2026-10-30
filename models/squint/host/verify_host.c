/* Current-image conditional verification hot path; no labels or stored geometry. */
#include <stdint.h>
#include <stddef.h>
#include "squint_platform.h"
extern uintptr_t sg_frame(void), sg_quad(void), sg_result(void), sg_crop(void);
extern int sg_scout(int, int, int, int), sg_fit(int, int, int, int, int), sg_rectify_fast(int, int, int, int);
extern uintptr_t sqw_input(void), sqw_quads(void);
extern int sqw_detect(int, int);
extern uintptr_t sa_current(void), sa_target(void), sa_aligned(void), sa_warp(void);
extern int sa_register(int, int, int), sa_known(int);
extern uintptr_t sa_known_mask(void), sg_known_mask(void);
extern int sa_polynomial(int, int);
static double sharpenAmount;
API int sv_sharpen(double amount) {
    if (!isfinite(amount) || amount < 0 || amount > 2) return -1;
    sharpenAmount = amount;
    return 0;
}
static int freshResample;
API int sv_fresh_resample(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    freshResample = enabled;
    return 0;
}
static int meshRegistration;
API int sv_mesh(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    meshRegistration = enabled;
    return 0;
}
static int knownRegistration;
API int sv_known_registration(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    knownRegistration = enabled;
    return 0;
}
#if 1 /* the browser API: always on */
extern uintptr_t sq_input(void), sq_result(void), sg_soft_grid(void), sg_visible(void), sq_visible(void);
extern int sg_sample_grid(int, int, int, int, double, double), sg_partial_sampling(int),
    sq_decode(int, int, double, int, int, double, double, int, double), sq_gmd(int), sq_reserve(int), sq_padding(int),
    sq_visibility(int), sq_reliability(int);
static int decodeRegistered = 0;
API int sv_decode_registered(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    decodeRegistered = enabled;
    return 0;
}
#endif

/* Evidence certificate (saturation-aware blur model + RS erasure certification): the only acceptance test for a
 * tracked payload that does not decode afresh. */
extern int se_run(const float *, int, int, double, double (*)(void));
extern double se_align_score(const float *, int, int);
extern int se_relocalize(const uint8_t *, int, int, const double *, int, double *);
extern uintptr_t se_stats(void);
static int evidenceMode;
static double evidenceMs;
static int lastRegistered, registeredQuadReady;
static double bestAlign = 1e9;
API double sv_best_align(void) {
    return bestAlign;
}
static double candidates[10 * 11];
static int candidateCount;
API uintptr_t sv_candidates(void) {
    return (uintptr_t)candidates;
}
API int sv_candidate_count(void) {
    return candidateCount;
}
API int sv_registered(void) {
    return registeredQuadReady;
}
API int sv_evidence(int mode) {
    if (mode < 0 || mode > 1) return -1;
    evidenceMode = mode;
    return 0;
}
API double sv_evidence_stats(int index) {
    return index >= 0 && index < 24 ? ((double *)se_stats())[index] : 0;
}
#define MAXFITS 10
static float crops[MAXFITS][196 * 196], chosen[196 * 196];
static int motionProposals;
static double alignGate = -1, evidenceReserve = 0;
static int roiDetect;
API int sv_evidence_reserve(double ms) {
    if (!(ms >= 0 && ms < 20)) return -1;
    evidenceReserve = ms / 1000;
    return 0;
}
API int sv_roi_detect(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    roiDetect = enabled;
    return 0;
}
API int sv_align_gate(double gate) {
    if (!(gate == -1 || (gate > 0 && gate < 10))) return -1;
    alignGate = gate;
    return 0;
}
static double relocalizeThreshold = -1;
API int sv_relocalize(double threshold) {
    if (!(threshold == -1 || (threshold > 0 && threshold < 10))) return -1;
    relocalizeThreshold = threshold;
    return 0;
}
API int sv_motion_proposals(int enabled) {
    if (enabled < 0 || enabled > 1) return -1;
    motionProposals = enabled;
    return 0;
}
extern int sp_phase_prepare(const float *, int);
extern double sp_phase_score(const float *, int);
static double phaseThreshold = -1, lastPhaseScore;
API int sv_phase_threshold(double threshold) {
    if (!isfinite(threshold) || threshold < -1 || threshold > 1) return -1;
    phaseThreshold = threshold;
    return 0;
}
API double sv_phase_score(void) {
    return lastPhaseScore;
}
static double verificationStart;
static int lastComplete;
static double acceptedQuad[8], lastScore, metrics[6], geometryGate = -1;
static int targetCount = 3;
static int proposalCount = 1;
API int sv_proposals(int count) {
    if (count < 1 || count > 4) return -1;
    proposalCount = count;
    return 0;
}
static int registrationIterations = 12, registrationStride = 4, registrationAll = 0;
API int sv_policy(int iterations, int stride, int all) {
    if (iterations < 1 || iterations > 40 || stride < 1 || stride > 4 || all < 0 || all > 3) return -1;
    registrationIterations = iterations;
    registrationStride = stride;
    registrationAll = all;
    return 0;
}
API int sv_geometry_gate(double threshold) {
    if (!isfinite(threshold) || threshold < -1 || threshold > 1) return -1;
    geometryGate = threshold;
    return 0;
}
API uintptr_t sv_metrics(void) {
    return (uintptr_t)metrics;
}
API uintptr_t sv_quad(void) {
    return (uintptr_t)acceptedQuad;
}
API double sv_score(void) {
    return lastScore;
}
static int mapping(const double *q, double *h) {
    double dx1 = q[2] - q[4], dx2 = q[6] - q[4], dx3 = q[0] - q[2] + q[4] - q[6], dy1 = q[3] - q[5], dy2 = q[7] - q[5],
           dy3 = q[1] - q[3] + q[5] - q[7], den = dx1 * dy2 - dx2 * dy1;
    if (fabs(den) < 1e-12) return 0;
    double g = (dx3 * dy2 - dx2 * dy3) / den, r = (dx1 * dy3 - dx3 * dy1) / den;
    h[0] = (q[2] - q[0] + g * q[2]) / 127;
    h[1] = (q[6] - q[0] + r * q[6]) / 127;
    h[2] = q[0];
    h[3] = (q[3] - q[1] + g * q[3]) / 127;
    h[4] = (q[7] - q[1] + r * q[7]) / 127;
    h[5] = q[1];
    h[6] = g / 127;
    h[7] = r / 127;
    return 1;
}
static void point(const double *h, double x, double y, double *out) {
    double d = h[6] * x + h[7] * y + 1;
    out[0] = (h[0] * x + h[1] * y + h[2]) / d;
    out[1] = (h[3] * x + h[4] * y + h[5]) / d;
}
static int inset(const double *q, double lo, double hi, double *out) {
    double h[8];
    if (!mapping(q, h)) return 0;
    point(h, lo, lo, out);
    point(h, hi, lo, out + 2);
    point(h, hi, hi, out + 4);
    point(h, lo, hi, out + 6);
    return 1;
}
static int normalize(const uint8_t *crop, float *out, int size) {
    uint32_t histogram[256] = {0}, total = 0;
    int lo = 0, hi = 255;
    for (int i = 0; i < size * size; i++) histogram[crop[i]]++;
    for (int i = 0; i < 256; i++) {
        total += histogram[i];
        if (total >= size * size * .01) {
            lo = i;
            break;
        }
    }
    total = 0;
    for (int i = 255; i >= 0; i--) {
        total += histogram[i];
        if (total >= size * size * .01) {
            hi = i;
            break;
        }
    }
    if (hi - lo < 20) return 0;
    for (int i = 0; i < size * size; i++) {
        double x = ((double)crop[i] - lo) / (hi - lo);
        out[i] = (float)fmax(0, fmin(1, x));
    }
    return 1;
}
static double distance(const float *a, const float *b, int size) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    int count = 0;
    for (int y = 16; y < size - 16; y += 2)
        for (int x = 16; x < size - 16; x += 2) {
            double av = a[y * size + x], bv = b[y * size + x];
            sa += av;
            sb += bv;
            saa += av * av;
            sbb += bv * bv;
            sab += av * bv;
            count++;
        }
    return 1 - (sab - sa * sb / count) / fmax(1e-6, sqrt((saa - sa * sa / count) * (sbb - sb * sb / count)));
}
/* Inputs are bounded by the current image and the tracked code's version. */
API int sv_verify(int width, int height, int version, int flowValid, const double *codeCorners, const float *candidate,
                  const float *targets, double deadline) {
    if (width < 64 || height < 64 || width > 1280 || height > 720 || width % 2 || height % 2 || version < 1 ||
        version > 6)
        return -1;
    memset(metrics, 0, sizeof(metrics));
    double start = seconds(), mark = start;
    registeredQuadReady = 0;
    bestAlign = 1e9;
    sa_known(knownRegistration ? 17 + 4 * version : 0);
    if (knownRegistration)
        memcpy((void *)sa_known_mask(), (void *)sg_known_mask(), (17 + 4 * version) * (17 + 4 * version));
    verificationStart = start;
    lastComplete = 0;
    lastScore = -1e30;
    int n = 17 + 4 * version, size = (n + 8) * 4;
    double fits[MAXFITS][8], quad[8], bestProposal[4][8], bestCost[4] = {1e5, 1e5, 1e5, 1e5};
    int fitCount = 0;
    if (flowValid && inset(codeCorners, -4. * 127 / n, 127 + 4. * 127 / n, fits[fitCount])) fitCount++;
    double *scratch = (double *)sg_quad(), *result = (double *)sg_result();
    int skipGeometry = 0;
    int precomputed0 = 0;
    double align0 = 1e9;
    if (evidenceMode && alignGate > 0 && fitCount) {
        memcpy(scratch, fits[0], 64);
        if (!sg_rectify_fast(width, height, width, size) && normalize((const uint8_t *)sg_crop(), crops[0], size)) {
            precomputed0 = 1;
            align0 = se_align_score(crops[0], size, version);
            if (align0 < alignGate) skipGeometry = 1;
        }
    }
    if (fitCount && geometryGate >= 0) {
        memcpy(scratch, fits[0], 64);
        if (!sg_scout(width, height, width, version) && result[1] < geometryGate) skipGeometry = 1;
    }
    if (fitCount && registrationAll == 3) {
        memcpy(scratch, fits[0], 64);
        if (!sg_fit(width, height, width, version, 1) && result[0]) memcpy(fits[0], result + 5, 64);
    }
    int count = 0, roiX = 0, roiY = 0;
    static int32_t shiftedPatches[16 * 8];
    const int32_t *patches = (const int32_t *)sqw_quads();
    if (!skipGeometry) {
        if (roiDetect && flowValid && fitCount) {
            double *f = fits[0], minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
            for (int k = 0; k < 4; k++) {
                if (f[k * 2] < minx) minx = f[k * 2];
                if (f[k * 2] > maxx) maxx = f[k * 2];
                if (f[k * 2 + 1] < miny) miny = f[k * 2 + 1];
                if (f[k * 2 + 1] > maxy) maxy = f[k * 2 + 1];
            }
            double cx = (minx + maxx) / 2, cy = (miny + maxy) / 2, hw = (maxx - minx) * 1.0, hh = (maxy - miny) * 1.0;
            int x0 = (int)(cx - hw), y0 = (int)(cy - hh), x1 = (int)(cx + hw), y1 = (int)(cy + hh);
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 > width) x1 = width;
            if (y1 > height) y1 = height;
            x0 &= ~1;
            y0 &= ~1;
            int rw = (x1 - x0) & ~1, rh = (y1 - y0) & ~1;
            if (rw >= 64 && rh >= 64 && rw * rh < width * height * 3 / 4) {
                uint8_t *in = (uint8_t *)sqw_input();
                const uint8_t *fr = (const uint8_t *)sg_frame();
                for (int y = 0; y < rh; y++) memcpy(in + y * rw, fr + (y0 + y) * width + x0, rw);
                count = sqw_detect(rw, rh);
                roiX = x0;
                roiY = y0;
                if (count > 0) {
                    const int32_t *q = (const int32_t *)sqw_quads();
                    for (int i = 0; i < count * 8; i++) shiftedPatches[i] = q[i] + ((i & 1) ? roiY : roiX);
                    patches = shiftedPatches;
                }
            } else {
                memcpy((void *)sqw_input(), (void *)sg_frame(), width * height);
                count = sqw_detect(width, height);
            }
        } else {
            memcpy((void *)sqw_input(), (void *)sg_frame(), width * height);
            count = sqw_detect(width, height);
        }
    }
    if (count < 0) return -1;
    for (int p = 0; p < count && p < 4; p++) {
        if (seconds() >= deadline) return 0;
        double patch[8];
        for (int k = 0; k < 8; k++) patch[k] = patches[p * 8 + k];
        for (int border = 1; border <= 4; border++) {
            double lo = (border - 4.) / (n + 2 * border) * 127;
            if (!inset(patch, lo, 127 - lo, quad)) continue;
            for (int rotation = 0; rotation < 4; rotation++) {
                for (int k = 0; k < 4; k++) {
                    scratch[k * 2] = quad[((k + rotation) % 4) * 2];
                    scratch[k * 2 + 1] = quad[((k + rotation) % 4) * 2 + 1];
                }
                if (!sg_scout(width, height, width, version) && result[1] < bestCost[proposalCount - 1]) {
                    int pos = proposalCount - 1;
                    while (pos > 0 && result[1] < bestCost[pos - 1]) {
                        bestCost[pos] = bestCost[pos - 1];
                        memcpy(bestProposal[pos], bestProposal[pos - 1], 64);
                        pos--;
                    }
                    bestCost[pos] = result[1];
                    memcpy(bestProposal[pos], scratch, 64);
                }
            }
        }
    }
    for (int p = 0; p < proposalCount; p++)
        if (bestCost[p] < 1e5) {
            memcpy(scratch, bestProposal[p], 64);
            if (!sg_fit(width, height, width, version, 1) && result[0]) {
                memcpy(fits[fitCount], result + 5, 64);
                fitCount++;
            }
        }
    /* Motion-consistent proposals: white patches near the flow prediction, corner order matched to the flow quad,
    quiet-zone border widths 1..4. Geometry candidates only; selection by the physical alignment score. */
    if (evidenceMode && motionProposals && flowValid && !skipGeometry) {
        double *f = fits[0], fc[2] = {(f[0] + f[2] + f[4] + f[6]) / 4, (f[1] + f[3] + f[5] + f[7]) / 4},
               fd = sqrt((f[4] - f[0]) * (f[4] - f[0]) + (f[5] - f[1]) * (f[5] - f[1]));
        for (int p = 0; p < count && p < 16 && fitCount < MAXFITS - 1; p++) {
            double patch[8];
            for (int k = 0; k < 8; k++) patch[k] = patches[p * 8 + k];
            double pc[2] = {(patch[0] + patch[2] + patch[4] + patch[6]) / 4,
                            (patch[1] + patch[3] + patch[5] + patch[7]) / 4},
                   pd = sqrt((patch[4] - patch[0]) * (patch[4] - patch[0]) +
                             (patch[5] - patch[1]) * (patch[5] - patch[1]));
            if (sqrt((pc[0] - fc[0]) * (pc[0] - fc[0]) + (pc[1] - fc[1]) * (pc[1] - fc[1])) > .6 * fd || pd < .6 * fd ||
                pd > 1.6 * fd)
                continue;
            int rot = 0;
            double bestd = 1e30;
            for (int r = 0; r < 4; r++) {
                double dsum = 0;
                for (int k = 0; k < 4; k++) {
                    double dx = patch[((k + r) % 4) * 2] - f[k * 2], dy = patch[((k + r) % 4) * 2 + 1] - f[k * 2 + 1];
                    dsum += dx * dx + dy * dy;
                }
                if (dsum < bestd) {
                    bestd = dsum;
                    rot = r;
                }
            }
            double ordered[8];
            for (int k = 0; k < 4; k++) {
                ordered[k * 2] = patch[((k + rot) % 4) * 2];
                ordered[k * 2 + 1] = patch[((k + rot) % 4) * 2 + 1];
            }
            for (int border = 1; border <= 4 && fitCount < MAXFITS - 1; border++) {
                double lo = (border - 4.) / (n + 2 * border) * 127;
                if (inset(ordered, lo, 127 - lo, fits[fitCount])) fitCount++;
            }
        }
    }
    metrics[0] = (seconds() - mark) * 1000;
    mark = seconds();
    int validCrop[MAXFITS] = {0};
    int bestCrop = -1, bestTarget = 0;
    double bestRank = 1e30, alignScore[MAXFITS];
    for (int i = 0; i < MAXFITS; i++) alignScore[i] = 1e9;
    for (int i = 0; i < fitCount; i++) {
        if (seconds() >= deadline) return 0;
        if (!(i == 0 && precomputed0)) {
            memcpy(scratch, fits[i], 64);
            if (sg_rectify_fast(width, height, width, size) || !normalize((const uint8_t *)sg_crop(), crops[i], size))
                continue;
        }
        validCrop[i] = 1;
        for (int t = 0; t < targetCount; t++) {
            double rank = distance(crops[i], targets + t * size * size, size);
            if (rank < bestRank) {
                bestRank = rank;
                bestCrop = i;
                bestTarget = t;
            }
        }
        if (evidenceMode) alignScore[i] = (i == 0 && precomputed0) ? align0 : se_align_score(crops[i], size, version);
    }
    metrics[1] = (seconds() - mark) * 1000;
    mark = seconds();
    if (evidenceMode && relocalizeThreshold > 0 && flowValid && fitCount < MAXFITS - 1 && seconds() < deadline) {
        double minAlign = 1e9;
        for (int i = 0; i < fitCount; i++)
            if (validCrop[i] && alignScore[i] < minAlign) minAlign = alignScore[i];
        if (minAlign > relocalizeThreshold &&
            se_relocalize((const uint8_t *)sg_frame(), width, height, fits[0], version, fits[fitCount])) {
            memcpy(scratch, fits[fitCount], 64);
            int i = fitCount;
            if (!sg_rectify_fast(width, height, width, size) && normalize((const uint8_t *)sg_crop(), crops[i], size)) {
                validCrop[i] = 1;
                alignScore[i] = se_align_score(crops[i], size, version);
                fitCount++;
                if (alignScore[i] < .75 * alignScore[bestCrop < 0 ? 0 : bestCrop] || bestCrop < 0) {
                    bestCrop = i;
                    bestRank = 1e30;
                    for (int t = 0; t < targetCount; t++) {
                        double z = distance(crops[i], targets + t * size * size, size);
                        if (z < bestRank) {
                            bestRank = z;
                            bestTarget = t;
                        }
                    }
                }
            }
        }
    }
    bestAlign = 1e9;
    for (int i = 0; i < fitCount; i++)
        if (validCrop[i] && alignScore[i] < bestAlign) bestAlign = alignScore[i];
    candidateCount = fitCount;
    for (int i = 0; i < fitCount; i++) {
        for (int k = 0; k < 8; k++) candidates[i * 11 + k] = fits[i][k];
        candidates[i * 11 + 8] = validCrop[i];
        candidates[i * 11 + 9] = alignScore[i];
        candidates[i * 11 + 10] = (i == 0 && flowValid);
    }
    lastRegistered = 0;
    if (bestCrop < 0) return 0;
    if (evidenceMode) {
        int alt = bestCrop;
        for (int i = 0; i < fitCount; i++)
            if (validCrop[i] && alignScore[i] < alignScore[alt]) alt = i;
        if (alt != bestCrop && alignScore[alt] < .75 * alignScore[bestCrop]) {
            bestCrop = alt;
            bestRank = 1e30;
            for (int t = 0; t < targetCount; t++) {
                double z = distance(crops[alt], targets + t * size * size, size);
                if (z < bestRank) {
                    bestRank = z;
                    bestTarget = t;
                }
            }
        }
    }
    memcpy(chosen, crops[bestCrop], size * size * sizeof(float));
    memcpy(quad, fits[bestCrop], 64);

    for (int i = 0; i < fitCount; i++) {
        if (!validCrop[i] || ((registrationAll == 0 || registrationAll == 3) && i != bestCrop)) continue;
        int targetIndex = bestTarget;
        if (i != bestCrop) {
            double rank = 1e30;
            for (int t = 0; t < targetCount; t++) {
                double z = distance(crops[i], targets + t * size * size, size);
                if (z < rank) {
                    rank = z;
                    targetIndex = t;
                }
            }
        }
        for (int attempt = 0; attempt < (registrationAll == 2 ? targetCount : 1); attempt++) {
            if (seconds() >= deadline) return 0;
            int t = registrationAll == 2 ? attempt : targetIndex;
            memcpy((void *)sa_current(), crops[i], size * size * sizeof(float));
            memcpy((void *)sa_target(), targets + t * size * size, size * size * sizeof(float));
            if (sa_register(size, registrationIterations, registrationStride) == 1) {
                const float *aligned = (const float *)sa_aligned();
                double rank = distance(aligned, targets + t * size * size, size);
                if (rank < bestRank) {
                    double h[8];
                    const double *w = (const double *)sa_warp();
                    if (mapping(fits[i], h)) {
                        double points[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
                        for (int k = 0; k < 4; k++) {
                            double x = points[k * 2], y = points[k * 2 + 1], d = w[6] * x + w[7] * y + w[8];
                            point(h, ((w[0] * x + w[1] * y + w[2]) / d + 1) * 63.5,
                                  ((w[3] * x + w[4] * y + w[5]) / d + 1) * 63.5, quad + k * 2);
                        }
                        memcpy(chosen, aligned, size * size * sizeof(float));
                        bestRank = rank;
                        lastRegistered = 1;
                    }
                }
            }
        }
    }

    if (seconds() >= deadline) {
        metrics[5] = (seconds() - start) * 1000;
        return 0;
    }
    if (freshResample) {
        memcpy(scratch, quad, 64);
        if (!sg_rectify_fast(width, height, width, size)) normalize((const uint8_t *)sg_crop(), chosen, size);
    }
    if (sharpenAmount > 0) {
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                double blurred = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy < 0) yy = 0;
                        if (yy >= size) yy = size - 1;
                        if (xx < 0) xx = 0;
                        if (xx >= size) xx = size - 1;
                        blurred += chosen[yy * size + xx] * (dy ? 1 : 2) * (dx ? 1 : 2) / 16.;
                    }
                double value = chosen[y * size + x] + sharpenAmount * (chosen[y * size + x] - blurred);
                crops[MAXFITS - 1][y * size + x] = (float)fmax(0, fmin(1, value));
            }
        memcpy(chosen, crops[MAXFITS - 1], size * size * sizeof(float));
    }
    if (meshRegistration) {
        sa_known(n);
        memcpy((void *)sa_known_mask(), (void *)sg_known_mask(), n * n);
        memcpy((void *)sa_current(), chosen, size * size * sizeof(float));
        memcpy((void *)sa_target(), targets + bestTarget * size * size, size * size * sizeof(float));
        if (sa_polynomial(size, 8) == 1) memcpy(chosen, (void *)sa_aligned(), size * size * sizeof(float));
    }
    if (phaseThreshold >= 0) {
        lastPhaseScore = sp_phase_score(chosen, size);
        if (lastPhaseScore >= phaseThreshold) {
            double lo = 4. * 127 / (n + 8);
            if (inset(quad, lo, 127 - lo, acceptedQuad)) {
                metrics[5] = (seconds() - start) * 1000;
                return 4;
            }
        }
    }
#if 1 /* the browser API: always on */
    if (decodeRegistered && seconds() < deadline) {
        double lo = 4. * 127 / (n + 8);
        if (inset(quad, lo, 127 - lo, acceptedQuad)) {
            memcpy(scratch, acceptedQuad, 64);
            sg_partial_sampling(1);
            if (sg_sample_grid(width, height, width, version, 0, 0) == 1) {
                memcpy((void *)sq_input(), (void *)sg_soft_grid(), n * n * sizeof(double));
                memcpy((void *)sq_visible(), (void *)sg_visible(), n * n);
                sq_visibility(1);
                sq_padding(1);
                sq_gmd(1);
                sq_reserve(2);
                sq_reliability(0);
                for (int reliability = 0; reliability < 3; reliability++) {
                    sq_reliability(reliability);
                    if (sq_decode(n, 1, 1., 30, 1, .12, .10, 1, .05) == 1) {
                        metrics[5] = (seconds() - start) * 1000;
                        return 2;
                    }
                }
            }
        }
    }
#endif
    metrics[2] = (seconds() - mark) * 1000;
    mark = seconds();
    if (evidenceMode && seconds() + evidenceReserve > deadline) {
        metrics[5] = (seconds() - start) * 1000;
        return 0;
    }
    if (evidenceMode) {
        double evidenceStart = seconds();
        int certified = se_run(chosen, size, version, deadline, seconds);
        evidenceMs = (seconds() - evidenceStart) * 1000;
        metrics[4] = evidenceMs;
        metrics[5] = (seconds() - start) * 1000;
        lastComplete = 1;
        lastScore = certified ? 1e3 : -1e3;
        if (certified < 0) return -1;
        double lo = 4. * 127 / (n + 8);
        if (!inset(quad, lo, 127 - lo, acceptedQuad)) return 0;
        registeredQuadReady = lastRegistered;
        if (certified) return 5;
    }
    return 0;
}

/* Browser host buffers share one fixed memory with geometry, registration and the evidence certificate. */
#if 1 /* the browser API: always on */
static float browserCandidate[196 * 196], browserTargets[15 * 196 * 196];
static double browserCorners[8];
API uintptr_t sv_candidate(void) {
    return (uintptr_t)browserCandidate;
}
API uintptr_t sv_targets(void) {
    return (uintptr_t)browserTargets;
}
API uintptr_t sv_corners(void) {
    return (uintptr_t)browserCorners;
}
API int sv_browser(int width, int height, int version, int flowValid, double deadline) {
    return sv_verify(width, height, version, flowValid, browserCorners, browserCandidate, browserTargets, deadline);
}
#endif

#if 1 /* the browser API: always on */
API int sv_motion_targets(int size, int enabled) {
    if (size < 116 || size > 196 || size % 4 || enabled < 0 || enabled > 1) return -1;
    targetCount = enabled ? 15 : 3;
    if (!enabled) return 0;
    int dx[4] = {1, 0, 1, 1}, dy[4] = {0, 1, 1, -1};
    for (int radius = 3, index = 3; radius <= 7; radius += 2)
        for (int direction = 0; direction < 4; direction++, index++) {
            float *out = browserTargets + index * size * size;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++) {
                    double sum = 0;
                    int count = 0;
                    for (int step = -radius; step <= radius; step++) {
                        int xx = x + dx[direction] * step, yy = y + dy[direction] * step;
                        sum += xx < 0 || yy < 0 || xx >= size || yy >= size ? 1 : browserTargets[yy * size + xx];
                        count++;
                    }
                    out[y * size + x] = (float)(sum / count);
                }
        }
    return 0;
}
#endif

#if 1 /* the browser API: always on */
API int sv_completed(void) {
    return lastComplete;
}
#endif

#if 1 /* the browser API: always on */
API int sv_phase_cache(int size) {
    if (phaseThreshold < 0) return 0;
    return sp_phase_prepare(browserCandidate, size);
}
#endif
