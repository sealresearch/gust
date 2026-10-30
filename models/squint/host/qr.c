/* Standalone bounded soft QR decoder. No allocator or imports. Unreleased. */
#include <stdint.h>
#include <stddef.h>
#include "qr_tables.h"
#include "squint_platform.h"
#define GRID   3249
#define WORDS  346
#define BLOCKS 8
static double probability[GRID], metrics[2];
static uint8_t visible[GRID];
static int visibilityEnabled, paddingEnabled;
API uintptr_t sq_visible(void) {
    return (uintptr_t)visible;
}
API int sq_visibility(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    visibilityEnabled = enabled;
    return 0;
}
API int sq_padding(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    paddingEnabled = enabled;
    return 0;
}
/* accepted, reason, version, ECC(format order M/L/H/Q), mask, payload bytes,
   mode(1 numeric/2 alpha/4 byte/8 mixed), erasures, errors, mode count */
static int result[11], minimumReserve = 1, adaptiveErasures = 0, formatRank = -1;
static int reliabilityMode = 0;
API int sq_reliability(int mode) {
    if (mode < 0 || mode > 2) return -1;
    reliabilityMode = mode;
    return 0;
}
API int sq_format_rank(int rank) {
    if (rank < -1 || rank > 3) return -1;
    formatRank = rank;
    return 0;
}
API int sq_gmd(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    adaptiveErasures = enabled;
    return 0;
}
API int sq_reserve(int reserve) {
    if (reserve < 0 || reserve > 30) return -1;
    minimumReserve = reserve;
    return 0;
}
static uint8_t payload[2048], grid[GRID], word[255], gfExp[512], gfLog[256];
static int positions[255], gfReady;
typedef struct {
    uint8_t fixed[GRID], values[GRID];
    uint16_t data[GRID], format[2][15];
    int count, n;
} Layout;
static Layout layouts[10];
static uint8_t raw[WORDS], parts[BLOCKS][255], encoded[BLOCKS][255], dataBytes[WORDS];
static float confidence[WORDS], quality[BLOCKS][255];
static int totals[BLOCKS], lengths[BLOCKS], blocks;
API uintptr_t sq_input(void) {
    return (uintptr_t)probability;
}
API uintptr_t sq_result(void) {
    return (uintptr_t)result;
}
API uintptr_t sq_metrics(void) {
    return (uintptr_t)metrics;
}
API uintptr_t sq_payload(void) {
    return (uintptr_t)payload;
}
API uintptr_t sq_grid(void) {
    return (uintptr_t)grid;
}
API uintptr_t sq_word(void) {
    return (uintptr_t)word;
}
API uintptr_t sq_positions(void) {
    return (uintptr_t)positions;
}
static double absolute(double v) {
    return v < 0 ? -v : v;
}
static int maximum(int a, int b) {
    return a > b ? a : b;
}
static int minimum(int a, int b) {
    return a < b ? a : b;
}
static int popcount(uint32_t v) {
    int n = 0;
    while (v) {
        v &= v - 1;
        n++;
    }
    return n;
}
static void init_gf(void) {
    if (gfReady) return;
    unsigned x = 1;
    for (int i = 0; i < 255; i++) {
        gfExp[i] = (uint8_t)x;
        gfLog[x] = (uint8_t)i;
        x <<= 1;
        if (x & 256) x ^= 0x11d;
    }
    for (int i = 255; i < 512; i++) gfExp[i] = gfExp[i - 255];
    gfReady = 1;
}
static uint8_t mul(int a, int b) {
    return a && b ? gfExp[gfLog[a] + gfLog[b]] : 0;
}
static uint8_t divide(int a, int b) {
    return a ? gfExp[(gfLog[a] - gfLog[b] + 255) % 255] : 0;
}
static uint8_t eval_desc(const uint8_t *p, int n, int x) {
    int y = 0;
    for (int i = 0; i < n; i++) y = mul(y, x) ^ p[i];
    return (uint8_t)y;
}
static uint8_t eval_asc(const uint8_t *p, int n, int x) {
    int y = 0;
    for (int i = n - 1; i >= 0; i--) y = mul(y, x) ^ p[i];
    return (uint8_t)y;
}
static void product(const uint8_t *a, int na, const uint8_t *b, int nb, uint8_t *out) {
    for (int i = 0; i < na + nb - 1; i++) out[i] = 0;
    for (int i = 0; i < na; i++)
        for (int j = 0; j < nb; j++) out[i + j] ^= mul(a[i], b[j]);
}
static int rs_correct(int n, int parity, int erased, int *unknownErrors) {
    if (n < 2 || n > 255 || parity < 1 || parity >= n || parity > 30 || erased < 0 || erased > parity) return 0;
    uint8_t seen[255] = {0}, syn[30], reduced[30], C[32] = {1}, B[32] = {1}, previous[32], temp[64];
    for (int i = 0; i < erased; i++) {
        if (positions[i] < 0 || positions[i] >= n || seen[positions[i]]) return 0;
        seen[positions[i]] = 1;
    }
    int any = 0;
    for (int i = 0; i < parity; i++) {
        syn[i] = eval_desc(word, n, gfExp[i]);
        reduced[i] = syn[i];
        any |= syn[i];
    }
    if (!any) {
        *unknownErrors = 0;
        return 1;
    }
    int reducedLength = parity;
    for (int i = 0; i < erased; i++) {
        int x = gfExp[n - 1 - positions[i]];
        for (int j = 0; j < reducedLength - 1; j++) reduced[j] = mul(reduced[j], x) ^ reduced[j + 1];
        reducedLength--;
    }
    int cLength = 1, bLength = 1, L = 0, m = 1, b = 1;
    for (int k = 0; k < reducedLength; k++) {
        int d = reduced[k];
        for (int i = 1; i <= L; i++) d ^= mul(i < cLength ? C[i] : 0, reduced[k - i]);
        if (!d) {
            m++;
            continue;
        }
        int oldLength = cLength;
        for (int i = 0; i < 32; i++) previous[i] = C[i];
        int factor = divide(d, b), need = bLength + m;
        if (need > 32) return 0;
        if (cLength < need) cLength = need;
        for (int j = 0; j < bLength; j++) C[j + m] ^= mul(factor, B[j]);
        if (2 * L <= k) {
            L = k + 1 - L;
            for (int j = 0; j < 32; j++) B[j] = previous[j];
            bLength = oldLength;
            b = d;
            m = 1;
        } else
            m++;
    }
    int count = L;
    if (2 * count + erased > parity) return 0;
    int found = erased;
    for (int p = 0; p < n; p++)
        if (!seen[p]) {
            int inverse = gfExp[(255 - (n - 1 - p)) % 255];
            if (eval_asc(C, count + 1, inverse) == 0) {
                if (found >= 30) return 0;
                positions[found++] = p;
            }
        }
    if (found != count + erased) return 0;
    uint8_t locator[32] = {1}, omega[64] = {0}, derivative[31] = {0};
    int locatorLength = 1;
    for (int i = 0; i < found; i++) {
        uint8_t factor[2] = {1, gfExp[n - 1 - positions[i]]};
        product(locator, locatorLength, factor, 2, temp);
        locatorLength++;
        for (int j = 0; j < locatorLength; j++) locator[j] = temp[j];
    }
    product(syn, parity, locator, locatorLength, omega);
    for (int i = 1; i < locatorLength; i += 2) derivative[i - 1] = locator[i];
    for (int i = 0; i < found; i++) {
        int x = gfExp[n - 1 - positions[i]], inverse = divide(1, x),
            den = eval_asc(derivative, maximum(1, locatorLength - 1), inverse);
        if (!den) return 0;
        word[positions[i]] ^= divide(mul(x, eval_asc(omega, parity, inverse)), den);
    }
    for (int i = 0; i < parity; i++)
        if (eval_desc(word, n, gfExp[i])) return 0;
    *unknownErrors = count;
    return 1;
}
API int sq_rs(int n, int parity, int erased) {
    init_gf();
    int errors = 0, ok = rs_correct(n, parity, erased, &errors);
    result[0] = ok;
    result[8] = errors;
    result[7] = erased;
    return ok;
}
static void encode_rs(const uint8_t *data, int count, int parity, uint8_t *out) {
    uint8_t generator[32] = {1}, temp[64], work[255] = {0};
    int length = 1;
    for (int i = 0; i < parity; i++) {
        uint8_t factor[2] = {1, gfExp[i]};
        product(generator, length, factor, 2, temp);
        length++;
        for (int j = 0; j < length; j++) generator[j] = temp[j];
    }
    for (int i = 0; i < count; i++) work[i] = data[i];
    for (int i = 0; i < count; i++) {
        int coefficient = work[i];
        if (coefficient)
            for (int j = 0; j < length; j++) work[i + j] ^= mul(coefficient, generator[j]);
    }
    for (int i = 0; i < count + parity; i++) out[i] = i < count ? data[i] : work[i];
}
static unsigned bch(unsigned data, int shift, unsigned poly, unsigned mask) {
    unsigned value = data << shift;
    for (int i = 31; i >= shift; i--)
        if (value & (1u << i)) value ^= poly << (i - shift);
    return ((data << shift) | value) ^ mask;
}
static void set(Layout *l, int x, int y, int value) {
    if (x >= 0 && y >= 0 && x < l->n && y < l->n) {
        l->fixed[y * l->n + x] = 1;
        l->values[y * l->n + x] = (uint8_t)!!value;
    }
}
static Layout *layout(int version) {
    Layout *l = layouts + version - 1;
    if (l->n) return l;
    int n = 17 + 4 * version;
    l->n = n;
    for (int i = 0; i < n; i++) {
        set(l, 6, i, i % 2 == 0);
        set(l, i, 6, i % 2 == 0);
    }
    int centres[6] = {3, 3, n - 4, 3, 3, n - 4};
    for (int k = 0; k < 3; k++)
        for (int y = -4; y <= 4; y++)
            for (int x = -4; x <= 4; x++) {
                int d = maximum(x < 0 ? -x : x, y < 0 ? -y : y);
                set(l, centres[k * 2] + x, centres[k * 2 + 1] + y, d != 2 && d != 4);
            }
    for (int a = 0; a < 3 && qrAlign[version - 1][a]; a++)
        for (int b = 0; b < 3 && qrAlign[version - 1][b]; b++) {
            int cx = qrAlign[version - 1][a], cy = qrAlign[version - 1][b];
            if ((cx == 6 && cy == 6) || (cx == 6 && cy == n - 7) || (cx == n - 7 && cy == 6)) continue;
            for (int y = -2; y <= 2; y++)
                for (int x = -2; x <= 2; x++) set(l, cx + x, cy + y, maximum(x < 0 ? -x : x, y < 0 ? -y : y) != 1);
        }
    for (int i = 0; i < 15; i++) {
        int x = i < 8 ? 8 : i == 8 ? 7 : 14 - i, y = i < 6 ? i : i == 6 ? 7 : 8;
        l->format[0][i] = (uint16_t)(y * n + x);
        set(l, x, y, 0);
        x = i < 8 ? n - 1 - i : 8;
        y = i < 8 ? 8 : n - 15 + i;
        l->format[1][i] = (uint16_t)(y * n + x);
        set(l, x, y, 0);
    }
    set(l, 8, n - 8, 1);
    if (version >= 7) {
        unsigned bits = bch(version, 12, 0x1f25, 0);
        for (int i = 0; i < 18; i++) {
            int a = n - 11 + i % 3, b = i / 3;
            set(l, a, b, (bits >> i) & 1);
            set(l, b, a, (bits >> i) & 1);
        }
    }
    for (int right = n - 1; right >= 1; right -= 2) {
        if (right == 6) right = 5;
        int up = ((right + 1) & 2) == 0;
        for (int v = 0; v < n; v++) {
            int y = up ? n - 1 - v : v;
            for (int j = 0; j < 2; j++) {
                int x = right - j;
                if (!l->fixed[y * n + x]) l->data[l->count++] = (uint16_t)(y * n + x);
            }
        }
    }
    return l;
}
static int mask_at(int mask, int x, int y) {
    switch (mask) {
    case 0: return (x + y) % 2 == 0;
    case 1: return y % 2 == 0;
    case 2: return x % 3 == 0;
    case 3: return (x + y) % 3 == 0;
    case 4: return (y / 2 + x / 3) % 2 == 0;
    case 5: return x * y % 2 + x * y % 3 == 0;
    case 6: return (x * y % 2 + x * y % 3) % 2 == 0;
    default: return ((x + y) % 2 + x * y % 3) % 2 == 0;
    }
}
static int reject(int reason) {
    result[0] = 0;
    result[1] = reason;
    result[5] = 0;
    return 0;
}
static int read_bits(int *offset, int end, int count, int *ok) {
    if (*offset + count > end) {
        *ok = 0;
        return 0;
    }
    int value = 0;
    for (int i = 0; i < count; i++, (*offset)++)
        value = (value << 1) | ((dataBytes[*offset >> 3] >> (7 - (*offset & 7))) & 1);
    return value;
}
static int parse(int dataLength, int version) {
    const char alpha[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
    int offset = 0, end = dataLength * 8, terminated = 0, modeCount = 0, firstMode = 0, out = 0, ok = 1;
    while (offset + 4 <= end) {
        int mode = read_bits(&offset, end, 4, &ok);
        if (!mode) {
            terminated = 1;
            break;
        }
        if (mode != 1 && mode != 2 && mode != 4) return 8;
        int count = read_bits(&offset, end,
                              mode == 1   ? (version < 10 ? 10 : 12)
                              : mode == 2 ? (version < 10 ? 9 : 11)
                                          : (version < 10 ? 8 : 16),
                              &ok);
        if (!ok) return 7;
        if (count > dataLength * 3) return 8;
        if (out + count > 2048) return 8;
        if (!modeCount) firstMode = mode;
        modeCount++;
        if (mode == 4) {
            for (int i = 0; i < count; i++) {
                payload[out++] = (uint8_t)read_bits(&offset, end, 8, &ok);
                if (!ok) return 7;
            }
        } else if (mode == 1) {
            for (int i = 0; i < count; i += 3) {
                int digits = minimum(3, count - i),
                    v = read_bits(&offset, end,
                                  digits == 3   ? 10
                                  : digits == 2 ? 7
                                                : 4,
                                  &ok),
                    limit = digits == 3   ? 1000
                            : digits == 2 ? 100
                                          : 10;
                if (!ok) return 7;
                if (v >= limit) return 8;
                int divisor = digits == 3 ? 100 : digits == 2 ? 10 : 1;
                for (int j = 0; j < digits; j++) {
                    payload[out++] = (uint8_t)('0' + v / divisor);
                    v %= divisor;
                    divisor /= 10;
                }
            }
        } else {
            for (int i = 0; i < count; i += 2) {
                int pair = i + 1 < count, v = read_bits(&offset, end, pair ? 11 : 6, &ok);
                if (!ok) return 7;
                if (v >= (pair ? 2025 : 45)) return 8;
                if (pair) payload[out++] = (uint8_t)alpha[v / 45];
                payload[out++] = (uint8_t)alpha[v % 45];
            }
        }
    }
    if (!modeCount || !out) return 8;
    if (terminated) {
        while (offset % 8)
            if (read_bits(&offset, end, 1, &ok)) return 8;
        if (offset + 8 <= end && dataBytes[offset >> 3] == 0) read_bits(&offset, end, 8, &ok);
        int pad = 0xec;
        while (offset + 8 <= end) {
            if (read_bits(&offset, end, 8, &ok) != pad) return 8;
            pad = pad == 0xec ? 0x11 : 0xec;
        }
    } else
        while (offset < end)
            if (read_bits(&offset, end, 1, &ok)) return 8;
    if (!ok) return 7;
    result[5] = out;
    result[6] = modeCount == 1 ? firstMode : 8;
    result[9] = modeCount;
    return 0;
}
static int complete_padding(int prefix, int version, int total) {
    int offset = 0, end = prefix * 8, ok = 1;
    for (int segment = 0; segment < 64; segment++) {
        if (offset + 4 > end) return 0;
        int mode = read_bits(&offset, end, 4, &ok);
        if (!mode) {
            int start = (offset + 7) / 8;
            while (offset % 8)
                if (read_bits(&offset, end, 1, &ok)) return 0;
            if (!ok) return 0;
            for (int extra = 0; extra < 2; extra++) {
                int valid = 1;
                for (int i = start; i < total; i++) {
                    int value = extra && i == start ? 0 : ((i - start - extra) & 1) ? 0x11 : 0xec;
                    if (i < prefix && dataBytes[i] != value) {
                        valid = 0;
                        break;
                    }
                    if (i >= prefix) dataBytes[i] = (uint8_t)value;
                }
                if (valid && !parse(total, version)) return 1;
            }
            return 0;
        }
        if (mode != 1 && mode != 2 && mode != 4) return 0;
        int count = read_bits(&offset, end,
                              mode == 1   ? (version < 10 ? 10 : 12)
                              : mode == 2 ? (version < 10 ? 9 : 11)
                                          : (version < 10 ? 8 : 16),
                              &ok);
        if (!ok) return 0;
        int bits = mode == 1   ? (count / 3) * 10 + (count % 3 == 1   ? 4
                                                     : count % 3 == 2 ? 7
                                                                      : 0)
                   : mode == 2 ? (count / 2) * 11 + (count % 2) * 6
                               : count * 8;
        if (offset + bits > end) return 0;
        offset += bits;
    }
    return 0;
}

/* Symbol/block map for evidence certification: per module, the RS symbol index within its block and the block. */
static int16_t symbolOf[GRID];
static int8_t blockOf[GRID];
static int blockTotals[BLOCKS], blockData[BLOCKS];
static uint16_t formatPositions[30];
API uintptr_t sq_symbol_of(void) {
    return (uintptr_t)symbolOf;
}
API uintptr_t sq_block_of(void) {
    return (uintptr_t)blockOf;
}
API uintptr_t sq_block_totals(void) {
    return (uintptr_t)blockTotals;
}
API uintptr_t sq_block_data(void) {
    return (uintptr_t)blockData;
}
API uintptr_t sq_format_positions(void) {
    return (uintptr_t)formatPositions;
}
API int sq_symbol_map(int version, int ecc) {
    if (version < 1 || version > 10 || ecc < 0 || ecc > 3) return -1;
    Layout *l = layout(version);
    int n = l->n, count = 0, maxData = 0, total = 0;
    for (int group = 0; group < 2; group++) {
        const unsigned short *spec = qrSpec[version - 1][ecc] + group * 3;
        for (int j = 0; j < spec[0]; j++) {
            if (count >= BLOCKS) return -1;
            blockTotals[count] = spec[1];
            blockData[count] = spec[2];
            maxData = maximum(maxData, spec[2]);
            total += spec[1];
            count++;
        }
    }
    for (int i = 0; i < n * n; i++) {
        symbolOf[i] = -1;
        blockOf[i] = -1;
    }
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 15; i++) formatPositions[k * 15 + i] = l->format[k][i];
    int at = 0, parity = blockTotals[0] - blockData[0];
    for (int i = 0; i < maxData; i++)
        for (int j = 0; j < count; j++)
            if (i < blockData[j]) {
                for (int bit = 0; bit < 8; bit++) {
                    int p = l->data[at * 8 + bit];
                    symbolOf[p] = (int16_t)i;
                    blockOf[p] = (int8_t)j;
                }
                at++;
            }
    for (int i = 0; i < parity; i++)
        for (int j = 0; j < count; j++) {
            for (int bit = 0; bit < 8; bit++) {
                int p = l->data[at * 8 + bit];
                symbolOf[p] = (int16_t)(blockData[j] + i);
                blockOf[p] = (int8_t)j;
            }
            at++;
        }
    if (at != total) return -1;
    return count;
}

/* Known (payload-independent) modules for sampling-level estimation: fixed patterns minus format bits. */
API int sq_known_layout(int version, uint8_t *mask, uint8_t *values) {
    if (version < 1 || version > 10 || !mask || !values) return -1;
    Layout *l = layout(version);
    int n = l->n;
    for (int i = 0; i < n * n; i++) {
        mask[i] = l->fixed[i];
        values[i] = l->values[i];
    }
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 15; i++) mask[l->format[k][i]] = 0;
    return n;
}
API int sq_decode(int n, int erasures, double erasureConfidence, int maxErasures, int ranked, double maxDisagreement,
                  double maxWeightedDisagreement, int softFormat, double formatErasureConfidence) {
    if (n < 21 || n > 57 || (n - 17) % 4 || (erasures != 0 && erasures != 1) || (ranked != 0 && ranked != 1) ||
        (softFormat != 0 && softFormat != 1) || maxErasures < 0 || maxErasures > 30 ||
        !(erasureConfidence >= 0 && erasureConfidence <= 1) || !(maxDisagreement >= 0 && maxDisagreement <= 1) ||
        !(maxWeightedDisagreement >= 0 && maxWeightedDisagreement <= 1) ||
        !(formatErasureConfidence >= 0 && formatErasureConfidence <= 1))
        return -1;
    int visibleCount = 0;
    for (int i = 0; i < n * n; i++) {
        if (!(probability[i] >= 0 && probability[i] <= 1)) return -1;
        if (visibilityEnabled) {
            if (visible[i] > 1 || (!visible[i] && probability[i] != .5)) return -1;
            visibleCount += visible[i];
        }
    }
    if (visibilityEnabled && visibleCount * 2 < n * n) return -1;
    for (int i = 0; i < 11; i++) result[i] = 0;
    metrics[0] = metrics[1] = 0;
    init_gf();
    int version = (n - 17) / 4;
    Layout *l = layout(version);
    unsigned observed[2] = {0}, erased[2] = {0};
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 15; i++) {
            double p = probability[l->format[k][i]];
            observed[k] |= (unsigned)(p >= .5) << i;
            if (softFormat && absolute(p - .5) * 2 < formatErasureConfidence) erased[k] |= 1u << i;
        }
    int winner = -1, tied = 0, formatCount = 0, formatIDs[32];
    double best = 1e30, formatCosts[32];
    for (int format = 0; format < 32; format++) {
        unsigned bits = bch(format, 10, 0x537, 0x5412);
        int valid = minimum(popcount(observed[0] ^ bits), popcount(observed[1] ^ bits)) <= 3;
        if (softFormat)
            for (int k = 0; k < 2; k++) {
                int s = popcount(erased[k]), e = popcount((observed[k] ^ bits) & (~erased[k] & 0x7fff));
                if (s <= 6 && 2 * e + s <= 6) valid = 1;
            }
        if (!valid) continue;
        double cost = 0;
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < 15; i++) {
                double p = probability[l->format[k][i]];
                cost += (bits >> i) & 1 ? 1 - p : p;
            }
        formatIDs[formatCount] = format;
        formatCosts[formatCount++] = cost;
        if (cost < best - 1e-7) {
            best = cost;
            winner = format;
            tied = 0;
        } else if (absolute(cost - best) <= 1e-7)
            tied = 1;
    }
    if (formatRank >= 0) {
        for (int i = 1; i < formatCount; i++) {
            int id = formatIDs[i], at = i;
            double cost = formatCosts[i];
            while (at > 0 && formatCosts[at - 1] > cost) {
                formatIDs[at] = formatIDs[at - 1];
                formatCosts[at] = formatCosts[at - 1];
                at--;
            }
            formatIDs[at] = id;
            formatCosts[at] = cost;
        }
        if (formatRank >= formatCount) return reject(1);
        winner = formatIDs[formatRank];
        tied = 0;
    }
    if (winner < 0 || tied) return reject(1);
    if (version >= 7) {
        unsigned expected = bch(version, 12, 0x1f25, 0), a = 0, b = 0;
        for (int i = 0; i < 18; i++) {
            int x = n - 11 + i % 3, y = i / 3;
            a |= (unsigned)(probability[y * n + x] >= .5) << i;
            b |= (unsigned)(probability[x * n + y] >= .5) << i;
        }
        if (minimum(popcount(a ^ expected), popcount(b ^ expected)) > 3) return reject(2);
    }
    int ecc = winner >> 3, mask = winner & 7, total = 0, maxData = 0;
    blocks = 0;
    for (int group = 0; group < 2; group++) {
        const unsigned short *spec = qrSpec[version - 1][ecc] + group * 3;
        for (int j = 0; j < spec[0]; j++) {
            if (blocks >= BLOCKS) return -1;
            totals[blocks] = spec[1];
            lengths[blocks] = spec[2];
            maxData = maximum(maxData, spec[2]);
            total += spec[1];
            blocks++;
        }
    }
    if (total > WORDS || total * 8 > l->count) return -1;
    for (int i = 0; i < total; i++) {
        raw[i] = 0;
        confidence[i] = reliabilityMode == 2 ? 0 : 1;
    }
    for (int i = 0; i < total * 8; i++) {
        int at = l->data[i], x = at % n, y = at / n;
        double p = probability[at];
        int bit = (p >= .5) ^ mask_at(mask, x, y);
        raw[i >> 3] |= (uint8_t)(bit << (7 - (i & 7)));
        double c = absolute(p - .5) * 2;
        if (reliabilityMode == 1)
            confidence[i >> 3] *= (float)((1 + c) * .5);
        else if (reliabilityMode == 2)
            confidence[i >> 3] += (float)(c * .125);
        else if (c < confidence[i >> 3])
            confidence[i >> 3] = (float)c;
    }
    int at = 0, parity = totals[0] - lengths[0];
    for (int i = 0; i < maxData; i++)
        for (int j = 0; j < blocks; j++)
            if (i < lengths[j]) {
                parts[j][i] = raw[at];
                quality[j][i] = confidence[at++];
            }
    for (int i = 0; i < parity; i++)
        for (int j = 0; j < blocks; j++) {
            parts[j][lengths[j] + i] = raw[at];
            quality[j][lengths[j] + i] = confidence[at++];
        }
    int dataLength = 0, used = 0, errors = 0, residualParity = 30;
    if (parity < minimumReserve) return reject(9);
    int erasureLimit = minimum(maxErasures, parity - minimumReserve);
    for (int j = 0; j < blocks; j++) {
        int count = 0;
        for (int i = 0; i < totals[j]; i++) {
            word[i] = parts[j][i];
            if (erasures && quality[j][i] < erasureConfidence) positions[count++] = i;
        }
        if (ranked) {
            for (int i = 1; i < count; i++) {
                int value = positions[i], k = i;
                while (k > 0 && (quality[j][positions[k - 1]] > quality[j][value] ||
                                 (quality[j][positions[k - 1]] == quality[j][value] && positions[k - 1] > value))) {
                    positions[k] = positions[k - 1];
                    k--;
                }
                positions[k] = value;
            }
            count = minimum(count, erasureLimit);
        } else if (count > erasureLimit)
            return reject(3);
        int unknown = 0, solved = 0;
        if (adaptiveErasures && ranked && erasures) {
            /* Independent, bounded GMD schedule per block. Never changes observations. */
            int limit = count;
            for (int trial = 0; trial <= limit; trial++) {
                for (int i = 0; i < totals[j]; i++) word[i] = parts[j][i];
                if (rs_correct(totals[j], parity, trial, &unknown) && parity - 2 * unknown - trial >= minimumReserve) {
                    count = trial;
                    solved = 1;
                    break;
                }
            }
        } else
            solved = rs_correct(totals[j], parity, count, &unknown);
        if (!solved) {
            int allData = 0;
            for (int k = 0; k < blocks; k++) allData += lengths[k];
            if (!paddingEnabled || !dataLength || !complete_padding(dataLength, version, allData)) return reject(4);
            int offset = dataLength;
            for (int k = j; k < blocks; k++) {
                encode_rs(dataBytes + offset, lengths[k], parity, encoded[k]);
                offset += lengths[k];
            }
            dataLength = allData;
            break;
        }
        int reserve = parity - 2 * unknown - count;
        if (reserve < minimumReserve) return reject(9);
        residualParity = minimum(residualParity, reserve);
        encode_rs(word, lengths[j], parity, encoded[j]);
        for (int i = 0; i < totals[j]; i++)
            if (encoded[j][i] != word[i]) return reject(5);
        for (int i = 0; i < lengths[j]; i++) dataBytes[dataLength++] = encoded[j][i];
        used += count;
        errors += unknown;
    }
    at = 0;
    for (int i = 0; i < maxData; i++)
        for (int j = 0; j < blocks; j++)
            if (i < lengths[j]) raw[at++] = encoded[j][i];
    for (int i = 0; i < parity; i++)
        for (int j = 0; j < blocks; j++) raw[at++] = encoded[j][lengths[j] + i];
    for (int i = 0; i < n * n; i++) grid[i] = l->values[i];
    for (int i = 0; i < l->count; i++) {
        int p = l->data[i], bit = i < total * 8 ? (raw[i >> 3] >> (7 - (i & 7))) & 1 : 0;
        grid[p] = (uint8_t)(bit ^ mask_at(mask, p % n, p / n));
    }
    unsigned formatBits = bch(winner, 10, 0x537, 0x5412);
    for (int k = 0; k < 2; k++)
        for (int i = 0; i < 15; i++) grid[l->format[k][i]] = (uint8_t)((formatBits >> i) & 1);
    int disagree = 0;
    double weighted = 0, weight = 0;
    for (int i = 0; i < n * n; i++) {
        if (visibilityEnabled && !visible[i]) continue;
        double p = probability[i], w = absolute(p - .5) * 2;
        weight += w;
        if ((p >= .5) != grid[i]) {
            disagree++;
            weighted += w;
        }
    }
    metrics[0] = (double)disagree / (visibilityEnabled ? visibleCount : n * n);
    metrics[1] = weight ? weighted / weight : 1;
    if (metrics[0] > maxDisagreement || metrics[1] > maxWeightedDisagreement) return reject(6);
    int parsed = parse(dataLength, version);
    if (parsed) return reject(parsed);
    result[0] = 1;
    result[2] = version;
    result[3] = ecc;
    result[4] = mask;
    result[7] = used;
    result[8] = errors;
    result[10] = residualParity;
    return 1;
}
