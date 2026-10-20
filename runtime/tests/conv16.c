/* Private-kernel regression: packed 2/3/4-bit rows, odd K, channel tails and every row remainder.
   Build with the same feature/ISA flags as gust.c; this includes that translation unit for its private kernels.
   No model artifacts required. Run on both x86 and ARM to check their respective implementations. */
#include "../src/gust.c"
#include <stdio.h>
#include <string.h>

static int16_t x[25 * 2048], slab[8 * 2048];
static uint8_t packed[8 * 2048 / 2];
static float out[25 * 8];
#if defined(__aarch64__)
static uint8_t packed_slab[8 * 2048 / 2];
#endif

static int code(uint32_t j, uint32_t k, uint32_t bits) {
    return (int)((j * 13u + k * 7u) & ((1u << bits) - 1u)) - (1 << (bits - 1u));
}

int main(void) {
    const uint32_t widths[] = {1, 2, 3, 7, 8, 9, 63, 64, 65, 2047, 2048};
    float scale[25], ws[8], bias[8];
    unsigned cases = 0;
    for (uint32_t bits = 2; bits <= 4; bits++)
        for (uint32_t ki = 0; ki < sizeof widths / sizeof widths[0]; ki++) {
            const uint32_t K = widths[ki], kk = (K + 1u) & ~1u;
            memset(packed, 0, sizeof packed);
            for (uint32_t j = 0; j < 8; j++) {
                ws[j] = (float)(j + 1u) * 0.0031f;
                bias[j] = (float)j * -0.17f;
                for (uint32_t k = 0; k < K; k++) {
                    const uint32_t v = (uint32_t)code(j, k, bits) & ((1u << bits) - 1u);
                    const uint32_t pos = (j * K + k) * bits;
                    for (uint32_t b = 0; b < bits; b++)
                        packed[(pos + b) >> 3] |= (uint8_t)(((v >> b) & 1u) << ((pos + b) & 7u));
                }
            }
            GfWeights w = {0};
            w.code = packed;
            w.K = K;
            w.bits = bits;
            for (uint32_t j = 0; j < 8; j++) {
                gq_codes_row(&w, j, K, slab, j);
                if (K & 1u) slab[(size_t)(kk / 2u - 1u) * 16u + 2u * j + 1u] = 0;
            }
            for (uint32_t r = 0; r < 25; r++) {
                scale[r] = r == 0 ? 0.0f : (float)r * 0.00013f;
                for (uint32_t k = 0; k < kk; k++)
                    x[r * kk + k] = k == K ? 0 : (int16_t)((int)((r * 7919u + k * 997u) % 65535u) - 32767);
            }
            for (uint32_t rows = 1; rows <= 25; rows++)
                for (uint32_t n = 1; n <= 8; n++) {
                    uint32_t modes = 1;
#if defined(__aarch64__)
                    if (bits == 4u && !(K & 1u)) modes = 2;
#endif
                    for (uint32_t mode = 0; mode < modes; mode++) {
#if defined(__aarch64__)
                        if (mode) {
                            gq4_codes_slab(&w, 0, n, K, packed_slab);
                            gq4_slab(x, kk, rows, packed_slab, kk / 2u, scale, ws, bias, out, n, n);
                        } else
#endif
                        gq_slab(x, kk, rows, slab, kk / 2u, scale, ws, bias, out, n, n);
                        for (uint32_t r = 0; r < rows; r++)
                            for (uint32_t j = 0; j < n; j++) {
                                int32_t sum = 0;
                                for (uint32_t k = 0; k < K; k++) sum += x[r * kk + k] * code(j, k, bits);
                                float expected = (float)sum * (scale[r] * ws[j]);
                                expected = expected + bias[j];
                                if (memcmp(&expected, out + r * n + j, sizeof expected)) {
                                    fprintf(stderr, "mismatch: bits=%u K=%u rows=%u n=%u r=%u j=%u\n", bits, K, rows, n, r, j);
                                    return 1;
                                }
                            }
                        cases++;
                    }
                }
        }
    printf("conv16: %u cases exact\n", cases);
    return 0;
}
