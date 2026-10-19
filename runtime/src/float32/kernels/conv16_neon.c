/* Short NEON convolutions: retain two signed 4-bit weights per byte in the existing slab scratch, then widen
   inside the tile. Activations stay int16, sums int32, and the epilogue is unchanged. No extra storage. */
#if defined(__aarch64__)
typedef uint8_t gq4u8 __attribute__((vector_size(8), aligned(1), may_alias));

static void gq4_codes_slab(const GfWeights *w, uint32_t n0, uint32_t jn, uint32_t K, uint8_t *dst) {
    const uint32_t pairs = K / 2u;
    for (uint32_t j = 0; j < 8u; j++) {
        if (j < jn) {
            const uint8_t *row = w->code + (size_t)(n0 + j) * pairs;
            for (uint32_t p = 0; p < pairs; p++) dst[(size_t)p * 8u + j] = row[p];
        } else {
            for (uint32_t p = 0; p < pairs; p++) dst[(size_t)p * 8u + j] = 0;
        }
    }
}

static inline __attribute__((always_inline)) void gq4_tile(const int16_t *x, uint32_t xk, const uint8_t *w,
                                                           uint32_t pairs, int32_t *a, const uint32_t R) {
    int32x4_t acc[GQ_TR][2];
    for (uint32_t r = 0; r < R; r++) {
        acc[r][0] = vld1q_s32(a + 8u * r);
        acc[r][1] = vld1q_s32(a + 8u * r + 4u);
    }
    for (uint32_t p = 0; p < pairs; p++) {
        const gq4u8 raw = *(const gq4u8 *)(w + (size_t)p * 8u);
        const gv8s low = (__builtin_convertvector(raw & 15u, gv8s) ^ 8) - 8;
        const gv8s high = (__builtin_convertvector(raw >> 4u, gv8s) ^ 8) - 8;
        const int16x8_t w0 = (int16x8_t)low, w1 = (int16x8_t)high;
        for (uint32_t r = 0; r < R; r++) {
            const int16x4_t xv = vcreate_s16((uint32_t)*(const GqI32 *)(x + (size_t)r * xk + 2u * p));
            acc[r][0] = vmlal_lane_s16(vmlal_lane_s16(acc[r][0], vget_low_s16(w0), xv, 0), vget_low_s16(w1), xv, 1);
            acc[r][1] = vmlal_high_lane_s16(vmlal_high_lane_s16(acc[r][1], w0, xv, 0), w1, xv, 1);
        }
    }
    for (uint32_t r = 0; r < R; r++) {
        vst1q_s32(a + 8u * r, acc[r][0]);
        vst1q_s32(a + 8u * r + 4u, acc[r][1]);
    }
}
#endif
