/* float32 graph kernels: reference attention inner loops. Part of the src/gust.c translation unit. */
/* ---- attention inner loops (head dimension D <= GUST_F32_MAX_HEAD) ---- */

/* Dot product of a float64-converted query row and a float32 key row: even and odd dimensions in two partial
   sums, a trailing odd dimension added to the even sum, result even + odd. */
static double gf_dot64(const double *q, const float *k, uint32_t D) {
    uint32_t d = 0;
#ifdef GUST_F32_SIMD
    gv2d s = {0.0, 0.0};
    for (; d + 1 < D; d += 2) s = s + *(const gv2d *)(q + d) * __builtin_convertvector(*(const gv2f *)(k + d), gv2d);
    double even = s[0], odd = s[1];
#else
    double even = 0.0, odd = 0.0;
    for (; d + 1 < D; d += 2) {
        even = even + q[d] * (double)k[d];
        odd = odd + q[d + 1] * (double)k[d + 1];
    }
#endif
    if (d < D) even = even + q[d] * (double)k[d];
    return even + odd;
}

/* acc[d] += e * v[d] for one key; callers apply keys in increasing order. Four keys at once keep that order. */
static void gf_axpy64_1(double *acc, double e, const float *v, uint32_t D) {
    uint32_t d = 0;
#ifdef GUST_F32_SIMD
    const gv2d ev = {e, e};
    for (; d + 1 < D; d += 2)
        *(gv2d *)(acc + d) = *(const gv2d *)(acc + d) + ev * __builtin_convertvector(*(const gv2f *)(v + d), gv2d);
#else
    for (; d + 1 < D; d += 2) {
        acc[d] = acc[d] + e * (double)v[d];
        acc[d + 1] = acc[d + 1] + e * (double)v[d + 1];
    }
#endif
    if (d < D) acc[d] = acc[d] + e * (double)v[d];
}

static void gf_axpy64_4(double *acc, const double *e, const float *v, size_t vrow, uint32_t D) {
    const float *v0 = v, *v1 = v + vrow, *v2 = v + 2 * vrow, *v3 = v + 3 * vrow;
    uint32_t d = 0;
#ifdef GUST_F32_SIMD
    const gv2d e0 = {e[0], e[0]}, e1 = {e[1], e[1]}, e2 = {e[2], e[2]}, e3 = {e[3], e[3]};
    for (; d + 1 < D; d += 2) {
        gv2d a = *(const gv2d *)(acc + d);
        a = a + e0 * __builtin_convertvector(*(const gv2f *)(v0 + d), gv2d);
        a = a + e1 * __builtin_convertvector(*(const gv2f *)(v1 + d), gv2d);
        a = a + e2 * __builtin_convertvector(*(const gv2f *)(v2 + d), gv2d);
        a = a + e3 * __builtin_convertvector(*(const gv2f *)(v3 + d), gv2d);
        *(gv2d *)(acc + d) = a;
    }
#else
    for (; d + 1 < D; d += 2) {
        double a0 = acc[d], a1 = acc[d + 1];
        a0 = a0 + e[0] * (double)v0[d];
        a1 = a1 + e[0] * (double)v0[d + 1];
        a0 = a0 + e[1] * (double)v1[d];
        a1 = a1 + e[1] * (double)v1[d + 1];
        a0 = a0 + e[2] * (double)v2[d];
        a1 = a1 + e[2] * (double)v2[d + 1];
        a0 = a0 + e[3] * (double)v3[d];
        a1 = a1 + e[3] * (double)v3[d + 1];
        acc[d] = a0;
        acc[d + 1] = a1;
    }
#endif
    if (d < D) {
        double a = acc[d];
        a = a + e[0] * (double)v0[d];
        a = a + e[1] * (double)v1[d];
        a = a + e[2] * (double)v2[d];
        a = a + e[3] * (double)v3[d];
        acc[d] = a;
    }
}
