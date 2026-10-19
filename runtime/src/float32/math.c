/* float32-graph-v1 math: deterministic expf, expm1f, tanhf and erff, plus bit-level helpers.
   No libm, no host math: only IEEE-754 +, -, *, / on float and double and integer bit casts, so every build
   (native, native without SSE2, WebAssembly SIMD and scalar) returns identical bits. Contraction is off.

   expf, expm1f, tanhf and erff are ports of musl 1.2.5 src/math (MIT, with the FreeBSD msun / Sun notice for
   erff and expm1f); see runtime/src/float32/NOTICE. Changes: libm macros replaced by local helpers, the
   FORCE_EVAL calls (inexact flag only) dropped, __math_oflowf(0) and __math_uflowf(0) written as their results
   (+inf, +0), double_t/float_t fixed to double/float. The polynomials, tables and constants are unchanged. */
#pragma clang fp contract(off)

typedef union {
    float f;
    uint32_t u;
} GfBits32;
typedef union {
    double f;
    uint64_t u;
} GfBits64;

static inline uint32_t gf_asuint(float f) {
    GfBits32 b;
    b.f = f;
    return b.u;
}
static inline float gf_asfloat(uint32_t u) {
    GfBits32 b;
    b.u = u;
    return b.f;
}
static inline uint64_t gf_asuint64(double f) {
    GfBits64 b;
    b.f = f;
    return b.u;
}
static inline double gf_asdouble(uint64_t u) {
    GfBits64 b;
    b.u = u;
    return b.f;
}
static inline float gf_fabsf(float x) {
    return gf_asfloat(gf_asuint(x) & 0x7fffffffu);
}
/* Not infinity, not NaN. */
static inline int gf_finite(float x) {
    return ((gf_asuint(x) >> 23) & 255u) != 255u;
}
/* Finite and strictly positive. */
static inline int gf_positive(float x) {
    const uint32_t u = gf_asuint(x);
    return gf_finite(x) && !(u >> 31) && (u & 0x7fffffffu);
}
/* float16 bits to float32 with integer arithmetic only; infinity and NaN (exponent 31) must be refused first. */
static float gf_f16(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    if (e == 0u) {
        if (m == 0u) return gf_asfloat(sign);
        int p = 9; /* subnormal m * 2^-24: normalise by the highest set bit */
        while (!((m >> p) & 1u)) p--;
        return gf_asfloat(sign | ((uint32_t)(p + 103) << 23) | ((m << (23 - p)) & 0x7FFFFFu));
    }
    return gf_asfloat(sign | ((e + 112u) << 23) | (m << 13));
}

/* ---- expf: origin musl src/math/expf.c and exp2f_data.c, Copyright (c) 2017-2018, Arm Limited, MIT. ---- */
static const uint64_t gf_exp2f_tab[32] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b,
    0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715, 0x3feef1a7373aa9cb,
    0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
    0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d, 0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};
/* invln2_scaled = 0x1.71547652b82fep+0 * 32; poly_scaled = {0x1.c6af84b912394p-5 / 32^3, 0x1.ebfce50fac4f3p-3 /
   32^2, 0x1.62e42ff0c52d6p-1 / 32}; shift = 0x1.8p+52 */
#define GF_EXP2F_INVLN2N (0x1.71547652b82fep+0 * 32)
#define GF_EXP2F_SHIFT   0x1.8p+52
#define GF_EXP2F_C0      (0x1.c6af84b912394p-5 / 32 / 32 / 32)
#define GF_EXP2F_C1      (0x1.ebfce50fac4f3p-3 / 32 / 32)
#define GF_EXP2F_C2      (0x1.62e42ff0c52d6p-1 / 32)

static float gf_expf(float x) {
    const uint32_t abstop = (gf_asuint(x) >> 20) & 0x7ffu;
    const double xd = (double)x;
    if (abstop >= 0x42bu) { /* top12(88.0f): |x| >= 88 or x is nan */
        if (gf_asuint(x) == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return gf_asfloat(0x7f800000u); /* __math_oflowf(0) */
        if (x < -0x1.9fe368p6f) return 0.0f;                   /* __math_uflowf(0) */
    }
    /* x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k */
    double z = GF_EXP2F_INVLN2N * xd;
    double kd = z + GF_EXP2F_SHIFT;
    const uint64_t ki = gf_asuint64(kd);
    kd -= GF_EXP2F_SHIFT;
    const double r = z - kd;
    /* exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
    uint64_t t = gf_exp2f_tab[ki % 32u];
    t += ki << (52 - 5);
    const double s = gf_asdouble(t);
    z = GF_EXP2F_C0 * r + GF_EXP2F_C1;
    const double r2 = r * r;
    double y = GF_EXP2F_C2 * r + 1;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* ---- expm1f: origin FreeBSD msun s_expm1f.c via musl (Sun Microsystems notice, see NOTICE). ---- */
static float gf_expm1f(float x) {
    const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f, invln2 = 1.4426950216e+00f;
    const float Q1 = -3.3333212137e-2f, Q2 = 1.5807170421e-3f;
    float y, hi, lo, c = 0.0f, t, e, hxs, hfx, r1, twopk;
    uint32_t hx = gf_asuint(x) & 0x7fffffffu;
    int k, sign = (int)(gf_asuint(x) >> 31);

    if (hx >= 0x4195b844u) { /* |x| >= 27*ln2 */
        if (hx > 0x7f800000u) return x;
        if (sign) return -1;
        if (hx > 0x42b17217u) { /* x > log(FLT_MAX) */
            x *= 0x1p127f;
            return x;
        }
    }
    if (hx > 0x3eb17218u) {     /* |x| > 0.5 ln2 */
        if (hx < 0x3F851592u) { /* and |x| < 1.5 ln2 */
            if (!sign) {
                hi = x - ln2_hi;
                lo = ln2_lo;
                k = 1;
            } else {
                hi = x + ln2_hi;
                lo = -ln2_lo;
                k = -1;
            }
        } else {
            k = (int)(invln2 * x + (sign ? -0.5f : 0.5f));
            t = (float)k;
            hi = x - t * ln2_hi; /* t*ln2_hi is exact here */
            lo = t * ln2_lo;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if (hx < 0x33000000u) { /* |x| < 2**-25: return x */
        return x;
    } else {
        k = 0;
    }
    /* x is now in the primary range */
    hfx = 0.5f * x;
    hxs = x * hfx;
    r1 = 1.0f + hxs * (Q1 + hxs * Q2);
    t = 3.0f - r1 * hfx;
    e = hxs * ((r1 - t) / (6.0f - x * t));
    if (k == 0) return x - (x * e - hxs); /* c is 0 */
    e = x * (e - c) - c;
    e -= hxs;
    /* exp(x) ~ 2^k (x_reduced - e + 1) */
    if (k == -1) return 0.5f * (x - e) - 0.5f;
    if (k == 1) {
        if (x < -0.25f) return -2.0f * (e - (x + 0.5f));
        return 1.0f + 2.0f * (x - e);
    }
    twopk = gf_asfloat((uint32_t)(0x7f + k) << 23); /* 2^k */
    if (k < 0 || k > 56) {                          /* suffices to return exp(x)-1 */
        y = x - e + 1.0f;
        if (k == 128)
            y = y * 2.0f * 0x1p127f;
        else
            y = y * twopk;
        return y - 1.0f;
    }
    const float twomk = gf_asfloat((uint32_t)(0x7f - k) << 23); /* 2^-k */
    if (k < 23)
        y = (x - e + (1 - twomk)) * twopk;
    else
        y = (x - (e + twomk) + 1) * twopk;
    return y;
}

/* ---- tanhf: origin FreeBSD msun s_tanhf.c via musl. ---- */
static float gf_tanhf(float x) {
    uint32_t w = gf_asuint(x);
    const int sign = (int)(w >> 31);
    float t;
    w &= 0x7fffffffu;
    x = gf_asfloat(w);
    if (w > 0x3f0c9f54u) {     /* |x| > log(3)/2 ~= 0.5493 or nan */
        if (w > 0x41200000u) { /* |x| > 10 */
            t = 1 + 0 / x;
        } else {
            t = gf_expm1f(2 * x);
            t = 1 - 2 / (t + 2);
        }
    } else if (w > 0x3e82c578u) { /* |x| > log(5/3)/2 ~= 0.2554 */
        t = gf_expm1f(2 * x);
        t = t / (t + 2);
    } else if (w >= 0x00800000u) { /* |x| >= 0x1p-126 */
        t = gf_expm1f(-2 * x);
        t = -t / (t + 2);
    } else { /* |x| is subnormal */
        t = x;
    }
    return sign ? -t : t;
}

/* ---- erff: origin FreeBSD msun s_erff.c via musl (Sun Microsystems notice, see NOTICE). ---- */
static const float gf_erx = 8.4506291151e-01f, gf_efx8 = 1.0270333290e+00f, gf_pp0 = 1.2837916613e-01f,
                   gf_pp1 = -3.2504209876e-01f, gf_pp2 = -2.8481749818e-02f, gf_pp3 = -5.7702702470e-03f,
                   gf_pp4 = -2.3763017452e-05f, gf_qq1 = 3.9791721106e-01f, gf_qq2 = 6.5022252500e-02f,
                   gf_qq3 = 5.0813062117e-03f, gf_qq4 = 1.3249473704e-04f, gf_qq5 = -3.9602282413e-06f,
                   gf_pa0 = -2.3621185683e-03f, gf_pa1 = 4.1485610604e-01f, gf_pa2 = -3.7220788002e-01f,
                   gf_pa3 = 3.1834661961e-01f, gf_pa4 = -1.1089469492e-01f, gf_pa5 = 3.5478305072e-02f,
                   gf_pa6 = -2.1663755178e-03f, gf_qa1 = 1.0642088205e-01f, gf_qa2 = 5.4039794207e-01f,
                   gf_qa3 = 7.1828655899e-02f, gf_qa4 = 1.2617121637e-01f, gf_qa5 = 1.3637083583e-02f,
                   gf_qa6 = 1.1984500103e-02f, gf_ra0 = -9.8649440333e-03f, gf_ra1 = -6.9385856390e-01f,
                   gf_ra2 = -1.0558626175e+01f, gf_ra3 = -6.2375331879e+01f, gf_ra4 = -1.6239666748e+02f,
                   gf_ra5 = -1.8460508728e+02f, gf_ra6 = -8.1287437439e+01f, gf_ra7 = -9.8143291473e+00f,
                   gf_sa1 = 1.9651271820e+01f, gf_sa2 = 1.3765776062e+02f, gf_sa3 = 4.3456588745e+02f,
                   gf_sa4 = 6.4538726807e+02f, gf_sa5 = 4.2900814819e+02f, gf_sa6 = 1.0863500214e+02f,
                   gf_sa7 = 6.5702495575e+00f, gf_sa8 = -6.0424413532e-02f, gf_rb0 = -9.8649431020e-03f,
                   gf_rb1 = -7.9928326607e-01f, gf_rb2 = -1.7757955551e+01f, gf_rb3 = -1.6063638306e+02f,
                   gf_rb4 = -6.3756646729e+02f, gf_rb5 = -1.0250950928e+03f, gf_rb6 = -4.8351919556e+02f,
                   gf_sb1 = 3.0338060379e+01f, gf_sb2 = 3.2579251099e+02f, gf_sb3 = 1.5367296143e+03f,
                   gf_sb4 = 3.1998581543e+03f, gf_sb5 = 2.5530502930e+03f, gf_sb6 = 4.7452853394e+02f,
                   gf_sb7 = -2.2440952301e+01f;

static float gf_erfc1(float x) {
    float s = gf_fabsf(x) - 1;
    float P = gf_pa0 + s * (gf_pa1 + s * (gf_pa2 + s * (gf_pa3 + s * (gf_pa4 + s * (gf_pa5 + s * gf_pa6)))));
    float Q = 1 + s * (gf_qa1 + s * (gf_qa2 + s * (gf_qa3 + s * (gf_qa4 + s * (gf_qa5 + s * gf_qa6)))));
    return 1 - gf_erx - P / Q;
}

static float gf_erfc2(uint32_t ix, float x) {
    float s, R, S, z;
    if (ix < 0x3fa00000u) return gf_erfc1(x); /* |x| < 1.25 */
    x = gf_fabsf(x);
    s = 1 / (x * x);
    if (ix < 0x4036db6du) { /* |x| < 1/0.35 */
        R = gf_ra0 +
            s * (gf_ra1 + s * (gf_ra2 + s * (gf_ra3 + s * (gf_ra4 + s * (gf_ra5 + s * (gf_ra6 + s * gf_ra7))))));
        S = 1.0f +
            s * (gf_sa1 +
                 s * (gf_sa2 + s * (gf_sa3 + s * (gf_sa4 + s * (gf_sa5 + s * (gf_sa6 + s * (gf_sa7 + s * gf_sa8)))))));
    } else { /* |x| >= 1/0.35 */
        R = gf_rb0 + s * (gf_rb1 + s * (gf_rb2 + s * (gf_rb3 + s * (gf_rb4 + s * (gf_rb5 + s * gf_rb6)))));
        S = 1.0f + s * (gf_sb1 + s * (gf_sb2 + s * (gf_sb3 + s * (gf_sb4 + s * (gf_sb5 + s * (gf_sb6 + s * gf_sb7))))));
    }
    ix = gf_asuint(x);
    z = gf_asfloat(ix & 0xffffe000u);
    return gf_expf(-z * z - 0.5625f) * gf_expf((z - x) * (z + x) + R / S) / x;
}

static float gf_erff(float x) {
    float r, s, z, y;
    uint32_t ix = gf_asuint(x);
    const int sign = (int)(ix >> 31);
    ix &= 0x7fffffffu;
    if (ix >= 0x7f800000u) return 1 - 2 * sign + 1 / x;              /* erf(nan)=nan, erf(+-inf)=+-1 */
    if (ix < 0x3f580000u) {                                          /* |x| < 0.84375 */
        if (ix < 0x31800000u) return 0.125f * (8 * x + gf_efx8 * x); /* |x| < 2**-28 */
        z = x * x;
        r = gf_pp0 + z * (gf_pp1 + z * (gf_pp2 + z * (gf_pp3 + z * gf_pp4)));
        s = 1 + z * (gf_qq1 + z * (gf_qq2 + z * (gf_qq3 + z * (gf_qq4 + z * gf_qq5))));
        y = r / s;
        return x + x * y;
    }
    if (ix < 0x40c00000u) /* |x| < 6 */
        y = 1 - gf_erfc2(ix, x);
    else
        y = 1 - 0x1p-120f;
    return sign ? -y : y;
}

/* The two activations of F_UNARY. GELU is the exact (erf) definition in the PyTorch float32 order:
   (x * 0.5) * (1 + erf(x * 0.70710678)). */
static inline float gf_gelu(float x) {
    const float e = gf_erff(x * 0.70710678118654752440f);
    return (x * 0.5f) * (1.0f + e);
}
