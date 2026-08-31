// Experimental ISPC backend for the frozen EXP53 mathematical machinery.
// No ISPC exp() or replacement transcendental math is used.
// Targeted at avx512skx-x8; processes four gangs (32 doubles) per uniform loop iteration.
// Exact fused operations are forced with LLVM fma intrinsics; bitwise gate remains mandatory.

static const uniform double INV128  = 0x1.71547652b82fep+7;
static const uniform double L128_HI = 0x1.62e42fefa39efp-8;
static const uniform double L128_MI = 0x1.abc9e3b39803fp-63;
static const uniform double L128_LO = 0x1.7b57a079a1934p-118;
static const uniform double MAGIC   = 0x1.8000000000000p+52;
static const uniform uint64 MAGIC_BITS = 0x4338000000000000;
static const uniform double Q1 = 0x1.0000000000000p-2;
static const uniform double Q2 = 0x1.aaaaaaaaaaaabp-5;
static const uniform double Q3 = 0x1.0000000000000p-7;
static const uniform double Q4 = 0x1.c16c16c16c16cp-11;

inline varying double vfma(varying double a, varying double b, varying double c)
{
    return @llvm.fma.v8f64(a, b, c);
}

inline void lane8(uniform double * uniform out,
                  uniform const double * uniform src,
                  uniform const uint64 * uniform tab_bits,
                  uniform int base)
{
    varying int idx = base + programIndex;
    varying double x = src[idx];
    varying double biased = vfma(x, INV128, MAGIC);
    varying double k = biased - MAGIC;
    varying int64 kn = (int64)intbits(biased) - (int64)MAGIC_BITS;
    varying int64 j = kn & 127;
    varying int64 q = kn >> 7;

    varying double r = vfma(-k, L128_HI, x);
    r = vfma(-k, L128_MI, r);
    r = vfma(-k, L128_LO, r);

    varying uint64 tb = tab_bits[j];
    varying uint64 sb = tb + (((uint64)q) << 52);
    varying double scale = doublebits(sb);

    varying double h = vfma(Q4, r, Q3);
    h = vfma(h, r, Q2);
    h = vfma(h, r, Q1);
    h = vfma(h, r, 1.0);
    varying double s = h * h;
    varying double er = vfma(r, s, 1.0);
    varying double el = vfma(r, s, 1.0 - er);
    varying double ph = er * scale;
    varying double y = vfma(el, scale, ph);
    out[idx] = y;
}

export void exp53_ispc_kernel(uniform double * uniform out,
                              uniform const double * uniform src,
                              uniform int count,
                              uniform const uint64 * uniform tab_bits)
{
    uniform int bulk = count & ~31;
    #pragma unroll(2)
    for (uniform int i = 0; i < bulk; i += 32) {
        lane8(out, src, tab_bits, i + 0);
        lane8(out, src, tab_bits, i + 8);
        lane8(out, src, tab_bits, i + 16);
        lane8(out, src, tab_bits, i + 24);
    }
}
