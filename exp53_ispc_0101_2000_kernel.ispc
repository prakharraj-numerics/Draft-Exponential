// Experimental ISPC backend for the frozen EXP53 mathematical machinery.
// No ISPC exp() or replacement transcendental math is used.
// Targeted at avx512skx-x8; processes four gangs (32 doubles) per uniform loop iteration.

static const uniform double INV128  = 0x1.71547652b82fep+7;
static const uniform double L128_HI = 0x1.62e42fefa39efp-8;
static const uniform double L128_MI = 0x1.abc9e3b39803fp-63;
static const uniform double L128_LO = 0x1.7b57a079a1934p-118;
static const uniform double MAGIC   = 0x1.8000000000000p+52;
static const uniform uint64 MAGIC_BITS = 0x4338000000000000;
static const uniform uint64 SIGN_BIT = 0x8000000000000000;
static const uniform double Q1 = 0x1.0000000000000p-2;
static const uniform double Q2 = 0x1.aaaaaaaaaaaabp-5;
static const uniform double Q3 = 0x1.0000000000000p-7;
static const uniform double Q4 = 0x1.c16c16c16c16cp-11;

inline varying double FMA(varying double a, varying double b, varying double c) {
    return @llvm.fma.v8f64(a,b,c);
}

// Exact IEEE sign flip. This avoids ISPC lowering unary '-' through a floating
// subtraction before llvm.fma; frozen AVX-512 fnmadd uses the original k bits
// with negated product semantics.
inline varying double NEG(varying double a) {
    return doublebits(intbits(a) ^ SIGN_BIT);
}

inline varying double FNMADD(varying double a, varying double b, varying double c) {
    return FMA(NEG(a), b, c);
}

inline void lane8(uniform double * uniform out,
                  uniform const double * uniform src,
                  uniform const uint64 * uniform tab_bits,
                  uniform int base)
{
    varying int idx = base + programIndex;
    varying double x = src[idx];
    varying double biased = FMA(x, INV128, MAGIC);
    varying double k = biased - MAGIC;
    varying int64 kn = (int64)intbits(biased) - (int64)MAGIC_BITS;
    varying int64 j = kn & 127;
    varying int64 q = kn >> 7;
    varying uint64 tb = tab_bits[j];

    varying double r = FNMADD(k, L128_HI, x);
    r = FNMADD(k, L128_MI, r);
    r = FNMADD(k, L128_LO, r);

    varying uint64 sb = tb + (((uint64)q) << 52);
    varying double scale = doublebits(sb);

    varying double h = FMA(Q4, r, Q3);
    h = FMA(h, r, Q2);
    h = FMA(h, r, Q1);
    h = FMA(h, r, 1.0);
    varying double s = h * h;
    varying double er = FMA(r, s, 1.0);
    varying double el = FMA(r, s, 1.0 - er);
    varying double ph = er * scale;
    varying double y = FMA(el, scale, ph);
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

// Diagnostic only: expose exact first-gang intermediates so C++ frozen intrinsics
// can identify the first bit-level divergence. Not used by timed candidate path.
export void exp53_ispc_debug8(uniform const double * uniform src,
                              uniform const uint64 * uniform tab_bits,
                              uniform uint64 * uniform dbg)
{
    varying int idx = programIndex;
    varying double x = src[idx];
    varying double biased = FMA(x, INV128, MAGIC);
    varying double k = biased - MAGIC;
    varying int64 kn = (int64)intbits(biased) - (int64)MAGIC_BITS;
    varying int64 j = kn & 127;
    varying int64 q = kn >> 7;
    varying uint64 tb = tab_bits[j];
    varying double r1 = FNMADD(k, L128_HI, x);
    varying double r2 = FNMADD(k, L128_MI, r1);
    varying double r = FNMADD(k, L128_LO, r2);
    varying uint64 sb = tb + (((uint64)q) << 52);
    varying double scale = doublebits(sb);
    varying double h1 = FMA(Q4, r, Q3);
    varying double h2 = FMA(h1, r, Q2);
    varying double h3 = FMA(h2, r, Q1);
    varying double h = FMA(h3, r, 1.0);
    varying double s = h*h;
    varying double er = FMA(r,s,1.0);
    varying double el = FMA(r,s,1.0-er);
    varying double ph = er*scale;
    varying double y = FMA(el,scale,ph);
#define PUT(stage,val) dbg[(stage)*8 + programIndex] = (uint64)(val)
    PUT(0,intbits(biased)); PUT(1,intbits(k)); PUT(2,(uint64)kn); PUT(3,(uint64)j); PUT(4,(uint64)q); PUT(5,tb);
    PUT(6,intbits(r1)); PUT(7,intbits(r2)); PUT(8,intbits(r)); PUT(9,sb); PUT(10,intbits(scale));
    PUT(11,intbits(h1)); PUT(12,intbits(h2)); PUT(13,intbits(h3)); PUT(14,intbits(h)); PUT(15,intbits(s));
    PUT(16,intbits(er)); PUT(17,intbits(el)); PUT(18,intbits(ph)); PUT(19,intbits(y));
#undef PUT
}
