#pragma once

/* EXP53 EXPERIMENT ONLY — no compensation, no helper, no tables.

   Family under test:
       exp(x) ~= P_{m,d}(x)^m
       P_{m,d}(x) = sum_{j=0}^d x^j/(m^j j!)

   Candidates:
       (m,d) = (32,8), (32,9), (16,9), (16,10), (8,11)

   m is a power of two, so the final power is only repeated squaring:
       m=32 -> 5 squarings
       m=16 -> 4 squarings
       m=8  -> 3 squarings

   Production and frozen EXP53 code are not touched.
*/

#include <immintrin.h>
#include <cstddef>

namespace exp53_power_family_raw {

static inline __m512d sq3(__m512d y) {
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    return y;
}
static inline __m512d sq4(__m512d y) {
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    return y;
}
static inline __m512d sq5(__m512d y) {
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    y = _mm512_mul_pd(y,y);
    return y;
}

static inline __m512d p32d8(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.a01a01a01a01ap-56);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-32));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-25));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-18));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-11));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-5));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}
static inline __m512d p32d9(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.71de3a556c734p-64);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-56));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-32));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-25));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-18));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-11));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-5));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}
static inline __m512d p16d9(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.71de3a556c734p-55);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-41));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-34));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-27));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-21));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-15));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-9));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-4));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}
static inline __m512d p16d10(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.27e4fb7789f5dp-62);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.71de3a556c734p-55));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-48));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-41));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-34));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-27));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-21));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-15));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-9));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-4));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}
static inline __m512d p8d11(__m512d x) {
    __m512d p = _mm512_set1_pd(0x1.ae64567f544e4p-59);
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.27e4fb7789f5dp-52));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.71de3a556c734p-46));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-40));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.a01a01a01a01ap-34));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.6c16c16c16c17p-28));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.1111111111111p-22));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-17));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.5555555555555p-12));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-7));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p-3));
    p=_mm512_fmadd_pd(p,x,_mm512_set1_pd(0x1.0000000000000p+0));
    return p;
}

enum class Variant { M32D8, M32D9, M16D9, M16D10, M8D11 };

template<Variant V>
static inline __m512d eval(__m512d x) {
    if constexpr (V == Variant::M32D8) return sq5(p32d8(x));
    if constexpr (V == Variant::M32D9) return sq5(p32d9(x));
    if constexpr (V == Variant::M16D9) return sq4(p16d9(x));
    if constexpr (V == Variant::M16D10) return sq4(p16d10(x));
    return sq3(p8d11(x));
}

template<Variant V>
__attribute__((target("avx512f,avx512dq,fma"), noinline, hot))
static void kernel(double* __restrict out, const double* __restrict in, size_t n) {
    size_t i=0;
    for (; i+32<=n; i+=32) {
        __m512d x0=_mm512_loadu_pd(in+i+0),  x1=_mm512_loadu_pd(in+i+8);
        __m512d x2=_mm512_loadu_pd(in+i+16), x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=eval<V>(x0), y1=eval<V>(x1), y2=eval<V>(x2), y3=eval<V>(x3);
        _mm512_storeu_pd(out+i+0,y0); _mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2); _mm512_storeu_pd(out+i+24,y3);
    }
    for (; i+8<=n; i+=8) {
        __m512d x=_mm512_loadu_pd(in+i);
        _mm512_storeu_pd(out+i,eval<V>(x));
    }
    if (i<n) {
        unsigned rem=(unsigned)(n-i);
        __mmask8 m=(__mmask8)((1u<<rem)-1u);
        __m512d x=_mm512_maskz_loadu_pd(m,in+i);
        _mm512_mask_storeu_pd(out+i,m,eval<V>(x));
    }
}

static inline void m32d8(double*o,const double*x,size_t n){kernel<Variant::M32D8>(o,x,n);}
static inline void m32d9(double*o,const double*x,size_t n){kernel<Variant::M32D9>(o,x,n);}
static inline void m16d9(double*o,const double*x,size_t n){kernel<Variant::M16D9>(o,x,n);}
static inline void m16d10(double*o,const double*x,size_t n){kernel<Variant::M16D10>(o,x,n);}
static inline void m8d11(double*o,const double*x,size_t n){kernel<Variant::M8D11>(o,x,n);}

} // namespace exp53_power_family_raw
