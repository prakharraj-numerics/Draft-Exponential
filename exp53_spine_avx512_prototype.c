#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*
  53-bit EXP prototype preserving the user's C/sqrt spine.

  For a tiny reduced residual r, u=r/2 and z=u^2:
      C(u) = z * (1/2! + z/4! + z^2/6! + ...)
      sinh(u) = u + u*z * (1/3! + z/5! + z^2/7! + ...)
      exp(r) = [1 + C(u) + sinh(u)]^2

  The sinh polynomial is the exact analytic positive-branch simplification of
  sqrt(C(C+2)); it is not a replacement exp polynomial.

  Range reduction is x = k*(ln2/16) + r, then exp(x)=2^(k/16)*exp(r).
  The 16-way table is static and setup-free. The hot batch path is AVX-512,
  8 doubles at a time, with fixed FMAs and one gather pair.
*/

static const double TAB_HI[16] = {
  0x1.0000000000000p+0, 0x1.0b5586cf9890fp+0,
  0x1.172b83c7d517bp+0, 0x1.2387a6e756238p+0,
  0x1.306fe0a31b715p+0, 0x1.3dea64c123422p+0,
  0x1.4bfdad5362a27p+0, 0x1.5ab07dd485429p+0,
  0x1.6a09e667f3bcdp+0, 0x1.7a11473eb0187p+0,
  0x1.8ace5422aa0dbp+0, 0x1.9c49182a3f090p+0,
  0x1.ae89f995ad3adp+0, 0x1.c199bdd85529cp+0,
  0x1.d5818dcfba487p+0, 0x1.ea4afa2a490dap+0
};
static const double TAB_LO[16] = {
  0x0.0p+0, 0x1.8a62e4adc610bp-54,
 -0x1.19041b9d78a76p-55, 0x1.9b07eb6c70573p-54,
  0x1.6f46ad23182e4p-55, 0x1.ada0911f09ebcp-55,
  0x1.d4397afec42e2p-56, 0x1.6324c054647adp-54,
 -0x1.bdd3413b26456p-54,-0x1.41577ee04992fp-55,
  0x1.6e9f156864b27p-54, 0x1.c7c46b071f2bep-56,
  0x1.7a1cd345dcc81p-54, 0x1.11065895048ddp-55,
  0x1.2ed02d75b3707p-55,-0x1.e9c23179c2893p-54
};

#define INV_LN2_16 0x1.71547652b82fep+4
#define L16_HI     0x1.62e42fe000000p-5
#define L16_MID    0x1.f473de6af278fp-34
#define L16_LO    -0x1.8cff81a12a17ep-89

static inline __m512d spine_residual8(__m512d r)
{
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d one  = _mm512_set1_pd(1.0);
    __m512d u = _mm512_mul_pd(r, half);
    __m512d z = _mm512_mul_pd(u, u);

    /* C(u)/z = 1/2! + z/4! + z^2/6! + z^3/8! + z^4/10! */
    __m512d p = _mm512_set1_pd(1.0/3628800.0);
    p = _mm512_fmadd_pd(p, z, _mm512_set1_pd(1.0/40320.0));
    p = _mm512_fmadd_pd(p, z, _mm512_set1_pd(1.0/720.0));
    p = _mm512_fmadd_pd(p, z, _mm512_set1_pd(1.0/24.0));
    p = _mm512_fmadd_pd(p, z, half);
    __m512d C = _mm512_mul_pd(z, p);

    /* sinh(u)=u+u*z*(1/3! + z/5! + ... + z^4/11!) */
    __m512d q = _mm512_set1_pd(1.0/39916800.0);
    q = _mm512_fmadd_pd(q, z, _mm512_set1_pd(1.0/362880.0));
    q = _mm512_fmadd_pd(q, z, _mm512_set1_pd(1.0/5040.0));
    q = _mm512_fmadd_pd(q, z, _mm512_set1_pd(1.0/120.0));
    q = _mm512_fmadd_pd(q, z, _mm512_set1_pd(1.0/6.0));
    __m512d uz = _mm512_mul_pd(u, z);
    __m512d s = _mm512_fmadd_pd(uz, q, u);

    __m512d eu = _mm512_add_pd(one, _mm512_add_pd(C, s));
    return _mm512_mul_pd(eu, eu);
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_avx512_batch(double *restrict out,
                              const double *restrict in,
                              size_t n)
{
    const __m512d inv = _mm512_set1_pd(INV_LN2_16);
    const __m512d lhi = _mm512_set1_pd(L16_HI);
    const __m512d lmi = _mm512_set1_pd(L16_MID);
    const __m512d llo = _mm512_set1_pd(L16_LO);
    size_t i=0;
    for(; i+8<=n; i+=8){
        __m512d x = _mm512_loadu_pd(in+i);
        __m512d kd = _mm512_mul_pd(x, inv);
        __m256i k32 = _mm512_cvt_roundpd_epi32(kd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512d k = _mm512_cvtepi32_pd(k32);

        __m512d r = _mm512_fnmadd_pd(k, lhi, x);
        r = _mm512_fnmadd_pd(k, lmi, r);
        r = _mm512_fnmadd_pd(k, llo, r);

        __m512d er = spine_residual8(r);

        __m256i j32 = _mm256_and_si256(k32, _mm256_set1_epi32(15));
        __m256i q32 = _mm256_srai_epi32(k32, 4);
        __m512d th = _mm512_i32gather_pd(j32, TAB_HI, 8);
        __m512d tl = _mm512_i32gather_pd(j32, TAB_LO, 8);
        __m512d t = _mm512_fmadd_pd(er, tl, _mm512_mul_pd(er, th));
        __m512d qd = _mm512_cvtepi32_pd(q32);
        __m512d y = _mm512_scalef_pd(t, qd);
        _mm512_storeu_pd(out+i, y);
    }
    for(; i<n; i++){
        double x=in[i];
        long k=lrint(x*INV_LN2_16);
        long q=k>>4; int j=(int)(k&15);
        double r=fma(-(double)k,L16_HI,x);
        r=fma(-(double)k,L16_MID,r);
        r=fma(-(double)k,L16_LO,r);
        double u=0.5*r,z=u*u;
        double p=fma(1.0/3628800.0,z,1.0/40320.0);
        p=fma(p,z,1.0/720.0);p=fma(p,z,1.0/24.0);p=fma(p,z,0.5);
        double C=z*p;
        double s0=fma(1.0/39916800.0,z,1.0/362880.0);
        s0=fma(s0,z,1.0/5040.0);s0=fma(s0,z,1.0/120.0);s0=fma(s0,z,1.0/6.0);
        double s=fma(u*z,s0,u);
        double eu=1.0+C+s;
        double er0=eu*eu;
        double t=fma(er0,TAB_LO[j],er0*TAB_HI[j]);
        out[i]=scalbn(t,(int)q);
    }
}
