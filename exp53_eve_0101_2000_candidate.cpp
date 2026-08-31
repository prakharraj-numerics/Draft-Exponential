// Experimental EVE SIMD backend for frozen EXP53 machinery. No eve::exp.
#define restrict __restrict__
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
#undef restrict
#include <eve/module/core.hpp>
#include <immintrin.h>
using D = eve::wide<double, eve::fixed<8>>;

__attribute__((noinline))
void exp53_eve_0101_2000_candidate(double* out, const double* in, size_t n)
{
  const D inv(N2F_INV128), hi(N2F_L128_HI), mi(N2F_L128_MI), lo(N2F_L128_LO),
          magic(N2F_MAGIC), one(1.0), nq1(N2F_Q1), nq2(N2F_Q2), nq3(N2F_Q3), nq4(N2F_Q4);
  const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127);
  const long long* tab=(const long long*)N2_FROZEN_TAB128;
  size_t i=0;
  for(;i+32<=n;i+=32)
  {
    D x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4],ph[4],y[4];
    __m512i kn[4],j[4],q[4],tb[4],sb[4];
    for(int L=0;L<4;++L) x[L]=eve::load(in+i+8*L,eve::as<D>{});
    for(int L=0;L<4;++L) biased[L]=eve::fma(x[L],inv,magic);
    for(int L=0;L<4;++L)
    {
      k[L]=biased[L]-magic;
      alignas(64) double bv[8]; eve::store(biased[L],bv);
      __m512d b=_mm512_load_pd(bv);
      kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(b),mb);
      j[L]=_mm512_and_si512(kn[L],mask);
      q[L]=_mm512_srai_epi64(kn[L],7);
      tb[L]=_mm512_i64gather_epi64(j[L],tab,8);
    }
    for(int L=0;L<4;++L){r[L]=eve::fnma(k[L],hi,x[L]);r[L]=eve::fnma(k[L],mi,r[L]);r[L]=eve::fnma(k[L],lo,r[L]);}
    for(int L=0;L<4;++L) h[L]=eve::fma(nq4,r[L],nq3);
    for(int L=0;L<4;++L) h[L]=eve::fma(h[L],r[L],nq2);
    for(int L=0;L<4;++L) h[L]=eve::fma(h[L],r[L],nq1);
    for(int L=0;L<4;++L) h[L]=eve::fma(h[L],r[L],one);
    for(int L=0;L<4;++L) s[L]=h[L]*h[L];
    for(int L=0;L<4;++L) er[L]=eve::fma(r[L],s[L],one);
    for(int L=0;L<4;++L) el[L]=eve::fma(r[L],s[L],one-er[L]);
    for(int L=0;L<4;++L)
    {
      sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
      alignas(64) double sv[8]; _mm512_store_pd(sv,_mm512_castsi512_pd(sb[L]));
      scale[L]=eve::load(sv,eve::as<D>{});
    }
    for(int L=0;L<4;++L){ph[L]=er[L]*scale[L];y[L]=eve::fma(el[L],scale[L],ph[L]);eve::store(y[L],out+i+8*L);}
  }
  if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
