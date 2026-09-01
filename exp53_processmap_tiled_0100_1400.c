/* EXP53 evidence-backed tiled execution-shape attack, n=100..1400.
   Math/constants/lane arithmetic are unchanged from the frozen kernel.
   The only change is execution shape: fast U3/U5/U6 vector bodies are used
   for the bulk and U3/U2/U1 tiles cover the remainder, so only n mod 8
   elements reach scalar exp().  This tests whether the earlier U3/U5/U6
   timing gains survive after eliminating block-width scalar-tail cliffs. */
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define PM_BLOCK(U) do { \
    __m512d x[U],biased[U],k[U],r[U],h[U],s[U],er[U],el[U],scale[U],ph[U],y[U]; \
    __m512i kn[U],j[U],q[U],tb[U],sb[U]; \
    for(int L=0;L<(U);L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
    for(int L=0;L<(U);L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
    for(int L=0;L<(U);L++){ \
      k[L]=_mm512_sub_pd(biased[L],magic); \
      kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); \
      j[L]=_mm512_and_epi64(kn[L],mask); \
      q[L]=_mm512_srai_epi64(kn[L],7); \
    } \
    for(int L=0;L<(U);L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
    for(int L=0;L<(U);L++){ \
      r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
      r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
      r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
    } \
    for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
    for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
    for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
    for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
    for(int L=0;L<(U);L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
    for(int L=0;L<(U);L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one); \
    for(int L=0;L<(U);L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L])); \
    for(int L=0;L<(U);L++){ \
      sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); \
      scale[L]=_mm512_castsi512_pd(sb[L]); \
    } \
    for(int L=0;L<(U);L++){ \
      ph[L]=_mm512_mul_pd(er[L],scale[L]); \
      y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); \
      _mm512_storeu_pd(out+i+8*L,y[L]); \
    } \
  } while(0)

#define DECL_CONSTS \
  const __m512d inv=_mm512_set1_pd(N2F_INV128), hi=_mm512_set1_pd(N2F_L128_HI), \
    mi=_mm512_set1_pd(N2F_L128_MI), lo=_mm512_set1_pd(N2F_L128_LO), \
    magic=_mm512_set1_pd(N2F_MAGIC), one=_mm512_set1_pd(1.0), \
    nq1=_mm512_set1_pd(N2F_Q1), nq2=_mm512_set1_pd(N2F_Q2), \
    nq3=_mm512_set1_pd(N2F_Q3), nq4=_mm512_set1_pd(N2F_Q4); \
  const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS), mask=_mm512_set1_epi64(127)

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_pm_tile5(double *restrict out,const double *restrict in,size_t n){
  DECL_CONSTS; size_t i=0;
  for(;i+40<=n;i+=40){ PM_BLOCK(5); }
  if(i+24<=n){ PM_BLOCK(3); i+=24; }
  if(i+16<=n){ PM_BLOCK(2); i+=16; }
  if(i+8<=n){ PM_BLOCK(1); i+=8; }
  for(;i<n;i++) out[i]=exp(in[i]);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_pm_tile3(double *restrict out,const double *restrict in,size_t n){
  DECL_CONSTS; size_t i=0;
  for(;i+24<=n;i+=24){ PM_BLOCK(3); }
  if(i+16<=n){ PM_BLOCK(2); i+=16; }
  if(i+8<=n){ PM_BLOCK(1); i+=8; }
  for(;i<n;i++) out[i]=exp(in[i]);
}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_pm_tile6(double *restrict out,const double *restrict in,size_t n){
  DECL_CONSTS; size_t i=0;
  for(;i+48<=n;i+=48){ PM_BLOCK(6); }
  if(i+40<=n){ PM_BLOCK(5); i+=40; }
  if(i+24<=n){ PM_BLOCK(3); i+=24; }
  if(i+16<=n){ PM_BLOCK(2); i+=16; }
  if(i+8<=n){ PM_BLOCK(1); i+=8; }
  for(;i<n;i++) out[i]=exp(in[i]);
}
