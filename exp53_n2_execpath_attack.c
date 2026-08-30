#include "exp53_n2_regpressure_attack.c"

/* Targeted tests of the remaining execution bottlenecks.
   1) i32gather: narrow gather indices, removing 64-bit j/mask work from gather address path.
   2) perm16x8: replace the 128-entry memory gather by 16x8 LUT decomposition held in ZMM regs.
   3) fmsubcorr: same gather path, algebraically identical ER-low correction with fmsub form.
*/

#define POLY_RECON_FMSUB(R,SCALE,Y) do{ \
 __m512d h=_mm512_fmadd_pd(q4,(R),q3); \
 h=_mm512_fmadd_pd(h,(R),q2); h=_mm512_fmadd_pd(h,(R),q1); h=_mm512_fmadd_pd(h,(R),one); \
 __m512d ss=_mm512_mul_pd(h,h); __m512d er=_mm512_fmadd_pd((R),ss,one); \
 __m512d d=_mm512_sub_pd(er,one); __m512d el=_mm512_fmsub_pd((R),ss,d); \
 __m512d ph=_mm512_mul_pd(er,(SCALE)); (Y)=_mm512_fmadd_pd(el,(SCALE),ph); \
}while(0)

#define SCALE_I32(KN,SCALE) do{ \
 __m256i j32=_mm512_cvtepi64_epi32((KN)); \
 j32=_mm256_and_si256(j32,_mm256_set1_epi32(127)); \
 __m512i qq=_mm512_srai_epi64((KN),7); \
 __m512i tb=_mm512_i32gather_epi64(j32,(const long long*)TAB128,8); \
 (SCALE)=_mm512_castsi512_pd(_mm512_add_epi64(tb,_mm512_slli_epi64(qq,52))); \
}while(0)

#define PERM_CONSTS \
 const __m512d a0=_mm512_setr_pd( \
  0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0, \
  0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0), \
 a1=_mm512_setr_pd( \
  0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0, \
  0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0), \
 bv=_mm512_setr_pd( \
  0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0, \
  0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0)

#define SCALE_PERM16X8(KN,SCALE) do{ \
 __m512i jj=_mm512_and_epi64((KN),mask); \
 __m512i ia=_mm512_srli_epi64(jj,3); \
 __m512i ib=_mm512_and_epi64(jj,_mm512_set1_epi64(7)); \
 __m512d aa=_mm512_permutex2var_pd(a0,ia,a1); \
 __m512d bb=_mm512_permutexvar_pd(ib,bv); \
 __m512i qq=_mm512_srai_epi64((KN),7); \
 __m512i abits=_mm512_add_epi64(_mm512_castpd_si512(aa),_mm512_slli_epi64(qq,52)); \
 (SCALE)=_mm512_mul_pd(_mm512_castsi512_pd(abits),bb); \
}while(0)

#define BODY4(SCALEMAC,RECONMAC) do{ \
 size_t i=0; for(;i+32<=n;i+=32){ \
  __m512d r0,r1,r2,r3,sc0,sc1,sc2,sc3,y0,y1,y2,y3; __m512i k0,k1,k2,k3; \
  __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24); \
  REDUCE_ONE(x0,r0,k0); REDUCE_ONE(x1,r1,k1); REDUCE_ONE(x2,r2,k2); REDUCE_ONE(x3,r3,k3); \
  SCALEMAC(k0,sc0); SCALEMAC(k1,sc1); SCALEMAC(k2,sc2); SCALEMAC(k3,sc3); \
  RECONMAC(r0,sc0,y0); RECONMAC(r1,sc1,y1); RECONMAC(r2,sc2,y2); RECONMAC(r3,sc3,y3); \
  _mm512_storeu_pd(out+i,y0); _mm512_storeu_pd(out+i+8,y1); _mm512_storeu_pd(out+i+16,y2); _mm512_storeu_pd(out+i+24,y3); \
 } for(;i<n;i++) out[i]=exp(in[i]); \
}while(0)

__attribute__((target("avx512f,avx512dq,fma,avx2"),noinline))
void exp53_n2_ep_i32gather4(double *restrict out,const double *restrict in,size_t n){CONSTS; BODY4(SCALE_I32,POLY_RECON);}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ep_fmsubcorr4(double *restrict out,const double *restrict in,size_t n){CONSTS; BODY4(SCALE_ONE,POLY_RECON_FMSUB);}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ep_perm16x8_4(double *restrict out,const double *restrict in,size_t n){CONSTS; PERM_CONSTS; BODY4(SCALE_PERM16X8,POLY_RECON);}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_ep_perm16x8_fmsub4(double *restrict out,const double *restrict in,size_t n){CONSTS; PERM_CONSTS; BODY4(SCALE_PERM16X8,POLY_RECON_FMSUB);}
