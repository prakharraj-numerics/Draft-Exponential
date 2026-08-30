/* Formula6 tuning sweep. Same outer architecture and same 4-FMA + 2-MUL residual DAG cost.
   Variants differ only in coefficient choice and whether C or S is fused into 1 first. */
#include "exp53_spine_v128_u4_frozen.c"

#define A0M 0x1.ffffffffffff5p-4
#define A1M 0x1.555556b330444p-9
#define B0M 0x1.fffffffffffe0p-2
#define B1M 0x1.555557621dbbbp-6
#define A0T 0x1.0000000000000p-3
#define A1T 0x1.5555555555555p-9
#define B0T 0x1.0000000000000p-1
#define B1T 0x1.5555555555555p-6

static inline __m512d f6_res(__m512d r,double a0,double a1,double b0,double b1,int cfirst){
    const __m512d one=_mm512_set1_pd(1.0);
    __m512d t=_mm512_mul_pd(r,r);
    __m512d pc=_mm512_fmadd_pd(_mm512_set1_pd(a1),t,_mm512_set1_pd(a0));
    __m512d ps=_mm512_fmadd_pd(_mm512_set1_pd(b1),t,_mm512_set1_pd(b0));
    __m512d h;
    if(cfirst){h=_mm512_fmadd_pd(t,pc,one);h=_mm512_fmadd_pd(r,ps,h);}else{h=_mm512_fmadd_pd(r,ps,one);h=_mm512_fmadd_pd(t,pc,h);}
    return _mm512_mul_pd(h,h);
}

#define DEF(NAME,A0,A1,B0,B1,CF) \
static inline __m512d NAME##_block(__m512d x,const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask){ \
 __m512d biased=_mm512_fmadd_pd(x,inv,magic); __m512d k=_mm512_sub_pd(biased,magic); __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb); \
 __m512d r=_mm512_fnmadd_pd(k,hi,x); r=_mm512_fnmadd_pd(k,mi,r); r=_mm512_fnmadd_pd(k,lo,r); \
 __m512d er=f6_res(r,A0,A1,B0,B1,CF); __m512i j=_mm512_and_epi64(kn,mask); __m512i q=_mm512_srai_epi64(kn,7); __m512d tab=_mm512_i64gather_pd(j,TAB128,8); \
 return _mm512_mul_pd(_mm512_mul_pd(er,tab),exp2_from_q(q)); } \
__attribute__((target("avx512f,avx512dq,fma"))) void NAME(double *restrict out,const double *restrict in,size_t n){ \
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC); \
 const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127); size_t i=0; \
 for(;i+32<=n;i+=32){__m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24); \
 __m512d y0=NAME##_block(x0,inv,hi,mi,lo,magic,mb,mask),y1=NAME##_block(x1,inv,hi,mi,lo,magic,mb,mask),y2=NAME##_block(x2,inv,hi,mi,lo,magic,mb,mask),y3=NAME##_block(x3,inv,hi,mi,lo,magic,mb,mask); \
 _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);} \
 for(;i<n;){if(n-i>=8){_mm512_storeu_pd(out+i,NAME##_block(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));i+=8;}else{for(;i<n;++i)out[i]=exp(in[i]);}} }

/* Base formula6 coefficients, only rounding order flipped. */
DEF(exp53_spine_v128_u4_f6_cfirst,A0M,A1M,B0M,B1M,1)
/* C-first with exact Taylor intercepts but minimax slopes. */
DEF(exp53_spine_v128_u4_f6_cfirst_i,A0T,A1M,B0T,B1M,1)
/* C-first with plain degree-1 Taylor coefficients. */
DEF(exp53_spine_v128_u4_f6_cfirst_t,A0T,A1T,B0T,B1T,1)
/* S-first plain Taylor control. */
DEF(exp53_spine_v128_u4_f6_sfirst_t,A0T,A1T,B0T,B1T,0)
