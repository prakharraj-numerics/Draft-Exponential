/* Anchor-density attack on the frozen n=2 baseline.
   Experimental only: v256/v512 use the SAME faithful Q4 spine but REMOVE ER-low.
   Tables are initialized once before benchmarking; table construction is not timed.
   If either path wins and stays <=1 ULP, it can later be converted to static hex tables. */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#define A_MAGIC 0x1.8000000000000p+52
#define A_MAGIC_BITS 0x4338000000000000ULL
#define A_Q1 0x1.0000000000000p-2
#define A_Q2 0x1.aaaaaaaaaaaabp-5
#define A_Q3 0x1.0000000000000p-7
#define A_Q4 0x1.c16c16c16c16cp-11

static double ATAB256[256] __attribute__((aligned(64)));
static double ATAB512[512] __attribute__((aligned(64)));
void exp53_n2_anchor_init(void){
    for(int j=0;j<256;j++) ATAB256[j]=exp2((double)j/256.0);
    for(int j=0;j<512;j++) ATAB512[j]=exp2((double)j/512.0);
}

#define DEF_ANCHOR_FAST(NAME,TAB,N,SH,INV,HI,MI,LO) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n){ \
 const __m512d inv=_mm512_set1_pd(INV),hi=_mm512_set1_pd(HI),mi=_mm512_set1_pd(MI),lo=_mm512_set1_pd(LO),magic=_mm512_set1_pd(A_MAGIC),one=_mm512_set1_pd(1.0),q1=_mm512_set1_pd(A_Q1),q2=_mm512_set1_pd(A_Q2),q3=_mm512_set1_pd(A_Q3),q4=_mm512_set1_pd(A_Q4); \
 const __m512i mb=_mm512_set1_epi64((long long)A_MAGIC_BITS),mask=_mm512_set1_epi64((N)-1); \
 size_t i=0; for(;i+32<=n;i+=32){ \
  __m512d x[4],b[4],k[4],r[4],h[4],s[4],er[4],scale[4],y[4]; __m512i kn[4],j[4],qq[4],tb[4],sb[4]; \
  for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
  for(int L=0;L<4;L++) b[L]=_mm512_fmadd_pd(x[L],inv,magic); \
  for(int L=0;L<4;L++){k[L]=_mm512_sub_pd(b[L],magic);kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(b[L]),mb);j[L]=_mm512_and_epi64(kn[L],mask);qq[L]=_mm512_srai_epi64(kn[L],SH);} \
  for(int L=0;L<4;L++){r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);} \
  for(int L=0;L<4;L++){tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)(TAB),8);sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(qq[L],52));scale[L]=_mm512_castsi512_pd(sb[L]);} \
  for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(q4,r[L],q3); \
  for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],q2); \
  for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],q1); \
  for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
  for(int L=0;L<4;L++){s[L]=_mm512_mul_pd(h[L],h[L]);er[L]=_mm512_fmadd_pd(r[L],s[L],one);y[L]=_mm512_mul_pd(er[L],scale[L]);_mm512_storeu_pd(out+i+8*L,y[L]);} \
 } for(;i<n;i++) out[i]=exp(in[i]); }

DEF_ANCHOR_FAST(exp53_n2_fast_v256_u4,ATAB256,256,8,0x1.71547652b82fep+8,0x1.62e42fefa39efp-9,0x1.abc9e3b39803fp-64,0x1.7b57a079a1934p-119)
DEF_ANCHOR_FAST(exp53_n2_fast_v512_u4,ATAB512,512,9,0x1.71547652b82fep+9,0x1.62e42fefa39efp-10,0x1.abc9e3b39803fp-65,0x1.7b57a079a1934p-120)
