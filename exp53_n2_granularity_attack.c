/* Fresh range-granularity attack for faithful n=2 spine.
   Only reduction/table granularity changes; reconstruction remains er=1+r*Q(r)^2. */
#include <immintrin.h>
#include <mpfr.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "exp53_spine_n2_integralpower.c"
#define Q5 0x1.3333333333333p-14 /* 3/40960 */
static double T32[32] __attribute__((aligned(64))),T64[64] __attribute__((aligned(64))),T256[256] __attribute__((aligned(64)));
static double I32,L32H,L32M,L32L,I64,L64H,L64M,L64L,I256,L256H,L256M,L256L;
static void split3(mpfr_t x,double*h,double*m,double*l){mpfr_t t;mpfr_init2(t,256);*h=mpfr_get_d(x,MPFR_RNDN);mpfr_sub_d(t,x,*h,MPFR_RNDN);*m=mpfr_get_d(t,MPFR_RNDN);mpfr_sub_d(t,t,*m,MPFR_RNDN);*l=mpfr_get_d(t,MPFR_RNDN);mpfr_clear(t);}
void exp53_n2_gran_init(void){mpfr_t ln2,a,e;mpfr_inits2(256,ln2,a,e,(mpfr_ptr)0);mpfr_const_log2(ln2,MPFR_RNDN);int Ms[3]={32,64,256};double*Ts[3]={T32,T64,T256};double*Is[3]={&I32,&I64,&I256},*Hs[3]={&L32H,&L64H,&L256H},*MM[3]={&L32M,&L64M,&L256M},*Ls[3]={&L32L,&L64L,&L256L};for(int z=0;z<3;z++){int M=Ms[z];mpfr_ui_div(a,(unsigned)M,ln2,MPFR_RNDN);*Is[z]=mpfr_get_d(a,MPFR_RNDN);mpfr_div_ui(a,ln2,(unsigned)M,MPFR_RNDN);split3(a,Hs[z],MM[z],Ls[z]);for(int j=0;j<M;j++){mpfr_set_si(a,j,MPFR_RNDN);mpfr_div_ui(a,a,(unsigned)M,MPFR_RNDN);mpfr_mul(a,a,ln2,MPFR_RNDN);mpfr_exp(e,a,MPFR_RNDN);Ts[z][j]=mpfr_get_d(e,MPFR_RNDN);}}mpfr_clears(ln2,a,e,(mpfr_ptr)0);}
#define F2(M) M(0) M(1)
#define F3(M) M(0) M(1) M(2)
#define F4(M) M(0) M(1) M(2) M(3)
#define LD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define BIAS(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define K(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb);
#define TAB(L) __m512i j##L=_mm512_and_epi64(ki##L,mask),qq##L=_mm512_srai_epi64(ki##L,SH); __m512d th##L=_mm512_i64gather_pd(j##L,TABP,8),sc##L=exp2_from_q(qq##L);
#define RED(L) __m512d r##L=_mm512_fnmadd_pd(k##L,lh,x##L); r##L=_mm512_fnmadd_pd(k##L,lm,r##L); r##L=_mm512_fnmadd_pd(k##L,ll,r##L);
#define H3A(L) __m512d h##L=_mm512_fmadd_pd(q3,r##L,q2);
#define H3B(L) h##L=_mm512_fmadd_pd(h##L,r##L,q1);
#define H3C(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define H4A(L) __m512d h##L=_mm512_fmadd_pd(q4,r##L,q3);
#define H4B(L) h##L=_mm512_fmadd_pd(h##L,r##L,q2);
#define H4C(L) h##L=_mm512_fmadd_pd(h##L,r##L,q1);
#define H4D(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define H5A(L) __m512d h##L=_mm512_fmadd_pd(q5,r##L,q4);
#define H5B(L) h##L=_mm512_fmadd_pd(h##L,r##L,q3);
#define H5C(L) h##L=_mm512_fmadd_pd(h##L,r##L,q2);
#define H5D(L) h##L=_mm512_fmadd_pd(h##L,r##L,q1);
#define H5E(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define REC(L) __m512d s##L=_mm512_mul_pd(h##L,h##L),er##L=_mm512_fmadd_pd(r##L,s##L,one); __m512d y##L=_mm512_mul_pd(_mm512_mul_pd(er##L,th##L),sc##L);
#define ST(L) _mm512_storeu_pd(out+i+8*(L),y##L);
#define PRE(F) F(LD) F(BIAS) F(K) F(TAB) F(RED)
#define D3(F) PRE(F) F(H3A) F(H3B) F(H3C) F(REC) F(ST)
#define D4(F) PRE(F) F(H4A) F(H4B) F(H4C) F(H4D) F(REC) F(ST)
#define D5(F) PRE(F) F(H5A) F(H5B) F(H5C) F(H5D) F(H5E) F(REC) F(ST)
#define DEF(NAME,U,F,MP,SHFT,DEG) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){const double *TABP=T##MP; const int SH=SHFT;const __m512d inv=_mm512_set1_pd(I##MP),lh=_mm512_set1_pd(L##MP##H),lm=_mm512_set1_pd(L##MP##M),ll=_mm512_set1_pd(L##MP##L),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),q1=_mm512_set1_pd(N2_Q1),q2=_mm512_set1_pd(N2_Q2),q3=_mm512_set1_pd(N2_Q3),q4=_mm512_set1_pd(N2_Q4),q5=_mm512_set1_pd(Q5);const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64((MP)-1);size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){DEG(F)}if(i<n)exp53_n2_fast_u2(out+i,in+i,n-i);}
DEF(exp53_n2_v32_d5_u2,2,F2,32,5,D5)
DEF(exp53_n2_v32_d5_u4,4,F4,32,5,D5)
DEF(exp53_n2_v64_d4_u2,2,F2,64,6,D4)
DEF(exp53_n2_v64_d4_u3,3,F3,64,6,D4)
DEF(exp53_n2_v64_d4_u4,4,F4,64,6,D4)
DEF(exp53_n2_v256_d3_u2,2,F2,256,8,D3)
DEF(exp53_n2_v256_d3_u3,3,F3,256,8,D3)
DEF(exp53_n2_v256_d3_u4,4,F4,256,8,D3)
