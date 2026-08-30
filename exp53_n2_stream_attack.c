/* Streaming-only microarchitecture attack. Math is frozen: Q4 -> Q4^2 -> ER-low repair. */
#include "exp53_spine_n2_integralpower.c"
#define F3(M) M(0) M(1) M(2)
#define F4(M) M(0) M(1) M(2) M(3)
#define LD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define BI(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define KI(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb); __m512i j##L=_mm512_and_epi64(ki##L,mask),qq##L=_mm512_srai_epi64(ki##L,7);
#define RR(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define TG(L) __m512d th##L=_mm512_i64gather_pd(j##L,TAB128,8);
#define SC(L) __m512d sc##L=exp2_from_q(qq##L);
#define H1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define H2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define H3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define H4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define RC(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one); __m512d el##L=_mm512_fmadd_pd(r##L,s##L,_mm512_sub_pd(one,er##L));
#define MU(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_fmadd_pd(el##L,th##L,ph##L); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define ST(L) _mm512_storeu_pd(out+i+8*(L),y##L);
#define C const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
static __attribute__((target("avx512f,avx512dq,fma"),always_inline)) inline void tail(double*out,const double*in,size_t n){while(n){unsigned c=n>=8?8:n;__mmask8 m=(1u<<c)-1;double z[8]={0};_mm512_mask_storeu_pd(z,m,_mm512_maskz_loadu_pd(m,in));for(unsigned t=0;t<c;t++){double x=z[t];double kd=nearbyint(x*INV128);long long kk=(long long)kd,j=kk&127,q=kk>>7;double r=fma(-kd,L128_HI,x);r=fma(-kd,L128_MI,r);r=fma(-kd,L128_LO,r);double h=fma(N2_Q4,r,N2_Q3);h=fma(h,r,N2_Q2);h=fma(h,r,N2_Q1);h=fma(h,r,1.0);double s=h*h,er=fma(r,s,1.0),el=fma(r,s,1.0-er);z[t]=ldexp(fma(el,TAB128[j],er*TAB128[j]),q);} _mm512_mask_storeu_pd(out,m,_mm512_loadu_pd(z));in+=c;out+=c;n-=c;}}
#define DEF(N,U,F,S) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void N(double*restrict out,const double*restrict in,size_t n){C size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){S}if(i<n)tail(out+i,in+i,n-i);}
#define MID(F) F(LD) F(BI) F(KI) F(RR) F(TG) F(SC) F(H1) F(H2) F(H3) F(H4) F(RC) F(MU) F(ST)
#define GEARLY_SLATE(F) F(LD) F(BI) F(KI) F(TG) F(RR) F(H1) F(H2) F(H3) F(H4) F(RC) F(SC) F(MU) F(ST)
#define SEARLY_GMID(F) F(LD) F(BI) F(KI) F(SC) F(RR) F(H1) F(TG) F(H2) F(H3) F(H4) F(RC) F(MU) F(ST)
#define GH2(F) F(LD) F(BI) F(KI) F(RR) F(H1) F(H2) F(TG) F(H3) F(SC) F(H4) F(RC) F(MU) F(ST)
#define GH3(F) F(LD) F(BI) F(KI) F(RR) F(H1) F(H2) F(H3) F(TG) F(SC) F(H4) F(RC) F(MU) F(ST)
DEF(exp53_stream_mid_u3,3,F3,MID(F3))
DEF(exp53_stream_mid_u4,4,F4,MID(F4))
DEF(exp53_stream_gearly_slate_u4,4,F4,GEARLY_SLATE(F4))
DEF(exp53_stream_searly_gmid_u4,4,F4,SEARLY_GMID(F4))
DEF(exp53_stream_gh2_u4,4,F4,GH2(F4))
DEF(exp53_stream_gh3_u4,4,F4,GH3(F4))
