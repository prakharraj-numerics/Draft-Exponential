/* EXP53 candidates from the integral-power identity.
   Let H_n(r)=((exp(r)-1)/r)^(1/n). Then the original integral identity gives
       exp(r) = 1 + r * H_n(r)^n.
   Runtime polynomials below are the Maclaurin polynomials of H_n, derived offline
   from that identity; no independent polynomial for exp(r) is fitted here.
   Same v128 reduction, TAB128 reconstruction, exact product residual, and scaling
   infrastructure as the established kernel are retained.
*/
#include "exp53_spine_v128_formula6_finalinfra.c"

#define IP_COMMON \
 const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0); \
 const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);

#define IP_LOAD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define IP_BIAS(L) __m512d bz##L=_mm512_fmadd_pd(x##L,inv,magic);
#define IP_K(L) __m512d k##L=_mm512_sub_pd(bz##L,magic); __m512i kn##L=_mm512_sub_epi64(_mm512_castpd_si512(bz##L),mb);
#define IP_RH(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L);
#define IP_RM(L) r##L=_mm512_fnmadd_pd(k##L,mi,r##L);
#define IP_RL(L) r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define IP_QJTAB(L) __m512i jj##L=_mm512_and_epi64(kn##L,mask),q##L=_mm512_srai_epi64(kn##L,7); __m512d th##L=_mm512_i64gather_pd(jj##L,TAB128,8);
#define IP_SCALE(L) __m512d sc##L=exp2_from_q(q##L);
#define IP_PROD(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d cr##L=_mm512_fmadd_pd(er##L,th##L,_mm512_sub_pd(_mm512_setzero_pd(),ph##L));
#define IP_STORE(L) _mm512_storeu_pd(out+i+8*(L),_mm512_mul_pd(_mm512_add_pd(ph##L,cr##L),sc##L));

/* H_2 = 1 + r/4 + 5r^2/96 + r^3/128 + 79r^4/92160 + 3r^5/40960 + ... */
#define N2_CONSTS const __m512d n2c1=_mm512_set1_pd(0x1.0000000000000p-2),n2c2=_mm512_set1_pd(0x1.aaaaaaaaaaaabp-5),n2c3=_mm512_set1_pd(0x1.0000000000000p-7),n2c4=_mm512_set1_pd(0x1.c16c16c16c16cp-11),n2c5=_mm512_set1_pd(0x1.3333333333333p-14);
#define N2_D3(L) __m512d h##L=_mm512_fmadd_pd(n2c3,r##L,n2c2); h##L=_mm512_fmadd_pd(h##L,r##L,n2c1); h##L=_mm512_fmadd_pd(h##L,r##L,one); __m512d hp##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,hp##L,one);
#define N2_D4(L) __m512d h##L=_mm512_fmadd_pd(n2c4,r##L,n2c3); h##L=_mm512_fmadd_pd(h##L,r##L,n2c2); h##L=_mm512_fmadd_pd(h##L,r##L,n2c1); h##L=_mm512_fmadd_pd(h##L,r##L,one); __m512d hp##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,hp##L,one);
#define N2_D5(L) __m512d h##L=_mm512_fmadd_pd(n2c5,r##L,n2c4); h##L=_mm512_fmadd_pd(h##L,r##L,n2c3); h##L=_mm512_fmadd_pd(h##L,r##L,n2c2); h##L=_mm512_fmadd_pd(h##L,r##L,n2c1); h##L=_mm512_fmadd_pd(h##L,r##L,one); __m512d hp##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,hp##L,one);

/* H_3 coefficients through r^4. */
#define N3_CONSTS const __m512d n3c1=_mm512_set1_pd(0x1.5555555555555p-3),n3c2=_mm512_set1_pd(0x1.c71c71c71c71cp-6),n3c3=_mm512_set1_pd(0x1.948b0fcd6e9e0p-9),n3c4=_mm512_set1_pd(0x1.af83440e53dbcp-13);
#define N3_D4(L) __m512d h##L=_mm512_fmadd_pd(n3c4,r##L,n3c3); h##L=_mm512_fmadd_pd(h##L,r##L,n3c2); h##L=_mm512_fmadd_pd(h##L,r##L,n3c1); h##L=_mm512_fmadd_pd(h##L,r##L,one); __m512d h2_##L=_mm512_mul_pd(h##L,h##L); __m512d hp##L=_mm512_mul_pd(h2_##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,hp##L,one);

/* H_4 coefficients through r^4. */
#define N4_CONSTS const __m512d n4c1=_mm512_set1_pd(0x1.0000000000000p-3),n4c2=_mm512_set1_pd(0x1.2aaaaaaaaaaabp-6),n4c3=_mm512_set1_pd(0x1.aaaaaaaaaaaabp-10),n4c4=_mm512_set1_pd(0x1.eeeeeeeeeeeefp-15);
#define N4_D4(L) __m512d h##L=_mm512_fmadd_pd(n4c4,r##L,n4c3); h##L=_mm512_fmadd_pd(h##L,r##L,n4c2); h##L=_mm512_fmadd_pd(h##L,r##L,n4c1); h##L=_mm512_fmadd_pd(h##L,r##L,one); __m512d h2_##L=_mm512_mul_pd(h##L,h##L); __m512d hp##L=_mm512_mul_pd(h2_##L,h2_##L); __m512d er##L=_mm512_fmadd_pd(r##L,hp##L,one);

/* Mid-gather: j/table known after k; issue gather after two reduction FMAs so table latency
   overlaps the final reduction FMA and the entire H_n polynomial/power chain. */
#define IP_PRE(F) F(IP_LOAD) F(IP_BIAS) F(IP_K) F(IP_RH) F(IP_RM) F(IP_QJTAB) F(IP_SCALE) F(IP_RL)
#define IP_POST(F,POLY) F(POLY) F(IP_PROD) F(IP_STORE)

#define DEF_IP(NAME,U,F,CONSTS,POLY) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n){ IP_COMMON CONSTS size_t i=0; for(;i+8*(U)<=n;i+=8*(U)){ IP_PRE(F) IP_POST(F,POLY) } if(i<n) exp53_spine_v128_formula6_compfastC(out+i,in+i,n-i); }

DEF_IP(exp53_ip_n2d3_u4,4,F4,N2_CONSTS,N2_D3)
DEF_IP(exp53_ip_n2d3_u6,6,F6,N2_CONSTS,N2_D3)
DEF_IP(exp53_ip_n2d4_u4,4,F4,N2_CONSTS,N2_D4)
DEF_IP(exp53_ip_n2d4_u5,5,F5,N2_CONSTS,N2_D4)
DEF_IP(exp53_ip_n2d4_u6,6,F6,N2_CONSTS,N2_D4)
DEF_IP(exp53_ip_n2d5_u4,4,F4,N2_CONSTS,N2_D5)
DEF_IP(exp53_ip_n3d4_u4,4,F4,N3_CONSTS,N3_D4)
DEF_IP(exp53_ip_n4d4_u4,4,F4,N4_CONSTS,N4_D4)
