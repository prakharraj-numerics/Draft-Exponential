/* EP-inspired attack on the frozen n=2 kernel.
   Numerical math MUST remain identical to exp53_n2_vmstyle_u4_0381_frozen.

   Intel VML EP structural clue on Xeon 6973P-C (run 33376776860 shard 17):
     dExp EP: gather=0, perm=9, zmm_regs=25
     dExp HA: gather=0, perm=18, zmm_regs=28
   Our older exact gatherless GLX used gather=0/perm=8 but zmm_regs=32 and spilled.

   This file keeps the exact gatherless anchor reconstruction but reduces live
   streams aggressively. The 16x8 factor product is repaired in integer ULPs
   to the exact frozen TAB128[j] bit pattern before exponent scaling.
   Same v128 reduction, same Q4 Horner, same square, same ER-low correction.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define EPX_TARGET __attribute__((target("avx512f,avx512dq,fma"),noinline,hot))

static const uint64_t EPX_PACK[8] = {
  0x1101001000000000ULL,0x1000101000000010ULL,
  0x00f0001010010000ULL,0x1101001000010000ULL,
  0x0001000000f0f000ULL,0x10f1f01011010010ULL,
  0x1100000010010020ULL,0x0f00ff100001f010ULL
};

#define EPX_CONSTS \
 const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI), \
 mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO), \
 magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0), \
 cq1=_mm512_set1_pd(N2F_Q1),cq2=_mm512_set1_pd(N2F_Q2), \
 cq3=_mm512_set1_pd(N2F_Q3),cq4=_mm512_set1_pd(N2F_Q4), \
 ga0=_mm512_setr_pd(0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0,0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0), \
 ga1=_mm512_setr_pd(0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0,0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0), \
 gbv=_mm512_setr_pd(0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0,0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0); \
 const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask127=_mm512_set1_epi64(127),mask7=_mm512_set1_epi64(7),mask15=_mm512_set1_epi64(15),mask8=_mm512_set1_epi64(8),gpack=_mm512_loadu_si512((const void*)EPX_PACK)

#define EPX_REDUCE(X,R,KN) do{ \
 __m512d _b=_mm512_fmadd_pd((X),inv,magic); \
 __m512d _k=_mm512_sub_pd(_b,magic); \
 (KN)=_mm512_sub_epi64(_mm512_castpd_si512(_b),mb); \
 (R)=_mm512_fnmadd_pd(_k,hi,(X)); \
 (R)=_mm512_fnmadd_pd(_k,mi,(R)); \
 (R)=_mm512_fnmadd_pd(_k,lo,(R)); \
}while(0)

#define EPX_SCALE_EXACT(KN,SC) do{ \
 __m512i _j=_mm512_and_epi64((KN),mask127); \
 __m512i _ia=_mm512_srli_epi64(_j,3); \
 __m512i _ib=_mm512_and_epi64(_j,mask7); \
 __m512d _aa=_mm512_permutex2var_pd(ga0,_ia,ga1); \
 __m512d _bb=_mm512_permutexvar_pd(_ib,gbv); \
 __m512d _pp=_mm512_mul_pd(_aa,_bb); \
 __m512i _grp=_mm512_srli_epi64(_j,4); \
 __m512i _pw=_mm512_permutexvar_epi64(_grp,gpack); \
 __m512i _sh=_mm512_slli_epi64(_mm512_and_epi64(_j,mask15),2); \
 __m512i _code=_mm512_and_epi64(_mm512_srlv_epi64(_pw,_sh),mask15); \
 __m512i _delta=_mm512_sub_epi64(_code,_mm512_slli_epi64(_mm512_and_epi64(_code,mask8),1)); \
 __m512i _eb=_mm512_add_epi64(_mm512_castpd_si512(_pp),_delta); \
 __m512i _q=_mm512_srai_epi64((KN),7); \
 _eb=_mm512_add_epi64(_eb,_mm512_slli_epi64(_q,52)); \
 (SC)=_mm512_castsi512_pd(_eb); \
}while(0)

#define EPX_RECON(R,SC,Y) do{ \
 __m512d _h=_mm512_fmadd_pd(cq4,(R),cq3); \
 _h=_mm512_fmadd_pd(_h,(R),cq2); \
 _h=_mm512_fmadd_pd(_h,(R),cq1); \
 _h=_mm512_fmadd_pd(_h,(R),one); \
 __m512d _s=_mm512_mul_pd(_h,_h); \
 __m512d _er=_mm512_fmadd_pd((R),_s,one); \
 __m512d _el=_mm512_fmadd_pd((R),_s,_mm512_sub_pd(one,_er)); \
 __m512d _ph=_mm512_mul_pd(_er,(SC)); \
 (Y)=_mm512_fmadd_pd(_el,(SC),_ph); \
}while(0)

/* Fully consume one vector before the next: minimum register pressure. */
EPX_TARGET void exp53_n2_epx_glx_stream1(double *restrict out,const double *restrict in,size_t n){
 EPX_CONSTS; size_t i=0;
 for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i),r,sc,y;__m512i kn;EPX_REDUCE(x,r,kn);EPX_SCALE_EXACT(kn,sc);EPX_RECON(r,sc,y);_mm512_storeu_pd(out+i,y);}
 if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}

/* Two independent vectors: intended EP-like balance between ILP and live ZMMs. */
EPX_TARGET void exp53_n2_epx_glx_pair2(double *restrict out,const double *restrict in,size_t n){
 EPX_CONSTS; size_t i=0;
 for(;i+16<=n;i+=16){
  __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),r0,r1,sc0,sc1,y0,y1;__m512i k0,k1;
  EPX_REDUCE(x0,r0,k0);EPX_REDUCE(x1,r1,k1);
  EPX_SCALE_EXACT(k0,sc0);EPX_SCALE_EXACT(k1,sc1);
  EPX_RECON(r0,sc0,y0);_mm512_storeu_pd(out+i,y0);
  EPX_RECON(r1,sc1,y1);_mm512_storeu_pd(out+i+8,y1);
 }
 if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}

/* Three streams: more ILP, still less live state than old U4 GLX. */
EPX_TARGET void exp53_n2_epx_glx_trip3(double *restrict out,const double *restrict in,size_t n){
 EPX_CONSTS; size_t i=0;
 for(;i+24<=n;i+=24){
  __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),r0,r1,r2,sc0,sc1,sc2,y;__m512i k0,k1,k2;
  EPX_REDUCE(x0,r0,k0);EPX_REDUCE(x1,r1,k1);EPX_REDUCE(x2,r2,k2);
  EPX_SCALE_EXACT(k0,sc0);EPX_SCALE_EXACT(k1,sc1);EPX_SCALE_EXACT(k2,sc2);
  EPX_RECON(r0,sc0,y);_mm512_storeu_pd(out+i,y);
  EPX_RECON(r1,sc1,y);_mm512_storeu_pd(out+i+8,y);
  EPX_RECON(r2,sc2,y);_mm512_storeu_pd(out+i+16,y);
 }
 if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}

/* Four reductions for ILP, but exact scale is created/consumed immediately. */
EPX_TARGET void exp53_n2_epx_glx_consume4(double *restrict out,const double *restrict in,size_t n){
 EPX_CONSTS; size_t i=0;
 for(;i+32<=n;i+=32){
  __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24),r0,r1,r2,r3,sc,y;__m512i k0,k1,k2,k3;
  EPX_REDUCE(x0,r0,k0);EPX_REDUCE(x1,r1,k1);EPX_REDUCE(x2,r2,k2);EPX_REDUCE(x3,r3,k3);
  EPX_SCALE_EXACT(k0,sc);EPX_RECON(r0,sc,y);_mm512_storeu_pd(out+i,y);
  EPX_SCALE_EXACT(k1,sc);EPX_RECON(r1,sc,y);_mm512_storeu_pd(out+i+8,y);
  EPX_SCALE_EXACT(k2,sc);EPX_RECON(r2,sc,y);_mm512_storeu_pd(out+i+16,y);
  EPX_SCALE_EXACT(k3,sc);EPX_RECON(r3,sc,y);_mm512_storeu_pd(out+i+24,y);
 }
 if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
