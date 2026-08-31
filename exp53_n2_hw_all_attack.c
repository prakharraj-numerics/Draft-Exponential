#include "exp53_n2_regpressure_attack.c"

/* Hardware-focused sweep after register-pressure and exact-gatherless rejection.
   Variants intentionally preserve the frozen mathematical spine and exact TAB128 anchors. */

static const double __attribute__((aligned(64))) TAB128_ALIGNED[128] = {
0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0,0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0,0x1.0b5586cf9890fp+0,0x1.0cc922b7247f7p+0,0x1.0e3ec32d3d1a2p+0,0x1.0fb66affed31bp+0,0x1.11301d0125b51p+0,0x1.12abdc06c31ccp+0,0x1.1429aaea92de0p+0,0x1.15a98c8a58e51p+0,0x1.172b83c7d517bp+0,0x1.18af9388c8deap+0,0x1.1a35beb6fcb75p+0,0x1.1bbe084045cd4p+0,0x1.1d4873168b9aap+0,0x1.1ed5022fcd91dp+0,0x1.2063b88628cd6p+0,0x1.21f49917ddc96p+0,0x1.2387a6e756238p+0,0x1.251ce4fb2a63fp+0,0x1.26b4565e27cddp+0,0x1.284dfe1f56381p+0,0x1.29e9df51fdee1p+0,0x1.2b87fd0dad990p+0,0x1.2d285a6e4030bp+0,0x1.2ecafa93e2f56p+0,0x1.306fe0a31b715p+0,0x1.32170fc4cd831p+0,0x1.33c08b26416ffp+0,0x1.356c55f929ff1p+0,0x1.371a7373aa9cbp+0,0x1.38cae6d05d866p+0,0x1.3a7db34e59ff7p+0,0x1.3c32dc313a8e5p+0,0x1.3dea64c123422p+0,0x1.3fa4504ac801cp+0,0x1.4160a21f72e2ap+0,0x1.431f5d950a897p+0,0x1.44e086061892dp+0,0x1.46a41ed1d0057p+0,0x1.486a2b5c13cd0p+0,0x1.4a32af0d7d3dep+0,0x1.4bfdad5362a27p+0,0x1.4dcb299fddd0dp+0,0x1.4f9b2769d2ca7p+0,0x1.516daa2cf6642p+0,0x1.5342b569d4f82p+0,0x1.551a4ca5d920fp+0,0x1.56f4736b527dap+0,0x1.58d12d497c7fdp+0,0x1.5ab07dd485429p+0,0x1.5c9268a5946b7p+0,0x1.5e76f15ad2148p+0,0x1.605e1b976dc09p+0,0x1.6247eb03a5585p+0,0x1.6434634ccc320p+0,0x1.6623882552225p+0,0x1.68155d44ca973p+0,0x1.6a09e667f3bcdp+0,0x1.6c012750bdabfp+0,0x1.6dfb23c651a2fp+0,0x1.6ff7df9519484p+0,0x1.71f75e8ec5f74p+0,0x1.73f9a48a58174p+0,0x1.75feb564267c9p+0,0x1.780694fde5d3fp+0,0x1.7a11473eb0187p+0,0x1.7c1ed0130c132p+0,0x1.7e2f336cf4e62p+0,0x1.80427543e1a12p+0,0x1.82589994cce13p+0,0x1.8471a4623c7adp+0,0x1.868d99b4492edp+0,0x1.88ac7d98a6699p+0,0x1.8ace5422aa0dbp+0,0x1.8cf3216b5448cp+0,0x1.8f1ae99157736p+0,0x1.9145b0b91ffc6p+0,0x1.93737b0cdc5e5p+0,0x1.95a44cbc8520fp+0,0x1.97d829fde4e50p+0,0x1.9a0f170ca07bap+0,0x1.9c49182a3f090p+0,0x1.9e86319e32323p+0,0x1.a0c667b5de565p+0,0x1.a309bec4a2d33p+0,0x1.a5503b23e255dp+0,0x1.a799e1330b358p+0,0x1.a9e6b5579fdbfp+0,0x1.ac36bbfd3f37ap+0,0x1.ae89f995ad3adp+0,0x1.b0e07298db666p+0,0x1.b33a2b84f15fbp+0,0x1.b59728de5593ap+0,0x1.b7f76f2fb5e47p+0,0x1.ba5b030a1064ap+0,0x1.bcc1e904bc1d2p+0,0x1.bf2c25bd71e09p+0,0x1.c199bdd85529cp+0,0x1.c40ab5fffd07ap+0,0x1.c67f12e57d14bp+0,0x1.c8f6d9406e7b5p+0,0x1.cb720dcef9069p+0,0x1.cdf0b555dc3fap+0,0x1.d072d4a07897cp+0,0x1.d2f87080d89f2p+0,0x1.d5818dcfba487p+0,0x1.d80e316c98398p+0,0x1.da9e603db3285p+0,0x1.dd321f301b460p+0,0x1.dfc97337b9b5fp+0,0x1.e264614f5a129p+0,0x1.e502ee78b3ff6p+0,0x1.e7a51fbc74c83p+0,0x1.ea4afa2a490dap+0,0x1.ecf482d8e67f1p+0,0x1.efa1bee615a27p+0,0x1.f252b376bba97p+0,0x1.f50765b6e4540p+0,0x1.f7bfdad9cbe14p+0,0x1.fa7c1819e90d8p+0,0x1.fd3c22b8f71f1p+0};

#define SCALE_ALIGNED(KN,SCALE) do{ __m512i j=_mm512_and_epi64((KN),mask); __m512i qq=_mm512_srai_epi64((KN),7); __m512i tb=_mm512_i64gather_epi64(j,(const long long*)TAB128_ALIGNED,8); (SCALE)=_mm512_castsi512_pd(_mm512_add_epi64(tb,_mm512_slli_epi64(qq,52))); }while(0)

/* Split one 8-lane gather into two independent 4-lane AVX2 gathers. */
#define SCALE_SPLIT256(KN,SCALE) do{ \
 __m512i jj=_mm512_and_epi64((KN),mask), qq=_mm512_srai_epi64((KN),7); \
 __m256i jl=_mm512_castsi512_si256(jj), jh=_mm512_extracti64x4_epi64(jj,1); \
 __m256i tl=_mm256_i64gather_epi64((const long long*)TAB128_ALIGNED,jl,8); \
 __m256i th=_mm256_i64gather_epi64((const long long*)TAB128_ALIGNED,jh,8); \
 __m512i tb=_mm512_castsi256_si512(tl); tb=_mm512_inserti64x4(tb,th,1); \
 (SCALE)=_mm512_castsi512_pd(_mm512_add_epi64(tb,_mm512_slli_epi64(qq,52))); \
}while(0)

#define RECON_PART(R,S,ER,EL) do{ __m512d h=_mm512_fmadd_pd(q4,(R),q3); h=_mm512_fmadd_pd(h,(R),q2); h=_mm512_fmadd_pd(h,(R),q1); h=_mm512_fmadd_pd(h,(R),one); __m512d ss=_mm512_mul_pd(h,h); (ER)=_mm512_fmadd_pd((R),ss,one); (EL)=_mm512_fmadd_pd((R),ss,_mm512_sub_pd(one,(ER))); }while(0)
#define FINISH_PART(SC,ER,EL,Y) do{ __m512d ph=_mm512_mul_pd((ER),(SC)); (Y)=_mm512_fmadd_pd((EL),(SC),ph); }while(0)

/* Cross-vector software pipeline: issue gather, do independent polynomial work, consume gather later. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_hw_swpipe4(double *restrict out,const double *restrict in,size_t n){CONSTS;size_t i=0;for(;i+32<=n;i+=32){
 __m512d r0,r1,r2,r3,sc0,sc1,sc2,sc3,e0,e1,e2,e3,l0,l1,l2,l3,y0,y1,y2,y3;__m512i k0,k1,k2,k3;
 REDUCE_ONE(_mm512_loadu_pd(in+i),r0,k0); SCALE_ALIGNED(k0,sc0);
 REDUCE_ONE(_mm512_loadu_pd(in+i+8),r1,k1); RECON_PART(r0,sc0,e0,l0); SCALE_ALIGNED(k1,sc1);
 REDUCE_ONE(_mm512_loadu_pd(in+i+16),r2,k2); RECON_PART(r1,sc1,e1,l1); SCALE_ALIGNED(k2,sc2); FINISH_PART(sc0,e0,l0,y0);
 REDUCE_ONE(_mm512_loadu_pd(in+i+24),r3,k3); RECON_PART(r2,sc2,e2,l2); SCALE_ALIGNED(k3,sc3); FINISH_PART(sc1,e1,l1,y1);
 RECON_PART(r3,sc3,e3,l3); FINISH_PART(sc2,e2,l2,y2); FINISH_PART(sc3,e3,l3,y3);
 _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
 }for(;i<n;i++)out[i]=exp(in[i]);}

__attribute__((target("avx512f,avx512dq,fma,avx2"),noinline))
void exp53_n2_hw_splitg256_u4(double *restrict out,const double *restrict in,size_t n){CONSTS;size_t i=0;for(;i+32<=n;i+=32){
 __m512d r0,r1,r2,r3,sc0,sc1,sc2,sc3,y0,y1,y2,y3;__m512i k0,k1,k2,k3;
 REDUCE_ONE(_mm512_loadu_pd(in+i),r0,k0);REDUCE_ONE(_mm512_loadu_pd(in+i+8),r1,k1);REDUCE_ONE(_mm512_loadu_pd(in+i+16),r2,k2);REDUCE_ONE(_mm512_loadu_pd(in+i+24),r3,k3);
 SCALE_SPLIT256(k0,sc0);SCALE_SPLIT256(k1,sc1);SCALE_SPLIT256(k2,sc2);SCALE_SPLIT256(k3,sc3);
 POLY_RECON(r0,sc0,y0);POLY_RECON(r1,sc1,y1);POLY_RECON(r2,sc2,y2);POLY_RECON(r3,sc3,y3);
 _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
 }for(;i<n;i++)out[i]=exp(in[i]);}

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_hw_aligned_u4(double *restrict out,const double *restrict in,size_t n){CONSTS;size_t i=0;for(;i+32<=n;i+=32){
 __m512d r0,r1,r2,r3,s0,s1,s2,s3,y0,y1,y2,y3;__m512i k0,k1,k2,k3;
 REDUCE_ONE(_mm512_loadu_pd(in+i),r0,k0);REDUCE_ONE(_mm512_loadu_pd(in+i+8),r1,k1);REDUCE_ONE(_mm512_loadu_pd(in+i+16),r2,k2);REDUCE_ONE(_mm512_loadu_pd(in+i+24),r3,k3);
 SCALE_ALIGNED(k0,s0);SCALE_ALIGNED(k1,s1);SCALE_ALIGNED(k2,s2);SCALE_ALIGNED(k3,s3);
 POLY_RECON(r0,s0,y0);POLY_RECON(r1,s1,y1);POLY_RECON(r2,s2,y2);POLY_RECON(r3,s3,y3);
 _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);_mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
 }for(;i<n;i++)out[i]=exp(in[i]);}

/* Hot-L1 table prefetch/control experiment. */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_hw_prefetch_u4(double *restrict out,const double *restrict in,size_t n){
 for(int z=0;z<16;z++) _mm_prefetch((const char*)TAB128_ALIGNED+64*z,_MM_HINT_T0); exp53_n2_hw_aligned_u4(out,in,n);
}

/* Full 256-bit arithmetic version: test sustained-width/frequency tradeoff. */
__attribute__((target("avx2,fma"),noinline))
void exp53_n2_hw_avx2_u4(double *restrict out,const double *restrict in,size_t n){
 const __m256d inv=_mm256_set1_pd(INV128),hi=_mm256_set1_pd(LHI),mi=_mm256_set1_pd(LMI),lo=_mm256_set1_pd(LLO),magic=_mm256_set1_pd(MAGIC),one=_mm256_set1_pd(1.0),a1=_mm256_set1_pd(Q1),a2=_mm256_set1_pd(Q2),a3=_mm256_set1_pd(Q3),a4=_mm256_set1_pd(Q4); const __m256i mb=_mm256_set1_epi64x((long long)MAGIC_BITS),msk=_mm256_set1_epi64x(127);
 size_t i=0;for(;i+16<=n;i+=16){for(int b=0;b<4;b++){__m256d x=_mm256_loadu_pd(in+i+4*b);__m256d bb=_mm256_fmadd_pd(x,inv,magic),kk=_mm256_sub_pd(bb,magic);__m256i kn=_mm256_sub_epi64(_mm256_castpd_si256(bb),mb);__m256d r=_mm256_fnmadd_pd(kk,hi,x);r=_mm256_fnmadd_pd(kk,mi,r);r=_mm256_fnmadd_pd(kk,lo,r);__m256i j=_mm256_and_si256(kn,msk);__m256i q=_mm256_srai_epi32(kn,7); /* replaced below with scalar-safe 64-bit arithmetic via shifts */
 long long kv[4],tv[4]; _mm256_storeu_si256((__m256i*)kv,kn); for(int t=0;t<4;t++){long long jj=kv[t]&127LL, qq=kv[t]>>7; uint64_t u; memcpy(&u,&TAB128_ALIGNED[jj],8); tv[t]=(long long)(u+((uint64_t)qq<<52));} __m256d sc=_mm256_castsi256_pd(_mm256_loadu_si256((const __m256i*)tv));
 __m256d h=_mm256_fmadd_pd(a4,r,a3);h=_mm256_fmadd_pd(h,r,a2);h=_mm256_fmadd_pd(h,r,a1);h=_mm256_fmadd_pd(h,r,one);__m256d ss=_mm256_mul_pd(h,h),er=_mm256_fmadd_pd(r,ss,one),el=_mm256_fmadd_pd(r,ss,_mm256_sub_pd(one,er)),ph=_mm256_mul_pd(er,sc),y=_mm256_fmadd_pd(el,sc,ph);_mm256_storeu_pd(out+i+4*b,y);}}
 for(;i<n;i++)out[i]=exp(in[i]);
}
