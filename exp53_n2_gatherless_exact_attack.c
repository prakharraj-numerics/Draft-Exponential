#include "exp53_n2_execpath_attack.c"
#include <stdio.h>
#include <string.h>

/* Gatherless exact-anchor repair.
   Fast 16x8 factor product is corrected in INTEGER ULP units back to the exact
   frozen TAB128[j] bit pattern before adding q<<52. Correction table is only
   128 signed nibbles = 64 bytes, packed as 8 uint64 words. Runtime lookup uses
   one zmm permute + variable shift, not a memory gather. */

static uint64_t GLX_PACK[8];
static int GLX_READY;
static long long GLX_DMIN,GLX_DMAX;

static inline uint64_t glx_bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}

void exp53_n2_gatherless_exact_init(void){
    if(GLX_READY) return;
    memset(GLX_PACK,0,sizeof(GLX_PACK));
    GLX_DMIN=999; GLX_DMAX=-999;
    for(int j=0;j<128;j++){
        double a=TAB128[(j>>3)*8];
        double b=TAB128[j&7];
        double p=a*b;
        long long d=(long long)glx_bits(TAB128[j])-(long long)glx_bits(p);
        if(d<GLX_DMIN)GLX_DMIN=d; if(d>GLX_DMAX)GLX_DMAX=d;
        if(d < -8 || d > 7){fprintf(stderr,"GLX correction out of nibble range j=%d d=%lld\n",j,d);GLX_READY=-1;return;}
        unsigned code=(unsigned)(d & 15);
        GLX_PACK[j>>4] |= (uint64_t)code << (4*(j&15));
    }
    GLX_READY=1;
}

void exp53_n2_gatherless_exact_diag(void){
    exp53_n2_gatherless_exact_init();
    printf("GLX_DIAG ready=%d dmin=%lld dmax=%lld packs=",GLX_READY,GLX_DMIN,GLX_DMAX);
    for(int i=0;i<8;i++)printf("%s%016llx",i?",":"",(unsigned long long)GLX_PACK[i]);
    putchar('\n');
}

#define GLX_CONSTS \
 const __m512d ga0=_mm512_setr_pd( \
  0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0, \
  0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0), \
 ga1=_mm512_setr_pd( \
  0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0, \
  0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0), \
 gbv=_mm512_setr_pd( \
  0x1.0000000000000p+0,0x1.0163da9fb3335p+0,0x1.02c9a3e778061p+0,0x1.04315e86e7f85p+0, \
  0x1.059b0d3158574p+0,0x1.0706b29ddf6dep+0,0x1.0874518759bc8p+0,0x1.09e3ecac6f383p+0); \
 const __m512i gpack=_mm512_loadu_si512((const void*)GLX_PACK)

#define SCALE_GLX(KN,SCALE) do{ \
 __m512i jj=_mm512_and_epi64((KN),mask); \
 __m512i ia=_mm512_srli_epi64(jj,3); \
 __m512i ib=_mm512_and_epi64(jj,_mm512_set1_epi64(7)); \
 __m512d aa=_mm512_permutex2var_pd(ga0,ia,ga1); \
 __m512d bb=_mm512_permutexvar_pd(ib,gbv); \
 __m512d pp=_mm512_mul_pd(aa,bb); \
 __m512i grp=_mm512_srli_epi64(jj,4); \
 __m512i pw=_mm512_permutexvar_epi64(grp,gpack); \
 __m512i sh=_mm512_slli_epi64(_mm512_and_epi64(jj,_mm512_set1_epi64(15)),2); \
 __m512i code=_mm512_and_epi64(_mm512_srlv_epi64(pw,sh),_mm512_set1_epi64(15)); \
 __m512i delta=_mm512_sub_epi64(code,_mm512_slli_epi64(_mm512_and_epi64(code,_mm512_set1_epi64(8)),1)); \
 __m512i exact=_mm512_add_epi64(_mm512_castpd_si512(pp),delta); \
 __m512i qq=_mm512_srai_epi64((KN),7); \
 exact=_mm512_add_epi64(exact,_mm512_slli_epi64(qq,52)); \
 (SCALE)=_mm512_castsi512_pd(exact); \
}while(0)

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_glx_u4(double *restrict out,const double *restrict in,size_t n){
    if(!GLX_READY) exp53_n2_gatherless_exact_init();
    CONSTS; GLX_CONSTS; BODY4(SCALE_GLX,POLY_RECON);
}
