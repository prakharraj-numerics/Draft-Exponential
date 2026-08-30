/* Three attacks on the frozen faithful n=2 spine.
   Mathematical core remains exactly er = 1 + r*Q4(r)^2.
   1) sign-conditioned single-gather table (256 doubles)
   2) hi+low table reconstruction
   3) residual-masked low-table correction
*/
#include <immintrin.h>
#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "exp53_spine_n2_integralpower.c"

static double N2_SIGNTAB[256] __attribute__((aligned(64)));
static double N2_TLO2[128] __attribute__((aligned(64)));
static double N2_RMAX;

static inline uint64_t n2_u64(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t n2_ord(double x){uint64_t u=n2_u64(x);return (u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t n2_udist(double a,double b){uint64_t A=n2_ord(a),B=n2_ord(b);return A>B?A-B:B-A;}
static inline double n2_er_scalar(double r){
    double h=fma(N2_Q4,r,N2_Q3); h=fma(h,r,N2_Q2); h=fma(h,r,N2_Q1); h=fma(h,r,1.0);
    return fma(r,h*h,1.0);
}

/* Offline/initialization fit: for each (j,sign(r)), choose base/down/up table double
   minimizing (max ULP, then >1 count, then total ULP) over a dense residual grid.
   This changes no hot-path arithmetic count versus a normal one-table lookup. */
void exp53_n2_lutres_init(void){
    mpfr_t ln2,a,e,rr,ref; mpfr_inits2(256,ln2,a,e,rr,ref,(mpfr_ptr)0);
    mpfr_const_log2(ln2,MPFR_RNDN); N2_RMAX=log(2.0)/256.0;
    for(int j=0;j<128;j++){
        mpfr_set_si(a,j,MPFR_RNDN); mpfr_div_ui(a,a,128,MPFR_RNDN); mpfr_mul(a,a,ln2,MPFR_RNDN); mpfr_exp(e,a,MPFR_RNDN);
        double base=TAB128[j], cand[3]={nextafter(base,-INFINITY),base,nextafter(base,INFINITY)};
        mpfr_set_d(rr,base,MPFR_RNDN); mpfr_sub(ref,e,rr,MPFR_RNDN); N2_TLO2[j]=mpfr_get_d(ref,MPFR_RNDN);
        for(int sg=0;sg<2;sg++){
            uint64_t bestmx=UINT64_MAX,bestsum=UINT64_MAX; int bestbad=0x7fffffff,best=1;
            for(int c=0;c<3;c++){
                uint64_t mx=0,sum=0; int bad=0;
                for(int g=0;g<=4096;g++){
                    double f=(double)g/4096.0;
                    double r=(sg?1.0:-1.0)*(f*N2_RMAX);
                    double er=n2_er_scalar(r), y=er*cand[c];
                    mpfr_set_d(rr,r,MPFR_RNDN); mpfr_exp(ref,rr,MPFR_RNDN); mpfr_mul(ref,ref,e,MPFR_RNDN);
                    double z=mpfr_get_d(ref,MPFR_RNDN); uint64_t d=n2_udist(y,z); if(d>mx)mx=d; if(d>1)bad++; sum+=d;
                }
                if(mx<bestmx || (mx==bestmx && (bad<bestbad || (bad==bestbad && sum<bestsum)))){bestmx=mx;bestbad=bad;bestsum=sum;best=c;}
            }
            N2_SIGNTAB[(j<<1)|sg]=cand[best];
        }
    }
    mpfr_clears(ln2,a,e,rr,ref,(mpfr_ptr)0);
}

#define LR_F2(M) M(0) M(1)
#define LR_F3(M) M(0) M(1) M(2)
#define LR_F4(M) M(0) M(1) M(2) M(3)
#define LR_LD(L) __m512d x##L=_mm512_loadu_pd(in+i+8*(L));
#define LR_BIAS(L) __m512d b##L=_mm512_fmadd_pd(x##L,inv,magic);
#define LR_K(L) __m512d k##L=_mm512_sub_pd(b##L,magic); __m512i ki##L=_mm512_sub_epi64(_mm512_castpd_si512(b##L),mb);
#define LR_JQ(L) __m512i j##L=_mm512_and_epi64(ki##L,mask),qq##L=_mm512_srai_epi64(ki##L,7); __m512d sc##L=exp2_from_q(qq##L);
#define LR_BASETAB(L) __m512d th##L=_mm512_i64gather_pd(j##L,TAB128,8);
#define LR_R(L) __m512d r##L=_mm512_fnmadd_pd(k##L,hi,x##L); r##L=_mm512_fnmadd_pd(k##L,mi,r##L); r##L=_mm512_fnmadd_pd(k##L,lo,r##L);
#define LR_SIGNTAB(L) __m512i sg##L=_mm512_srli_epi64(_mm512_castpd_si512(r##L),63); __m512i cj##L=_mm512_or_epi64(_mm512_slli_epi64(j##L,1),sg##L); __m512d th##L=_mm512_i64gather_pd(cj##L,N2_SIGNTAB,8);
#define LR_H1(L) __m512d h##L=_mm512_fmadd_pd(nq4,r##L,nq3);
#define LR_H2(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq2);
#define LR_H3(L) h##L=_mm512_fmadd_pd(h##L,r##L,nq1);
#define LR_H4(L) h##L=_mm512_fmadd_pd(h##L,r##L,one);
#define LR_ER(L) __m512d s##L=_mm512_mul_pd(h##L,h##L); __m512d er##L=_mm512_fmadd_pd(r##L,s##L,one);
#define LR_PLAIN(L) __m512d y##L=_mm512_mul_pd(_mm512_mul_pd(er##L,th##L),sc##L);
#define LR_TLO(L) __m512d tl##L=_mm512_i64gather_pd(j##L,N2_TLO2,8);
#define LR_HILO(L) __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_fmadd_pd(er##L,tl##L,ph##L); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define LR_MASKLOW_Q(L,QTR) __m512d ar##L=_mm512_andnot_pd(signmask,r##L); __mmask8 km##L=_mm512_cmp_pd_mask(ar##L,thr##QTR,_CMP_GE_OQ); __m512d tl##L=_mm512_mask_i64gather_pd(zero,km##L,j##L,N2_TLO2,8); __m512d ph##L=_mm512_mul_pd(er##L,th##L); __m512d p##L=_mm512_mask3_fmadd_pd(er##L,tl##L,ph##L,km##L); __m512d y##L=_mm512_mul_pd(p##L,sc##L);
#define LR_M25(L) LR_MASKLOW_Q(L,25)
#define LR_M50(L) LR_MASKLOW_Q(L,50)
#define LR_M75(L) LR_MASKLOW_Q(L,75)
#define LR_ST(L) _mm512_storeu_pd(out+i+8*(L),y##L);
#define LR_C const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2_Q1),nq2=_mm512_set1_pd(N2_Q2),nq3=_mm512_set1_pd(N2_Q3),nq4=_mm512_set1_pd(N2_Q4),zero=_mm512_setzero_pd(),signmask=_mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000ULL)),thr25=_mm512_set1_pd(N2_RMAX*0.25),thr50=_mm512_set1_pd(N2_RMAX*0.50),thr75=_mm512_set1_pd(N2_RMAX*0.75); const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);
#define LR_CORE(F) F(LR_H1) F(LR_H2) F(LR_H3) F(LR_H4) F(LR_ER)
#define LR_SIGN_ST(F) F(LR_LD) F(LR_BIAS) F(LR_K) F(LR_JQ) F(LR_R) F(LR_SIGNTAB) LR_CORE(F) F(LR_PLAIN) F(LR_ST)
#define LR_HILO_ST(F) F(LR_LD) F(LR_BIAS) F(LR_K) F(LR_JQ) F(LR_BASETAB) F(LR_R) LR_CORE(F) F(LR_TLO) F(LR_HILO) F(LR_ST)
#define LR_M25_ST(F) F(LR_LD) F(LR_BIAS) F(LR_K) F(LR_JQ) F(LR_BASETAB) F(LR_R) LR_CORE(F) F(LR_M25) F(LR_ST)
#define LR_M50_ST(F) F(LR_LD) F(LR_BIAS) F(LR_K) F(LR_JQ) F(LR_BASETAB) F(LR_R) LR_CORE(F) F(LR_M50) F(LR_ST)
#define LR_M75_ST(F) F(LR_LD) F(LR_BIAS) F(LR_K) F(LR_JQ) F(LR_BASETAB) F(LR_R) LR_CORE(F) F(LR_M75) F(LR_ST)
#define LR_DEF(NAME,U,F,ST) __attribute__((target("avx512f,avx512dq,fma"),noinline)) void NAME(double*restrict out,const double*restrict in,size_t n){LR_C size_t i=0;for(;i+8*(U)<=n;i+=8*(U)){ST(F)}if(i<n)exp53_n2_fast_u2(out+i,in+i,n-i);}

LR_DEF(exp53_n2_signlut_u2,2,LR_F2,LR_SIGN_ST)
LR_DEF(exp53_n2_signlut_u4,4,LR_F4,LR_SIGN_ST)
LR_DEF(exp53_n2_hilo_u2,2,LR_F2,LR_HILO_ST)
LR_DEF(exp53_n2_hilo_u3,3,LR_F3,LR_HILO_ST)
LR_DEF(exp53_n2_hilo_u4,4,LR_F4,LR_HILO_ST)
LR_DEF(exp53_n2_mask25_u4,4,LR_F4,LR_M25_ST)
LR_DEF(exp53_n2_mask50_u4,4,LR_F4,LR_M50_ST)
LR_DEF(exp53_n2_mask75_u4,4,LR_F4,LR_M75_ST)
