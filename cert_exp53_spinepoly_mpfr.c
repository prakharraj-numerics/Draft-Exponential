#define _GNU_SOURCE
#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "exp53_spine_v128_spinepoly.c"

static inline uint64_t U64(double x){uint64_t u; memcpy(&u,&x,8); return u;}
static inline uint64_t ORD(double x){uint64_t u=U64(x); return (u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ULPD(double a,double b){uint64_t A=ORD(a),B=ORD(b); return A>=B?A-B:B-A;}
static uint64_t sm64(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm64(s)>>11)*0x1p-53;}

/* Certified correctly-rounded binary64 exp(x): x is imported exactly; exp bounds are
   computed directed at increasing MPFR precision until both bounds round-to-nearest
   to the same binary64 value. */
static int cr_exp(double x,double *out){
  mpfr_prec_t p=192;
  for(int tries=0;tries<5;tries++,p*=2){
    mpfr_t mx,lo,hi; mpfr_init2(mx,p);mpfr_init2(lo,p);mpfr_init2(hi,p);
    mpfr_set_d(mx,x,MPFR_RNDN);
    mpfr_exp(lo,mx,MPFR_RNDD); mpfr_exp(hi,mx,MPFR_RNDU);
    double dl=mpfr_get_d(lo,MPFR_RNDN), dh=mpfr_get_d(hi,MPFR_RNDN);
    mpfr_clear(mx);mpfr_clear(lo);mpfr_clear(hi);
    if(U64(dl)==U64(dh)){*out=dl; return 1;}
  }
  return 0;
}

static void add_base(double *x,size_t *n,size_t cap){
  const size_t H=6144; uint64_t s=0x243f6a8885a308d3ULL;
  for(size_t i=0;i<H/2 && *n<cap;i++) x[(*n)++]=0x1p-20+uni(&s)*(1.0-0x1p-20);
  for(size_t i=H/2;i<H && *n<cap;i++) x[(*n)++]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));
  for(size_t i=H;i<H+H/2 && *n<cap;i++) x[(*n)++]=1.0+uni(&s)*99.0;
  for(size_t i=H+H/2;i<2*H && *n<cap;i++) x[(*n)++]=-(1.0+uni(&s)*99.0);
}

/* Adversarial set around every v128 cell: residual endpoints, zero, half/end-neighbours,
   and one-ULP neighbours of reconstructed x. q spans the complete benchmark domain. */
static void add_adversarial(double *x,size_t *n,size_t cap){
  const long double L=logl(2.0L)/128.0L;
  const long double R=logl(2.0L)/256.0L;
  const long double rf[]={-R,nextafterl(-R,0.0L),-0.75L*R,-0.5L*R,-0.25L*R,
                          -0x1p-60L,0.0L,0x1p-60L,0.25L*R,0.5L*R,0.75L*R,
                          nextafterl(R,0.0L),R};
  for(long long q=-145;q<=144;q++){
    for(int j=0;j<128;j++){
      long long k=128*q+j;
      long double base=(long double)k*L;
      for(size_t z=0;z<sizeof(rf)/sizeof(rf[0]);z++){
        long double xx=base+rf[z]; if(xx < -100.0L || xx > 100.0L) continue;
        double d=(double)xx;
        if(*n<cap)x[(*n)++]=d;
        if(*n<cap)x[(*n)++]=nextafter(d,-INFINITY);
        if(*n<cap)x[(*n)++]=nextafter(d, INFINITY);
      }
    }
  }
}

static void run_set(const char *name,double *x,size_t n){
  double *y=aligned_alloc(64,((n+7)/8*8)*sizeof(double));
  if(!y){perror("alloc");exit(2);} exp53_spinepoly_mid_u4(y,x,n);
  uint64_t maxulp=0,gt0=0,gt1=0,gt2=0,gt3=0; size_t uncert=0; double worstx=0,worsty=0,worstr=0;
  long long spos=0,sneg=0;
  for(size_t i=0;i<n;i++){
    double ref; if(!cr_exp(x[i],&ref)){uncert++;continue;}
    uint64_t u=ULPD(y[i],ref); if(u>maxulp){maxulp=u;worstx=x[i];worsty=y[i];worstr=ref;}
    if(u>0)gt0++; if(u>1)gt1++; if(u>2)gt2++; if(u>3)gt3++;
    if(u>1){ if(ORD(y[i])>ORD(ref))spos++; else sneg++; }
  }
  printf("MPFR_CERT set=%s n=%zu uncert=%zu maxulp=%llu gt0=%llu gt1=%llu gt2=%llu gt3=%llu bad_pos=%lld bad_neg=%lld worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",
    name,n,uncert,(unsigned long long)maxulp,(unsigned long long)gt0,(unsigned long long)gt1,
    (unsigned long long)gt2,(unsigned long long)gt3,spos,sneg,worstx,worsty,worstr);
  free(y);
}

int main(void){
  size_t cap=1600000,n=0; double *x=aligned_alloc(64,cap*sizeof(double)); if(!x){perror("alloc");return 2;}
  add_base(x,&n,cap); run_set("benchmark_exact",x,n);
  n=0; add_adversarial(x,&n,cap); run_set("v128_adversarial",x,n);
  free(x); return 0;
}
