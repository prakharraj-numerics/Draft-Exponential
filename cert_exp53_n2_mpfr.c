#define _GNU_SOURCE
#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void exp53_n2_rc_u4(double*,const double*,size_t);
static inline uint64_t U64(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static inline uint64_t ORD(double x){uint64_t u=U64(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t ULPD(double a,double b){uint64_t A=ORD(a),B=ORD(b);return A>=B?A-B:B-A;}
static uint64_t sm64(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm64(s)>>11)*0x1p-53;}
static int cr_exp(double x,double*out){mpfr_prec_t p=192;for(int t=0;t<5;t++,p*=2){mpfr_t mx,lo,hi;mpfr_init2(mx,p);mpfr_init2(lo,p);mpfr_init2(hi,p);mpfr_set_d(mx,x,MPFR_RNDN);mpfr_exp(lo,mx,MPFR_RNDD);mpfr_exp(hi,mx,MPFR_RNDU);double dl=mpfr_get_d(lo,MPFR_RNDN),dh=mpfr_get_d(hi,MPFR_RNDN);mpfr_clear(mx);mpfr_clear(lo);mpfr_clear(hi);if(U64(dl)==U64(dh)){*out=dl;return 1;}}return 0;}
static void add_base(double*x,size_t*n){const size_t H=6144;uint64_t s=0x9e3779b97f4a7c15ULL;for(size_t i=0;i<H/2;i++)x[(*n)++]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(size_t i=H/2;i<H;i++)x[(*n)++]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(size_t i=H;i<H+H/2;i++)x[(*n)++]=1.0+uni(&s)*99.0;for(size_t i=H+H/2;i<2*H;i++)x[(*n)++]=-(1.0+uni(&s)*99.0);}
static void add_adv(double*x,size_t*n,size_t cap){const long double L=logl(2.0L)/128.0L,R=logl(2.0L)/256.0L;const long double rf[]={-R,nextafterl(-R,0.0L),-.75L*R,-.5L*R,-.25L*R,-0x1p-60L,0,0x1p-60L,.25L*R,.5L*R,.75L*R,nextafterl(R,0.0L),R};for(long long q=-145;q<=144;q++)for(int j=0;j<128;j++){long long k=128*q+j;long double base=(long double)k*L;for(size_t z=0;z<sizeof(rf)/sizeof(rf[0]);z++){long double xx=base+rf[z];if(xx<-100||xx>100)continue;double d=(double)xx;if(*n<cap)x[(*n)++]=d;if(*n<cap)x[(*n)++]=nextafter(d,-INFINITY);if(*n<cap)x[(*n)++]=nextafter(d,INFINITY);}}}
static void run(const char*name,double*x,size_t n){size_t np=(n+7)&~(size_t)7;double*y=aligned_alloc(64,np*sizeof(double));exp53_n2_rc_u4(y,x,n);uint64_t mx=0,g0=0,g1=0,g2=0;size_t un=0;long long pos=0,neg=0;double wx=0,wy=0,wr=0;for(size_t i=0;i<n;i++){double r;if(!cr_exp(x[i],&r)){un++;continue;}uint64_t d=ULPD(y[i],r);if(d>mx){mx=d;wx=x[i];wy=y[i];wr=r;}if(d>0)g0++;if(d>1){g1++;if(ORD(y[i])>ORD(r))pos++;else neg++;}if(d>2)g2++;}printf("N2_MPFR set=%s n=%zu uncert=%zu maxulp=%llu gt0=%llu gt1=%llu gt2=%llu badpos=%lld badneg=%lld worst_x=%.17g worst_y=%.17g worst_ref=%.17g\n",name,n,un,(unsigned long long)mx,(unsigned long long)g0,(unsigned long long)g1,(unsigned long long)g2,pos,neg,wx,wy,wr);free(y);}
int main(void){size_t cap=1600000,n=0;double*x=aligned_alloc(64,cap*sizeof(double));if(!x)return 2;add_base(x,&n);run("benchmark_exact",x,n);n=0;add_adv(x,&n,cap);run("v128_adversarial",x,n);free(x);return 0;}
