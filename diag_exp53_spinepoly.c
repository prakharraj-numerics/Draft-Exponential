#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define INV128   0x1.71547652b82fep+7
#define L128_HI  0x1.62e42fefa39efp-8
#define L128_MI  0x1.abc9e3b39803fp-63
#define L128_LO  0x1.7b57a079a1934p-118
#define MAGIC     0x1.8000000000000p+52
#define MAGIC_BITS 0x4338000000000000ULL
#define SP_A2 0x1.1111111111111p-7
#define SP_A1 0x1.5555555555555p-3
#define SP_B2 0x1.6c16c16c16c17p-10
#define SP_B1 0x1.5555555555555p-5
#define SP_B0 0x1.0000000000000p-1
extern const double TAB128[128];
void exp53_spinepoly_mid_u4(double*,const double*,size_t);

static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline long long SD(double a,double b){uint64_t A=O(a),B=O(b);return A>=B?(long long)(A-B):-(long long)(B-A);}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}

static void reduce(double x,long long *kn,int *j,long long *q,double *r){
    double bz=fma(x,INV128,MAGIC), k=bz-MAGIC;
    uint64_t bits=U(bz); *kn=(long long)(bits-MAGIC_BITS); *j=(int)(*kn&127); *q=*kn>>7;
    double rr=fma(-k,L128_HI,x); rr=fma(-k,L128_MI,rr); rr=fma(-k,L128_LO,rr); *r=rr;
}
static double poly(double r){double t=r*r; double A=fma(SP_A2,t,SP_A1); double B=fma(SP_B2,t,SP_B1); A=fma(A,t,1.0); B=fma(B,t,SP_B0); double y=fma(r,A,1.0); return fma(t,B,y);}

int main(void){
 enum{N=12288,H=6144}; static double x[N],y[N]; uint64_t s=0x243f6a8885a308d3ULL;
 for(int i=0;i<H/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20); for(int i=H/2;i<H;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20)); for(int i=H;i<H+H/2;i++)x[i]=1.0+uni(&s)*99.0; for(int i=H+H/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);
 exp53_spinepoly_mid_u4(y,x,N);
 long long out_pos=0,out_neg=0,res_pos=0,res_neg=0,tab_pos=0,tab_neg=0; int bad=0,bad3=0,jbad[128]={0},jpos[128]={0},jneg[128]={0}; int rpos=0,rneg=0;
 long double max_res_ulps=0,max_tab_ulps=0; int printed=0;
 for(int i=0;i<N;i++){
   double ref=exp(x[i]); long long du=SD(y[i],ref); if(du>1||du<-1){
     bad++; if(du>=3||du<=-3)bad3++; if(du>0)out_pos++; else out_neg++;
     long long kn,q; int j; double r; reduce(x[i],&kn,&j,&q,&r); jbad[j]++; if(du>0)jpos[j]++; else jneg[j]++; if(r>=0)rpos++; else rneg++;
     double er=poly(r), th=TAB128[j]; double ph=er*th; double cr=fma(er,th,-ph); double prod=ph+cr;
     long double true_er=expl((long double)x[i]-((long double)q*logl(2.0L)+(long double)j*logl(2.0L)/128.0L));
     long double erulp=((long double)er-true_er)/(long double)nextafter(er,INFINITY)-(long double)er; /* fixed below */
     long double ulper=(long double)nextafter(er,INFINITY)-(long double)er; erulp=((long double)er-true_er)/ulper;
     long double true_prod=expl((long double)x[i]-(long double)q*logl(2.0L)); long double ulpp=(long double)nextafter(prod,INFINITY)-(long double)prod; long double pulps=((long double)prod-true_prod)/ulpp;
     if(fabsl(erulp)>max_res_ulps)max_res_ulps=fabsl(erulp); if(fabsl(pulps)>max_tab_ulps)max_tab_ulps=fabsl(pulps);
     if(erulp>0)res_pos++; else res_neg++; if(pulps>0)tab_pos++; else tab_neg++;
     if(printed<40){printf("BAD i=%d x=%.17g out_sulp=%lld j=%d q=%lld r=%+.17g er_err_ulp=%+.4Lf prod_err_ulp=%+.4Lf\n",i,x[i],du,j,q,r,erulp,pulps);printed++;}
   }
 }
 printf("SUMMARY bad=%d bad3=%d out_pos=%lld out_neg=%lld rpos=%d rneg=%d\n",bad,bad3,out_pos,out_neg,rpos,rneg);
 printf("STAGES residual_pos=%lld residual_neg=%lld maxabs_res_ulp=%.4Lf product_pos=%lld product_neg=%lld maxabs_prod_ulp=%.4Lf\n",res_pos,res_neg,max_res_ulps,tab_pos,tab_neg,max_tab_ulps);
 printf("JBUCKETS\n"); for(int j=0;j<128;j++)if(jbad[j])printf("j=%d bad=%d pos=%d neg=%d\n",j,jbad[j],jpos[j],jneg[j]);
 return 0;
}
