#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "exp53_spine_v128_spinepoly.c"

static inline uint64_t DU(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t DO(double a){uint64_t u=DU(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline long long SD(double a,double b){uint64_t A=DO(a),B=DO(b);return A>=B?(long long)(A-B):-(long long)(B-A);}
static uint64_t smd(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);} static double uni_d(uint64_t*s){return(smd(s)>>11)*0x1p-53;}

static void reduce_d(double x,long long *kn,int *j,long long *q,double *r){
    double bz=fma(x,INV128,MAGIC), k=bz-MAGIC;
    uint64_t bits=DU(bz); *kn=(long long)(bits-MAGIC_BITS); *j=(int)(*kn&127); *q=*kn>>7;
    double rr=fma(-k,L128_HI,x); rr=fma(-k,L128_MI,rr); rr=fma(-k,L128_LO,rr); *r=rr;
}
static double poly_d(double r){double t=r*r; double A=fma(SP_A2,t,SP_A1); double B=fma(SP_B2,t,SP_B1); A=fma(A,t,1.0); B=fma(B,t,SP_B0); double y=fma(r,A,1.0); return fma(t,B,y);}

int main(void){
 enum{N=12288,H=6144}; static double x[N],y[N]; uint64_t s=0x243f6a8885a308d3ULL;
 for(int i=0;i<H/2;i++)x[i]=0x1p-20+uni_d(&s)*(1.0-0x1p-20); for(int i=H/2;i<H;i++)x[i]=-(0x1p-20+uni_d(&s)*(1.0-0x1p-20)); for(int i=H;i<H+H/2;i++)x[i]=1.0+uni_d(&s)*99.0; for(int i=H+H/2;i<N;i++)x[i]=-(1.0+uni_d(&s)*99.0);
 exp53_spinepoly_mid_u4(y,x,N);
 long long out_pos=0,out_neg=0,res_pos=0,res_neg=0,prod_pos=0,prod_neg=0; int bad=0,bad3=0,jbad[128]={0},jpos[128]={0},jneg[128]={0}; int rpos=0,rneg=0;
 long double max_res_ulps=0,max_prod_ulps=0; int printed=0;
 for(int i=0;i<N;i++){
   double ref=exp(x[i]); long long du=SD(y[i],ref); if(du>1||du<-1){
     bad++; if(llabs(du)>=3)bad3++; if(du>0)out_pos++; else out_neg++;
     long long kn,q; int j; double r; reduce_d(x[i],&kn,&j,&q,&r); jbad[j]++; if(du>0)jpos[j]++; else jneg[j]++; if(r>=0)rpos++; else rneg++;
     double er=poly_d(r), th=TAB128[j]; double ph=er*th; double cr=fma(er,th,-ph); double prod=ph+cr;
     long double true_er=expl((long double)x[i]-((long double)q*logl(2.0L)+(long double)j*logl(2.0L)/128.0L));
     long double ulper=(long double)nextafter(er,INFINITY)-(long double)er; long double erulp=((long double)er-true_er)/ulper;
     long double true_prod=expl((long double)x[i]-(long double)q*logl(2.0L)); long double ulpp=(long double)nextafter(prod,INFINITY)-(long double)prod; long double pulps=((long double)prod-true_prod)/ulpp;
     if(fabsl(erulp)>max_res_ulps)max_res_ulps=fabsl(erulp); if(fabsl(pulps)>max_prod_ulps)max_prod_ulps=fabsl(pulps);
     if(erulp>0)res_pos++; else res_neg++; if(pulps>0)prod_pos++; else prod_neg++;
     if(printed<48){printf("BAD i=%d x=%.17g out_sulp=%lld j=%d q=%lld r=%+.17g er_err_ulp=%+.4Lf prod_err_ulp=%+.4Lf\n",i,x[i],du,j,q,r,erulp,pulps);printed++;}
   }
 }
 printf("SUMMARY bad=%d bad3=%d out_pos=%lld out_neg=%lld rpos=%d rneg=%d\n",bad,bad3,out_pos,out_neg,rpos,rneg);
 printf("STAGES residual_pos=%lld residual_neg=%lld maxabs_res_ulp=%.4Lf product_pos=%lld product_neg=%lld maxabs_prod_ulp=%.4Lf\n",res_pos,res_neg,max_res_ulps,prod_pos,prod_neg,max_prod_ulps);
 printf("JBUCKETS\n"); for(int j=0;j<128;j++)if(jbad[j])printf("j=%d bad=%d pos=%d neg=%d\n",j,jbad[j],jpos[j],jneg[j]);
 return 0;
}
