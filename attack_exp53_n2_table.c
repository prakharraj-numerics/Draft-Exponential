/* MPFR-guided table attack for faithful n=2 spine. Hot math is unchanged:
   er = 1 + r Q4(r)^2. Only choose TAB[j] among nextdown/current/nextup. */
#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "exp53_spine_n2_integralpower.c"
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}
static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
static double eval_with_tab(double x,double th,int *jout){double bz=fma(x,INV128,MAGIC),kd=bz-MAGIC;int64_t kn=(int64_t)(U(bz)-MAGIC_BITS);int j=(int)(kn&127),q=(int)(kn>>7);double r=fma(-kd,L128_HI,x);r=fma(-kd,L128_MI,r);r=fma(-kd,L128_LO,r);double h=fma(N2_Q4,r,N2_Q3);h=fma(h,r,N2_Q2);h=fma(h,r,N2_Q1);h=fma(h,r,1.0);double er=fma(r,h*h,1.0);if(jout)*jout=j;return ldexp(er*th,q);}
static double ref_mpfr(mpfr_t X,mpfr_t E,double x){mpfr_set_d(X,x,MPFR_RNDN);mpfr_exp(E,X,MPFR_RNDN);return mpfr_get_d(E,MPFR_RNDN);}
int main(void){
 enum{N=12288,H=6144}; static double x[N]; uint64_t s=0x9e3779b97f4a7c15ULL;
 for(int i=0;i<H/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20); for(int i=H/2;i<H;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));
 for(int i=H;i<H+H/2;i++)x[i]=1.0+uni(&s)*99.0; for(int i=H+H/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);
 mpfr_t X,E;mpfr_inits2(256,X,E,(mpfr_ptr)0);
 unsigned long long cnt[128][3]={{0}},sum[128][3]={{0}},mx[128][3]={{0}}; int badbase[128]={0};
 for(int i=0;i<N;i++){double ref=ref_mpfr(X,E,x[i]);int j;eval_with_tab(x[i],TAB128[0],&j);double tv[3]={nextafter(TAB128[j],-INFINITY),TAB128[j],nextafter(TAB128[j],INFINITY)};for(int z=0;z<3;z++){uint64_t d=D(eval_with_tab(x[i],tv[z],0),ref);sum[j][z]+=d;if(d>mx[j][z])mx[j][z]=d;if(d>1)cnt[j][z]++;}if(cnt[j][1])badbase[j]=1;}
 double tuned[128];int changed=0;unsigned long long before=0,after=0;for(int j=0;j<128;j++){int best=1;for(int z=0;z<3;z++)if(cnt[j][z]<cnt[j][best]||(cnt[j][z]==cnt[j][best]&&(mx[j][z]<mx[j][best]||(mx[j][z]==mx[j][best]&&sum[j][z]<sum[j][best]))))best=z;tuned[j]=best==0?nextafter(TAB128[j],-INFINITY):best==2?nextafter(TAB128[j],INFINITY):TAB128[j];before+=cnt[j][1];after+=cnt[j][best];if(best!=1)changed++;if(cnt[j][1]||best!=1)printf("J %3d base_bad=%llu down_bad=%llu up_bad=%llu base_mx=%llu down_mx=%llu up_mx=%llu pick=%d base=%a tuned=%a\n",j,cnt[j][1],cnt[j][0],cnt[j][2],mx[j][1],mx[j][0],mx[j][2],best-1,TAB128[j],tuned[j]);}
 printf("TABLE_ATTACK sample_before=%llu sample_after=%llu changed=%d\n",before,after,changed);
 printf("TUNED_INIT={\n");for(int j=0;j<128;j++)printf("  %a%s\n",tuned[j],j==127?"":",");printf("};\n");
 /* independent deterministic stress set, 262144 inputs over [-100,100] */
 s=0xd1b54a32d192ed03ULL;unsigned long long b2=0,a2=0,mxb=0,mxa=0;for(int i=0;i<262144;i++){double xx=-100.0+200.0*uni(&s),ref=ref_mpfr(X,E,xx);int j;double yb=eval_with_tab(xx,TAB128[0],&j); /* j only; recompute correct base below */ yb=eval_with_tab(xx,TAB128[j],0);double ya=eval_with_tab(xx,tuned[j],0);uint64_t db=D(yb,ref),da=D(ya,ref);if(db>mxb)mxb=db;if(da>mxa)mxa=da;if(db>1)b2++;if(da>1)a2++;}
 printf("TABLE_STRESS n=262144 base_max=%llu base_gt1=%llu tuned_max=%llu tuned_gt1=%llu\n",mxb,b2,mxa,a2);
 mpfr_clears(X,E,(mpfr_ptr)0);return 0;
}
