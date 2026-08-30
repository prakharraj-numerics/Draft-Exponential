#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "exp53_spine_v128_formula6_compfast.c"
#define Q1 0x1.0000000000000p-2
#define Q2 0x1.aaaaaaaaaaaabp-5
#define Q3 0x1.0000000000000p-7
#define Q4 0x1.c16c16c16c16cp-11
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
enum{N=12288,H=6144};static double x[N];static void inputs(void){uint64_t s=0x9e3779b97f4a7c15ULL;for(int i=0;i<H/2;i++)x[i]=0x1p-20+uni(&s)*(1.0-0x1p-20);for(int i=H/2;i<H;i++)x[i]=-(0x1p-20+uni(&s)*(1.0-0x1p-20));for(int i=H;i<H+H/2;i++)x[i]=1.0+uni(&s)*99.0;for(int i=H+H/2;i<N;i++)x[i]=-(1.0+uni(&s)*99.0);}
struct C{const char*n;uint64_t mx;int g1;};static void hit(struct C*c,double y,double ref){uint64_t d=D(y,ref);if(d>c->mx)c->mx=d;if(d>1)c->g1++;}
static double rnd(mpfr_t v){return mpfr_get_d(v,MPFR_RNDN);} 
int main(void){inputs();struct C c[5]={{"after_r",0,0},{"after_Q",0,0},{"after_ER",0,0},{"after_tabconst",0,0},{"actual_fast",0,0}};mpfr_t X,L,R,E,Q,S,V,T;mpfr_inits2(256,X,L,R,E,Q,S,V,T,(mpfr_ptr)0);mpfr_const_log2(L,MPFR_RNDN);mpfr_div_ui(L,L,128,MPFR_RNDN);
for(int i=0;i<N;i++){double xd=x[i];double bz=fma(xd,INV128,MAGIC),kd=bz-MAGIC;int64_t kn=(int64_t)(U(bz)-MAGIC_BITS);int j=(int)(kn&127),qq=(int)(kn>>7);double rd=fma(-kd,L128_HI,xd);rd=fma(-kd,L128_MI,rd);rd=fma(-kd,L128_LO,rd);double h=fma(Q4,rd,Q3);h=fma(h,rd,Q2);h=fma(h,rd,Q1);h=fma(h,rd,1.0);double ss=h*h,er=fma(rd,ss,1.0),th=TAB128[j];double sc=ldexp(1.0,qq),fast=(er*th)*sc;
mpfr_set_d(X,xd,MPFR_RNDN);mpfr_exp(E,X,MPFR_RNDN);double ref=rnd(E);
/* after r: exact exp(rd) then exact 2^(kn/128) */mpfr_set_d(R,rd,MPFR_RNDN);mpfr_exp(V,R,MPFR_RNDN);mpfr_set_si(T,kn,MPFR_RNDN);mpfr_mul(T,T,L,MPFR_RNDN);mpfr_exp(T,T,MPFR_RNDN);mpfr_mul(V,V,T,MPFR_RNDN);hit(&c[0],rnd(V),ref);
/* after Q: exact 1+r*h^2 and exact table/scale factor */mpfr_set_d(Q,h,MPFR_RNDN);mpfr_mul(S,Q,Q,MPFR_RNDN);mpfr_set_d(R,rd,MPFR_RNDN);mpfr_mul(V,R,S,MPFR_RNDN);mpfr_add_ui(V,V,1,MPFR_RNDN);mpfr_set_si(T,kn,MPFR_RNDN);mpfr_mul(T,T,L,MPFR_RNDN);mpfr_exp(T,T,MPFR_RNDN);mpfr_mul(V,V,T,MPFR_RNDN);hit(&c[1],rnd(V),ref);
/* after ER: exact multiply er * exact 2^(kn/128) */mpfr_set_d(V,er,MPFR_RNDN);mpfr_set_si(T,kn,MPFR_RNDN);mpfr_mul(T,T,L,MPFR_RNDN);mpfr_exp(T,T,MPFR_RNDN);mpfr_mul(V,V,T,MPFR_RNDN);hit(&c[2],rnd(V),ref);
/* after table constant: exact er*binary64 table*2^q */mpfr_set_d(V,er,MPFR_RNDN);mpfr_set_d(T,th,MPFR_RNDN);mpfr_mul(V,V,T,MPFR_RNDN);mpfr_mul_2si(V,V,qq,MPFR_RNDN);hit(&c[3],rnd(V),ref);hit(&c[4],fast,ref);
}
for(int k=0;k<5;k++)printf("STAGE_MPFR %-14s maxulp=%llu gt1=%d\n",c[k].n,(unsigned long long)c[k].mx,c[k].g1);mpfr_clears(X,L,R,E,Q,S,V,T,(mpfr_ptr)0);return 0;}
