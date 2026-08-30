#include <mpfr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "exp53_spine_n2_integralpower.c"
static inline uint64_t U(double a){uint64_t u;memcpy(&u,&a,8);return u;}
static inline uint64_t O(double a){uint64_t u=U(a);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static inline uint64_t D(double a,double b){uint64_t A=O(a),B=O(b);return A>B?A-B:B-A;}
static uint64_t sm(uint64_t*s){uint64_t v=(*s+=0x9e3779b97f4a7c15ULL);v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL;v=(v^(v>>27))*0x94d049bb133111ebULL;return v^(v>>31);}static double uni(uint64_t*s){return(sm(s)>>11)*0x1p-53;}
int main(void){enum{N=12288,H=6144};double x[N];uint64_t seed=0x9e3779b97f4a7c15ULL;for(int i=0;i<H/2;i++)x[i]=0x1p-20+uni(&seed)*(1.0-0x1p-20);for(int i=H/2;i<H;i++)x[i]=-(0x1p-20+uni(&seed)*(1.0-0x1p-20));for(int i=H;i<H+H/2;i++)x[i]=1.0+uni(&seed)*99.0;for(int i=H+H/2;i<N;i++)x[i]=-(1.0+uni(&seed)*99.0);
int bad=0,pcfix=0,elfix=0,rcfix=0,elonlyfix=0;double maxrel_el=0;int posel=0,negel=0;mpfr_t X,E;mpfr_inits2(256,X,E,(mpfr_ptr)0);
for(int i=0;i<N;i++){double xd=x[i];double bz=fma(xd,INV128,MAGIC),kd=bz-MAGIC;int64_t kn=(int64_t)(U(bz)-MAGIC_BITS);int j=(int)(kn&127),qq=(int)(kn>>7);double r=fma(-kd,L128_HI,xd);r=fma(-kd,L128_MI,r);r=fma(-kd,L128_LO,r);double h=fma(N2_Q4,r,N2_Q3);h=fma(h,r,N2_Q2);h=fma(h,r,N2_Q1);h=fma(h,r,1.0);double s=h*h;double er=fma(r,s,1.0);double el=fma(r,s,1.0-er);double th=TAB128[j],sc=ldexp(1.0,qq);double ph=er*th;double fast=ph*sc;double pl=fma(er,th,-ph);double pc=(ph+pl)*sc;double elonly=fma(el,th,ph)*sc;double pl2=fma(el,th,pl);double rc=(ph+pl2)*sc;mpfr_set_d(X,xd,MPFR_RNDN);mpfr_exp(E,X,MPFR_RNDN);double ref=mpfr_get_d(E,MPFR_RNDN);uint64_t df=D(fast,ref);if(df>1){bad++;if(D(pc,ref)<=1)pcfix++;if(D(elonly,ref)<=1)elonlyfix++;if(D(rc,ref)<=1)rcfix++;if(el>0)posel++;else if(el<0)negel++;double rel=fabs(el/er);if(rel>maxrel_el)maxrel_el=rel;printf("FAIL i=%d x=%a j=%d q=%d r=%a er=%a el=%a th=%a fastd=%llu pcd=%llu elonlyd=%llu rcd=%llu\n",i,xd,j,qq,r,er,el,th,(unsigned long long)df,(unsigned long long)D(pc,ref),(unsigned long long)D(elonly,ref),(unsigned long long)D(rc,ref));}}
printf("SUMMARY bad=%d pcfix=%d elonlyfix=%d rcfix=%d elpos=%d elneg=%d max_rel_el=%.17g\n",bad,pcfix,elonlyfix,rcfix,posel,negel,maxrel_el);mpfr_clears(X,E,(mpfr_ptr)0);return 0;}
