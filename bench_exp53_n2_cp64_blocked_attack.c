#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mkl_vml.h>
void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_n2_cp64_blocked_attack(double*,const double*,size_t);
typedef void(*fn_t)(double*,const double*,size_t);static volatile double sink;
static void*xalloc(size_t n){void*p=NULL;if(posix_memalign(&p,64,n)!=0||!p){perror("posix_memalign");exit(2);}return p;}
static uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}static uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}static uint64_t ud(double a,double b){uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static uint64_t rng=0x2545f4914f6cdd1dULL;static uint64_t ru(void){uint64_t x=rng;x^=x>>12;x^=x<<25;x^=x>>27;rng=x;return x*2685821657736338717ULL;}static double rd(void){double z=(double)(ru()>>11)*(1.0/9007199254740992.0);return -100.0+200.0*z;}
static void vha(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec*1e9+t.tv_nsec;}
static void screen(void){size_t n=200000;double*in=xalloc(n*8),*f=xalloc(n*8),*c=xalloc(n*8);for(size_t i=0;i<n;i++)in[i]=rd();exp53_n2_vmstyle_u4_0381_frozen(f,in,n);exp53_n2_cp64_blocked_attack(c,in,n);uint64_t mx=0,g=0,bd=0;for(size_t i=0;i<n;i++){double ref=(double)expl((long double)in[i]);uint64_t d=ud(c[i],ref);if(d>mx)mx=d;if(d>1)g++;if(bits(c[i])!=bits(f[i]))bd++;}printf("CP64B_ACCURACY maxULP=%llu gt1=%llu bitdiff=%llu\n",(unsigned long long)mx,(unsigned long long)g,(unsigned long long)bd);free(in);free(f);free(c);}
static double bench(fn_t fn,size_t n){double*in=xalloc(n*8),*out=xalloc(n*8);for(size_t i=0;i<n;i++)in[i]=rd();for(int w=0;w<20;w++)fn(out,in,n);int calls=(int)(16000000ULL/n);if(calls<80)calls=80;if(calls>5000)calls=5000;double best=1e300;for(int q=0;q<15;q++){double a=now();for(int r=0;r<calls;r++)fn(out,in,n);double b=now();double v=(b-a)/((double)calls*n);if(v<best)best=v;sink+=out[(q*157u)%n];}free(in);free(out);return best;}
int main(void){screen();const size_t ns[]={10079,12288,65536,262144};for(int z=0;z<4;z++){size_t n=ns[z];double f=bench(exp53_n2_vmstyle_u4_0381_frozen,n),c=bench(exp53_n2_cp64_blocked_attack,n),v=bench(vha,n);printf("CP64B_TIME n=%zu frozen=%.9f blocked=%.9f vml_ha=%.9f frozen_over_blocked=%.6f\n",n,f,c,v,f/c);}return sink==1234567.0;}
