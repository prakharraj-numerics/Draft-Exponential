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
void exp53_n2_cp56_attack(double*,const double*,size_t);
void exp53_n2_cp56_split52(double*,const double*,size_t);
void exp53_n2_cp56_split61(double*,const double*,size_t);
typedef void(*fn_t)(double*,const double*,size_t);
static volatile double sink;
static void*xalloc(size_t n){void*p=NULL;if(posix_memalign(&p,64,n)!=0||!p){perror("posix_memalign");exit(2);}return p;}
static uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static uint64_t ord(double x){uint64_t u=bits(x);return(u>>63)?~u:(u|0x8000000000000000ULL);}
static uint64_t ud(double a,double b){uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static uint64_t rng=0x94d049bb133111ebULL;
static uint64_t ru(void){uint64_t x=rng;x^=x>>12;x^=x<<25;x^=x>>27;rng=x;return x*2685821657736338717ULL;}
static double rd(void){double z=(double)(ru()>>11)*(1.0/9007199254740992.0);return -100.0+200.0*z;}
static void vha(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}
static void screen(const char*name,fn_t f,const double*in,const double*fr,size_t n){double*out=xalloc(n*8);f(out,in,n);uint64_t mx=0,g=0,bd=0;for(size_t i=0;i<n;i++){double ref=(double)expl((long double)in[i]);uint64_t d=ud(out[i],ref);if(d>mx)mx=d;if(d>1)g++;if(bits(out[i])!=bits(fr[i]))bd++;}printf("CPSPLIT_ACCURACY name=%s maxULP=%llu gt1=%llu bitdiff=%llu\n",name,(unsigned long long)mx,(unsigned long long)g,(unsigned long long)bd);free(out);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return(double)t.tv_sec*1e9+t.tv_nsec;}
static double bench(fn_t f,size_t n){double*in=xalloc(n*8),*out=xalloc(n*8);for(size_t i=0;i<n;i++)in[i]=rd();for(int w=0;w<20;w++)f(out,in,n);int calls=(int)(15000000ULL/n);if(calls<80)calls=80;if(calls>5000)calls=5000;double best=1e300;for(int q=0;q<13;q++){double a=now();for(int r=0;r<calls;r++)f(out,in,n);double b=now();double v=(b-a)/((double)calls*n);if(v<best)best=v;sink+=out[(q*149u)%n];}free(in);free(out);return best;}
int main(void){const size_t an=200000;double*in=xalloc(an*8),*fr=xalloc(an*8);for(size_t i=0;i<an;i++)in[i]=rd();exp53_n2_vmstyle_u4_0381_frozen(fr,in,an);screen("CP56",exp53_n2_cp56_attack,in,fr,an);screen("SPLIT52",exp53_n2_cp56_split52,in,fr,an);screen("SPLIT61",exp53_n2_cp56_split61,in,fr,an);free(in);free(fr);const size_t ns[]={10079,12288,65536,262144};for(int z=0;z<4;z++){size_t n=ns[z];double f=bench(exp53_n2_vmstyle_u4_0381_frozen,n),c=bench(exp53_n2_cp56_attack,n),a=bench(exp53_n2_cp56_split52,n),b=bench(exp53_n2_cp56_split61,n),v=bench(vha,n);printf("CPSPLIT_TIME n=%zu frozen=%.9f cp56=%.9f split52=%.9f split61=%.9f vml_ha=%.9f\n",n,f,c,a,b,v);}return sink==1234567.0;}
