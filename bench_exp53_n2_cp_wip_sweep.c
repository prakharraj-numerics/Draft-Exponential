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
void exp53_n2_cp40_attack(double*,const double*,size_t);
void exp53_n2_cp48_attack(double*,const double*,size_t);
void exp53_n2_cp56_attack(double*,const double*,size_t);
void exp53_n2_cp64_attack(double*,const double*,size_t);

typedef void (*fn_t)(double*,const double*,size_t);
static volatile double sink;

static void *xalloc(size_t bytes){ void *p=NULL; if(posix_memalign(&p,64,bytes)!=0||!p){perror("posix_memalign");exit(2);} return p; }
static uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
static uint64_t ordered(double x){uint64_t u=bits(x);return (u>>63)?~u:(u|0x8000000000000000ULL);}
static uint64_t ulpdiff(double a,double b){uint64_t x=ordered(a),y=ordered(b);return x>y?x-y:y-x;}
static uint64_t rng=0xd1b54a32d192ed03ULL;
static uint64_t nextu(void){uint64_t x=rng;x^=x>>12;x^=x<<25;x^=x>>27;rng=x;return x*2685821657736338717ULL;}
static double rnd_domain(void){double z=(double)(nextu()>>11)*(1.0/9007199254740992.0);return -100.0+200.0*z;}
static void intel_ha(double*out,const double*in,size_t n){vmdExp((MKL_INT)n,in,out,VML_HA);}

static void screen_one(const char*name,fn_t f,const double*in,const double*frozen,size_t n){
    double *out=xalloc(n*sizeof(double)); f(out,in,n); uint64_t mx=0,gt1=0,bd=0;
    for(size_t i=0;i<n;i++){double ref=(double)expl((long double)in[i]);uint64_t d=ulpdiff(out[i],ref);if(d>mx)mx=d;if(d>1)gt1++;if(bits(out[i])!=bits(frozen[i]))bd++;}
    printf("CPWIP_ACCURACY name=%s n=%zu maxULP=%llu gt1=%llu bitdiff_vs_frozen=%llu\n",name,n,(unsigned long long)mx,(unsigned long long)gt1,(unsigned long long)bd);free(out);
}
static double nsnow(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (double)t.tv_sec*1e9+(double)t.tv_nsec;}
static double bench(fn_t f,size_t n){
    double*in=xalloc(n*sizeof(double)),*out=xalloc(n*sizeof(double));for(size_t i=0;i<n;i++)in[i]=rnd_domain();for(int w=0;w<20;w++)f(out,in,n);
    int calls=(int)(14000000ULL/n);if(calls<80)calls=80;if(calls>5000)calls=5000;double best=1e300;
    for(int trial=0;trial<13;trial++){double t0=nsnow();for(int r=0;r<calls;r++)f(out,in,n);double t1=nsnow();double v=(t1-t0)/((double)calls*n);if(v<best)best=v;sink+=out[(trial*137u)%n];}
    free(in);free(out);return best;
}
int main(void){
    const size_t an=200000;double*ain=xalloc(an*sizeof(double)),*fro=xalloc(an*sizeof(double));for(size_t i=0;i<an;i++)ain[i]=rnd_domain();exp53_n2_vmstyle_u4_0381_frozen(fro,ain,an);
    screen_one("CP40",exp53_n2_cp40_attack,ain,fro,an);screen_one("CP48",exp53_n2_cp48_attack,ain,fro,an);screen_one("CP56",exp53_n2_cp56_attack,ain,fro,an);screen_one("CP64",exp53_n2_cp64_attack,ain,fro,an);free(ain);free(fro);
    const size_t ns[]={10079,12288,65536,262144};
    for(size_t z=0;z<4;z++){size_t n=ns[z];double f=bench(exp53_n2_vmstyle_u4_0381_frozen,n),a=bench(exp53_n2_cp40_attack,n),b=bench(exp53_n2_cp48_attack,n),c=bench(exp53_n2_cp56_attack,n),d=bench(exp53_n2_cp64_attack,n),v=bench(intel_ha,n);
      printf("CPWIP_TIME n=%zu frozen=%.9f cp40=%.9f cp48=%.9f cp56=%.9f cp64=%.9f vml_ha=%.9f\n",n,f,a,b,c,d,v);
    }
    return sink==1234567.0;
}
