#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <mkl_vml.h>
extern "C" void exp53_highway_fixedtail_0101_2000_candidate(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void* exp53_hwy2_create();
extern "C" void exp53_hwy2_destroy(void*);
extern "C" void exp53_hwy2_run(void*,double*,const double*,size_t,int);
struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double ur(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x9e3779b97f4a7c15ULL^((uint64_t)n<<21)^((uint64_t)d<<59);for(size_t i=0;i<n;i++){double q=ur(sm(s+i*0x94d049bb133111ebULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return (u>>63)?~u:(u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}static size_t calls(size_t n){size_t c=180000/n;return std::max<size_t>(64,std::min<size_t>(1500,c));}
template<class F>static double tim(F f,size_t n,size_t c){for(int i=0;i<10;i++)f();std::vector<double>v;for(int r=0;r<11;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}
int main(){B in(2048),fro(2048),ser(2048),syn(2048),intelout(2048);void*ex=exp53_hwy2_create();const size_t sizes[]={101,125,150,160,192,200,224,250,256,300,320,384,400,448,500,512,600,640,750,768,900,960,1000,1024,1100,1152,1250,1280,1400,1440,1500,1536,1600,1664,1750,1792,1800,1850,1900,1920,1950,1984,2000};const int shares[]={20,25,30,35,40,45};std::cout<<std::fixed<<std::setprecision(9)<<"HWY_FIXEDTAIL_SYNC exact_formula_tail cpu0_cpu2\n";for(int d=0;d<2;d++){const char*dn=d?"mid":"unit";for(size_t n:sizes){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(fro.p,in.p,n);exp53_highway_fixedtail_0101_2000_candidate(ser.p,in.p,n);uint64_t maxu=0;size_t gt1=0,bulkdiff=0;for(size_t i=0;i<n;i++){double ref=std::exp(in.p[i]);uint64_t u=ulp(ser.p[i],ref);if(u>maxu)maxu=u;if(u>1)gt1++;if(i<(n/32)*32 && std::memcmp(ser.p+i,fro.p+i,8))bulkdiff++;}size_t c=calls(n);double sers=tim([&]{exp53_highway_fixedtail_0101_2000_candidate(ser.p,in.p,n);},n,c);double fros=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(fro.p,in.p,n);},n,c);double ints=tim([&]{vmdExp((MKL_INT)n,in.p,intelout.p,VML_HA);},n,c);double best=1e9;int bsh=0;for(int sh:shares){double t=tim([&]{exp53_hwy2_run(ex,syn.p,in.p,n,sh);},n,c);if(t<best){best=t;bsh=sh;}}std::cout<<"FINAL domain="<<dn<<" n="<<n<<" serial_hwy_ns="<<sers<<" sync_hwy_ns="<<best<<" best_share="<<bsh<<" frozen_ns="<<fros<<" intel_ns="<<ints<<" intel_over_serial="<<ints/sers<<" intel_over_sync="<<ints/best<<" maxulp_stdexp="<<maxu<<" gt1="<<gt1<<" bulk_bitdiff="<<bulkdiff<<"\n";}}exp53_hwy2_destroy(ex);}
