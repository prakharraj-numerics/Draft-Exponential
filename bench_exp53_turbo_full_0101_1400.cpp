#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
extern "C" void exp53_turbo_full_u2_0101_1400(double*,const double*,size_t);
extern "C" void exp53_turbo_full_u4_0101_1400(double*,const double*,size_t);
extern "C" void exp53_turbo_full_u6_0101_1400(double*,const double*,size_t);
extern "C" void exp53_turbo_full_u8_0101_1400(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double ur(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=ur(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=300000/n;return std::max<size_t>(128,std::min<size_t>(2600,c));}
template<class F>static double tim(F f,size_t n,size_t c){for(int i=0;i<12;i++)f();std::vector<double>v;for(int r=0;r<13;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}
int main(){const size_t sizes[]={101,125,150,192,200,256,320,384,512,640,768,896,1024,1152,1200,1280,1344,1399,1400};using F=void(*)(double*,const double*,size_t);const char*nm[]={"u2","u4","u6","u8"};F fn[]={exp53_turbo_full_u2_0101_1400,exp53_turbo_full_u4_0101_1400,exp53_turbo_full_u6_0101_1400,exp53_turbo_full_u8_0101_1400};B in(1408),out(1408),intel(1408),frozen(1408);std::cout<<std::fixed<<std::setprecision(9)<<"TURBO_FULL_101_1400 pinned=9b212dedfaa4d4e08a854776e1ddaca7d746444c formula_tail=1\n";for(int d=0;d<2;d++){const char*dn=d?"mid":"unit";for(size_t n:sizes){fill(in.p,n,d);size_t c=calls(n);double best=1e9;int bi=-1;uint64_t maxu=0;size_t gt1=0;for(int k=0;k<4;k++){fn[k](out.p,in.p,n);uint64_t mu=0;size_t g=0;for(size_t i=0;i<n;i++){auto u=ulp(out.p[i],std::exp(in.p[i]));mu=std::max(mu,u);if(u>1)g++;}maxu=std::max(maxu,mu);gt1+=g;double t=tim([&]{fn[k](out.p,in.p,n);},n,c);if(t<best){best=t;bi=k;}}double fr=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(frozen.p,in.p,n);},n,c);double it=tim([&]{vmdExp((MKL_INT)n,in.p,intel.p,VML_HA);},n,c);std::cout<<"FINAL domain="<<dn<<" n="<<n<<" turbo_ns="<<best<<" turbo_variant="<<nm[bi]<<" frozen_ns="<<fr<<" intel_ns="<<it<<" intel_over_turbo="<<it/best<<" frozen_over_turbo="<<fr/best<<" maxulp="<<maxu<<" gt1="<<gt1<<"\n";}}}
