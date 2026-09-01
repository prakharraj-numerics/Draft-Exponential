#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_highway_u2z_0100_2000_candidate(double*,const double*,size_t);
struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}static size_t calls(size_t n){size_t c=300000/n;return std::max<size_t>(96,std::min<size_t>(3000,c));}
template<class F>static double tim(F f,size_t n,size_t c){for(int i=0;i<16;i++)f();std::vector<double>v;for(int r=0;r<13;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}
int main(){B in(2000),ref(2000),out(2000);const size_t sizes[]={100,125,150,200,250,300,400,500,600,750,900,1000,1100,1250,1400,1500,1600,1700,1750,1800,1850,1900,1950,2000};std::cout<<std::fixed<<std::setprecision(9)<<"HIGHWAY_U2Z_SERIAL exact_bitwise=required cpu=0 granularity=16\n";for(int d=0;d<2;d++){const char*dn=d?"mid":"unit";for(size_t n:sizes){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);exp53_highway_u2z_0100_2000_candidate(out.p,in.p,n);if(std::memcmp(out.p,ref.p,n*sizeof(double))){size_t bad=0;while(bad<n&&std::memcmp(out.p+bad,ref.p+bad,sizeof(double))==0)++bad;std::cout<<"FAIL domain="<<dn<<" n="<<n<<" index="<<bad<<" bitdiff=1\n";return 9;}size_t c=calls(n);double ser=tim([&]{exp53_highway_u2z_0100_2000_candidate(out.p,in.p,n);},n,c);double fro=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);double intel=tim([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);std::cout<<"FINAL domain="<<dn<<" n="<<n<<" highway_u2z_ns="<<ser<<" frozen_ns="<<fro<<" intel_ns="<<intel<<" intel_over_highway_u2z="<<intel/ser<<" frozen_over_highway_u2z="<<fro/ser<<" bitdiff=0\n";}}return 0;}
