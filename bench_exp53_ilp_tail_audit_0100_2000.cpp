#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
extern "C" void exp53_u4_vectail_candidate(double*,const double*,size_t);
extern "C" void exp53_u6_vectail_candidate(double*,const double*,size_t);
extern "C" void exp53_u8_vectail_candidate(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}static size_t calls(size_t n){size_t c=180000/n;return std::max<size_t>(64,std::min<size_t>(1600,c));}
template<class F>static double tim(F f,size_t n,size_t c){for(int i=0;i<10;i++)f();std::vector<double>v;for(int r=0;r<11;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}
static size_t diffs(const double*a,const double*b,size_t n,size_t&fb){size_t c=0;fb=n;for(size_t i=0;i<n;i++)if(std::memcmp(a+i,b+i,8)){if(fb==n)fb=i;c++;}return c;}
int main(){B in(2000),ref(2000),a(2000),b(2000),c(2000);const size_t sizes[]={100,101,125,128,150,160,192,200,224,250,256,300,320,384,400,448,500,512,600,640,750,768,900,960,1000,1024,1100,1152,1250,1280,1400,1440,1500,1536,1600,1664,1750,1792,1800,1850,1900,1920,1950,1984,2000};std::cout<<std::fixed<<std::setprecision(9)<<"ILP_TAIL_AUDIT cpu=0 exact_bitwise=required\n";for(int d=0;d<2;d++){const char*dn=d?"mid":"unit";for(size_t n:sizes){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);exp53_u4_vectail_candidate(a.p,in.p,n);exp53_u6_vectail_candidate(b.p,in.p,n);exp53_u8_vectail_candidate(c.p,in.p,n);size_t f4,f6,f8;size_t d4=diffs(a.p,ref.p,n,f4),d6=diffs(b.p,ref.p,n,f6),d8=diffs(c.p,ref.p,n,f8);size_t cc=calls(n);double t4=tim([&]{exp53_u4_vectail_candidate(a.p,in.p,n);},n,cc);double t6=tim([&]{exp53_u6_vectail_candidate(b.p,in.p,n);},n,cc);double t8=tim([&]{exp53_u8_vectail_candidate(c.p,in.p,n);},n,cc);double tf=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(a.p,in.p,n);},n,cc);double ti=tim([&]{vmdExp((MKL_INT)n,in.p,a.p,VML_HA);},n,cc);std::cout<<"FINAL domain="<<dn<<" n="<<n<<" u4vt_ns="<<t4<<" u6vt_ns="<<t6<<" u8vt_ns="<<t8<<" frozen_ns="<<tf<<" intel_ns="<<ti<<" d4="<<d4<<" d6="<<d6<<" d8="<<d8; if(d4)std::cout<<" f4="<<f4; if(d6)std::cout<<" f6="<<f6; if(d8)std::cout<<" f8="<<f8; std::cout<<"\n";}}}
