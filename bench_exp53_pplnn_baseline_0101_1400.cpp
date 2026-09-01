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

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
struct B { double* p; B(size_t n){ if(posix_memalign((void**)&p,64,n*sizeof(double))) std::abort(); } ~B(){ free(p); } };
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static double ur(uint64_t h){return ((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=ur(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=350000/n;return std::max<size_t>(128,std::min<size_t>(3000,c));}
template<class F> static double tim(F f,size_t n,size_t c){for(int i=0;i<20;i++)f();std::vector<double>v;for(int r=0;r<15;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}
int main(){
 const size_t sizes[]={101,125,150,192,200,256,320,384,512,640,768,896,1024,1152,1200,1280,1344,1399,1400};
 B in(1408),out(1408),intel(1408);
 std::cout<<std::fixed<<std::setprecision(9);
 for(int d=0;d<2;d++) for(size_t n:sizes){fill(in.p,n,d);auto c=calls(n);double fr=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);double it=tim([&]{vmdExp((MKL_INT)n,in.p,intel.p,VML_HA);},n,c);std::cout<<"BASE domain="<<(d?"mid":"unit")<<" n="<<n<<" frozen_ns="<<fr<<" intel_ns="<<it<<"\n";}
}
