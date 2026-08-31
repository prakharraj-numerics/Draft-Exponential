#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>

extern "C" void exp53_nvcxx(double*, const double*, size_t);
#define restrict __restrict__
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
#undef restrict

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*8))abort();}~B(){free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=260000/n;return std::max<size_t>(64,std::min<size_t>(2600,c));}
template<class G>static double tim(G g,size_t n,size_t c){for(int i=0;i<10;i++)g();std::vector<double>v;for(int r=0;r<11;r++){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<c;k++)g();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

int main(){
  const size_t sizes[]={101,150,250,500,750,1000,1250,1500,1750,2000};
  B in(2000),ref(2000),out(2000);
  std::cout<<std::fixed<<std::setprecision(9)<<"NVCXX_EXP53 exact_bitwise=required sizes=101,150,250,500,750,1000,1250,1500,1750,2000\n";
  for(int d=0;d<2;d++) for(size_t n:sizes){
    fill(in.p,n,d); exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);
    exp53_nvcxx(out.p,in.p,n);
    if(std::memcmp(out.p,ref.p,n*8)){std::cout<<"FAIL domain="<<(d?"mid":"unit")<<" n="<<n<<" bitdiff=1\n";return 9;}
    size_t c=calls(n);
    double nt=tim([&]{exp53_nvcxx(out.p,in.p,n);},n,c);
    double ft=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);
    double it=tim([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);
    std::cout<<"FINAL domain="<<(d?"mid":"unit")<<" n="<<n<<" nvcxx_ns="<<nt<<" frozen_ns="<<ft<<" intel_ns="<<it<<" intel_over_nvcxx="<<it/nt<<" frozen_over_nvcxx="<<ft/nt<<" bitdiff=0\n";
  }
}
