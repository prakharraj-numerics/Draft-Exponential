#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
#include "exp53_fastflow_batch.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*8))abort();}~B(){free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static inline double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=240000/n;return std::max<size_t>(80,std::min<size_t>(2400,c));}
template<class F>static double tim(F f,size_t n,size_t c,int k=11){for(int i=0;i<12;i++)f();std::vector<double>v;v.reserve(k);for(int j=0;j<k;j++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

int main(){
  constexpr size_t M=2000; B in(M),ref(M),out(M);
  const size_t sizes[]={100,125,150,200,250,300,400,500,600,750,900,1000,1100,1250,1400,1500,1600,1700,1750,1800,1850,1900,1950,2000};
  Exp53FastFlowExecutor ff(2, exp53_n2_vmstyle_u4_0381_frozen);
  std::cout<<std::fixed<<std::setprecision(9)
           <<"FASTFLOW_SYNC_EXP53 exact_bitwise=required cpuset=0,2 persistent_parallelfor=1 spin_workers=1 spin_barrier=1 workers=2 static_32aligned=1\n";
  for(int d=0;d<2;d++){
    const char*dn=d?"mid":"unit";
    for(size_t n:sizes){
      fill(in.p,n,d); exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);
      std::memset(out.p,0,n*8); ff.run_static(out.p,in.p,n,2);
      if(std::memcmp(out.p,ref.p,n*8)){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" bitdiff=1\n";return 9;}
      const size_t c=calls(n);
      const double fft=tim([&]{ff.run_static(out.p,in.p,n,2);},n,c);
      const double serial=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);
      const double intel=tim([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);
      const bool useff=fft<serial;
      const double effective=useff?fft:serial;
      std::cout<<"FINAL domain="<<dn<<" n="<<n
               <<" ff_ns="<<fft<<" serial_ns="<<serial<<" effective_ns="<<effective
               <<" effective_mode="<<(useff?"ff2":"serial")
               <<" intel_ns="<<intel<<" intel_over_effective="<<intel/effective
               <<" serial_over_ff="<<serial/fft<<" bitdiff=0\n";
    }
  }
}
