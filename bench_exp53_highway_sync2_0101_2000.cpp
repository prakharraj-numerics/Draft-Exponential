#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>

void exp53_highway_0101_2000_candidate(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
#include "exp53_highway_sync2_0101_2000_candidate.hpp"

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*8))abort();}~B(){free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=220000/n;return std::max<size_t>(64,std::min<size_t>(2200,c));}
template<class F>static double tim(F f,size_t n,size_t c,int k=11){for(int i=0;i<12;i++)f();std::vector<double>v;v.reserve(k);for(int j=0;j<k;j++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

int main(){
  constexpr size_t M=2000; B in(M),ref(M),out(M);
  const size_t sizes[]={101,150,250,500,750,1000,1250,1500,1750,2000};
  const unsigned shares[]={20,25,30,35,40,45,50};
  Exp53HighwaySync2Candidate sync2;
  std::cout<<std::fixed<<std::setprecision(9)
           <<"HIGHWAY_SYNC2_EXP53 exact_bitwise=required cpus=0+2 permanent_helper=1 shares=20,25,30,35,40,45,50\n";

  for(int d=0;d<2;d++){
    const char*dn=d?"mid":"unit";
    for(size_t n:sizes){
      fill(in.p,n,d); exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);
      const size_t c=calls(n);

      double best=1e300; unsigned best_share=0;
      for(unsigned sh:shares){
        std::memset(out.p,0,n*8); sync2.run(out.p,in.p,n,sh);
        if(std::memcmp(out.p,ref.p,n*8)){
          std::cout<<"FAIL domain="<<dn<<" n="<<n<<" share="<<sh<<" bitdiff=1\n"; return 9;
        }
        const double t=tim([&]{sync2.run(out.p,in.p,n,sh);},n,c);
        std::cout<<"TRY domain="<<dn<<" n="<<n<<" share="<<sh<<" sync2_ns="<<t<<" bitdiff=0\n";
        if(t<best){best=t;best_share=sh;}
      }

      std::memset(out.p,0,n*8); exp53_highway_0101_2000_candidate(out.p,in.p,n);
      if(std::memcmp(out.p,ref.p,n*8)){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" serial_highway bitdiff=1\n";return 9;}
      const double serial=tim([&]{exp53_highway_0101_2000_candidate(out.p,in.p,n);},n,c);
      const double frozen=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);
      const double intel=tim([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);
      std::cout<<"FINAL domain="<<dn<<" n="<<n<<" best_share="<<best_share
               <<" sync2_ns="<<best<<" serial_highway_ns="<<serial
               <<" frozen_ns="<<frozen<<" intel_ns="<<intel
               <<" intel_over_sync2="<<intel/best
               <<" serial_over_sync2="<<serial/best
               <<" frozen_over_sync2="<<frozen/best<<" bitdiff=0\n";
    }
  }
}
