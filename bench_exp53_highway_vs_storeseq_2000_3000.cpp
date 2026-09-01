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
#include "exp53_custom2_storeseq_2000_3000_frozen.hpp"

extern "C" void exp53_highway_fixedtail_0101_2000_candidate(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void* exp53_hwy2_create();
extern "C" void exp53_hwy2_destroy(void*);
extern "C" void exp53_hwy2_run(void*,double*,const double*,size_t,int);

struct B { double* p; B(size_t n){ if(posix_memalign((void**)&p,64,n*sizeof(double))) std::abort(); } ~B(){ free(p); } };
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static double ur(uint64_t h){return ((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<17)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=ur(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return (u>>63)?~u:(u|0x8000000000000000ULL);} static uint64_t ulp(double a,double b){uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=220000/n;return std::max<size_t>(64,std::min<size_t>(1200,c));}
template<class F> static double tim(F f,size_t n,size_t c){for(int i=0;i<12;i++)f();std::vector<double> v;for(int r=0;r<13;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

int main(){
  B in(3072), ref(3072), frozenout(3072), hwyout(3072), serialout(3072), intelout(3072);
  const size_t sizes[]={2000,2050,2100,2150,2200,2250,2300,2400,2500,2600,2700,2800,2850,2900,2950,3000};
  const int shares[]={30,32,34,35,36,38,40,41,42,44};
  std::cout<<std::fixed<<std::setprecision(9)<<"HWY_VS_STORESEQ_2K3K separate_helper_lifetimes cpu0_cpu2\n";
  for(int d=0;d<2;d++){
    const char*dn=d?"mid":"unit";
    for(size_t n:sizes){
      fill(in.p,n,d); size_t c=calls(n);
      exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);
      exp53_highway_fixedtail_0101_2000_candidate(serialout.p,in.p,n);
      uint64_t maxu=0; size_t gt1=0,bulkdiff=0;
      for(size_t i=0;i<n;i++){uint64_t u=ulp(serialout.p[i],std::exp(in.p[i]));if(u>maxu)maxu=u;if(u>1)gt1++;if(i<(n/32)*32 && std::memcmp(serialout.p+i,ref.p+i,8))bulkdiff++;}
      double serial_ns=tim([&]{exp53_highway_fixedtail_0101_2000_candidate(serialout.p,in.p,n);},n,c);
      double intel_ns=tim([&]{vmdExp((MKL_INT)n,in.p,intelout.p,VML_HA);},n,c);
      double store_ns;
      {
        Exp53Custom2StoreSeq2000_3000Frozen ex;
        store_ns=tim([&]{ex.run(frozenout.p,in.p,n);},n,c);
      }
      double hwy_ns=1e9; int best_share=0;
      {
        void* ex=exp53_hwy2_create();
        for(int sh:shares){double t=tim([&]{exp53_hwy2_run(ex,hwyout.p,in.p,n,sh);},n,c);if(t<hwy_ns){hwy_ns=t;best_share=sh;}}
        exp53_hwy2_destroy(ex);
      }
      std::cout<<"FINAL domain="<<dn<<" n="<<n
               <<" hwy_sync_ns="<<hwy_ns<<" hwy_share="<<best_share
               <<" storeseq_ns="<<store_ns<<" intel_ns="<<intel_ns<<" serial_hwy_ns="<<serial_ns
               <<" store_over_hwy="<<store_ns/hwy_ns<<" intel_over_hwy="<<intel_ns/hwy_ns
               <<" maxulp_stdexp="<<maxu<<" gt1="<<gt1<<" bulk_bitdiff="<<bulkdiff<<"\n";
    }
  }
}
