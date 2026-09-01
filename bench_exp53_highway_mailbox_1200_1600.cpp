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
#include "exp53_highway_mailbox_1200_1600_candidate.hpp"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void* exp53_hwy2_create();
extern "C" void exp53_hwy2_destroy(void*);
extern "C" void exp53_hwy2_run(void*,double*,const double*,size_t,int);
extern "C" void exp53_highway_fixedtail_0101_2000_candidate(double*,const double*,size_t);

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double ur(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x6a09e667f3bcc909ULL^((uint64_t)n<<23)^((uint64_t)d<<59);for(size_t i=0;i<n;i++){double q=ur(sm(s+i*0x94d049bb133111ebULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return (u>>63)?~u:(u|0x8000000000000000ULL);}static uint64_t ulp(double a,double b){uint64_t x=ord(a),y=ord(b);return x>y?x-y:y-x;}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=240000/n;return std::max<size_t>(96,std::min<size_t>(1200,c));}
template<class F>static double tim(F f,size_t n,size_t c){for(int i=0;i<12;i++)f();std::vector<double>v;for(int r=0;r<13;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

int main(){
  const size_t sizes[]={1200,1216,1248,1250,1280,1312,1344,1376,1400,1408,1440,1472,1500,1504,1536,1568,1599,1600};
  const int shares[]={25,27,28,29,30,31,32,33,34,35};
  B in(1664),ref(1664),mail(1664),oldsync(1664),ser(1664),intel(1664);
  std::cout<<std::fixed<<std::setprecision(9)<<"HWY_MAILBOX_1200_1600 exact_formula cpu0_cpu2\n";
  for(int d=0;d<2;d++){
    const char*dn=d?"mid":"unit";
    for(size_t n:sizes){
      fill(in.p,n,d);
      exp53_highway_fixedtail_0101_2000_candidate(ref.p,in.p,n);
      uint64_t maxu=0; size_t gt1=0,bitdiff=0;
      {
        Exp53HighwayMailbox1200_1600Candidate ex;
        ex.run(mail.p,in.p,n);
        for(size_t i=0;i<n;i++){uint64_t u=ulp(mail.p[i],std::exp(in.p[i])); if(u>maxu)maxu=u; if(u>1)gt1++; if(std::memcmp(mail.p+i,ref.p+i,8))bitdiff++;}
      }
      size_t c=calls(n);
      double mail_ns;
      {
        Exp53HighwayMailbox1200_1600Candidate ex;
        mail_ns=tim([&]{ex.run(mail.p,in.p,n);},n,c);
      }
      double oldbest=1e9; int oldshare=0;
      {
        void* ex=exp53_hwy2_create();
        for(int sh:shares){double t=tim([&]{exp53_hwy2_run(ex,oldsync.p,in.p,n,sh);},n,c); if(t<oldbest){oldbest=t;oldshare=sh;}}
        exp53_hwy2_destroy(ex);
      }
      double serial_ns=tim([&]{exp53_highway_fixedtail_0101_2000_candidate(ser.p,in.p,n);},n,c);
      double frozen_ns=tim([&]{exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);},n,c);
      double intel_ns=tim([&]{vmdExp((MKL_INT)n,in.p,intel.p,VML_HA);},n,c);
      std::cout<<"FINAL domain="<<dn<<" n="<<n
               <<" mailbox_ns="<<mail_ns
               <<" oldsync_ns="<<oldbest<<" oldshare="<<oldshare
               <<" serial_hwy_ns="<<serial_ns
               <<" frozen_ns="<<frozen_ns
               <<" intel_ns="<<intel_ns
               <<" intel_over_mailbox="<<intel_ns/mail_ns
               <<" old_over_mailbox="<<oldbest/mail_ns
               <<" helper_blocks="<<Exp53HighwayMailbox1200_1600Candidate::helper_blocks_for(n)
               <<" maxulp_stdexp="<<maxu<<" gt1="<<gt1<<" bitdiff_serial_hwy="<<bitdiff<<"\n";
    }
  }
}
