#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <pthread.h>
#include <sched.h>
#include <mkl_vml.h>
#include "TaskScheduler.h"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*8))abort();}~B(){free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static void pin_cpu(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);} 
static void enki_thread_start(uint32_t threadnum){if(threadnum==1)pin_cpu(2);} 
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=180000/n;return std::max<size_t>(32,std::min<size_t>(1800,c));}
template<class F>static double tim(F f,size_t n,size_t c,int k=7){for(int i=0;i<5;i++)f();std::vector<double>v;v.reserve(k);for(int j=0;j<k;j++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);} 

struct ExpTask: enki::ITaskSet{
  double*out=nullptr; const double*in=nullptr;
  void ExecuteRange(enki::TaskSetPartition r,uint32_t) override { exp53_n2_vmstyle_u4_0381_frozen(out+r.start,in+r.start,(size_t)(r.end-r.start)); }
};

class EnkiExp53{
 public:
  EnkiExp53(){pin_cpu(0);enki::TaskSchedulerConfig cfg;cfg.numTaskThreadsToCreate=1;cfg.numExternalTaskThreads=0;cfg.profilerCallbacks.threadStart=enki_thread_start;ts.Initialize(cfg);} 
  ~EnkiExp53(){ts.WaitforAllAndShutdown();}
  void run(double*out,const double*in,size_t n,uint32_t grain){task.out=out;task.in=in;task.m_SetSize=(uint32_t)n;task.m_MinRange=grain;ts.AddTaskSetToPipe(&task);ts.WaitforTask(&task);} 
 private: enki::TaskScheduler ts; ExpTask task;
};

int main(){
  constexpr size_t M=2000;B in(M),ref(M),out(M);EnkiExp53 enki;
  const uint32_t grains[]={32,64,96,128,160,192,256,320,384,512,768,1024};
  std::vector<size_t> sizes;sizes.push_back(101);for(size_t n=150;n<=2000;n+=50)sizes.push_back(n);
  std::cout<<std::fixed<<std::setprecision(9)<<"ENKITS_EXP53 version=v1.12 sizes=101,150..2000 threads=2 cpu0+cpu2 grains=12 math=frozen bitwise_gate=required\n";
  for(int d=0;d<2;d++){
    const char*dn=d?"mid":"unit";
    for(size_t n:sizes){
      fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);size_t c=calls(n);double best=1e99;uint32_t bg=0;
      for(uint32_t g:grains){
        std::memset(out.p,0,n*8);enki.run(out.p,in.p,n,g);if(std::memcmp(out.p,ref.p,n*8)){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" grain="<<g<<" bitdiff=1\n";return 9;}
        auto ef=[&]{enki.run(out.p,in.p,n,g);};double t=tim(ef,n,c,7);std::cout<<"GRAIN domain="<<dn<<" n="<<n<<" grain="<<g<<" enki_ns="<<t<<" bitdiff=0\n";if(t<best){best=t;bg=g;}
      }
      auto ff=[&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};auto ii=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};double ft=tim(ff,n,c,7),it=tim(ii,n,c,7);
      std::cout<<"FINAL domain="<<dn<<" n="<<n<<" best_grain="<<bg<<" enki_ns="<<best<<" frozen_ns="<<ft<<" intel_ns="<<it<<" intel_over_enki="<<it/best<<" frozen_over_enki="<<ft/best<<" bitdiff=0\n";
    }
  }
}
