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
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*8))abort();}~B(){free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static void pin_cpu(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls(size_t n){size_t c=180000/n;return std::max<size_t>(32,std::min<size_t>(1800,c));}
template<class F>static double tim(F f,size_t n,size_t c,int k=7){for(int i=0;i<5;i++)f();std::vector<double>v;v.reserve(k);for(int j=0;j<k;j++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/c/n);}return med(v);}

class MarlExp53{
 public:
  MarlExp53():scheduler_(marl::Scheduler::Config().setWorkerThreadCount(1).setWorkerThreadInitializer([](int){pin_cpu(2);})) {pin_cpu(0);scheduler_.bind();}
  ~MarlExp53(){marl::Scheduler::unbind();}
  void run(double*out,const double*in,size_t n,uint32_t requested_tasks){
    size_t blocks=n/32,prefix=blocks*32,tail=n-prefix;
    if(!blocks){exp53_n2_vmstyle_u4_0381_frozen(out,in,n);return;}
    uint32_t tasks=std::min<uint32_t>(requested_tasks,(uint32_t)blocks);
    marl::WaitGroup wg(tasks);
    for(uint32_t t=0;t<tasks;t++){
      size_t b0=blocks*t/tasks,b1=blocks*(t+1)/tasks,off=b0*32,len=(b1-b0)*32;
      marl::schedule([=]{exp53_n2_vmstyle_u4_0381_frozen(out+off,in+off,len);wg.done();});
    }
    wg.wait();
    if(tail)exp53_n2_vmstyle_u4_0381_frozen(out+prefix,in+prefix,tail);
  }
 private:marl::Scheduler scheduler_;
};

int main(){constexpr size_t M=2000;B in(M),ref(M),out(M);MarlExp53 marl;const uint32_t counts[]={2,4,8};std::vector<size_t> sizes;sizes.push_back(101);for(size_t n=150;n<=2000;n+=50)sizes.push_back(n);std::cout<<std::fixed<<std::setprecision(9)<<"MARL_EXP53 sizes=101,150..2000 worker_threads=1 bound_main=1 cpu0+cpu2 task_counts=2,4,8 block=32 math=frozen bitwise_gate=required\n";for(int d=0;d<2;d++){const char*dn=d?"mid":"unit";for(size_t n:sizes){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);size_t c=calls(n);double best=1e99;uint32_t bt=0;for(uint32_t tc:counts){std::memset(out.p,0,n*8);marl.run(out.p,in.p,n,tc);if(std::memcmp(out.p,ref.p,n*8)){std::cout<<"FAIL domain="<<dn<<" n="<<n<<" tasks="<<tc<<" bitdiff=1\n";return 9;}auto mf=[&]{marl.run(out.p,in.p,n,tc);};double t=tim(mf,n,c,7);std::cout<<"TASKS domain="<<dn<<" n="<<n<<" tasks="<<tc<<" marl_ns="<<t<<" bitdiff=0\n";if(t<best){best=t;bt=tc;}}auto ff=[&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};auto ii=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};double ft=tim(ff,n,c,7),it=tim(ii,n,c,7);std::cout<<"FINAL domain="<<dn<<" n="<<n<<" best_tasks="<<bt<<" marl_ns="<<best<<" frozen_ns="<<ft<<" intel_ns="<<it<<" intel_over_marl="<<it/best<<" frozen_over_marl="<<ft/best<<" bitdiff=0\n";}}}
