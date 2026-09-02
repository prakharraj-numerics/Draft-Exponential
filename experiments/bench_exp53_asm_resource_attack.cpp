#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <mkl_vml.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
extern "C" void exp53_attack_asm_gather_loop(double*,const double*,size_t);
using fn_t=void(*)(double*,const double*,size_t);
struct A{double*p=nullptr;explicit A(size_t n){if(posix_memalign((void**)&p,64,n*8)||!p)std::exit(2);}~A(){std::free(p);}};
static inline uint64_t mix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u01(uint64_t h){return(double(h>>11)+.5)*(1.0/9007199254740992.0);}static void fill(double*x,size_t n){const double e=0x1p-20;uint64_t s=0xd1b54a32d192ed03ULL^(uint64_t(n)<<17);for(size_t i=0;i<n;++i){double u=u01(mix64(s+i*0x9e3779b97f4a7c15ULL));unsigned k=i&3;double v=k<2?e+u*(1-2*e):1+e+u*(99-2*e);if(k&1)v=-v;x[i]=v;}}static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}static uint64_t ulp(const double*a,const double*b,size_t n){uint64_t m=0;for(size_t i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX;uint64_t x=bits(a[i]),y=bits(b[i]);m=std::max(m,x>y?x-y:y-x);}return m;}static uint64_t diff(const double*a,const double*b,size_t n){uint64_t d=0;for(size_t i=0;i<n;++i)d+=bits(a[i])!=bits(b[i]);return d;}static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}static size_t calls_for(size_t n){size_t c=180000000ULL/n;c=std::max<size_t>(200,c);return std::min<size_t>(50000,c);}template<class F>static double tm(F&&f,size_t calls,size_t n){for(int i=0;i<24;++i)f();double b=1e300;for(int r=0;r<5;++r){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)f();auto z=std::chrono::steady_clock::now();b=std::min(b,std::chrono::duration<double,std::nano>(z-a).count()/(double(calls)*n));}return b;}

static inline void wait_equal_asm(const std::atomic<uint64_t>*p,uint64_t seen){
    const uint64_t* q=reinterpret_cast<const uint64_t*>(p);
    __asm__ volatile("1:\n\tpause\n\tcmpq %[v],(%[p])\n\tje 1b\n\t"::[p]"r"(q),[v]"r"(seen):"cc","memory");
}
static inline void wait_not_equal_asm(const std::atomic<uint64_t>*p,uint64_t wanted){
    const uint64_t* q=reinterpret_cast<const uint64_t*>(p);
    __asm__ volatile("1:\n\tpause\n\tcmpq %[v],(%[p])\n\tjne 1b\n\t"::[p]"r"(q),[v]"r"(wanted):"cc","memory");
}
extern "C" __attribute__((noinline,used)) void exp53_shape_wait_equal_asm(const std::atomic<uint64_t>*p,uint64_t v){wait_equal_asm(p,v);}extern "C" __attribute__((noinline,used)) void exp53_shape_wait_not_equal_asm(const std::atomic<uint64_t>*p,uint64_t v){wait_not_equal_asm(p,v);}

template<fn_t F,bool ASMWAIT>
class Dispatcher{
public:Dispatcher():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0),fn_(nullptr){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}~Dispatcher(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}void run(double*out,const double*in,size_t n){size_t full=n/32;if(full<2){F(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){F(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;fn_=F;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;F(out,in,split);if constexpr(ASMWAIT)wait_not_equal_asm(&completed_,g);else while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
private:void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;if constexpr(ASMWAIT){wait_equal_asm(&generation_,seen);g=generation_.load(std::memory_order_relaxed);std::atomic_thread_fence(std::memory_order_acquire);}else{while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();}seen=g;if(stop_.load(std::memory_order_relaxed))return;fn_t f=fn_;double*o=out_;const double*x=in_;size_t z=n2_;f(o,x,z);completed_.store(g,std::memory_order_release);}}
std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;double*out_;const double*in_;size_t n2_;fn_t fn_;};

template<class D>static double dt(double*out,const double*in,size_t n,size_t c){D d;return tm([&]{d.run(out,in,n);},c,n);}static double kt(fn_t f,double*out,const double*in,size_t n,size_t c){return tm([&]{f(out,in,n);},c,n);}
int main(int ac,char**av){if(ac!=2)return 2;size_t n=strtoull(av[1],0,10);std::vector<size_t>ok={3500,15000,50000,1000000,2000000};if(std::find(ok.begin(),ok.end(),n)==ok.end())return 2;pin(0);size_t c=calls_for(n);A in(n),base(n),att(n),ref(n);fill(in.p,n);vmdExp((MKL_INT)n,in.p,ref.p,VML_HA);exp53_n2_vmstyle_u4_0381_frozen(base.p,in.p,n);exp53_attack_asm_gather_loop(att.p,in.p,n);uint64_t u=ulp(att.p,ref.p,n),d=diff(att.p,base.p,n);if(u>2||d)return 4;double kb=kt(exp53_n2_vmstyle_u4_0381_frozen,base.p,in.p,n,c),ka=kt(exp53_attack_asm_gather_loop,att.p,in.p,n,c);double frozen_base=dt<Dispatcher<exp53_n2_vmstyle_u4_0381_frozen,false>>(base.p,in.p,n,c);double frozen_att=dt<Dispatcher<exp53_attack_asm_gather_loop,false>>(att.p,in.p,n,c);double asmwait_att=dt<Dispatcher<exp53_attack_asm_gather_loop,true>>(att.p,in.p,n,c);double intel=tm([&]{vmdExp((MKL_INT)n,in.p,ref.p,VML_HA);},c,n);std::cout.setf(std::ios::fixed);std::cout.precision(9);std::cout<<"ASMATTACK n="<<n<<" kernel_base="<<kb<<" kernel_attack="<<ka<<" frozen_base="<<frozen_base<<" frozen_attack="<<frozen_att<<" asmwait_attack="<<asmwait_att<<" intel="<<intel<<" attack_vs_base="<<frozen_base/frozen_att<<" asmwait_vs_frozen_attack="<<frozen_att/asmwait_att<<" speed_attack_vs_intel="<<intel/frozen_att<<" speed_asmwait_vs_intel="<<intel/asmwait_att<<" bitdiff="<<d<<" maxulp="<<u<<"\n";}
