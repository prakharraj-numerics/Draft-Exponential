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
extern "C" void exp53_attack_gather(double*,const double*,size_t);
extern "C" void exp53_attack_scalef(double*,const double*,size_t);
extern "C" void exp53_attack_full(double*,const double*,size_t);
extern "C" void exp53_attack_factored(double*,const double*,size_t);

using fn_t=void(*)(double*,const double*,size_t);

struct Aligned { double* p=nullptr; explicit Aligned(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double))||!p)std::exit(2);} ~Aligned(){std::free(p);} };
static inline uint64_t mix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static inline double u01(uint64_t h){return (double(h>>11)+0.5)*(1.0/9007199254740992.0);}
static void fill_mixed(double* x,size_t n){const double e=0x1p-20;const uint64_t s=0xd1b54a32d192ed03ULL^(uint64_t(n)<<17);for(size_t i=0;i<n;++i){double u=u01(mix64(s+i*0x9e3779b97f4a7c15ULL));unsigned k=i&3u;double v=k<2?e+u*(1-2*e):1+e+u*(99-2*e);if(k&1)v=-v;x[i]=v;}}
static inline uint64_t bits(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
static uint64_t maxulp(const double*a,const double*b,size_t n){uint64_t m=0;for(size_t i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX;uint64_t x=bits(a[i]),y=bits(b[i]);m=std::max(m,x>y?x-y:y-x);}return m;}
static uint64_t bitdiff(const double*a,const double*b,size_t n){uint64_t d=0;for(size_t i=0;i<n;++i)d+=(bits(a[i])!=bits(b[i]));return d;}
static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}

class FrozenStyleDispatcher {
public:
    explicit FrozenStyleDispatcher(fn_t fn):fn_const_(fn),generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0),fn_(nullptr){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
    ~FrozenStyleDispatcher(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}
    void run(double*out,const double*in,size_t n){run_with(fn_const_,out,in,n);}
private:
    void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;fn_t f=fn_;double*o=out_;const double*x=in_;size_t z=n2_;f(o,x,z);completed_.store(g,std::memory_order_release);}}
    void run_with(fn_t f,double*out,const double*in,size_t n){size_t full=n/32;if(full<2){f(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){f(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;fn_=f;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;f(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
    fn_t fn_const_;std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;double*out_;const double*in_;size_t n2_;fn_t fn_;
};

class AttackDispatcher {
public:
    AttackDispatcher():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
    ~AttackDispatcher(){stop_.store(true,std::memory_order_relaxed);generation_.store(++seq_,std::memory_order_release);if(helper_.joinable())helper_.join();}
    void run(double*out,const double*in,size_t n){size_t full=n/32;if(full<2){exp53_attack_full(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){exp53_attack_full(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;const uint64_t g=++seq_;generation_.store(g,std::memory_order_release);exp53_attack_full(out,in,split);while(completed_.load(std::memory_order_relaxed)!=g)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}
private:
    void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){while(generation_.load(std::memory_order_relaxed)==seen)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);uint64_t g=generation_.load(std::memory_order_relaxed);seen=g;if(stop_.load(std::memory_order_relaxed))return;double*o=out_;const double*x=in_;size_t z=n2_;exp53_attack_full(o,x,z);completed_.store(g,std::memory_order_release);}}
    std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;uint64_t seq_=0;double*out_;const double*in_;size_t n2_;
};

class AttackDispatcherCounted {
public:
    AttackDispatcherCounted():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0){pin(0);helper_thread_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
    ~AttackDispatcherCounted(){stop_.store(true,std::memory_order_relaxed);generation_.store(++seq_,std::memory_order_release);if(helper_thread_.joinable())helper_thread_.join();}
    void run(double*out,const double*in,size_t n){size_t full=n/32;if(full<2){exp53_attack_full(out,in,n);return;}size_t split=(full/2)*32;out_=out+split;in_=in+split;n2_=n-split;uint64_t g=++seq_;generation_.store(g,std::memory_order_release);exp53_attack_full(out,in,split);uint64_t s=0;while(completed_.load(std::memory_order_relaxed)!=g){_mm_pause();++s;}std::atomic_thread_fence(std::memory_order_acquire);caller_spins_+=s;++calls_;}
    uint64_t helper()const{return helper_spins_;}uint64_t caller()const{return caller_spins_;}uint64_t calls()const{return calls_;}
private:
    void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t s=0;while(generation_.load(std::memory_order_relaxed)==seen){_mm_pause();++s;}std::atomic_thread_fence(std::memory_order_acquire);helper_spins_+=s;uint64_t g=generation_.load(std::memory_order_relaxed);seen=g;if(stop_.load(std::memory_order_relaxed))return;exp53_attack_full(out_,in_,n2_);completed_.store(g,std::memory_order_release);}}
    std::thread helper_thread_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;uint64_t seq_=0;double*out_;const double*in_;size_t n2_;uint64_t helper_spins_=0,caller_spins_=0,calls_=0;
};

extern "C" __attribute__((noinline,used)) void exp53_attack_shape_generation_wait(const std::atomic<uint64_t>*p,uint64_t seen){while(p->load(std::memory_order_relaxed)==seen)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}
extern "C" __attribute__((noinline,used)) void exp53_attack_shape_completion_wait(const std::atomic<uint64_t>*p,uint64_t wanted){while(p->load(std::memory_order_relaxed)!=wanted)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}

static size_t calls_for(size_t n){size_t c=180000000ULL/n;c=std::max<size_t>(200,c);c=std::min<size_t>(50000,c);return c;}
template<class F> static double timeit(F&&f,size_t calls,size_t n){for(int i=0;i<24;++i)f();double best=1e300;for(int r=0;r<5;++r){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)f();auto b=std::chrono::steady_clock::now();double t=std::chrono::duration<double,std::nano>(b-a).count()/(double(calls)*n);best=std::min(best,t);}return best;}
static double timefn(fn_t f,double*out,const double*in,size_t n,size_t calls){return timeit([&]{f(out,in,n);},calls,n);}

int main(int argc,char**argv){if(argc!=2){std::cerr<<"usage: bench N\n";return 2;}size_t n=strtoull(argv[1],nullptr,10);const std::vector<size_t>ok={700,3500,15000,50000,1000000,2000000};if(std::find(ok.begin(),ok.end(),n)==ok.end())return 2;pin(0);size_t calls=calls_for(n);Aligned in(n),base(n),intel(n),g(n),s(n),full(n),fact(n);fill_mixed(in.p,n);vmdExp((MKL_INT)n,in.p,intel.p,VML_HA);exp53_n2_vmstyle_u4_0381_frozen(base.p,in.p,n);exp53_attack_gather(g.p,in.p,n);exp53_attack_scalef(s.p,in.p,n);exp53_attack_full(full.p,in.p,n);exp53_attack_factored(fact.p,in.p,n);
    uint64_t ug=maxulp(g.p,intel.p,n),us=maxulp(s.p,intel.p,n),uf=maxulp(full.p,intel.p,n),ux=maxulp(fact.p,intel.p,n);uint64_t dg=bitdiff(g.p,base.p,n),ds=bitdiff(s.p,base.p,n),df=bitdiff(full.p,base.p,n),dx=bitdiff(fact.p,base.p,n);if(ug>2||us>2||uf>2)return 4;
    double kb=timefn(exp53_n2_vmstyle_u4_0381_frozen,base.p,in.p,n,calls);double kg=timefn(exp53_attack_gather,g.p,in.p,n,calls);double ks=timefn(exp53_attack_scalef,s.p,in.p,n,calls);double kf=timefn(exp53_attack_full,full.p,in.p,n,calls);double kx=(ux<=2)?timefn(exp53_attack_factored,fact.p,in.p,n,calls):0.0;
    double prod,attack,intel_ns,legacy_full=0;if(n<=3000){prod=kb;attack=kf;}else{FrozenStyleDispatcher p(exp53_n2_vmstyle_u4_0381_frozen);prod=timeit([&]{p.run(base.p,in.p,n);},calls,n);FrozenStyleDispatcher oldfull(exp53_attack_full);legacy_full=timeit([&]{oldfull.run(full.p,in.p,n);},calls,n);AttackDispatcher a;attack=timeit([&]{a.run(full.p,in.p,n);},calls,n);}intel_ns=timeit([&]{vmdExp((MKL_INT)n,in.p,intel.p,VML_HA);},calls,n);
    double counted_ratio=1;uint64_t hs=0,cs=0,dc=0;if(n>3000){AttackDispatcherCounted c;for(int i=0;i<16;++i)c.run(full.p,in.p,n);uint64_t h0=c.helper(),c0=c.caller(),d0=c.calls();double ct=timeit([&]{c.run(full.p,in.p,n);},calls,n);hs=c.helper()-h0;cs=c.caller()-c0;dc=c.calls()-d0;counted_ratio=ct/attack;}
    std::cout.setf(std::ios::fixed);std::cout.precision(9);std::cout<<"ATTACK n="<<n<<" calls="<<calls<<" kernel_base="<<kb<<" kernel_g="<<kg<<" kernel_s="<<ks<<" kernel_full="<<kf<<" kernel_fact="<<kx<<" prod_wall="<<prod<<" legacy_full_wall="<<legacy_full<<" attack_wall="<<attack<<" intel_wall="<<intel_ns<<" speed_attack_vs_prod="<<prod/attack<<" speed_attack_vs_intel="<<intel_ns/attack<<" scheduler_new_vs_old="<<(legacy_full?legacy_full/attack:1.0)<<" g_bitdiff="<<dg<<" s_bitdiff="<<ds<<" full_bitdiff="<<df<<" fact_bitdiff="<<dx<<" g_maxulp="<<ug<<" s_maxulp="<<us<<" full_maxulp="<<uf<<" fact_maxulp="<<ux<<" helper_spins="<<hs<<" caller_spins="<<cs<<" dispatch_calls="<<dc<<" counted_perturb="<<counted_ratio<<"\n";return 0;}
