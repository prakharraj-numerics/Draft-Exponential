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

struct A{double*p=nullptr;explicit A(size_t n){if(posix_memalign((void**)&p,64,n*8)||!p)std::exit(2);}~A(){std::free(p);}};
static inline uint64_t mix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static inline double u01(uint64_t h){return(double(h>>11)+.5)*(1.0/9007199254740992.0);}static void fill(double*x,size_t n){const double e=0x1p-20;uint64_t s=0xd1b54a32d192ed03ULL^(uint64_t(n)<<17);for(size_t i=0;i<n;++i){double u=u01(mix64(s+i*0x9e3779b97f4a7c15ULL));unsigned k=i&3;double v=k<2?e+u*(1-2*e):1+e+u*(99-2*e);if(k&1)v=-v;x[i]=v;}}static inline uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}static uint64_t ulp(const double*a,const double*b,size_t n){uint64_t m=0;for(size_t i=0;i<n;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]<=0||b[i]<=0)return UINT64_MAX;uint64_t x=bits(a[i]),y=bits(b[i]);m=std::max(m,x>y?x-y:y-x);}return m;}static uint64_t diff(const double*a,const double*b,size_t n){uint64_t d=0;for(size_t i=0;i<n;++i)d+=bits(a[i])!=bits(b[i]);return d;}static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}static size_t calls_for(size_t n){size_t c=180000000ULL/n;c=std::max<size_t>(200,c);return std::min<size_t>(50000,c);}template<class F>static double tm(F&&f,size_t calls,size_t n){for(int i=0;i<24;++i)f();double b=1e300;for(int r=0;r<5;++r){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)f();auto z=std::chrono::steady_clock::now();b=std::min(b,std::chrono::duration<double,std::nano>(z-a).count()/(double(calls)*n));}return b;}

template<fn_t F>
class FrozenExactClone{
public:
 FrozenExactClone():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0),fn_(nullptr){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
 ~FrozenExactClone(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}
 void run(double*out,const double*in,size_t n){run_with(F,out,in,n);}
private:
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;fn_t f=fn_;double*o=out_;const double*x=in_;size_t z=n2_;f(o,x,z);completed_.store(g,std::memory_order_release);}}
 void run_with(fn_t f,double*out,const double*in,size_t n){if(!n)return;size_t full=n/32;if(full<2){f(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){f(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;fn_=f;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;f(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
 std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;double*out_;const double*in_;size_t n2_;fn_t fn_;
};

template<fn_t F>
class FrozenSpecialized{
public:
 FrozenSpecialized():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
 ~FrozenSpecialized(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}
 void run(double*out,const double*in,size_t n){if(!n)return;size_t full=n/32;if(full<2){F(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){F(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;F(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
private:
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;F(out_,in_,n2_);completed_.store(g,std::memory_order_release);}}
 std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;double*out_;const double*in_;size_t n2_;
};

class StoreSpecialized{
public:
 StoreSpecialized():generation_(0),completed_(0),stop_(false),out_(nullptr),in_(nullptr),n2_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
 ~StoreSpecialized(){stop_.store(true,std::memory_order_relaxed);generation_.store(++seq_,std::memory_order_release);if(helper_.joinable())helper_.join();}
 void run(double*out,const double*in,size_t n){size_t full=n/32;if(full<2){exp53_attack_full(out,in,n);return;}size_t split=(full/2)*32;if(!split||split>=n){exp53_attack_full(out,in,n);return;}out_=out+split;in_=in+split;n2_=n-split;uint64_t g=++seq_;generation_.store(g,std::memory_order_release);exp53_attack_full(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
private:
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;exp53_attack_full(out_,in_,n2_);completed_.store(g,std::memory_order_release);}}
 std::thread helper_;alignas(64)std::atomic<uint64_t>generation_,completed_;alignas(64)std::atomic<bool>ready_{false};std::atomic<bool>stop_;uint64_t seq_=0;double*out_;const double*in_;size_t n2_;
};

template<class D>static double dispatch_time(double*out,const double*in,size_t n,size_t c){D d;return tm([&]{d.run(out,in,n);},c,n);}static double kernel_time(fn_t f,double*out,const double*in,size_t n,size_t c){return tm([&]{f(out,in,n);},c,n);}
int main(int ac,char**av){if(ac!=2)return 2;size_t n=strtoull(av[1],0,10);std::vector<size_t>ok={700,3500,15000,50000,1000000,2000000};if(std::find(ok.begin(),ok.end(),n)==ok.end())return 2;pin(0);size_t c=calls_for(n);A in(n),base(n),ref(n),s(n),full(n),fact(n);fill(in.p,n);vmdExp((MKL_INT)n,in.p,ref.p,VML_HA);exp53_n2_vmstyle_u4_0381_frozen(base.p,in.p,n);exp53_attack_scalef(s.p,in.p,n);exp53_attack_full(full.p,in.p,n);exp53_attack_factored(fact.p,in.p,n);uint64_t us=ulp(s.p,ref.p,n),uf=ulp(full.p,ref.p,n),ux=ulp(fact.p,ref.p,n);uint64_t ds=diff(s.p,base.p,n),df=diff(full.p,base.p,n),dx=diff(fact.p,base.p,n);if(us>2||uf>2)return 4;
 double kb=kernel_time(exp53_n2_vmstyle_u4_0381_frozen,base.p,in.p,n,c),ks=kernel_time(exp53_attack_scalef,s.p,in.p,n,c),kf=kernel_time(exp53_attack_full,full.p,in.p,n,c),kx=ux<=2?kernel_time(exp53_attack_factored,fact.p,in.p,n,c):0;
 double db=0,dsched=0,dfull=0,dfact=0,spec=0,store=0;if(n>3000){db=dispatch_time<FrozenExactClone<exp53_n2_vmstyle_u4_0381_frozen>>(base.p,in.p,n,c);dsched=dispatch_time<FrozenExactClone<exp53_attack_scalef>>(s.p,in.p,n,c);dfull=dispatch_time<FrozenExactClone<exp53_attack_full>>(full.p,in.p,n,c);if(ux<=2)dfact=dispatch_time<FrozenExactClone<exp53_attack_factored>>(fact.p,in.p,n,c);spec=dispatch_time<FrozenSpecialized<exp53_attack_full>>(full.p,in.p,n,c);store=dispatch_time<StoreSpecialized>(full.p,in.p,n,c);}else{db=kb;dsched=ks;dfull=kf;dfact=kx;spec=kf;store=kf;}double intel=tm([&]{vmdExp((MKL_INT)n,in.p,ref.p,VML_HA);},c,n);
 std::cout.setf(std::ios::fixed);std::cout.precision(9);std::cout<<"ATTACK2 n="<<n<<" calls="<<c<<" kernel_base="<<kb<<" kernel_s="<<ks<<" kernel_full="<<kf<<" kernel_fact="<<kx<<" frozen_base="<<db<<" frozen_s="<<dsched<<" frozen_full="<<dfull<<" frozen_fact="<<dfact<<" specialized_full="<<spec<<" store_full="<<store<<" intel="<<intel<<" frozen_s_vs_base="<<db/dsched<<" frozen_full_vs_base="<<db/dfull<<" frozen_fact_vs_base="<<(dfact?db/dfact:0)<<" specialized_vs_frozen_full="<<dfull/spec<<" store_vs_frozen_full="<<dfull/store<<" speed_full_vs_intel="<<intel/dfull<<" speed_fact_vs_intel="<<(dfact?intel/dfact:0)<<" s_bitdiff="<<ds<<" full_bitdiff="<<df<<" fact_bitdiff="<<dx<<" s_maxulp="<<us<<" full_maxulp="<<uf<<" fact_maxulp="<<ux<<"\n";}
