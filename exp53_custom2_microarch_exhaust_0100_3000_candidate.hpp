#pragma once
/* EXPERIMENTAL ONLY. Exhausts low-overhead two-core handshake variants.
   Frozen production/kernel files remain untouched. */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

class Exp53Custom2MicroarchExhaustCandidate {
private:
 struct alignas(128) IsoAtomic{std::atomic<uint64_t> v;char pad[120];IsoAtomic():v(0),pad{0}{}};
public:
 enum Mode : unsigned { BASE=0, STORESEQ=1, RELAXPOLL=2, ISO128=3, PAUSE2=4, PAUSE4=5, PAUSE8=6, COMBINED=7 };
 Exp53Custom2MicroarchExhaustCandidate():gen64_(0),done64_(0),stop_(false),ready_(false),out2_(nullptr),in2_(nullptr),n2_(0),mode_(BASE),local_gen_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
 ~Exp53Custom2MicroarchExhaustCandidate(){stop_.store(true,std::memory_order_relaxed);uint64_t g=++local_gen_;gen64_.store(g,std::memory_order_release);gen128_.v.store(g,std::memory_order_release);if(helper_.joinable())helper_.join();}
 static unsigned schedule(size_t n){if(n<800)return 0;if(n<1200)return 20;if(n<1600)return 25;if(n<2150)return 34;if(n<2850)return 38;return 41;}
 void run(double*out,const double*in,size_t n,Mode mode,unsigned pct=999){if(!n)return;unsigned hp=pct==999?schedule(n):pct;if(!hp){exp53_n2_vmstyle_u4_0381_frozen(out,in,n);return;}size_t h=n*(size_t)hp/100;size_t split=n-h;split=((split+31)/32)*32;if(split<32||split>=n||n-split<32){exp53_n2_vmstyle_u4_0381_frozen(out,in,n);return;}out2_=out+split;in2_=in+split;n2_=n-split;mode_.store((unsigned)mode,std::memory_order_relaxed);uint64_t g;if(mode==BASE||mode==RELAXPOLL||mode==PAUSE2||mode==PAUSE4||mode==PAUSE8){g=gen64_.fetch_add(1,std::memory_order_release)+1;local_gen_=g;}else{g=++local_gen_;if(mode==ISO128||mode==COMBINED)gen128_.v.store(g,std::memory_order_release);else gen64_.store(g,std::memory_order_release);}exp53_n2_vmstyle_u4_0381_frozen(out,in,split);wait_done(g,mode);}
private:
 static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
 static inline void pauses(unsigned k){for(unsigned i=0;i<k;++i)_mm_pause();}
 void wait_done(uint64_t g,Mode m){if(m==ISO128||m==COMBINED){if(m==COMBINED){while(done128_.v.load(std::memory_order_relaxed)!=g)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}else while(done128_.v.load(std::memory_order_acquire)!=g)_mm_pause();return;}unsigned p=(m==PAUSE2?2:m==PAUSE4?4:m==PAUSE8?8:1);if(m==RELAXPOLL){while(done64_.load(std::memory_order_relaxed)!=g)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}else while(done64_.load(std::memory_order_acquire)!=g)pauses(p);}
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t s64=gen64_.load(std::memory_order_relaxed),s128=gen128_.v.load(std::memory_order_relaxed);for(;;){unsigned m=mode_.load(std::memory_order_relaxed);uint64_t g;if(m==ISO128||m==COMBINED){if(m==COMBINED){while((g=gen128_.v.load(std::memory_order_relaxed))==s128){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}std::atomic_thread_fence(std::memory_order_acquire);}else while((g=gen128_.v.load(std::memory_order_acquire))==s128){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}s128=g;}else{if(m==RELAXPOLL){while((g=gen64_.load(std::memory_order_relaxed))==s64){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}std::atomic_thread_fence(std::memory_order_acquire);}else while((g=gen64_.load(std::memory_order_acquire))==s64){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}s64=g;}if(stop_.load(std::memory_order_relaxed))return;exp53_n2_vmstyle_u4_0381_frozen(out2_,in2_,n2_);if(m==ISO128||m==COMBINED)done128_.v.store(g,std::memory_order_release);else done64_.store(g,std::memory_order_release);}}
 std::thread helper_;alignas(64)std::atomic<uint64_t>gen64_,done64_;IsoAtomic gen128_,done128_;alignas(128)std::atomic<bool>stop_,ready_;alignas(128)double*out2_;const double*in2_;size_t n2_;std::atomic<unsigned>mode_;uint64_t local_gen_;
};
