#pragma once
/* EXPERIMENTAL ONLY. Exhausts low-overhead two-core handshake variants.
   Each instance has one immutable mode, so the helper can never poll the wrong
   generation channel. Frozen production/kernel files remain untouched. */
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
 struct alignas(128) IsoAtomic {
   std::atomic<uint64_t> v;
   char pad[120];
   IsoAtomic():v(0),pad{0}{}
 };
public:
 enum Mode : unsigned { BASE=0, STORESEQ=1, RELAXPOLL=2, ISO128=3, PAUSE2=4, PAUSE4=5, PAUSE8=6, COMBINED=7 };
 explicit Exp53Custom2MicroarchExhaustCandidate(Mode mode)
   : gen64_(0), done64_(0), stop_(false), ready_(false), out2_(nullptr), in2_(nullptr), n2_(0), mode_(mode), local_gen_(0) {
   pin(0);
   helper_=std::thread([this]{loop();});
   while(!ready_.load(std::memory_order_acquire)) _mm_pause();
 }
 ~Exp53Custom2MicroarchExhaustCandidate(){
   stop_.store(true,std::memory_order_relaxed);
   uint64_t g=++local_gen_;
   publish_generation(g);
   if(helper_.joinable()) helper_.join();
 }
 Exp53Custom2MicroarchExhaustCandidate(const Exp53Custom2MicroarchExhaustCandidate&)=delete;
 Exp53Custom2MicroarchExhaustCandidate& operator=(const Exp53Custom2MicroarchExhaustCandidate&)=delete;
 static unsigned schedule(size_t n){if(n<800)return 0;if(n<1200)return 20;if(n<1600)return 25;if(n<2150)return 34;if(n<2850)return 38;return 41;}
 void run(double*out,const double*in,size_t n,unsigned pct=999){
   if(!n)return;
   unsigned hp=pct==999?schedule(n):pct;
   if(!hp){exp53_n2_vmstyle_u4_0381_frozen(out,in,n);return;}
   size_t h=n*(size_t)hp/100;
   size_t split=n-h;
   split=((split+31)/32)*32;
   if(split<32||split>=n||n-split<32){exp53_n2_vmstyle_u4_0381_frozen(out,in,n);return;}
   out2_=out+split; in2_=in+split; n2_=n-split;
   uint64_t g;
   if(mode_==BASE||mode_==RELAXPOLL||mode_==PAUSE2||mode_==PAUSE4||mode_==PAUSE8){
     g=gen64_.fetch_add(1,std::memory_order_release)+1;
     local_gen_=g;
   } else {
     g=++local_gen_;
     publish_generation(g);
   }
   exp53_n2_vmstyle_u4_0381_frozen(out,in,split);
   wait_done(g);
 }
private:
 static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
 static inline void pauses(unsigned k){for(unsigned i=0;i<k;++i)_mm_pause();}
 bool iso_mode() const {return mode_==ISO128||mode_==COMBINED;}
 void publish_generation(uint64_t g){if(iso_mode())gen128_.v.store(g,std::memory_order_release);else gen64_.store(g,std::memory_order_release);}
 void wait_done(uint64_t g){
   if(iso_mode()){
     if(mode_==COMBINED){while(done128_.v.load(std::memory_order_relaxed)!=g)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}
     else while(done128_.v.load(std::memory_order_acquire)!=g)_mm_pause();
     return;
   }
   unsigned p=(mode_==PAUSE2?2:mode_==PAUSE4?4:mode_==PAUSE8?8:1);
   if(mode_==RELAXPOLL){while(done64_.load(std::memory_order_relaxed)!=g)_mm_pause();std::atomic_thread_fence(std::memory_order_acquire);}
   else while(done64_.load(std::memory_order_acquire)!=g)pauses(p);
 }
 void loop(){
   pin(2);
   ready_.store(true,std::memory_order_release);
   uint64_t seen=iso_mode()?gen128_.v.load(std::memory_order_relaxed):gen64_.load(std::memory_order_relaxed);
   for(;;){
     uint64_t g;
     if(iso_mode()){
       if(mode_==COMBINED){
         while((g=gen128_.v.load(std::memory_order_relaxed))==seen){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}
         std::atomic_thread_fence(std::memory_order_acquire);
       } else {
         while((g=gen128_.v.load(std::memory_order_acquire))==seen){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}
       }
     } else if(mode_==RELAXPOLL){
       while((g=gen64_.load(std::memory_order_relaxed))==seen){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}
       std::atomic_thread_fence(std::memory_order_acquire);
     } else {
       while((g=gen64_.load(std::memory_order_acquire))==seen){if(stop_.load(std::memory_order_relaxed))return;_mm_pause();}
     }
     seen=g;
     if(stop_.load(std::memory_order_relaxed))return;
     exp53_n2_vmstyle_u4_0381_frozen(out2_,in2_,n2_);
     if(iso_mode())done128_.v.store(g,std::memory_order_release);else done64_.store(g,std::memory_order_release);
   }
 }
 std::thread helper_;
 alignas(64) std::atomic<uint64_t> gen64_,done64_;
 IsoAtomic gen128_,done128_;
 alignas(128) std::atomic<bool> stop_,ready_;
 alignas(128) double*out2_;
 const double*in2_;
 size_t n2_;
 const Mode mode_;
 uint64_t local_gen_;
};
