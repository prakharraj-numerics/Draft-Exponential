#pragma once
/* EXPERIMENTAL ONLY. Permanent CPU0/CPU2 custom2 handoff. Both sides use the
   immutable frozen u4 worker. Caller/helper boundary is supplied explicitly
   and MUST be a multiple of 32, preserving frozen 32-value decomposition. */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
class Exp53Custom2Split32SearchCandidate {
public:
 Exp53Custom2Split32SearchCandidate():gen_(0),done_(0),stop_(false),ready_(false),o2_(nullptr),i2_(nullptr),n2_(0){pin(0); th_=std::thread([this]{loop();}); while(!ready_.load(std::memory_order_acquire)) _mm_pause();}
 ~Exp53Custom2Split32SearchCandidate(){stop_.store(true,std::memory_order_relaxed);gen_.fetch_add(1,std::memory_order_release);if(th_.joinable())th_.join();}
 Exp53Custom2Split32SearchCandidate(const Exp53Custom2Split32SearchCandidate&)=delete;
 void run_serial(double*o,const double*i,size_t n){exp53_n2_vmstyle_u4_0381_frozen(o,i,n);}
 void run_split(double*o,const double*i,size_t n,size_t split){
  if(split<32||split>=n||split%32||n-split<32){run_serial(o,i,n);return;}
  o2_=o+split;i2_=i+split;n2_=n-split;uint64_t g=gen_.fetch_add(1,std::memory_order_release)+1;
  exp53_n2_vmstyle_u4_0381_frozen(o,i,split);while(done_.load(std::memory_order_acquire)!=g)_mm_pause();
 }
private:
 static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=gen_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=gen_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;exp53_n2_vmstyle_u4_0381_frozen(o2_,i2_,n2_);done_.store(g,std::memory_order_release);}}
 std::thread th_;alignas(64)std::atomic<uint64_t>gen_,done_;alignas(64)std::atomic<bool>stop_,ready_;double*o2_;const double*i2_;size_t n2_;
};
