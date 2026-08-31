#pragma once
/* EXPERIMENTAL ONLY — identical conservative-safe custom2 policy, replacing
   only the worker by the GCC vector-function candidate. */
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
extern "C" void exp53_gcc_vector_0100_3000(double*,const double*,size_t);
class Exp53Custom2GCCVectorCandidate {
public:
 Exp53Custom2GCCVectorCandidate():generation_(0),completed_(0),stop_(false),ready_(false),out2_(nullptr),in2_(nullptr),n2_(0){pin(0);helper_=std::thread([this]{loop();});while(!ready_.load(std::memory_order_acquire))_mm_pause();}
 ~Exp53Custom2GCCVectorCandidate(){stop_.store(true,std::memory_order_relaxed);generation_.fetch_add(1,std::memory_order_release);if(helper_.joinable())helper_.join();}
 static unsigned conservative_share(size_t n){if(n<900)return 0;if(n<1300)return 25;if(n<1800)return 34;if(n<2300)return 41;return 46;}
 void run(double*out,const double*in,size_t n){unsigned hp=conservative_share(n);if(!n||hp==0){if(n)exp53_gcc_vector_0100_3000(out,in,n);return;}size_t dh=n*(size_t)hp/100;size_t split=n-dh;split=((split+31u)/32u)*32u;if(split<32||split>=n||n-split<32){exp53_gcc_vector_0100_3000(out,in,n);return;}out2_=out+split;in2_=in+split;n2_=n-split;uint64_t g=generation_.fetch_add(1,std::memory_order_release)+1;exp53_gcc_vector_0100_3000(out,in,split);while(completed_.load(std::memory_order_acquire)!=g)_mm_pause();}
private:
 static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);(void)pthread_setaffinity_np(pthread_self(),sizeof(s),&s);}
 void loop(){pin(2);ready_.store(true,std::memory_order_release);uint64_t seen=generation_.load(std::memory_order_relaxed);for(;;){uint64_t g;while((g=generation_.load(std::memory_order_acquire))==seen)_mm_pause();seen=g;if(stop_.load(std::memory_order_relaxed))return;exp53_gcc_vector_0100_3000(out2_,in2_,n2_);completed_.store(g,std::memory_order_release);}}
 std::thread helper_;alignas(64) std::atomic<uint64_t> generation_,completed_;alignas(64) std::atomic<bool> stop_,ready_;double*out2_;const double*in2_;size_t n2_;
};
