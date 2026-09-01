#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mkl_vml.h>

extern "C" {
void exp53_icx_frozen(double*,const double*,size_t);
void exp53_icx_pref0(double*,const double*,size_t);
void exp53_icx_pref1(double*,const double*,size_t);
void exp53_icx_pref3(double*,const double*,size_t);
void exp53_icx_pref5(double*,const double*,size_t);
void exp53_icx_throughput(double*,const double*,size_t);
void exp53_icx_gshuffle(double*,const double*,size_t);
void exp53_icx_gnoshuffle(double*,const double*,size_t);
void exp53_icx_omp(double*,const double*,size_t);
void exp53_icx_unroll(double*,const double*,size_t);
void exp53_icx_nounroll(double*,const double*,size_t);
void exp53_icx_align32(double*,const double*,size_t);
void exp53_icx_align64(double*,const double*,size_t);
void exp53_icx_dynalign(double*,const double*,size_t);
void exp53_icx_native(double*,const double*,size_t);
void exp53_icx_lto(double*,const double*,size_t);
void exp53_icx_pgo(double*,const double*,size_t);
void exp53_icx_unrolljam(double*,const double*,size_t);
void exp53_icx_nostream(double*,const double*,size_t);
}

using Fn=void(*)(double*,const double*,size_t);
struct Entry { const char* name; Fn fn; };

static const Entry ENTRIES[] = {
 {"frozen",exp53_icx_frozen}, {"pref0",exp53_icx_pref0}, {"pref1",exp53_icx_pref1},
 {"pref3",exp53_icx_pref3}, {"pref5",exp53_icx_pref5}, {"throughput",exp53_icx_throughput},
 {"gshuffle",exp53_icx_gshuffle}, {"gnoshuffle",exp53_icx_gnoshuffle}, {"omp",exp53_icx_omp},
 {"unroll",exp53_icx_unroll}, {"nounroll",exp53_icx_nounroll}, {"align32",exp53_icx_align32},
 {"align64",exp53_icx_align64}, {"dynalign",exp53_icx_dynalign}, {"native",exp53_icx_native},
 {"lto",exp53_icx_lto}, {"pgo",exp53_icx_pgo}, {"unrolljam",exp53_icx_unrolljam},
 {"nostream",exp53_icx_nostream}
};

static uint64_t sm(uint64_t& x){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static void fill(std::vector<double>& a,const char* dom){uint64_t s=0x51376a9d4c2b1e0fULL;for(size_t i=0;i<a.size();++i){double u=(sm(s)>>11)*0x1.0p-53;double m=!std::strcmp(dom,"unit")?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);a[i]=(i&1)?-m:m;}}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls_for(size_t n){size_t c=3000000/n;if(c<1000)c=1000;if(c>40000)c=40000;return c;}
static size_t bitdiff(const std::vector<double>&a,const std::vector<double>&b){size_t z=0;for(size_t i=0;i<a.size();++i){uint64_t x,y;std::memcpy(&x,&a[i],8);std::memcpy(&y,&b[i],8);z+=(x!=y);}return z;}

int main(int argc,char**argv){
 if(argc!=2){std::fprintf(stderr,"stack required\n");return 2;}
 std::string st=argv[1]; Fn fn=nullptr;
 for(const auto& e:ENTRIES) if(st==e.name){fn=e.fn;break;}
 if(!fn && st!="intel") return 3;
 std::vector<int> ns; for(int n=100;n<=1400;n+=50) ns.push_back(n); if(std::getenv("SWEEP_REVERSE")) std::reverse(ns.begin(),ns.end());
 for(const char* dom:{"unit","mid"}) for(int ni:ns){
   size_t n=(size_t)ni,calls=calls_for(n); std::vector<double> in(n),out(n),ref(n); fill(in,dom);
   if(st!="intel" && st!="frozen"){
     exp53_icx_frozen(ref.data(),in.data(),n); fn(out.data(),in.data(),n);
     std::printf("CHECK stack=%s domain=%s n=%d bitdiff=%zu\n",st.c_str(),dom,ni,bitdiff(out,ref));
   }
   auto run=[&](){ if(st=="intel") vmdExp((MKL_INT)n,in.data(),out.data(),VML_HA); else fn(out.data(),in.data(),n); };
   for(int w=0;w<12;++w) run(); std::vector<double> ts; ts.reserve(5);
   for(int r=0;r<5;++r){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)run();auto b=std::chrono::steady_clock::now();ts.push_back(std::chrono::duration<double,std::nano>(b-a).count()/calls/n);}
   std::printf("RESULT stack=%s domain=%s n=%d calls=%zu ns=%.9f\n",st.c_str(),dom,ni,calls,median(ts));
 }
 return 0;
}
