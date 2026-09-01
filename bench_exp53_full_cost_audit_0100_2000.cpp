#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <x86intrin.h>
#include <mkl_vml.h>
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

struct B{double*p;B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))std::abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x13198a2e03707344ULL^((uint64_t)n<<17)^((uint64_t)d<<61);for(size_t i=0;i<n;i++){double q=u(sm(s+i*0x9e3779b97f4a7c15ULL)),a=d?1.000001+q*98.999998:0x1p-20+q*(1.-0x1p-19);x[i]=(i&1)?-a:a;}}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}

template<class F> static double ns(F f,size_t n,size_t c){for(int i=0;i<20;i++)f();std::vector<double> v;for(int r=0;r<11;r++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/double(c*n));}return med(v);} 
template<class F> static double cyc(F f,size_t n,size_t c){for(int i=0;i<20;i++)f();std::vector<double> v;for(int r=0;r<11;r++){_mm_lfence();uint64_t a=__rdtsc();_mm_lfence();for(size_t z=0;z<c;z++)f();_mm_lfence();uint64_t b=__rdtsc();_mm_lfence();v.push_back(double(b-a)/double(c*n));}return med(v);} 

int main(int argc,char**argv){
  B in(2048),out(2048),ref(2048);
  if(argc>=4){size_t n=strtoull(argv[2],0,10);int d=atoi(argv[3]);fill(in.p,n,d);std::string m=argv[1];size_t c=std::max<size_t>(5000,20000000/n); if(m=="ours"){for(size_t z=0;z<c;z++)exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);}else if(m=="intel"){for(size_t z=0;z<c;z++)vmdExp((MKL_INT)n,in.p,out.p,VML_HA);}else return 2; volatile double sink=out.p[n-1];(void)sink;return 0;}
  const size_t sizes[]={256,512,1024,2048};
  std::cout<<std::fixed<<std::setprecision(9)<<"FULL_COST_AUDIT clean_multiples_only\n";
  for(int d=0;d<2;d++)for(size_t n:sizes){fill(in.p,n,d);size_t c=std::max<size_t>(128,1200000/n);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);vmdExp((MKL_INT)n,in.p,out.p,VML_HA);size_t diff=0;for(size_t i=0;i<n;i++)if(std::memcmp(ref.p+i,out.p+i,8))diff++;double on=ns([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);double inx=ns([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);double oc=cyc([&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);},n,c);double ic=cyc([&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);},n,c);std::cout<<"FINAL domain="<<(d?"mid":"unit")<<" n="<<n<<" ours_ns="<<on<<" intel_ns="<<inx<<" intel_over_ours="<<inx/on<<" ours_cycles="<<oc<<" intel_cycles="<<ic<<" intel_cycle_ratio="<<ic/oc<<" result_bitdiff="<<diff<<"\n";}
}
