#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
#include "exp53_custom2_split32_search_0100_3000_candidate.hpp"
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
struct B{double*p;explicit B(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double)))abort();}~B(){free(p);}};
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);for(size_t j=0;j<n;j++){double z=u(sm(s+j*0x9e3779b97f4a7c15ULL));double a=d?1.000001+z*98.999998:0x1p-20+z*(1.-0x1p-19);x[j]=(j&1)?-a:a;}}
static double med(std::vector<double>v){sort(v.begin(),v.end());return v[v.size()/2];}static size_t calls(size_t n){size_t c=180000/n;return c<24?24:c>1800?1800:c;}
template<class F>static double tm(F&&f,size_t n,size_t c,int k){for(int w=0;w<3;w++)f();std::vector<double>v;for(int q=0;q<k;q++){auto a=std::chrono::steady_clock::now();for(size_t z=0;z<c;z++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/((double)c*n));}return med(v);}
int main(){constexpr size_t M=3000;B in(M),ref(M),out(M);Exp53Custom2Split32SearchCandidate x;std::cout<<std::fixed<<std::setprecision(9);std::cout<<"SPLIT32_SEARCH sizes=100..3000 step=50 all_32_boundaries bitwise_gate=required\n";
 for(size_t n=100;n<=3000;n+=50){
  /* Search one geometry that is valid and fast across BOTH domains. Score is worst
     ratio to Intel across unit/mid, preventing domain-specific cherry-picking. */
  struct R{size_t sp;double score,cu,cm,iu,im;};std::vector<R>rs;size_t c=calls(n);
  for(size_t sp=32;sp+32<=n;sp+=32){double cv[2],iv[2];bool ok=true;for(int d=0;d<2;d++){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);x.run_split(out.p,in.p,n,sp);if(memcmp(out.p,ref.p,n*sizeof(double))){ok=false;break;}auto fc=[&]{x.run_split(out.p,in.p,n,sp);};auto fi=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};cv[d]=tm(fc,n,c,3);iv[d]=tm(fi,n,c,3);}if(ok)rs.push_back({sp,std::max(cv[0]/iv[0],cv[1]/iv[1]),cv[0],cv[1],iv[0],iv[1]});}
  if(rs.empty()){std::cout<<"FAIL n="<<n<<" reason=no_bitwise_safe_split\n";return 9;}auto best=*std::min_element(rs.begin(),rs.end(),[](const R&a,const R&b){return a.score<b.score;});
  double cv[2],iv[2],fv[2];for(int d=0;d<2;d++){fill(in.p,n,d);exp53_n2_vmstyle_u4_0381_frozen(ref.p,in.p,n);x.run_split(out.p,in.p,n,best.sp);if(memcmp(out.p,ref.p,n*sizeof(double))){std::cout<<"FAIL n="<<n<<" final_bitdiff=1\n";return 10;}auto fc=[&]{x.run_split(out.p,in.p,n,best.sp);};auto fi=[&]{vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};auto ff=[&]{exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);};cv[d]=tm(fc,n,c,7);iv[d]=tm(fi,n,c,7);fv[d]=tm(ff,n,c,5);}
  std::cout<<"FINAL n="<<n<<" split="<<best.sp<<" helper="<<(n-best.sp)<<" helper_pct="<<(100.0*(n-best.sp)/n)<<" unit_custom="<<cv[0]<<" unit_intel="<<iv[0]<<" unit_ratio="<<(iv[0]/cv[0])<<" mid_custom="<<cv[1]<<" mid_intel="<<iv[1]<<" mid_ratio="<<(iv[1]/cv[1])<<" frozen_unit="<<fv[0]<<" frozen_mid="<<fv[1]<<" bitdiff=0\n";
 }
 return 0;}
