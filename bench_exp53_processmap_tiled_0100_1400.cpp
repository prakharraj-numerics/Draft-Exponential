#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpfr.h>
#include <mkl_vml.h>

extern "C" {
void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);
void exp53_pm_tile3(double*,const double*,size_t);
void exp53_pm_tile5(double*,const double*,size_t);
void exp53_pm_tile6(double*,const double*,size_t);
}
using Fn=void(*)(double*,const double*,size_t);

static uint64_t sm(uint64_t& x){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static void fill(std::vector<double>& a,const char* dom){uint64_t s=0x51376a9d4c2b1e0fULL ^ (uint64_t)a.size()*0x9e3779b97f4a7c15ULL;for(size_t i=0;i<a.size();++i){double u=(sm(s)>>11)*0x1.0p-53;double m=!std::strcmp(dom,"unit")?(0x1p-20+u*(1.0-0x1p-20)):(1.0+u*99.0);a[i]=(i&1)?-m:m;}}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls_for(size_t n){size_t c=4000000/n;if(c<1500)c=1500;if(c>50000)c=50000;return c;}
static uint64_t bits(double x){uint64_t u;std::memcpy(&u,&x,8);return u;}
static uint64_t ulpd(double a,double b){uint64_t x=bits(a),y=bits(b);return x>y?x-y:y-x;}
static size_t bitdiff(const std::vector<double>&a,const std::vector<double>&b){size_t z=0;for(size_t i=0;i<a.size();++i)z+=(bits(a[i])!=bits(b[i]));return z;}

static Fn getfn(const std::string& s){
 if(s=="tile3")return exp53_pm_tile3;
 if(s=="tile5")return exp53_pm_tile5;
 if(s=="tile6")return exp53_pm_tile6;
 if(s=="frozen")return exp53_n2_vmstyle_u4_0381_frozen;
 return nullptr;
}

static int validate_one(const std::string& st){
 Fn fn=getfn(st); if(!fn)return 3;
 mpfr_t mx,my; mpfr_init2(mx,256); mpfr_init2(my,256);
 uint64_t maxulp=0,gt1=0,total=0,diffs=0;
 for(const char* dom:{"unit","mid"}) for(int ni=100;ni<=1400;ni+=50){
   size_t n=(size_t)ni; std::vector<double> in(n),out(n),fro(n); fill(in,dom); fn(out.data(),in.data(),n);
   exp53_n2_vmstyle_u4_0381_frozen(fro.data(),in.data(),n); diffs+=bitdiff(out,fro);
   for(size_t i=0;i<n;++i){
     mpfr_set_d(mx,in[i],MPFR_RNDN); mpfr_exp(my,mx,MPFR_RNDN); double ref=mpfr_get_d(my,MPFR_RNDN);
     uint64_t u=ulpd(out[i],ref); if(u>maxulp)maxulp=u; if(u>1)++gt1; ++total;
   }
 }
 mpfr_clear(mx); mpfr_clear(my);
 std::printf("ACCURACY stack=%s total=%llu maxulp=%llu gt1=%llu bitdiff_vs_frozen=%llu\n",st.c_str(),(unsigned long long)total,(unsigned long long)maxulp,(unsigned long long)gt1,(unsigned long long)diffs);
 return gt1?1:0;
}

int main(int argc,char**argv){
 if(argc!=2){std::fprintf(stderr,"stack or validate:<stack> required\n");return 2;}
 std::string st=argv[1];
 if(st.rfind("validate:",0)==0)return validate_one(st.substr(9));
 Fn fn=getfn(st); if(!fn && st!="intel")return 3;
 std::vector<int> ns;for(int n=100;n<=1400;n+=50)ns.push_back(n);if(std::getenv("SWEEP_REVERSE"))std::reverse(ns.begin(),ns.end());
 for(const char* dom:{"unit","mid"})for(int ni:ns){
   size_t n=(size_t)ni,calls=calls_for(n);std::vector<double> in(n),out(n);fill(in,dom);
   auto run=[&](){if(st=="intel")vmdExp((MKL_INT)n,in.data(),out.data(),VML_HA);else fn(out.data(),in.data(),n);};
   for(int w=0;w<16;++w)run();std::vector<double> ts;ts.reserve(7);
   for(int r=0;r<7;++r){auto a=std::chrono::steady_clock::now();for(size_t k=0;k<calls;++k)run();auto b=std::chrono::steady_clock::now();ts.push_back(std::chrono::duration<double,std::nano>(b-a).count()/calls/n);}
   std::printf("RESULT stack=%s domain=%s n=%d calls=%zu ns=%.9f\n",st.c_str(),dom,ni,calls,median(ts));
 }
 return 0;
}
