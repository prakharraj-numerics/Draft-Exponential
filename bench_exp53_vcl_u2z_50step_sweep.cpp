#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>

extern "C" void exp53_vcl_u2z_candidate(double*,const double*,size_t);
extern "C" void exp53_small_u2z_0100_frozen(double*,const double*,size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

struct A{double*p=nullptr;explicit A(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double))||!p)std::exit(2);}~A(){std::free(p);}};
static inline uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u01(uint64_t h){return ((double)(h>>11)+.5)*(1.0/9007199254740992.0);}static const char*dn(int d){return d?"mid":"unit";}static void fill(double*x,size_t n,int d){uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<17)^((uint64_t)d<<57);for(size_t i=0;i<n;i++){double u=u01(sm(s+i*0x9e3779b97f4a7c15ULL)),e=0x1p-20,m=d?(1.0+e+u*(99.0-2*e)):(e+u*(1.0-2*e));x[i]=(i&1)?-m:m;}}static size_t cc(size_t n){size_t c=4000000ULL/n;if(c<2)c=2;if(c>100000)c=100000;return c;}static double med(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){if(argc!=2)return 2;std::string st=argv[1];if(st!="vclu2z"&&st!="u2z"&&st!="frozen"&&st!="intel")return 2;bool rev=getenv("SWEEP_REVERSE")&&std::string(getenv("SWEEP_REVERSE"))=="1";std::vector<size_t>sz;for(size_t n=50;n<=3000;n+=50)sz.push_back(n);if(rev)std::reverse(sz.begin(),sz.end());std::vector<int>ds={0,1};if(rev)std::reverse(ds.begin(),ds.end());volatile double sink=0;std::cout<<std::fixed<<std::setprecision(9)<<"STACK="<<st<<" REVERSE="<<(rev?1:0)<<"\n";for(int d:ds)for(size_t n:sz){A in(n),out(n);fill(in.p,n,d);size_t calls=cc(n);auto f=[&](){if(st=="vclu2z")exp53_vcl_u2z_candidate(out.p,in.p,n);else if(st=="u2z")exp53_small_u2z_0100_frozen(out.p,in.p,n);else if(st=="frozen")exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);else vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};for(int w=0;w<5;w++)f();std::vector<double>v;for(int k=0;k<7;k++){auto a=std::chrono::steady_clock::now();for(size_t c=0;c<calls;c++)f();auto b=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/((double)calls*n));}sink+=out.p[(n*13u+d)%n]*0x1p-1022;std::cout<<"RESULT stack="<<st<<" domain="<<dn(d)<<" n="<<n<<" calls="<<calls<<" ns="<<med(v)<<"\n";}if(sink==123456789.0)std::fprintf(stderr,"sink=%g\n",(double)sink);return 0;}
