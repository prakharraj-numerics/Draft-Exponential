#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "exp53_range2core_0101_2999_candidates.hpp"
#include "exp53_batch_custom2_u2z_candidate.hpp"

extern "C" void exp53_small_u2z_0100_frozen(double*, const double*, size_t);
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);

struct A { double* p=nullptr; explicit A(size_t n){if(posix_memalign((void**)&p,64,n*sizeof(double))||!p)std::abort();} ~A(){std::free(p);} };
static inline unsigned long long sm(unsigned long long x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);} 
static double u01(unsigned long long h){return ((double)(h>>11)+.5)*(1.0/9007199254740992.0);} 
static void fill(double*x,size_t n,int d){unsigned long long s=0x243f6a8885a308d3ULL^((unsigned long long)n<<19)^((unsigned long long)d<<61);for(size_t i=0;i<n;i++){double u=u01(sm(s+i*0x9e3779b97f4a7c15ULL));double m=d?1.000001+u*98.999998:1e-6+u*.999998;x[i]=(i&1)?-m:m;}}
static size_t calls(size_t n){size_t c=4000000ULL/n;if(c<2)c=2;if(c>100000)c=100000;return c;}
static double med(std::vector<double>&v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(int argc,char**argv){
 if(argc!=2)return 2; std::string s=argv[1]; bool rev=getenv("SWEEP_REVERSE")&&std::string(getenv("SWEEP_REVERSE"))=="1";
 std::vector<size_t> ns; for(size_t n=101;n<=2951;n+=50)ns.push_back(n); ns.push_back(2999); if(rev)std::reverse(ns.begin(),ns.end());
 std::vector<int> ds={0,1}; if(rev)std::reverse(ds.begin(),ds.end());
 volatile double sink=0;
 auto run_points=[&](auto &ex, const char* name){for(int d:ds)for(size_t n:ns){A in(n),out(n);fill(in.p,n,d);size_t c=calls(n);auto f=[&](){ex.run(out.p,in.p,n);};for(int w=0;w<5;w++)f();std::vector<double>v;for(int k=0;k<7;k++){auto t0=std::chrono::steady_clock::now();for(size_t j=0;j<c;j++)f();auto t1=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)c*n));}sink+=out.p[n/3]*0x1p-1022;std::cout<<std::fixed<<std::setprecision(9)<<"RESULT stack="<<name<<" domain="<<(d?"mid":"unit")<<" n="<<n<<" ns="<<med(v)<<"\n";}};
 if(s=="cons"){Exp53Range2CoreConservative ex;run_points(ex,"cons");}
 else if(s=="bal"){Exp53Range2CoreBalanced ex;run_points(ex,"bal");}
 else if(s=="agg"){Exp53Range2CoreAggressive ex;run_points(ex,"agg");}
 else if(s=="custom2u2z"){Exp53Custom2U2ZCandidate ex;run_points(ex,"custom2u2z");}
 else if(s=="u2z"||s=="frozen"||s=="intel"){
   for(int d:ds)for(size_t n:ns){A in(n),out(n);fill(in.p,n,d);size_t c=calls(n);auto f=[&](){if(s=="u2z")exp53_small_u2z_0100_frozen(out.p,in.p,n);else if(s=="frozen")exp53_n2_vmstyle_u4_0381_frozen(out.p,in.p,n);else vmdExp((MKL_INT)n,in.p,out.p,VML_HA);};for(int w=0;w<5;w++)f();std::vector<double>v;for(int k=0;k<7;k++){auto t0=std::chrono::steady_clock::now();for(size_t j=0;j<c;j++)f();auto t1=std::chrono::steady_clock::now();v.push_back(std::chrono::duration<double,std::nano>(t1-t0).count()/((double)c*n));}sink+=out.p[n/3]*0x1p-1022;std::cout<<std::fixed<<std::setprecision(9)<<"RESULT stack="<<s<<" domain="<<(d?"mid":"unit")<<" n="<<n<<" ns="<<med(v)<<"\n";}
 } else return 2;
 if(sink==1234567.0)std::cerr<<sink; return 0;
}
