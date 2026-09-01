#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mkl_vml.h>
#include "exp53_hybrid_0101_1400.hpp"

struct A {
    double* p=nullptr;
    explicit A(size_t n){ if(posix_memalign((void**)&p,64,n*sizeof(double))||!p) std::exit(2); }
    ~A(){ std::free(p); }
};

static inline uint64_t sm(uint64_t x){
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
static double u01(uint64_t h){ return ((double)(h>>11)+0.5)*(1.0/9007199254740992.0); }
static void fill(double* x,size_t n,int d){
    uint64_t s=0x243f6a8885a308d3ULL^((uint64_t)n<<19)^((uint64_t)d<<61);
    for(size_t i=0;i<n;i++){
        double u=u01(sm(s+i*0x9e3779b97f4a7c15ULL));
        double e=0x1p-20;
        double m=d==0 ? (e+u*(1.0-2*e)) : (1.0+e+u*(99.0-2*e));
        x[i]=(i&1)?-m:m;
    }
}
static size_t calls(size_t n){
    size_t c=5000000ULL/n;
    if(c<1000)c=1000;
    if(c>50000)c=50000;
    return c;
}
static double med(std::vector<double>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc,char** argv){
    if(argc!=2) return 2;
    std::string st=argv[1];
    if(st!="hybrid"&&st!="intel") return 2;
    bool rev=getenv("SWEEP_REVERSE")&&std::string(getenv("SWEEP_REVERSE"))=="1";
    std::vector<size_t> ns;
    for(size_t n=100;n<=1400;n+=50) ns.push_back(n);
    if(rev) std::reverse(ns.begin(),ns.end());
    std::vector<int> ds={0,1};
    if(rev) std::reverse(ds.begin(),ds.end());

    Exp53Hybrid0101_1400 hybrid(2);
    volatile double sink=0;
    std::cout<<std::fixed<<std::setprecision(9);
    for(int d:ds){
        for(size_t n:ns){
            A in(n),out(n);
            fill(in.p,n,d);
            size_t c=calls(n);
            auto f=[&](){
                if(st=="hybrid") hybrid.run(out.p,in.p,n,2);
                else vmdExp((MKL_INT)n,in.p,out.p,VML_HA);
            };
            for(int w=0;w<10;w++) f();
            std::vector<double> v;
            for(int s=0;s<9;s++){
                auto a=std::chrono::steady_clock::now();
                for(size_t k=0;k<c;k++) f();
                auto b=std::chrono::steady_clock::now();
                v.push_back(std::chrono::duration<double,std::nano>(b-a).count()/((double)c*n));
            }
            sink += out.p[(n*13u+d)%n]*0x1p-1022;
            std::cout<<"RESULT stack="<<st<<" domain="<<(d?"mid":"unit")
                     <<" n="<<n<<" calls="<<c<<" ns="<<med(v)<<"\n";
        }
    }
    if(sink==123) std::cerr<<sink;
    return 0;
}
