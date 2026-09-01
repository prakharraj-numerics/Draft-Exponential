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
void exp53_n2_vmstyle_u4_0381_frozen(double*, const double*, size_t);
void exp53_frozen_robinhood(double*, const double*, size_t);
}

using Fn = void(*)(double*, const double*, size_t);

static uint64_t sm(uint64_t& x){
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z=x;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
static void fill(std::vector<double>& a,const char* dom){
    uint64_t s=0x51376a9d4c2b1e0fULL;
    for(size_t i=0;i<a.size();++i){
        double u=(sm(s)>>11)*0x1.0p-53;
        double m=!std::strcmp(dom,"unit") ? (0x1p-20+u*(1.0-0x1p-20)) : (1.0+u*99.0);
        a[i]=(i&1)?-m:m;
    }
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t calls_for(size_t n){size_t c=3000000/n;if(c<1000)c=1000;if(c>40000)c=40000;return c;}
static size_t bitdiff(const std::vector<double>&a,const std::vector<double>&b){
    size_t z=0;
    for(size_t i=0;i<a.size();++i){uint64_t x,y;std::memcpy(&x,&a[i],8);std::memcpy(&y,&b[i],8);z+=(x!=y);}
    return z;
}

int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"stack required\n");return 2;}
    std::string st=argv[1];
    Fn fn=nullptr;
    if(st=="robin") fn=exp53_frozen_robinhood;
    else if(st=="frozen") fn=exp53_n2_vmstyle_u4_0381_frozen;
    else if(st!="intel") return 3;

    std::vector<int> ns;
    for(int n=100;n<=1400;n+=50) ns.push_back(n);
    if(std::getenv("SWEEP_REVERSE")) std::reverse(ns.begin(),ns.end());

    for(const char* dom:{"unit","mid"}) for(int ni:ns){
        size_t n=(size_t)ni,calls=calls_for(n);
        std::vector<double> in(n),out(n),ref(n);
        fill(in,dom);
        if(st=="robin"){
            exp53_n2_vmstyle_u4_0381_frozen(ref.data(),in.data(),n);
            fn(out.data(),in.data(),n);
            std::printf("CHECK stack=robin domain=%s n=%d bitdiff=%zu\n",dom,ni,bitdiff(out,ref));
        }
        auto run=[&](){
            if(st=="intel") vmdExp((MKL_INT)n,in.data(),out.data(),VML_HA);
            else fn(out.data(),in.data(),n);
        };
        for(int w=0;w<12;++w)run();
        std::vector<double> ts;ts.reserve(5);
        for(int r=0;r<5;++r){
            auto a=std::chrono::steady_clock::now();
            for(size_t k=0;k<calls;++k)run();
            auto b=std::chrono::steady_clock::now();
            ts.push_back(std::chrono::duration<double,std::nano>(b-a).count()/calls/n);
        }
        std::printf("RESULT stack=%s domain=%s n=%d calls=%zu ns=%.9f\n",st.c_str(),dom,ni,calls,median(ts));
    }
    return 0;
}
