#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void exp53_icx_pgo(double*,const double*,size_t);

static uint64_t sm(uint64_t& x){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

int main(){
 volatile double sink=0.0;
 for(int dom=0;dom<2;++dom){
   for(int n=100;n<=1400;n+=50){
     std::vector<double> in((size_t)n),out((size_t)n);
     uint64_t s=0x71d24a936bc50ef1ULL ^ (uint64_t)n ^ ((uint64_t)dom<<32);
     for(int i=0;i<n;++i){double u=(sm(s)>>11)*0x1.0p-53;double m=dom?(1.0+u*99.0):(0x1p-20+u*(1.0-0x1p-20));in[(size_t)i]=(i&1)?-m:m;}
     for(int r=0;r<128;++r) exp53_icx_pgo(out.data(),in.data(),(size_t)n);
     sink += out[(size_t)(n/2)];
   }
 }
 std::printf("PGO_TRAIN sink=%.17g\n",(double)sink);
 return 0;
}
