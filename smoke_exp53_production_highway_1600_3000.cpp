#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "exp53_batch_production.hpp"

extern "C" void exp53_vcl_u2z_0100_frozen(double*o,const double*,size_t n){for(size_t i=0;i<n;i++)o[i]=22.0;}
extern "C" void exp53_n2_vmstyle_u4_0381_frozen(double*o,const double*,size_t n){for(size_t i=0;i<n;i++)o[i]=11.0;}
extern "C" void exp53_n2_vmstyle_u4_0381_nt_sfence(double*o,const double*,size_t n){for(size_t i=0;i<n;i++)o[i]=33.0;}

static uint64_t ord(double x){uint64_t u;std::memcpy(&u,&x,8);return(u>>63)?~u:(u|0x8000000000000000ULL);} 
static uint64_t ulp(double a,double b){auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}

int main(){
  const size_t sizes[]={99,100,101,1599,1600,1999,2000,2999,3000,3001};
  Exp53BatchProductionExecutor ex(2);
  for(size_t n:sizes){
    std::vector<double> in(n),out(n);
    for(size_t i=0;i<n;i++) in[i]=(i&1)?-(0.01+0.9*(double)i/(double)n):(0.01+0.9*(double)i/(double)n);
    ex.run(out.data(),in.data(),n,2);
    if(n<=100){ if(out[0]!=22.0){std::cerr<<"FAIL small n="<<n<<"\n";return 2;} }
    else if(n<1600){ if(out[0]!=11.0){std::cerr<<"FAIL serial n="<<n<<"\n";return 3;} }
    else if(n<=3000){ uint64_t m=0; for(size_t i=0;i<n;i++){auto u=ulp(out[i],std::exp(in[i])); if(u>m)m=u;} if(m>1){std::cerr<<"FAIL hwy n="<<n<<" maxulp="<<m<<"\n";return 4;} std::cout<<"PASS highway n="<<n<<" maxulp="<<m<<"\n"; }
    else { if(out[0]!=11.0 || out[n-1]!=11.0){std::cerr<<"FAIL custom_gt3k n="<<n<<"\n";return 5;} }
  }
  std::cout<<"PRODUCTION_BOUNDARY_SMOKE_PASS\n";
  return 0;
}
