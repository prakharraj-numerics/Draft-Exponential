#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <mkl_vml.h>
#include "exp53_resource_aware_dispatch.hpp"

static inline uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

int main() {
    const std::vector<size_t> sizes = {100,700,3500,15000,49999,50000,1000000,2000000};
    Exp53ResourceAwareExecutor resource(2);
    Exp53BatchProductionExecutor canonical(2);

    for (size_t n : sizes) {
        double *in=nullptr,*balanced=nullptr,*expected=nullptr,*intel=nullptr;
        if (posix_memalign((void**)&in,64,n*sizeof(double)) ||
            posix_memalign((void**)&balanced,64,n*sizeof(double)) ||
            posix_memalign((void**)&expected,64,n*sizeof(double)) ||
            posix_memalign((void**)&intel,64,n*sizeof(double))) return 2;
        for (size_t i=0;i<n;++i) {
            const double u=(static_cast<double>((i*104729ULL)%1000003)+0.5)/1000003.0;
            const double mag=(i&2)?1.0+98.0*u:0x1p-20+(1.0-0x1p-19)*u;
            in[i]=(i&1)?-mag:mag;
        }

        const long chosen=resource.workers_for(n,Exp53ResourcePolicy::Balanced);
        const long required=n>=50000?2:1;
        if (chosen!=required) {
            std::cerr<<"route failure n="<<n<<" chosen="<<chosen<<" required="<<required<<"\n";
            return 3;
        }

        resource.run(balanced,in,n,Exp53ResourcePolicy::Balanced);
        canonical.run(expected,in,n,required);
        vmdExp((MKL_INT)n,in,intel,VML_HA);
        uint64_t maxulp=0;
        for(size_t i=0;i<n;++i) {
            if(bits(balanced[i])!=bits(expected[i])) {
                std::cerr<<"wrapper mismatch n="<<n<<" i="<<i<<"\n";
                return 4;
            }
            const uint64_t a=bits(balanced[i]),b=bits(intel[i]);
            maxulp=std::max(maxulp,a>b?a-b:b-a);
        }
        if(maxulp>2) {
            std::cerr<<"accuracy failure n="<<n<<" maxulp="<<maxulp<<"\n";
            return 5;
        }
        std::cout<<"PASS n="<<n<<" balanced_workers="<<chosen<<" maxulp="<<maxulp<<"\n";
        std::free(in);std::free(balanced);std::free(expected);std::free(intel);
    }

    if(resource.workers_for(3500,Exp53ResourcePolicy::LowestLatency)!=2 ||
       resource.workers_for(2000000,Exp53ResourcePolicy::LowestCompute)!=1) return 6;
    std::cout<<"RESOURCE_AWARE_DISPATCH_PASS\n";
    return 0;
}
