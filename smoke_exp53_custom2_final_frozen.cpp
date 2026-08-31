#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "exp53_batch_production.hpp"

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

int main() {
    const size_t sizes[] = {50, 100, 250, 1000, 3000, 65000, 1000000};
    const size_t maxn = 1000000;
    double *in=nullptr, *ref=nullptr, *out=nullptr;
    if (posix_memalign((void**)&in, 64, maxn*sizeof(double)) ||
        posix_memalign((void**)&ref,64, maxn*sizeof(double)) ||
        posix_memalign((void**)&out,64, maxn*sizeof(double))) return 2;

    for (size_t i=0;i<maxn;++i) {
        uint64_t u=mix64(0x9e3779b97f4a7c15ULL + i);
        double t=(double)(u & ((1ULL<<53)-1)) * (1.0/9007199254740992.0);
        in[i] = -80.0 + 160.0*t;
    }

    Exp53BatchProductionExecutor ex(2);
    for (size_t n: sizes) {
        exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
        std::memset(out,0,n*sizeof(double));
        ex.run(out,in,n);
        if (std::memcmp(out,ref,n*sizeof(double)) != 0) {
            std::printf("FINAL_FROZEN_SMOKE FAIL mode=default n=%zu\n",n);
            return 3;
        }

        std::memset(out,0,n*sizeof(double));
        ex.run_streaming_write_once(out,in,n);
        if (std::memcmp(out,ref,n*sizeof(double)) != 0) {
            std::printf("FINAL_FROZEN_SMOKE FAIL mode=streaming_nt n=%zu\n",n);
            return 4;
        }

        std::memset(out,0,n*sizeof(double));
        ex.run(out,in,n,1);
        if (std::memcmp(out,ref,n*sizeof(double)) != 0) {
            std::printf("FINAL_FROZEN_SMOKE FAIL mode=serial_escape n=%zu\n",n);
            return 5;
        }
    }

    std::printf("FINAL_FROZEN_SMOKE PASS default_bitdiff=0 streaming_bitdiff=0 serial_escape_bitdiff=0\n");
    std::free(in); std::free(ref); std::free(out);
    return 0;
}
