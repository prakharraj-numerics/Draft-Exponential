#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "exp53_batch_production.hpp"

/* This smoke targets only n>100. Resolve the unrelated frozen <=100 VCL
   production symbol without pulling a second translation unit that embeds the
   same baseline C implementation. */
extern "C" void exp53_vcl_u2z_0100_frozen(double *out,
                                           const double *in,
                                           size_t n) {
    exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

static int check(const char *mode, size_t n, const double *ref, const double *out) {
    if (std::memcmp(out, ref, n * sizeof(double)) != 0) {
        std::printf("RANGE2CORE_FROZEN_SMOKE FAIL mode=%s n=%zu\n", mode, n);
        return 1;
    }
    return 0;
}

int main() {
    const size_t sizes[] = {2499, 2500, 2501, 2600, 2700, 2800, 2900, 2999, 3000, 3001};
    const size_t maxn = 3001;
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
        ex.run(out,in,n,2);
        if (check("default2",n,ref,out)) return 3;

        std::memset(out,0,n*sizeof(double));
        ex.run_streaming_write_once(out,in,n,2);
        if (check("streaming2",n,ref,out)) return 4;

        std::memset(out,0,n*sizeof(double));
        ex.run(out,in,n,1);
        if (check("workers1",n,ref,out)) return 5;
    }

    std::printf("RANGE2CORE_FROZEN_SMOKE PASS bitdiff=0 boundaries=2499,2500,3000,3001\n");
    std::free(in); std::free(ref); std::free(out);
    return 0;
}
