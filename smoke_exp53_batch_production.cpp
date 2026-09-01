#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "exp53_batch_production.hpp"

static void* xalloc(size_t n){ void* p=nullptr; if(posix_memalign(&p,64,n)!=0||!p) std::abort(); return p; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }

static size_t bitdiff(const double* a,const double* b,size_t n){
    size_t d=0; for(size_t i=0;i<n;i++) d += bits(a[i]) != bits(b[i]); return d;
}

int main(){
    const std::vector<size_t> sizes={101,125,160,192,224,256,320,384,448,512,640,768,896,1024,1152,1280,1400};
    const size_t cap=200000;
    double *in=(double*)xalloc(cap*8), *ref=(double*)xalloc(cap*8), *a=(double*)xalloc(cap*8), *b=(double*)xalloc(cap*8);
    for(size_t i=0;i<cap;i++) in[i]=-100.0 + 200.0*(double)i/(double)(cap-1);

    Exp53BatchProductionExecutor ex(2);
    size_t hybrid_diff=0, streaming_diff=0;
    for(size_t n:sizes){
        exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
        ex.run(a,in,n);
        ex.run_streaming_write_once(b,in,n);
        hybrid_diff += bitdiff(a,ref,n);
        streaming_diff += bitdiff(b,ref,n);
    }

    const size_t n=cap;
    exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
    ex.run(a,in,n);
    ex.run_streaming_write_once(b,in,n);
    const size_t large_default_diff=bitdiff(a,ref,n);
    const size_t large_streaming_diff=bitdiff(b,ref,n);

    std::printf("PRODUCTION_SMOKE hybrid_bitdiff=%zu streaming_bitdiff=%zu large_default_bitdiff=%zu large_streaming_bitdiff=%zu\n",
                hybrid_diff,streaming_diff,large_default_diff,large_streaming_diff);
    std::free(in);std::free(ref);std::free(a);std::free(b);
    return (hybrid_diff||streaming_diff||large_default_diff||large_streaming_diff) ? 1 : 0;
}
