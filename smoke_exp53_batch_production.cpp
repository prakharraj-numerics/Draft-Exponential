#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "exp53_batch_production.hpp"

static void* xalloc(size_t n){ void* p=nullptr; if(posix_memalign(&p,64,n)!=0||!p) std::abort(); return p; }
static uint64_t bits(double x){ uint64_t u; std::memcpy(&u,&x,8); return u; }

int main(){
    const size_t n=200000;
    double *in=(double*)xalloc(n*8), *ref=(double*)xalloc(n*8), *a=(double*)xalloc(n*8), *b=(double*)xalloc(n*8);
    for(size_t i=0;i<n;i++) in[i]=-100.0 + 200.0*(double)i/(double)(n-1);

    exp53_n2_vmstyle_u4_0381_frozen(ref,in,n);
    Exp53BatchProductionExecutor ex(2);
    ex.run(a,in,n);
    ex.run_streaming_write_once(b,in,n);

    size_t da=0,db=0;
    for(size_t i=0;i<n;i++){
        da += bits(a[i]) != bits(ref[i]);
        db += bits(b[i]) != bits(ref[i]);
    }
    std::printf("PRODUCTION_SMOKE default_bitdiff=%zu streaming_bitdiff=%zu\n",da,db);
    std::free(in);std::free(ref);std::free(a);std::free(b);
    return (da||db) ? 1 : 0;
}
