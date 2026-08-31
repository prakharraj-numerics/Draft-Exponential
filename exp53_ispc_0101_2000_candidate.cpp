// Experimental ISPC wrapper for frozen EXP53 machinery.
#define restrict __restrict__
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
#undef restrict
#include <cstdint>
#include <cstddef>
#include "exp53_ispc_generated.h"

__attribute__((noinline))
void exp53_ispc_0101_2000_candidate(double* out, const double* in, size_t n) {
    const size_t bulk = n & ~(size_t)31;
    if (bulk) {
        ispc::exp53_ispc_kernel(out, const_cast<double*>(in), (int)bulk,
                                reinterpret_cast<uint64_t*>(const_cast<double*>(N2_FROZEN_TAB128)));
    }
    if (bulk < n) exp53_n2_fused_u4_038_frozen(out + bulk, in + bulk, n - bulk);
}
