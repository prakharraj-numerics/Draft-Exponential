/* Frozen accuracy-safe compfastC baseline.
   Source architecture is exactly compfastC from the optimization sweep that
   measured maxulp=1, gt1=0 on the 6400-case Xeon 6973P-C harness.
*/
#include "exp53_spine_v128_formula6_compfast.c"

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_formula6_compfastC_frozen(double *restrict out,
                                                 const double *restrict in,
                                                 size_t n)
{
    exp53_spine_v128_formula6_compfastC(out,in,n);
}
