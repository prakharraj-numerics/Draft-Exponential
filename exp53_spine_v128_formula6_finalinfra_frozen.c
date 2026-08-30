/* FINAL frozen infrastructure winner, 2026-08-31.
   Source winner: exp53_final_mid_u6 from exp53_spine_v128_formula6_finalinfra.c
   Best measured toolchain in final sweep: GCC 13, -O3 -march=native -mtune=native
   -ffp-contract=fast -fno-math-errno -DNDEBUG.
   Xeon 6973P-C run 33331008924, job 99309445821:
     ACC maxulp=1 gt1=0 gt2=0 on current 12288-case harness
     small=0.648376 ns, wide=0.646902 ns, all=0.645608 ns.
   Frozen baseline C in same GCC run: all=0.687923 ns.
   Intel VML HA same run: all=0.322292 ns.
*/
#include "exp53_spine_v128_formula6_finalinfra.c"

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_formula6_final_frozen(double *restrict out,
                                            const double *restrict in,
                                            size_t n)
{
    exp53_final_mid_u6(out,in,n);
}
