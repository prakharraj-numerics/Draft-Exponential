/* Frozen checkpoint: faithful n=2 integral-power EXP spine, SAFE path.

   Mathematical spine preserved:
     e^r - 1 = r Q4(r)^2
     Q4(r)=1+r/4+5r^2/96+r^3/128+79r^4/92160.

   Canonical kernel: exp53_n2_rc_u4 from exp53_spine_n2_integralpower.c.
   Fresh-audit ICX result on Intel Xeon 6973P-C:
     all = 0.467545 ns/input
     max ULP = 1, gt1 = 0 on the audit accuracy set.
   Separate directed-MPFR certification of the n=2 compensated path also
   established max ULP = 1, gt1 = 0 on the benchmark set and adversarial sweep.

   IMPORTANT: this file is a frozen checkpoint. Do not edit it for experiments;
   create a new source file instead.
*/
#include "exp53_spine_n2_integralpower.c"

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_n2_safe_rc_u4_frozen(double *restrict out,
                                const double *restrict in,
                                size_t n)
{
    exp53_n2_rc_u4(out,in,n);
}
