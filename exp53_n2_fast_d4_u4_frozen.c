/* Frozen checkpoint: faithful n=2 integral-power EXP spine, FAST path.

   Mathematical spine preserved:
     e^r - 1 = r Q4(r)^2
     Q4(r)=1+r/4+5r^2/96+r^3/128+79r^4/92160.

   Canonical kernel: exp53_n2_d4_u4a from exp53_n2_freshaudit.c.
   Fresh-audit ICX result on Intel Xeon 6973P-C:
     all = 0.360488 ns/input
     max ULP = 2, gt1 = 34 on the ICX audit screen.
   This is the aggressive speed checkpoint, NOT the certified-safe production path.

   IMPORTANT: this file is a frozen checkpoint. Do not edit it for experiments;
   create a new source file instead.
*/
#include "exp53_n2_freshaudit.c"

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_n2_fast_d4_u4_frozen(double *restrict out,
                                const double *restrict in,
                                size_t n)
{
    exp53_n2_d4_u4a(out,in,n);
}
