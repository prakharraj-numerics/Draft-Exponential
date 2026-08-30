/* FROZEN best safe n=2 checkpoint, 2026-08-31.

   Mathematical spine preserved:
     er = 1 + r * Q4(r)^2
   with the faithful degree-4 Q(r).

   Minimal accuracy repair:
     el = fma(r, Q4(r)^2, 1 - er)
     y  = fma(el, T[j], er*T[j]) * 2^q

   This intentionally omits the ordinary product-error compensation used by the
   older RC path; diagnosis showed ER-low alone fixed every observed FAST >1-ULP
   case on the current 12288-input screen.

   Best measured toolchain/run:
     Intel icx -O3 -xHost -qopt-zmm-usage=high -fp-model=precise
     -fno-math-errno -DNDEBUG
     Intel Xeon 6973P-C, workflow run 33336052293, job 99323023372

   Accuracy screen:
     max ULP = 1
     >1 ULP  = 0

   Timing, all inputs:
     exp53_n2_el_u4 = 0.402858 ns/input
     raw FAST u4    = 0.355638 ns/input (max ULP 2)
     full RC u4     = 0.455283 ns/input
     Intel VML HA   = 0.322104 ns/input

   Immutable checkpoint wrapper. Do not modify for experiments; create a new file.
*/
#include "exp53_n2_elonly.c"

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_n2_elonly_u4_frozen(double *restrict out,
                               const double *restrict in,
                               size_t n)
{
    exp53_n2_el_u4(out,in,n);
}
