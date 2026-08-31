/* FROZEN INTEL COMPARATOR — oneMKL VM double exp, VML_LA.
   This is NOT the project's n=2 EXP math. It is the Intel comparator wrapper.

   Validation/benchmark snapshot:
     GitHub run 33374715828, shard 2
     CPU: Intel Xeon 6973P-C
     oneMKL: cached 2026.1.x environment used by project workflows
     threads: 1 (mkl_sequential / mkl_set_num_threads_local(1) in harness)
     mixed-domain accuracy screen: 200000 inputs, max ULP observed = 1
     n=12288: 0.303744294 ns/input
     n=65536: 0.305159738 ns/input

   Important: max ULP = 1 is an observed result for this screen, not a universal
   VML_LA accuracy guarantee.
   Mode is deliberately explicit and frozen: VML_LA.
*/
#include <mkl.h>
#include <mkl_vml.h>
#include <stddef.h>

void exp53_intel_vml_la_frozen(double *out, const double *in, size_t n)
{
    vmdExp((MKL_INT)n, in, out, VML_LA);
}
