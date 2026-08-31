/* FROZEN INTEL COMPARATOR — oneMKL VM double exp, VML_HA.
   This is NOT the project's n=2 EXP math. It is the Intel comparator wrapper.

   Validation/benchmark snapshot:
     GitHub run 33374715828, shard 2
     CPU: Intel Xeon 6973P-C
     oneMKL: cached 2026.1.x environment used by project workflows
     threads: 1 (mkl_sequential / mkl_set_num_threads_local(1) in harness)
     mixed-domain accuracy screen: 200000 inputs, max ULP observed = 1
     n=12288: 0.328757458 ns/input
     n=65536: 0.328295687 ns/input

   Mode is deliberately explicit and frozen: VML_HA.
*/
#include <mkl.h>
#include <mkl_vml.h>
#include <stddef.h>

void exp53_intel_vml_ha_frozen(double *out, const double *in, size_t n)
{
    vmdExp((MKL_INT)n, in, out, VML_HA);
}
