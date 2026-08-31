/* Three saved VML-mode environments around the CURRENT frozen EXP53 n=2 kernel.

   IMPORTANT: the numerical kernel is NOT modified. All three entry points call
   exp53_n2_vmstyle_u4_0381_frozen() exactly. The only difference is the one-time
   oneMKL VML global mode selected before a benchmark/use block.

   This intentionally tests whether VML_HA / VML_LA / VML_EP can influence an
   arbitrary hand-written AVX-512 kernel that does not itself call a VML math
   routine. vmlSetMode() is kept OUTSIDE the timed kernel calls.
*/
#include <mkl_vml.h>
#include <stddef.h>

void exp53_n2_vmstyle_u4_0381_frozen(double*,const double*,size_t);

void exp53_n2_vml_ha_prepare(void){ (void)vmlSetMode(VML_HA); }
void exp53_n2_vml_la_prepare(void){ (void)vmlSetMode(VML_LA); }
void exp53_n2_vml_ep_prepare(void){ (void)vmlSetMode(VML_EP); }

void exp53_n2_vml_ha(double *out,const double *in,size_t n){
    exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
}
void exp53_n2_vml_la(double *out,const double *in,size_t n){
    exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
}
void exp53_n2_vml_ep(double *out,const double *in,size_t n){
    exp53_n2_vmstyle_u4_0381_frozen(out,in,n);
}
