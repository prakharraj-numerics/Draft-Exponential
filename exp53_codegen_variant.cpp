#ifndef EXP53_VARIANT_FN
#error EXP53_VARIANT_FN required
#endif
#ifndef EXP53_TAIL_FN
#error EXP53_TAIL_FN required
#endif
#define restrict __restrict__
#define exp53_n2_vmstyle_u4_0381_frozen EXP53_VARIANT_FN
#define exp53_n2_fused_u4_038_frozen EXP53_TAIL_FN
extern "C" {
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
}
#undef exp53_n2_fused_u4_038_frozen
#undef exp53_n2_vmstyle_u4_0381_frozen
#undef restrict
