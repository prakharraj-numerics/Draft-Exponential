#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* Spine-preserving EXP53 variants.
   C(u)=z P(z), z=u^2.
   sqrt(C(C+2)) = 2u(P+zP') for the analytic branch.
   exp(r)=[1+C(u)+sqrt(C(C+2))]^2.
*/
static const double TAB64_HI[64] = {
  0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
  0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0,
  0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
  0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
  0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
  0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
  0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
  0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
  0x1.6a09e667f3bcdp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
  0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.869d99b4492edp+0,
  0x1.8ace5422aa0dbp+0, 0x1.8f0a6a7a2959ap+0, 0x1.935d8dddaaa78p+0, 0x1.97b6514c9d9b0p+0,
  0x1.9c49182a3f090p+0, 0x1.a0e3ec32d3d1ap+0, 0x1.a589994cce13bp+0, 0x1.aa39c3f4bdc8ap+0,
  0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
  0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0,
  0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
  0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0, 0x1.fa7c1819e90d8p+0,
};
static const double TAB64_LO[64] = {
  0x0.0p+0, -0x1.19041b9d78a76p-56, -0x1.0e1203176f1fdp-56, -0x1.0d67cb4bce3f3p-57,
  0x1.8a62e4adc610bp-54, 0x1.e62dbe8c18cdcp-56, -0x1.9f7490e4bb40bp-55, -0x1.c02be1d1d9bfep-57,
 -0x1.19041b9d78a76p-55, 0x1.e5b4c7b4968e4p-55, 0x1.641c4c902e3b4p-54, 0x1.35cf734bad3d2p-55,
  0x1.9b07eb6c70573p-54, -0x1.4f6b2a7609f71p-55, 0x1.11161a0e6d122p-54, -0x1.07abe1db13c90p-54,
  0x1.6f46ad23182e4p-55, 0x1.7f09c0a6ab6d0p-54, -0x1.8f7e49738c3c9p-54, 0x1.6f2fb5e47a22ap-55,
  0x1.ada0911f09ebcp-55, -0x1.95a2f97aa4e5bp-54, 0x1.1a1e9cb7285f7p-54, -0x1.0d67cb4bce3f3p-55,
  0x1.d4397afec42e2p-56, -0x1.2558d17f4f0fep-56, -0x1.5b45c6c0d53b7p-54, 0x1.7a1cd345dcc81p-55,
  0x1.6324c054647adp-54, -0x1.19041b9d78a76p-54, -0x1.9f7490e4bb40bp-54, 0x1.1a1e9cb7285f7p-55,
 -0x1.bdd3413b26456p-54, 0x1.18b7abb5569a4p-54, 0x1.d2ac258f87d03p-55, -0x1.15b45c6c0d53cp-54,
 -0x1.41577ee04992fp-55, -0x1.7f09c0a6ab6d0p-55, 0x1.5b45c6c0d53b7p-54, 0x1.2558d17f4f0fep-54,
  0x1.6e9f156864b27p-54, -0x1.1bcb7b1526e50p-55, 0x1.1a1e9cb7285f7p-54, 0x1.4f6b2a7609f71p-54,
  0x1.c7c46b071f2bep-56, -0x1.ef35793c76730p-54, -0x1.0e1203176f1fdp-54, -0x1.2558d17f4f0fep-54,
  0x1.7a1cd345dcc81p-54, 0x1.741aa5d4a0b48p-54, -0x1.0d67cb4bce3f3p-54, -0x1.4f6b2a7609f71p-54,
  0x1.11065895048ddp-55, -0x1.5d8f18e9c4cf3p-54, 0x1.35793c7673008p-54, 0x1.56f4736b527dap-54,
  0x1.2ed02d75b3707p-55, 0x1.118b7abb5569ap-54, -0x1.5d8f18e9c4cf3p-54, 0x1.456d49b62e93bp-54,
 -0x1.e9c23179c2893p-54, -0x1.39c3f4bdc8a13p-54, 0x1.0e1203176f1fdp-54, -0x1.6f2fb5e47a22ap-55,
};
static const double TAB128_HI[128] = {
  0x1.0000000000000p+0, 0x1.0163da9fb3335p+0, 0x1.02c9a3e778061p+0, 0x1.04315e86e7f85p+0,
  0x1.059b0d3158574p+0, 0x1.0706b29ddf6dep+0, 0x1.0874518759bc8p+0, 0x1.09e3ecac6f383p+0,
  0x1.0b5586cf9890fp+0, 0x1.0cc922b7247f7p+0, 0x1.0e3ec32d3d1a2p+0, 0x1.0fb66affed31bp+0,
  0x1.11301d0125b51p+0, 0x1.12abdc06c31ccp+0, 0x1.1429aaea92de0p+0, 0x1.15a98c8a58e51p+0,
  0x1.172b83c7d517bp+0, 0x1.18afa0d40a537p+0, 0x1.1a35beb6fcb75p+0, 0x1.1bbe084045cd4p+0,
  0x1.1d4873168b9aap+0, 0x1.1ed5022fcd91dp+0, 0x1.2063b88628cd6p+0, 0x1.21f49917ddc96p+0,
  0x1.2387a6e756238p+0, 0x1.251ce4fb2a63fp+0, 0x1.26b4565e27cddp+0, 0x1.284dfe1f56381p+0,
  0x1.29e9df51fdee1p+0, 0x1.2b87fd0dad990p+0, 0x1.2d285a6e4030bp+0, 0x1.2ecafa93e2f56p+0,
  0x1.306fe0a31b715p+0, 0x1.32170fc4cd831p+0, 0x1.33c08b26416ffp+0, 0x1.356c55f929ff1p+0,
  0x1.371a7373aa9cbp+0, 0x1.38cae6d05d866p+0, 0x1.3a7db34e59ff7p+0, 0x1.3c32dc313a8e5p+0,
  0x1.3dea64c123422p+0, 0x1.3fa4504ac801cp+0, 0x1.4160a21f72e2ap+0, 0x1.431f5d950a897p+0,
  0x1.44e086061892dp+0, 0x1.46a3c95ba71d1p+0, 0x1.486a2b5c13cd0p+0, 0x1.4a32f213b8efbp+0,
  0x1.4bfdad5362a27p+0, 0x1.4dcb299fddd0dp+0, 0x1.4f9b2769d2ca7p+0, 0x1.516daa2cf6642p+0,
  0x1.5342b569d4f82p+0, 0x1.551a4ca5d920fp+0, 0x1.56f4736b527dap+0, 0x1.58d12d497c7fdp+0,
  0x1.5ab07dd485429p+0, 0x1.5c9268a5946b7p+0, 0x1.5e76f15ad2148p+0, 0x1.605ddf7431d9bp+0,
  0x1.6247eb03a5585p+0, 0x1.6434d0055eb2cp+0, 0x1.6623882552225p+0, 0x1.68155d44ca973p+0,
  0x1.6a09e667f3bcdp+0, 0x1.6c012750bdabfp+0, 0x1.6dfb23c651a2fp+0, 0x1.6ff7df9519484p+0,
  0x1.71f75e8ec5f74p+0, 0x1.73f9a48a58174p+0, 0x1.75feb564267c9p+0, 0x1.7806fd2e5b50bp+0,
  0x1.7a11473eb0187p+0, 0x1.7c1ed0130c132p+0, 0x1.7e2f336cf4e62p+0, 0x1.80427543e1a12p+0,
  0x1.82589994cce13p+0, 0x1.8471a4623c7adp+0, 0x1.869d99b4492edp+0, 0x1.88cbdf0d8f00cp+0,
  0x1.8ace5422aa0dbp+0, 0x1.8cf3216b5448cp+0, 0x1.8f0a6a7a2959ap+0, 0x1.9144570cfd0f0p+0,
  0x1.935d8dddaaa78p+0, 0x1.957e0bd6e192ep+0, 0x1.97b6514c9d9b0p+0, 0x1.99f6e824b7bf0p+0,
  0x1.9c49182a3f090p+0, 0x1.9e9df51fdee12p+0, 0x1.a0e3ec32d3d1ap+0, 0x1.a3410b8d5b9c0p+0,
  0x1.a589994cce13bp+0, 0x1.a7e774e35f9fep+0, 0x1.aa39c3f4bdc8ap+0, 0x1.acae5e9d502aap+0,
  0x1.ae89f995ad3adp+0, 0x1.b0f3cb3f0e5dep+0, 0x1.b33a2b84f15fbp+0, 0x1.b5a3c79f48f04p+0,
  0x1.b7f76f2fb5e47p+0, 0x1.ba5b030a1064ap+0, 0x1.bcc1e904bc1d2p+0, 0x1.bf2b968ed74aap+0,
  0x1.c199bdd85529cp+0, 0x1.c40ab5fffd07ap+0, 0x1.c67f12e57d14bp+0, 0x1.c8f6d9406e7b5p+0,
  0x1.cb720dcef9069p+0, 0x1.cdf0b555dc3fap+0, 0x1.d072d4a07897cp+0, 0x1.d2f87080d89f2p+0,
  0x1.d5818dcfba487p+0, 0x1.d80e316c98398p+0, 0x1.da9e603db3285p+0, 0x1.dd3214c67a2c6p+0,
  0x1.dfc97337b9b5fp+0, 0x1.e264614f5a129p+0, 0x1.e502ee78b3ff6p+0, 0x1.e7a51fbc74c83p+0,
  0x1.ea4afa2a490dap+0, 0x1.ecf482d8e67f1p+0, 0x1.efa1bee615a27p+0, 0x1.f252b376bba97p+0,
  0x1.f50765b6e4540p+0, 0x1.f7bfdad9cbe14p+0, 0x1.fa7c1819e90d8p+0, 0x1.fd3c22b8f71f1p+0,
};
static const double TAB128_LO[128] = {
  0x0.0p+0, 0x1.b610a5d9300a2p-54, -0x1.19041b9d78a76p-56, 0x1.e9b9d588e4c59p-54,
 -0x1.0e1203176f1fdp-56, -0x1.234b73d9a08c9p-54, -0x1.0d67cb4bce3f3p-57, -0x1.7180d3db2b3f5p-54,
  0x1.8a62e4adc610bp-54, 0x1.e5b4c7b4968e4p-56, 0x1.e62dbe8c18cdcp-56, -0x1.09f84512f8b9ep-54,
 -0x1.9f7490e4bb40bp-55, -0x1.436bdaf94d8e5p-54, -0x1.c02be1d1d9bfep-57, 0x1.a6c8f32b2a9e3p-54,
 -0x1.19041b9d78a76p-55, -0x1.b3d2e6f3d00a5p-54, 0x1.e5b4c7b4968e4p-55, 0x1.01a9a3e778061p-54,
  0x1.641c4c902e3b4p-54, 0x1.2f2b98c7d517bp-54, 0x1.35cf734bad3d2p-55, -0x1.1ffb7f22f3f0cp-54,
  0x1.9b07eb6c70573p-54, -0x1.428aa3ba9e8f0p-54, -0x1.4f6b2a7609f71p-55, -0x1.4636e2a22f5d6p-55,
  0x1.11161a0e6d122p-54, -0x1.64b0f9b05e5b9p-54, -0x1.07abe1db13c90p-54, -0x1.7d7f4e0f30b19p-54,
  0x1.6f46ad23182e4p-55, -0x1.8f43f6f4f85e8p-54, 0x1.7f09c0a6ab6d0p-54, 0x1.4af8f2d7b6e1cp-54,
 -0x1.8f7e49738c3c9p-54, 0x1.0de5bd8c3a8f4p-54, 0x1.6f2fb5e47a22ap-55, 0x1.7386c48f1c15bp-54,
  0x1.ada0911f09ebcp-55, 0x1.c61164d5d4a17p-54, -0x1.95a2f97aa4e5bp-54, 0x1.0f7bb4e8f3d1ap-54,
  0x1.1a1e9cb7285f7p-54, 0x1.86b4aa1c96f6cp-55, -0x1.0d67cb4bce3f3p-55, -0x1.14c2f47c11c0ep-54,
  0x1.d4397afec42e2p-56, -0x1.3339b2e7e8a35p-55, -0x1.2558d17f4f0fep-56, 0x1.8f5a2b7e3f0b4p-54,
 -0x1.5b45c6c0d53b7p-54, 0x1.5f5e6f1b1b8a3p-54, 0x1.7a1cd345dcc81p-55, -0x1.29d4d6b7f6a28p-54,
  0x1.6324c054647adp-54, 0x1.7b1a8e3a9b1b0p-54, -0x1.19041b9d78a76p-54, -0x1.0fddfbf4f7c71p-54,
 -0x1.9f7490e4bb40bp-54, -0x1.58f5c42b6c8b6p-54, 0x1.1a1e9cb7285f7p-55, -0x1.26d8c1b6c0b31p-54,
 -0x1.bdd3413b26456p-54, -0x1.91d7aa8b6e8e1p-54, 0x1.18b7abb5569a4p-54, 0x1.2c8b0d5de775dp-54,
  0x1.d2ac258f87d03p-55, 0x1.66b7f83a0d12cp-54, -0x1.15b45c6c0d53cp-54, -0x1.1e6f4b54f79d0p-54,
 -0x1.41577ee04992fp-55, -0x1.0aa2b6ad9c151p-54, -0x1.7f09c0a6ab6d0p-55, -0x1.48e8f6a0bd1fdp-54,
  0x1.5b45c6c0d53b7p-54, -0x1.50f5b8ec8e904p-54, 0x1.2558d17f4f0fep-54, 0x1.1c1a36f1db0e9p-54,
  0x1.6e9f156864b27p-54, -0x1.60450db0f2b76p-54, -0x1.1bcb7b1526e50p-55, -0x1.7683fc16e7074p-54,
  0x1.1a1e9cb7285f7p-54, -0x1.0db8b70d0a0b7p-54, 0x1.4f6b2a7609f71p-54, 0x1.13b5965f70413p-54,
  0x1.c7c46b071f2bep-56, 0x1.c40411e7f0d3ep-54, -0x1.ef35793c76730p-54, 0x1.4c1d0a2b1fbc4p-54,
 -0x1.0e1203176f1fdp-54, -0x1.83f4d6dc8f1b2p-54, -0x1.2558d17f4f0fep-54, 0x1.9f26769a0b10ep-54,
  0x1.7a1cd345dcc81p-54, -0x1.5db8b25d1a226p-54, 0x1.741aa5d4a0b48p-54, -0x1.b6f84e6c1b215p-54,
 -0x1.0d67cb4bce3f3p-54, 0x1.26d1e94b65d11p-54, -0x1.4f6b2a7609f71p-54, 0x1.76f0d2b8d5a0cp-54,
  0x1.11065895048ddp-55, 0x1.6f6d5ac7f9c29p-54, -0x1.5d8f18e9c4cf3p-54, -0x1.74a21b5aab6c3p-54,
  0x1.35793c7673008p-54, -0x1.2c37b83b14c94p-54, 0x1.56f4736b527dap-54, -0x1.392ed0d1cf983p-54,
  0x1.2ed02d75b3707p-55, 0x1.15d2554f2ba66p-54, 0x1.118b7abb5569ap-54, -0x1.202b04c75b494p-54,
 -0x1.5d8f18e9c4cf3p-54, -0x1.49f4ca8a76413p-54, 0x1.456d49b62e93bp-54, -0x1.557a5cb55f857p-54,
 -0x1.e9c23179c2893p-54, 0x1.a2cb4b7555d84p-54, -0x1.39c3f4bdc8a13p-54, -0x1.64cfe4d9ec252p-54,
  0x1.0e1203176f1fdp-54, 0x1.06c64d9d0be1cp-54, -0x1.6f2fb5e47a22ap-55, 0x1.331996dfaa3f6p-54,
};
#define INV64  0x1.71547652b82fep+6
#define L64_HI 0x1.62e42fefa39efp-7
#define L64_MI 0x1.abc9e3b39803fp-62
#define L64_LO 0x1.7b57a079a1934p-117
#define INV128  0x1.71547652b82fep+7
#define L128_HI 0x1.62e42fefa39efp-8
#define L128_MI 0x1.abc9e3b39803fp-63
#define L128_LO 0x1.7b57a079a1934p-118
static inline __m512d spine_shared_small(__m512d r){const __m512d one=_mm512_set1_pd(1.0),half=_mm512_set1_pd(0.5),a0=half,a1=_mm512_set1_pd(1.0/24.0),a2=_mm512_set1_pd(1.0/720.0);__m512d u=_mm512_mul_pd(r,half),z=_mm512_mul_pd(u,u),p=a2,dp=_mm512_setzero_pd();dp=_mm512_fmadd_pd(dp,z,p);p=_mm512_fmadd_pd(p,z,a1);dp=_mm512_fmadd_pd(dp,z,p);p=_mm512_fmadd_pd(p,z,a0);__m512d C=_mm512_mul_pd(z,p),pp=_mm512_fmadd_pd(z,dp,p),s=_mm512_mul_pd(_mm512_add_pd(u,u),pp),h=_mm512_add_pd(one,_mm512_add_pd(C,s));return _mm512_mul_pd(h,h);}
static inline __m512d expbits_from_q64(__m512i q){__m512i e=_mm512_add_epi64(q,_mm512_set1_epi64(1023));return _mm512_castsi512_pd(_mm512_slli_epi64(e,52));}
#define MAGIC 0x1.8000000000000p+52
#define MAGIC_BITS 0x4338000000000000ULL
__attribute__((target("avx512f,avx512dq,fma"))) static inline void core64(double*out,const double*in,size_t n,int bits){const __m512d inv=_mm512_set1_pd(INV64),hi=_mm512_set1_pd(L64_HI),mi=_mm512_set1_pd(L64_MI),lo=_mm512_set1_pd(L64_LO);size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i),kd=_mm512_mul_pd(x,inv);__m256i k32=_mm512_cvt_roundpd_epi32(kd,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512d k=_mm512_cvtepi32_pd(k32);__m512d r=_mm512_fnmadd_pd(k,hi,x);r=_mm512_fnmadd_pd(k,mi,r);r=_mm512_fnmadd_pd(k,lo,r);__m512d er=spine_shared_small(r);__m256i j=_mm256_and_si256(k32,_mm256_set1_epi32(63)),q32=_mm256_srai_epi32(k32,6);__m512d th=_mm512_i32gather_pd(j,TAB64_HI,8),tl=_mm512_i32gather_pd(j,TAB64_LO,8),t=_mm512_fmadd_pd(er,tl,_mm512_mul_pd(er,th)),y;if(bits)y=_mm512_mul_pd(t,expbits_from_q64(_mm512_cvtepi32_epi64(q32)));else y=_mm512_scalef_pd(t,_mm512_cvtepi32_pd(q32));_mm512_storeu_pd(out+i,y);}for(;i<n;i++)out[i]=exp(in[i]);}
__attribute__((target("avx512f,avx512dq,fma"))) static inline void core128(double*out,const double*in,size_t n,int hilo){const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),magic=_mm512_set1_pd(MAGIC);const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(127);size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i),biased=_mm512_fmadd_pd(x,inv,magic),k=_mm512_sub_pd(biased,magic);__m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);__m512d r=_mm512_fnmadd_pd(k,hi,x);r=_mm512_fnmadd_pd(k,mi,r);r=_mm512_fnmadd_pd(k,lo,r);__m512d er=spine_shared_small(r);__m512i j=_mm512_and_epi64(kn,mask),q=_mm512_srai_epi64(kn,7);__m512d th=_mm512_i64gather_pd(j,TAB128_HI,8),t;if(hilo){__m512d tl=_mm512_i64gather_pd(j,TAB128_LO,8);t=_mm512_fmadd_pd(er,tl,_mm512_mul_pd(er,th));}else t=_mm512_mul_pd(er,th);_mm512_storeu_pd(out+i,_mm512_mul_pd(t,expbits_from_q64(q)));}for(;i<n;i++)out[i]=exp(in[i]);}
void exp53_spine_v64_scalef(double*out,const double*in,size_t n){core64(out,in,n,0);}void exp53_spine_v64_bits(double*out,const double*in,size_t n){core64(out,in,n,1);}void exp53_spine_v128_magic_hi(double*out,const double*in,size_t n){core128(out,in,n,0);}void exp53_spine_v128_magic_hilo(double*out,const double*in,size_t n){core128(out,in,n,1);}
static const double TAB16_HI[16]={0x1.0000000000000p+0,0x1.0b5586cf9890fp+0,0x1.172b83c7d517bp+0,0x1.2387a6e756238p+0,0x1.306fe0a31b715p+0,0x1.3dea64c123422p+0,0x1.4bfdad5362a27p+0,0x1.5ab07dd485429p+0,0x1.6a09e667f3bcdp+0,0x1.7a11473eb0187p+0,0x1.8ace5422aa0dbp+0,0x1.9c49182a3f090p+0,0x1.ae89f995ad3adp+0,0x1.c199bdd85529cp+0,0x1.d5818dcfba487p+0,0x1.ea4afa2a490dap+0};
static const double TAB16_LO[16]={0x0p+0,0x1.8a62e4adc610bp-54,-0x1.19041b9d78a76p-55,0x1.9b07eb6c70573p-54,0x1.6f46ad23182e4p-55,0x1.ada0911f09ebcp-55,0x1.d4397afec42e2p-56,0x1.6324c054647adp-54,-0x1.bdd3413b26456p-54,-0x1.41577ee04992fp-55,0x1.6e9f156864b27p-54,0x1.c7c46b071f2bep-56,0x1.7a1cd345dcc81p-54,0x1.11065895048ddp-55,0x1.2ed02d75b3707p-55,-0x1.e9c23179c2893p-54};
#define INV16 0x1.71547652b82fep+4
#define L16_HI 0x1.62e42fefa39efp-5
#define L16_MI 0x1.abc9e3b39803fp-60
#define L16_LO 0x1.7b57a079a1934p-115
static inline __m512d spine_old(__m512d r){const __m512d h=_mm512_set1_pd(.5),o=_mm512_set1_pd(1.0);__m512d u=_mm512_mul_pd(r,h),z=_mm512_mul_pd(u,u),p=_mm512_set1_pd(1.0/3628800.0);p=_mm512_fmadd_pd(p,z,_mm512_set1_pd(1.0/40320.0));p=_mm512_fmadd_pd(p,z,_mm512_set1_pd(1.0/720.0));p=_mm512_fmadd_pd(p,z,_mm512_set1_pd(1.0/24.0));p=_mm512_fmadd_pd(p,z,h);__m512d C=_mm512_mul_pd(z,p),q=_mm512_set1_pd(1.0/39916800.0);q=_mm512_fmadd_pd(q,z,_mm512_set1_pd(1.0/362880.0));q=_mm512_fmadd_pd(q,z,_mm512_set1_pd(1.0/5040.0));q=_mm512_fmadd_pd(q,z,_mm512_set1_pd(1.0/120.0));q=_mm512_fmadd_pd(q,z,_mm512_set1_pd(1.0/6.0));__m512d s=_mm512_fmadd_pd(_mm512_mul_pd(u,z),q,u),e=_mm512_add_pd(o,_mm512_add_pd(C,s));return _mm512_mul_pd(e,e);}
__attribute__((target("avx512f,avx512dq,fma"))) void exp53_spine_v16_baseline(double*out,const double*in,size_t n){const __m512d inv=_mm512_set1_pd(INV16),hi=_mm512_set1_pd(L16_HI),mi=_mm512_set1_pd(L16_MI),lo=_mm512_set1_pd(L16_LO);size_t i=0;for(;i+8<=n;i+=8){__m512d x=_mm512_loadu_pd(in+i),kd=_mm512_mul_pd(x,inv);__m256i k32=_mm512_cvt_roundpd_epi32(kd,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);__m512d k=_mm512_cvtepi32_pd(k32),r=_mm512_fnmadd_pd(k,hi,x);r=_mm512_fnmadd_pd(k,mi,r);r=_mm512_fnmadd_pd(k,lo,r);__m512d er=spine_old(r);__m256i j=_mm256_and_si256(k32,_mm256_set1_epi32(15)),q=_mm256_srai_epi32(k32,4);__m512d th=_mm512_i32gather_pd(j,TAB16_HI,8),tl=_mm512_i32gather_pd(j,TAB16_LO,8),t=_mm512_fmadd_pd(er,tl,_mm512_mul_pd(er,th));_mm512_storeu_pd(out+i,_mm512_scalef_pd(t,_mm512_cvtepi32_pd(q)));}for(;i<n;i++)out[i]=exp(in[i]);}
