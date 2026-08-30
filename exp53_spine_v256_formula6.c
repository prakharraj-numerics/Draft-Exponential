/* v256 reduction + unchanged Formula6 six-op residual DAG.
   Preserves C/S spine and compares against the current v128 compensated winner.
   Residual bound halves to |r| <= ln2/512.
*/
#include "exp53_spine_v128_u4_formula6_tabcomp.c"

#define INV256  0x1.71547652b82fep+8
#define L256_HI 0x1.62e42fefa39efp-9
#define L256_MI 0x1.abc9e3b39803fp-64
#define L256_LO 0x1.7b57a079a1934p-119

static const double TAB256[256] = {
  0x1.0000000000000p+0, 0x1.00b1afa5abcbfp+0, 0x1.0163da9fb3335p+0, 0x1.02168143b0281p+0,
  0x1.02c9a3e778061p+0, 0x1.037d42e11bbccp+0, 0x1.04315e86e7f85p+0, 0x1.04e5f72f654b1p+0,
  0x1.059b0d3158574p+0, 0x1.0650a0e3c1f89p+0, 0x1.0706b29ddf6dep+0, 0x1.07bd42b72a836p+0,
  0x1.0874518759bc8p+0, 0x1.092bdf66607e0p+0, 0x1.09e3ecac6f383p+0, 0x1.0a9c79b1f3919p+0,
  0x1.0b5586cf9890fp+0, 0x1.0c0f145e46c85p+0, 0x1.0cc922b7247f7p+0, 0x1.0d83b23395decp+0,
  0x1.0e3ec32d3d1a2p+0, 0x1.0efa55fdfa9c5p+0, 0x1.0fb66affed31bp+0, 0x1.1073028d7233ep+0,
  0x1.11301d0125b51p+0, 0x1.11edbab5e2ab6p+0, 0x1.12abdc06c31ccp+0, 0x1.136a814f204abp+0,
  0x1.1429aaea92de0p+0, 0x1.14e95934f312ep+0, 0x1.15a98c8a58e51p+0, 0x1.166a45471c3c2p+0,
  0x1.172b83c7d517bp+0, 0x1.17ed48695bbc0p+0, 0x1.18af9388c8deap+0, 0x1.1972658375d2fp+0,
  0x1.1a35beb6fcb75p+0, 0x1.1af99f8138a1cp+0, 0x1.1bbe084045cd4p+0, 0x1.1c82f95281c6bp+0,
  0x1.1d4873168b9aap+0, 0x1.1e0e75eb44027p+0, 0x1.1ed5022fcd91dp+0, 0x1.1f9c18438ce4dp+0,
  0x1.2063b88628cd6p+0, 0x1.212be3578a819p+0, 0x1.21f49917ddc96p+0, 0x1.22bdda27912d1p+0,
  0x1.2387a6e756238p+0, 0x1.2451ffb82140ap+0, 0x1.251ce4fb2a63fp+0, 0x1.25e85711ece75p+0,
  0x1.26b4565e27cddp+0, 0x1.2780e341ddf29p+0, 0x1.284dfe1f56381p+0, 0x1.291ba7591bb70p+0,
  0x1.29e9df51fdee1p+0, 0x1.2ab8a66d10f13p+0, 0x1.2b87fd0dad990p+0, 0x1.2c57e39771b2fp+0,
  0x1.2d285a6e4030bp+0, 0x1.2df961f641589p+0, 0x1.2ecafa93e2f56p+0, 0x1.2f9d24abd886bp+0,
  0x1.306fe0a31b715p+0, 0x1.31432edeeb2fdp+0, 0x1.32170fc4cd831p+0, 0x1.32eb83ba8ea32p+0,
  0x1.33c08b26416ffp+0, 0x1.3496266e3fa2dp+0, 0x1.356c55f929ff1p+0, 0x1.36431a2de883bp+0,
  0x1.371a7373aa9cbp+0, 0x1.37f26231e754ap+0, 0x1.38cae6d05d866p+0, 0x1.39a401b7140efp+0,
  0x1.3a7db34e59ff7p+0, 0x1.3b57fbfec6cf4p+0, 0x1.3c32dc313a8e5p+0, 0x1.3d0e544ede173p+0,
  0x1.3dea64c123422p+0, 0x1.3ec70df1c5175p+0, 0x1.3fa4504ac801cp+0, 0x1.40822c367a024p+0,
  0x1.4160a21f72e2ap+0, 0x1.423fb2709468ap+0, 0x1.431f5d950a897p+0, 0x1.43ffa3f84b9d4p+0,
  0x1.44e086061892dp+0, 0x1.45c2042a7d232p+0, 0x1.46a41ed1d0057p+0, 0x1.4786d668b3237p+0,
  0x1.486a2b5c13cd0p+0, 0x1.494e1e192aed2p+0, 0x1.4a32af0d7d3dep+0, 0x1.4b17dea6db7d7p+0,
  0x1.4bfdad5362a27p+0, 0x1.4ce41b817c114p+0, 0x1.4dcb299fddd0dp+0, 0x1.4eb2d81d8abffp+0,
  0x1.4f9b2769d2ca7p+0, 0x1.508417f4531eep+0, 0x1.516daa2cf6642p+0, 0x1.5257de83f4eefp+0,
  0x1.5342b569d4f82p+0, 0x1.542e2f4f6ad27p+0, 0x1.551a4ca5d920fp+0, 0x1.56070dde910d2p+0,
  0x1.56f4736b527dap+0, 0x1.57e27dbe2c4cfp+0, 0x1.58d12d497c7fdp+0, 0x1.59c0827ff07ccp+0,
  0x1.5ab07dd485429p+0, 0x1.5ba11fba87a03p+0, 0x1.5c9268a5946b7p+0, 0x1.5d84590998b93p+0,
  0x1.5e76f15ad2148p+0, 0x1.5f6a320dceb71p+0, 0x1.605e1b976dc09p+0, 0x1.6152ae6cdf6f4p+0,
  0x1.6247eb03a5585p+0, 0x1.633dd1d1929fdp+0, 0x1.6434634ccc320p+0, 0x1.652b9febc8fb7p+0,
  0x1.6623882552225p+0, 0x1.671c1c70833f6p+0, 0x1.68155d44ca973p+0, 0x1.690f4b19e9538p+0,
  0x1.6a09e667f3bcdp+0, 0x1.6b052fa75173ep+0, 0x1.6c012750bdabfp+0, 0x1.6cfdcddd47645p+0,
  0x1.6dfb23c651a2fp+0, 0x1.6ef9298593ae5p+0, 0x1.6ff7df9519484p+0, 0x1.70f7466f42e87p+0,
  0x1.71f75e8ec5f74p+0, 0x1.72f8286ead08ap+0, 0x1.73f9a48a58174p+0, 0x1.74fbd35d7cbfdp+0,
  0x1.75feb564267c9p+0, 0x1.77024b1ab6e09p+0, 0x1.780694fde5d3fp+0, 0x1.790b938ac1cf6p+0,
  0x1.7a11473eb0187p+0, 0x1.7b17b0976cfdbp+0, 0x1.7c1ed0130c132p+0, 0x1.7d26a62ff86f0p+0,
  0x1.7e2f336cf4e62p+0, 0x1.7f3878491c491p+0, 0x1.80427543e1a12p+0, 0x1.814d2add106d9p+0,
  0x1.82589994cce13p+0, 0x1.8364c1eb941f7p+0, 0x1.8471a4623c7adp+0, 0x1.857f4179f5b21p+0,
  0x1.868d99b4492edp+0, 0x1.879cad931a436p+0, 0x1.88ac7d98a6699p+0, 0x1.89bd0a478580fp+0,
  0x1.8ace5422aa0dbp+0, 0x1.8be05bad61778p+0, 0x1.8cf3216b5448cp+0, 0x1.8e06a5e0866d9p+0,
  0x1.8f1ae99157736p+0, 0x1.902fed0282c8ap+0, 0x1.9145b0b91ffc6p+0, 0x1.925c353aa2fe2p+0,
  0x1.93737b0cdc5e5p+0, 0x1.948b82b5f98e5p+0, 0x1.95a44cbc8520fp+0, 0x1.96bdd9a7670b3p+0,
  0x1.97d829fde4e50p+0, 0x1.98f33e47a22a2p+0, 0x1.9a0f170ca07bap+0, 0x1.9b2bb4d53fe0dp+0,
  0x1.9c49182a3f090p+0, 0x1.9d674194bb8d5p+0, 0x1.9e86319e32323p+0, 0x1.9fa5e8d07f29ep+0,
  0x1.a0c667b5de565p+0, 0x1.a1e7aed8eb8bbp+0, 0x1.a309bec4a2d33p+0, 0x1.a42c980460ad8p+0,
  0x1.a5503b23e255dp+0, 0x1.a674a8af46052p+0, 0x1.a799e1330b358p+0, 0x1.a8bfe53c12e59p+0,
  0x1.a9e6b5579fdbfp+0, 0x1.ab0e521356ebap+0, 0x1.ac36bbfd3f37ap+0, 0x1.ad5ff3a3c2774p+0,
  0x1.ae89f995ad3adp+0, 0x1.afb4ce622f2ffp+0, 0x1.b0e07298db666p+0, 0x1.b20ce6c9a8952p+0,
  0x1.b33a2b84f15fbp+0, 0x1.b468415b749b1p+0, 0x1.b59728de5593ap+0, 0x1.b6c6e29f1c52ap+0,
  0x1.b7f76f2fb5e47p+0, 0x1.b928cf22749e4p+0, 0x1.ba5b030a1064ap+0, 0x1.bb8e0b79a6f1fp+0,
  0x1.bcc1e904bc1d2p+0, 0x1.bdf69c3f3a207p+0, 0x1.bf2c25bd71e09p+0, 0x1.c06286141b33dp+0,
  0x1.c199bdd85529cp+0, 0x1.c2d1cd9fa652cp+0, 0x1.c40ab5fffd07ap+0, 0x1.c544778fafb22p+0,
  0x1.c67f12e57d14bp+0, 0x1.c7ba88988c933p+0, 0x1.c8f6d9406e7b5p+0, 0x1.ca3405751c4dbp+0,
  0x1.cb720dcef9069p+0, 0x1.ccb0f2e6d1675p+0, 0x1.cdf0b555dc3fap+0, 0x1.cf3155b5bab74p+0,
  0x1.d072d4a07897cp+0, 0x1.d1b532b08c968p+0, 0x1.d2f87080d89f2p+0, 0x1.d43c8eacaa1d6p+0,
  0x1.d5818dcfba487p+0, 0x1.d6c76e862e6d3p+0, 0x1.d80e316c98398p+0, 0x1.d955d71ff6075p+0,
  0x1.da9e603db3285p+0, 0x1.dbe7cd63a8315p+0, 0x1.dd321f301b460p+0, 0x1.de7d5641c0658p+0,
  0x1.dfc97337b9b5fp+0, 0x1.e11676b197d17p+0, 0x1.e264614f5a129p+0, 0x1.e3b333b16ee12p+0,
  0x1.e502ee78b3ff6p+0, 0x1.e653924676d76p+0, 0x1.e7a51fbc74c83p+0, 0x1.e8f7977cdb740p+0,
  0x1.ea4afa2a490dap+0, 0x1.eb9f4867cca6ep+0, 0x1.ecf482d8e67f1p+0, 0x1.ee4aaa2188510p+0,
  0x1.efa1bee615a27p+0, 0x1.f0f9c1cb6412ap+0, 0x1.f252b376bba97p+0, 0x1.f3ac948dd7274p+0,
  0x1.f50765b6e4540p+0, 0x1.f6632798844f8p+0, 0x1.f7bfdad9cbe14p+0, 0x1.f91d802243c89p+0,
  0x1.fa7c1819e90d8p+0, 0x1.fbdba3692d514p+0, 0x1.fd3c22b8f71f1p+0, 0x1.fe9d96b2a23d9p+0,
};

static inline __m512d exp53_v256_formula6_block(__m512d x,
    const __m512d inv,const __m512d hi,const __m512d mi,const __m512d lo,
    const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);
    __m512d r=_mm512_fnmadd_pd(k,hi,x);
    r=_mm512_fnmadd_pd(k,mi,r);
    r=_mm512_fnmadd_pd(k,lo,r);
    __m512d er=spine_residual128_formula6(r);
    __m512i j=_mm512_and_epi64(kn,mask);
    __m512i q=_mm512_srai_epi64(kn,8);
    __m512d tab=_mm512_i64gather_pd(j,TAB256,8);
    return _mm512_mul_pd(_mm512_mul_pd(er,tab),exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v256_formula6(double *restrict out,const double *restrict in,size_t n)
{
    const __m512d inv=_mm512_set1_pd(INV256),hi=_mm512_set1_pd(L256_HI),
      mi=_mm512_set1_pd(L256_MI),lo=_mm512_set1_pd(L256_LO),
      magic=_mm512_set1_pd(MAGIC);
    const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),mask=_mm512_set1_epi64(255);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),
                x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=exp53_v256_formula6_block(x0,inv,hi,mi,lo,magic,mb,mask),
                y1=exp53_v256_formula6_block(x1,inv,hi,mi,lo,magic,mb,mask),
                y2=exp53_v256_formula6_block(x2,inv,hi,mi,lo,magic,mb,mask),
                y3=exp53_v256_formula6_block(x3,inv,hi,mi,lo,magic,mb,mask);
        _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
    }
    for(;i<n;){
        if(n-i>=8){
            _mm512_storeu_pd(out+i,exp53_v256_formula6_block(_mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));
            i+=8;
        } else {
            for(;i<n;++i) out[i]=exp(in[i]);
        }
    }
}
