/* Accuracy-first compensated Formula6 direct path.
   Removes the two dominant audited error sources:
   1) preserves low parts of near-1 reconstruction via TwoSum/FMA residuals;
   2) represents 2^(j/128) as TAB128[j] + TAB128_LO[j] and recovers
      the final product rounding residual with FMA.

   Speed is deliberately secondary. This file includes the hi-accuracy direct
   candidate so all shared symbols are emitted exactly once.
*/
#include "exp53_spine_v128_formula6_direct_hiacc.c"

static const double TAB128_LO[128] = {
  0x0.0p+0, 0x1.b61299ab8cdb7p-54, -0x1.19083535b085dp-56, -0x1.0a31c1977c96ep-54,
  0x1.d73e2a475b465p-55, -0x1.c91dfe2b13c27p-55, 0x1.186be4bb284ffp-57, 0x1.1487818316136p-54,
  0x1.8a62e4adc610bp-54, 0x1.01edc16e24f71p-54, 0x1.03a1727c57b53p-59, -0x1.b9bedc44ebd7bp-57,
  -0x1.6c51039449b3ap-54, -0x1.1b514b36ca5c7p-58, -0x1.32fbf9af1369ep-54, 0x1.2406ab9eeab0ap-55,
  -0x1.19041b9d78a76p-55, -0x1.11023d1970f6cp-54, 0x1.e5b4c7b4968e4p-55, -0x1.95386352ef607p-54,
  0x1.e016e00a2643cp-54, -0x1.1df98027bb78cp-54, 0x1.dc775814a8495p-55, 0x1.2a97e9494a5eep-55,
  0x1.9b07eb6c70573p-54, 0x1.ac155bef4f4a4p-55, 0x1.2bd339940e9d9p-55, -0x1.a4c3a8c3f0d7ep-54,
  0x1.612e8afad1255p-55, -0x1.10adcd6381aa4p-59, 0x1.0024754db41d5p-54, 0x1.1ca0f45d52383p-56,
  0x1.6f46ad23182e4p-55, 0x1.a9ce78e18047cp-55, 0x1.32721843659a6p-54, -0x1.b5cee5c4e4628p-55,
  -0x1.63aeabf42eae2p-54, -0x1.e958d3c9904bdp-54, -0x1.5e436d661f5e3p-56, -0x1.efff8375d29c3p-54,
  0x1.ada0911f09ebcp-55, -0x1.7d023f956f9f3p-54, -0x1.e9c23179c2893p-54, -0x1.02cf22526545ap-54,
  0x1.1e94e2f8ff106p-54, 0x1.0536cfc22b1e6p-54, -0x1.c7c46b071f2bep-54, 0x1.539edcb6c04c3p-54,
  0x1.26af9aac49f27p-54, -0x1.17b3c5d7eec5dp-54, 0x1.1bf3115591bcap-55, -0x1.0aee54f6192f5p-54,
  0x1.1fc539f7f5217p-55, -0x1.83b54f67dbe13p-54, 0x1.49922907495ccp-54, 0x1.9638c8b67b686p-57,
  0x1.1c78a1a8a5b59p-55, 0x1.60b06e6e40824p-55, 0x1.2017de96c029dp-54, -0x1.d7c54004c4cf1p-55,
  0x1.461fa260c9c54p-54, -0x1.b45e58292f369p-54, 0x1.86c95422e64a1p-55, -0x1.32bf3c2f4f07cp-54,
  0x1.90337887e94b8p-55, -0x1.b7c5fc0e21ca3p-54, 0x1.7bc0b1cd750a8p-54, -0x1.2257c8a7e0e0bp-54,
  0x1.e0a6b77a7a7c7p-54, -0x1.6f63bb2d39cd7p-55, -0x1.20f01f22d7d2cp-54, 0x1.530d6ff8663bcp-56,
  -0x1.2cf5b0c48e716p-54, 0x1.84e9989b110e0p-55, -0x1.fc6f89bd4f6bap-54, -0x1.10fc12a321d4fp-54,
  0x1.2ab1a6a9f2638p-54, 0x1.999b81c26d90dp-54, 0x1.1fa86f4e4aa09p-56, 0x1.d99c19ad0cb36p-54,
  -0x1.6787c5d913f38p-55, -0x1.396d990fbb644p-54, -0x1.881ecd211658ap-57, -0x1.7f6e1f0ad4ff0p-54,
  -0x1.5d1010ac9f586p-57, 0x1.8f6e9a9f0d0f8p-54, -0x1.4dbdf6c6a1e52p-55, -0x1.e8b3816f54db7p-54,
  -0x1.638feb7a7fb40p-54, -0x1.efcd30e54292ep-54, -0x1.072cb6ae6403dp-56, -0x1.30b8d6f0f7f1fp-54,
  -0x1.a2e1b9774f9bep-55, 0x1.44f52f3f9fe49p-54, 0x1.1377e94e8c7d2p-54, -0x1.0852a7d6b6f44p-54,
  -0x1.1fdd2e9c328dap-54, 0x1.73ef4d7cbed12p-54, -0x1.b2c9fbf3e9d0fp-55, 0x1.37ac66f98b9b4p-54,
  0x1.ce254794b17dap-55, -0x1.4472a6f5f0143p-54, -0x1.5d634a3bc0a78p-54, 0x1.2d1e154a5fcd2p-54,
  -0x1.16d95090ff654p-54, 0x1.f1a0a9464e8fdp-54, -0x1.ad5f5a2752f9ap-55, -0x1.2de3e13ef2b01p-55,
  -0x1.88f3c631516f2p-54, -0x1.0e6b041625994p-54, 0x1.5c415ae1b2b45p-54, 0x1.6b0f6fba59107p-56,
  0x1.a0f0787d1a415p-54, 0x1.01a5a4b9a59c9p-56, -0x1.1262f8d7dbefep-54, 0x1.c2b8b13cdd8acp-55,
  -0x1.4f8b00fd6bca0p-54, 0x1.396f813a9db53p-55, 0x1.2eb74966579e7p-57, -0x1.2f21bf9f16614p-54
};

static inline void two_sum_pd(__m512d a, __m512d b, __m512d *s, __m512d *e)
{
    __m512d z = _mm512_add_pd(a,b);
    __m512d bb = _mm512_sub_pd(z,a);
    __m512d aa = _mm512_sub_pd(z,bb);
    __m512d br = _mm512_sub_pd(b,bb);
    __m512d ar = _mm512_sub_pd(a,aa);
    *s = z;
    *e = _mm512_add_pd(ar,br);
}

static inline void residual_hi_lo(__m512d r, __m512d *rh, __m512d *rl)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d A0 = _mm512_set1_pd(1.0/8.0);
    const __m512d A1 = _mm512_set1_pd(1.0/384.0);
    const __m512d A2 = _mm512_set1_pd(1.0/46080.0);
    const __m512d A3 = _mm512_set1_pd(1.0/10321920.0);
    const __m512d B0 = _mm512_set1_pd(1.0/2.0);
    const __m512d B1 = _mm512_set1_pd(1.0/48.0);
    const __m512d B2 = _mm512_set1_pd(1.0/3840.0);
    const __m512d B3 = _mm512_set1_pd(1.0/645120.0);

    __m512d t = _mm512_mul_pd(r,r);
    __m512d Pc = _mm512_fmadd_pd(A3,t,A2);
    Pc = _mm512_fmadd_pd(Pc,t,A1);
    Pc = _mm512_fmadd_pd(Pc,t,A0);
    __m512d Ps = _mm512_fmadd_pd(B3,t,B2);
    Ps = _mm512_fmadd_pd(Ps,t,B1);
    Ps = _mm512_fmadd_pd(Ps,t,B0);

    __m512d C = _mm512_mul_pd(t,Pc);
    __m512d S = _mm512_mul_pd(r,Ps);

    __m512d twoS = _mm512_add_pd(S,S);
    __m512d ah, al;
    two_sum_pd(one,twoS,&ah,&al);

    __m512d cs = _mm512_add_pd(C,S);
    __m512d inner = _mm512_add_pd(two,cs);
    __m512d twoC = _mm512_add_pd(C,C);
    __m512d bh = _mm512_mul_pd(twoC,inner);
    __m512d bl = _mm512_fmadd_pd(twoC,inner,_mm512_sub_pd(_mm512_setzero_pd(),bh));

    __m512d sh, sl;
    two_sum_pd(ah,bh,&sh,&sl);
    *rh = sh;
    *rl = _mm512_add_pd(_mm512_add_pd(sl,al),bl);
}

static inline __m512d exp53_v128_formula6_compacc_block(
    __m512d x,const __m512d inv,const __m512d hi,const __m512d mi,
    const __m512d lo,const __m512d magic,const __m512i mb,const __m512i mask)
{
    __m512d biased=_mm512_fmadd_pd(x,inv,magic);
    __m512d k=_mm512_sub_pd(biased,magic);
    __m512i kn=_mm512_sub_epi64(_mm512_castpd_si512(biased),mb);

    __m512d r=_mm512_fnmadd_pd(k,hi,x);
    r=_mm512_fnmadd_pd(k,mi,r);
    r=_mm512_fnmadd_pd(k,lo,r);

    __m512d erh,erl;
    residual_hi_lo(r,&erh,&erl);

    __m512i j=_mm512_and_epi64(kn,mask);
    __m512i q=_mm512_srai_epi64(kn,7);
    __m512d th=_mm512_i64gather_pd(j,TAB128,8);
    __m512d tl=_mm512_i64gather_pd(j,TAB128_LO,8);

    __m512d ph=_mm512_mul_pd(erh,th);
    __m512d pe=_mm512_fmadd_pd(erh,th,_mm512_sub_pd(_mm512_setzero_pd(),ph));
    __m512d corr=pe;
    corr=_mm512_fmadd_pd(erh,tl,corr);
    corr=_mm512_fmadd_pd(erl,th,corr);
    corr=_mm512_fmadd_pd(erl,tl,corr);

    __m512d y=_mm512_add_pd(ph,corr);
    return _mm512_mul_pd(y,exp2_from_q(q));
}

__attribute__((target("avx512f,avx512dq,fma")))
void exp53_spine_v128_formula6_compacc(double *restrict out,const double *restrict in,size_t n)
{
    const __m512d inv=_mm512_set1_pd(INV128),hi=_mm512_set1_pd(L128_HI),
        mi=_mm512_set1_pd(L128_MI),lo=_mm512_set1_pd(L128_LO),
        magic=_mm512_set1_pd(MAGIC);
    const __m512i mb=_mm512_set1_epi64((long long)MAGIC_BITS),
        mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+32<=n;i+=32){
        __m512d x0=_mm512_loadu_pd(in+i),x1=_mm512_loadu_pd(in+i+8),
            x2=_mm512_loadu_pd(in+i+16),x3=_mm512_loadu_pd(in+i+24);
        __m512d y0=exp53_v128_formula6_compacc_block(x0,inv,hi,mi,lo,magic,mb,mask);
        __m512d y1=exp53_v128_formula6_compacc_block(x1,inv,hi,mi,lo,magic,mb,mask);
        __m512d y2=exp53_v128_formula6_compacc_block(x2,inv,hi,mi,lo,magic,mb,mask);
        __m512d y3=exp53_v128_formula6_compacc_block(x3,inv,hi,mi,lo,magic,mb,mask);
        _mm512_storeu_pd(out+i,y0);_mm512_storeu_pd(out+i+8,y1);
        _mm512_storeu_pd(out+i+16,y2);_mm512_storeu_pd(out+i+24,y3);
    }
    for(;i<n;){
        if(n-i>=8){
            _mm512_storeu_pd(out+i,exp53_v128_formula6_compacc_block(
                _mm512_loadu_pd(in+i),inv,hi,mi,lo,magic,mb,mask));
            i+=8;
        } else {
            for(;i<n;++i) out[i]=exp(in[i]);
        }
    }
}
