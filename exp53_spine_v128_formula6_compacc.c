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
  0x1.ada0911f09ebcp-55, -0x1.7d023f956f9f3p-54, -0x1.ef3691c309278p-58, -0x1.1c7dde35f7999p-55,
  0x1.89b7a04ef80d0p-59, 0x1.c944bd1648a76p-54, 0x1.3c1a3b69062f0p-56, 0x1.9cb62f3d1be56p-54,
  0x1.d4397afec42e2p-56, 0x1.8ecdbbc6a7833p-54, -0x1.4b309d25957e3p-54, -0x1.f768569bd93efp-55,
  -0x1.07abe1db13cadp-55, -0x1.d689cefede59bp-55, 0x1.9bb2c011d93adp-54, 0x1.295e15b9a1de8p-55,
  0x1.6324c054647adp-54, 0x1.c4b1b816986a2p-60, 0x1.ba6f93080e65ep-54, -0x1.3e2429b56de47p-54,
  -0x1.383c17e40b497p-54, -0x1.c483c759d8933p-55, -0x1.bb60987591c34p-54, 0x1.038ae44f73e65p-57,
  -0x1.bdd3413b26456p-54, -0x1.2895667ff0b0dp-56, -0x1.bbe3a683c88abp-57, -0x1.83c0f25860ef6p-55,
  -0x1.16e4786887a99p-55, -0x1.0a8d96c65d53cp-54, -0x1.0245957316dd3p-54, 0x1.866b80a02162dp-54,
  -0x1.41577ee04992fp-55, 0x1.f124cd1164dd6p-54, 0x1.05d02ba15797ep-56, -0x1.27c86626d972bp-54,
  -0x1.d4c1dd41532d8p-54, -0x1.8d684a341cdfbp-55, -0x1.fc6f89bd4f6bap-54, 0x1.994c2f37cb53ap-54,
  0x1.6e9f156864b27p-54, -0x1.0d55e32e9e3aap-56, 0x1.5cc13a2e3976cp-55, -0x1.dd6792e582524p-54,
  -0x1.75fc781b57ebcp-57, -0x1.64b7c96a5f039p-56, -0x1.d185b7c1b85d1p-54, -0x1.173bd91cee632p-54,
  0x1.c7c46b071f2bep-56, 0x1.824ca78e64c6ep-56, -0x1.359495d1cd533p-54, 0x1.6305c7ddc36abp-54,
  -0x1.d2f6edb8d41e1p-54, 0x1.bcb7ecac563c7p-54, 0x1.0fac90ef7fd31p-54, -0x1.f9234cae76cd0p-55,
  0x1.7a1cd345dcc81p-54, -0x1.bdef54c80e425p-54, -0x1.2805e3084d708p-57, -0x1.c71dfbbba6de3p-54,
  -0x1.5584f7e54ac3bp-56, -0x1.efcd30e54292ep-54, 0x1.23dd07a2d9e84p-55, -0x1.efdca3f6b9c73p-54,
  0x1.11065895048ddp-55, 0x1.b4537e083c60ap-54, 0x1.2884dff483cadp-54, 0x1.1acbc48805c44p-56,
  0x1.503cbd1e949dbp-56, -0x1.dd83b53829d72p-55, -0x1.cbc3743797a9cp-54, -0x1.d487b719d8578p-54,
  0x1.2ed02d75b3707p-55, -0x1.11ec18beddfe8p-54, 0x1.c2300696db532p-54, 0x1.2da5778f018c3p-54,
  -0x1.1a5cd4f184b5cp-54, -0x1.7b627817a1496p-54, 0x1.39e8980a9cc8fp-55, 0x1.2d522ca0c8de2p-54,
  -0x1.e9c23179c2893p-54, -0x1.c93f3b411ad8cp-54, 0x1.dc7f486a4b6b0p-54, 0x1.3a1a5bf0d8e43p-54,
  0x1.9d3e12dd8a18bp-54, -0x1.dbb12d006350ap-54, 0x1.74853f3a5931ep-55, 0x1.2eb74966579e7p-57,
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
