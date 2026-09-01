#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#define restrict __restrict__
extern "C" {
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
}
#undef restrict
namespace hn=hwy::HWY_NAMESPACE;
static uint64_t sm(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}static double u(uint64_t h){return((double)(h>>11)+.5)/9007199254740992.;}
static uint64_t bits(double x){uint64_t b;memcpy(&b,&x,8);return b;}
int main(){
  double in[125]; uint64_t seed=0x243f6a8885a308d3ULL^((uint64_t)125<<19); for(size_t i=0;i<125;i++){double q=u(sm(seed+i*0x9e3779b97f4a7c15ULL)),a=0x1p-20+q*(1.-0x1p-19);in[i]=(i&1)?-a:a;}
  const size_t base=96,lane=3; alignas(64) double x8[8]; for(int i=0;i<8;i++)x8[i]=in[base+i];
  // intrinsic exact path
  __m512d x=_mm512_load_pd(x8),inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI),mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO),magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0),q1=_mm512_set1_pd(N2F_Q1),q2=_mm512_set1_pd(N2F_Q2),q3=_mm512_set1_pd(N2F_Q3),q4=_mm512_set1_pd(N2F_Q4);
  __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127);
  __m512d ib=_mm512_fmadd_pd(x,inv,magic),ik=_mm512_sub_pd(ib,magic); __m512i ikn=_mm512_sub_epi64(_mm512_castpd_si512(ib),mb),ij=_mm512_and_epi64(ikn,mask),iq=_mm512_srai_epi64(ikn,7),itb=_mm512_i64gather_epi64(ij,(const long long*)N2_FROZEN_TAB128,8);
  __m512d ir=_mm512_fnmadd_pd(ik,hi,x);ir=_mm512_fnmadd_pd(ik,mi,ir);ir=_mm512_fnmadd_pd(ik,lo,ir); __m512d ih=_mm512_fmadd_pd(q4,ir,q3);ih=_mm512_fmadd_pd(ih,ir,q2);ih=_mm512_fmadd_pd(ih,ir,q1);ih=_mm512_fmadd_pd(ih,ir,one);__m512d is=_mm512_mul_pd(ih,ih),ier=_mm512_fmadd_pd(ir,is,one),iel=_mm512_fmadd_pd(ir,is,_mm512_sub_pd(one,ier));__m512i isb=_mm512_add_epi64(itb,_mm512_slli_epi64(iq,52));__m512d isc=_mm512_castsi512_pd(isb),iph=_mm512_mul_pd(ier,isc),iy=_mm512_fmadd_pd(iel,isc,iph);
  alignas(64) double I[10][8];_mm512_store_pd(I[0],ib);_mm512_store_pd(I[1],ik);_mm512_store_pd(I[2],ir);_mm512_store_pd(I[3],ih);_mm512_store_pd(I[4],is);_mm512_store_pd(I[5],ier);_mm512_store_pd(I[6],iel);_mm512_store_pd(I[7],isc);_mm512_store_pd(I[8],iph);_mm512_store_pd(I[9],iy);
  // Highway path
  const hn::FixedTag<double,8>d;const hn::RebindToSigned<decltype(d)>di;auto hx=hn::Load(d,x8),hinv=hn::Set(d,N2F_INV128),hhi=hn::Set(d,N2F_L128_HI),hmi=hn::Set(d,N2F_L128_MI),hlo=hn::Set(d,N2F_L128_LO),hmagic=hn::Set(d,N2F_MAGIC),hone=hn::Set(d,1.0),hq1=hn::Set(d,N2F_Q1),hq2=hn::Set(d,N2F_Q2),hq3=hn::Set(d,N2F_Q3),hq4=hn::Set(d,N2F_Q4);auto hmb=hn::Set(di,(int64_t)N2F_MAGIC_BITS),hmask=hn::Set(di,127);auto hb=hn::MulAdd(hx,hinv,hmagic),hk=hn::Sub(hb,hmagic),hkn=hn::Sub(hn::BitCast(di,hb),hmb),hj=hn::And(hkn,hmask),hqq=hn::ShiftRight<7>(hkn),htb=hn::GatherIndex(di,(const int64_t*)N2_FROZEN_TAB128,hj);auto hr=hn::NegMulAdd(hk,hhi,hx);hr=hn::NegMulAdd(hk,hmi,hr);hr=hn::NegMulAdd(hk,hlo,hr);auto hh=hn::MulAdd(hq4,hr,hq3);hh=hn::MulAdd(hh,hr,hq2);hh=hn::MulAdd(hh,hr,hq1);hh=hn::MulAdd(hh,hr,hone);auto hs=hn::Mul(hh,hh),her=hn::MulAdd(hr,hs,hone),hel=hn::MulAdd(hr,hs,hn::Sub(hone,her));auto hsb=hn::Add(htb,hn::ShiftLeft<52>(hqq)),hsc=hn::BitCast(d,hsb),hph=hn::Mul(her,hsc),hy=hn::MulAdd(hel,hsc,hph);
  alignas(64) double H[10][8];hn::Store(hb,d,H[0]);hn::Store(hk,d,H[1]);hn::Store(hr,d,H[2]);hn::Store(hh,d,H[3]);hn::Store(hs,d,H[4]);hn::Store(her,d,H[5]);hn::Store(hel,d,H[6]);hn::Store(hsc,d,H[7]);hn::Store(hph,d,H[8]);hn::Store(hy,d,H[9]);
  const char*name[]={"biased","k","r","h","s","er","el","scale","ph","y"};printf("index=99 x=%.17g bits=%016llx\n",in[99],(unsigned long long)bits(in[99]));for(int st=0;st<10;st++)printf("STAGE %s intrinsic=%016llx highway=%016llx diff=%d\n",name[st],(unsigned long long)bits(I[st][lane]),(unsigned long long)bits(H[st][lane]),bits(I[st][lane])!=bits(H[st][lane]));
}
