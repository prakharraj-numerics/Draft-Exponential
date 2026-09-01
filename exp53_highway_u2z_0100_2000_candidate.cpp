// Experimental Highway u2z backend preserving frozen EXP53 arithmetic.
#define HWY_COMPILE_ONLY_STATIC
#include "hwy/highway.h"
#define restrict __restrict__
extern "C" {
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
}
#undef restrict
namespace hn = hwy::HWY_NAMESPACE;

__attribute__((noinline))
void exp53_highway_u2z_0100_2000_candidate(double* out,const double* in,size_t n){
  const hn::FixedTag<double,8> d; const hn::RebindToSigned<decltype(d)> di;
  const auto inv=hn::Set(d,N2F_INV128),hi=hn::Set(d,N2F_L128_HI),mi=hn::Set(d,N2F_L128_MI),lo=hn::Set(d,N2F_L128_LO),magic=hn::Set(d,N2F_MAGIC),one=hn::Set(d,1.0),nq1=hn::Set(d,N2F_Q1),nq2=hn::Set(d,N2F_Q2),nq3=hn::Set(d,N2F_Q3),nq4=hn::Set(d,N2F_Q4);
  const auto mb=hn::Set(di,(int64_t)N2F_MAGIC_BITS),mask=hn::Set(di,127); const auto*tab_bits=reinterpret_cast<const int64_t*>(N2_FROZEN_TAB128);
  size_t i=0;
  for(;i+16<=n;i+=16){
    hn::Vec<decltype(d)> x[2],biased[2],k[2],r[2],h[2],s[2],er[2],el[2],scale[2],ph[2],y[2]; hn::Vec<decltype(di)> kn[2],j[2],q[2],tb[2],sb[2];
    for(int L=0;L<2;++L)x[L]=hn::LoadU(d,in+i+8*L);
    for(int L=0;L<2;++L)biased[L]=hn::MulAdd(x[L],inv,magic);
    for(int L=0;L<2;++L){k[L]=hn::Sub(biased[L],magic);kn[L]=hn::Sub(hn::BitCast(di,biased[L]),mb);j[L]=hn::And(kn[L],mask);q[L]=hn::ShiftRight<7>(kn[L]);}
    for(int L=0;L<2;++L)tb[L]=hn::GatherIndex(di,tab_bits,j[L]);
    for(int L=0;L<2;++L){r[L]=hn::NegMulAdd(k[L],hi,x[L]);r[L]=hn::NegMulAdd(k[L],mi,r[L]);r[L]=hn::NegMulAdd(k[L],lo,r[L]);}
    for(int L=0;L<2;++L)h[L]=hn::MulAdd(nq4,r[L],nq3); for(int L=0;L<2;++L)h[L]=hn::MulAdd(h[L],r[L],nq2); for(int L=0;L<2;++L)h[L]=hn::MulAdd(h[L],r[L],nq1); for(int L=0;L<2;++L)h[L]=hn::MulAdd(h[L],r[L],one);
    for(int L=0;L<2;++L)s[L]=hn::Mul(h[L],h[L]); for(int L=0;L<2;++L)er[L]=hn::MulAdd(r[L],s[L],one); for(int L=0;L<2;++L)el[L]=hn::MulAdd(r[L],s[L],hn::Sub(one,er[L]));
    for(int L=0;L<2;++L){sb[L]=hn::Add(tb[L],hn::ShiftLeft<52>(q[L]));scale[L]=hn::BitCast(d,sb[L]);}
    for(int L=0;L<2;++L){ph[L]=hn::Mul(er[L],scale[L]);y[L]=hn::MulAdd(el[L],scale[L],ph[L]);hn::StoreU(y[L],d,out+i+8*L);}
  }
  if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
