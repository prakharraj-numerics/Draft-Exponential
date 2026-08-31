/* VM-style execution attack on the frozen n=2 kernel.
   MATH MUST NOT CHANGE. This file includes the untouched frozen checkpoint and
   only changes instruction scheduling / batch organization.

   Exact common-path DAG per lane remains:
     k/r reduction -> Q4(r) Horner -> s=Q4^2 -> er=1+r*s
     -> el=fma(r,s,1-er) -> scale(TAB128[j],q) -> er*scale + el*scale.

   Main scheduling change: j/q are available before residual reduction, so the
   TAB128 gather is issued immediately, then residual/Q4/ER-low work is done
   while the gather can be in flight. U2/U3/U4/U5 sweep the independent-vector
   bank width. Tails delegate to the frozen function, so no alternate math is
   introduced anywhere.
*/
#include "exp53_n2_fused_u4_038_frozen.c"

#define VM_TARGET __attribute__((target("avx512f,avx512dq,fma"),noinline))

#define DEFINE_VM_EARLY(NAME,U) \
VM_TARGET void NAME(double *restrict out,const double *restrict in,size_t n){ \
    const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI), \
      mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO), \
      magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0), \
      nq1=_mm512_set1_pd(N2F_Q1),nq2=_mm512_set1_pd(N2F_Q2), \
      nq3=_mm512_set1_pd(N2F_Q3),nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127); \
    size_t i=0; \
    for(;i+8*(U)<=n;i+=8*(U)){ \
      __m512d x[U],biased[U],k[U],r[U],h[U],s[U],er[U],el[U],scale[U],ph[U],y[U]; \
      __m512i kn[U],j[U],q[U],tb[U],sb[U]; \
      for(int L=0;L<(U);L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
      for(int L=0;L<(U);L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
      for(int L=0;L<(U);L++){ \
        k[L]=_mm512_sub_pd(biased[L],magic); \
        kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); \
        j[L]=_mm512_and_epi64(kn[L],mask); \
        q[L]=_mm512_srai_epi64(kn[L],7); \
      } \
      /* VM-style latency hiding: launch every independent gather ASAP. */ \
      for(int L=0;L<(U);L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
      for(int L=0;L<(U);L++){ \
        r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
        r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
        r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
      } \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
      for(int L=0;L<(U);L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
      for(int L=0;L<(U);L++) s[L]=_mm512_mul_pd(h[L],h[L]); \
      for(int L=0;L<(U);L++) er[L]=_mm512_fmadd_pd(r[L],s[L],one); \
      for(int L=0;L<(U);L++) el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L])); \
      for(int L=0;L<(U);L++){ \
        sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); \
        scale[L]=_mm512_castsi512_pd(sb[L]); \
      } \
      for(int L=0;L<(U);L++){ \
        ph[L]=_mm512_mul_pd(er[L],scale[L]); \
        y[L]=_mm512_fmadd_pd(el[L],scale[L],ph[L]); \
        _mm512_storeu_pd(out+i+8*L,y[L]); \
      } \
    } \
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i); \
}

DEFINE_VM_EARLY(exp53_n2_vm_early_u2,2)
DEFINE_VM_EARLY(exp53_n2_vm_early_u3,3)
DEFINE_VM_EARLY(exp53_n2_vm_early_u4,4)
DEFINE_VM_EARLY(exp53_n2_vm_early_u5,5)

/* Same U4 schedule, but use aligned memory ops when both arrays are 64B aligned.
   This is an API-level fast-path dispatch only; otherwise it calls early_u4. */
VM_TARGET void exp53_n2_vm_early_u4_aligned(double *restrict out,const double *restrict in,size_t n){
    if((((uintptr_t)out|(uintptr_t)in)&63u)!=0){ exp53_n2_vm_early_u4(out,in,n); return; }
    const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI),
      mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO),magic=_mm512_set1_pd(N2F_MAGIC),
      one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2F_Q1),nq2=_mm512_set1_pd(N2F_Q2),
      nq3=_mm512_set1_pd(N2F_Q3),nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+32<=n;i+=32){
      __m512d x[4],biased[4],k[4],r[4],h[4],s[4],er[4],el[4],scale[4];
      __m512i kn[4],j[4],q[4],tb[4];
      for(int L=0;L<4;L++) x[L]=_mm512_load_pd(in+i+8*L);
      for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
      for(int L=0;L<4;L++){k[L]=_mm512_sub_pd(biased[L],magic);kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);j[L]=_mm512_and_epi64(kn[L],mask);q[L]=_mm512_srai_epi64(kn[L],7);}
      for(int L=0;L<4;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
      for(int L=0;L<4;L++){r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);}
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
      for(int L=0;L<4;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);
      for(int L=0;L<4;L++){s[L]=_mm512_mul_pd(h[L],h[L]);er[L]=_mm512_fmadd_pd(r[L],s[L],one);el[L]=_mm512_fmadd_pd(r[L],s[L],_mm512_sub_pd(one,er[L]));}
      for(int L=0;L<4;L++){__m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));scale[L]=_mm512_castsi512_pd(sb);}
      for(int L=0;L<4;L++){__m512d ph=_mm512_mul_pd(er[L],scale[L]);__m512d y=_mm512_fmadd_pd(el[L],scale[L],ph);_mm512_store_pd(out+i+8*L,y);}
    }
    if(i<n) exp53_n2_fused_u4_038_frozen(out+i,in+i,n-i);
}
