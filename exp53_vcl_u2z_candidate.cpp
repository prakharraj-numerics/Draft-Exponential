/* Experimental VCL re-expression of the frozen u2z small-batch kernel.
   VCL is used only as the 512-bit floating-vector abstraction. Integer/table
   bit manipulation stays intrinsic because the fused table/exponent layout is
   part of the existing kernel design. Mathematical constants, FMA order,
   reduction, Q4, ER-low correction and table are identical to frozen u2z.
   Production/frozen files are untouched.
*/
#include "vectorclass.h"
extern "C" {
#include "exp53_n2_vmstyle_u4_0381_frozen.c"
}

extern "C" __attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_vcl_u2z_candidate(double *restrict out,const double *restrict in,size_t n){
    const Vec8d inv(N2F_INV128),hi(N2F_L128_HI),mi(N2F_L128_MI),lo(N2F_L128_LO),
                magic(N2F_MAGIC),one(1.0),nq1(N2F_Q1),nq2(N2F_Q2),nq3(N2F_Q3),nq4(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+16<=n;i+=16){
        Vec8d x[2],biased[2],k[2],r[2],h[2],s[2],er[2],el[2],scale[2],ph[2],y[2];
        __m512i kn[2],j[2],q[2],tb[2],sb[2];
        for(int L=0;L<2;L++) x[L].load(in+i+8*L);
        for(int L=0;L<2;L++) biased[L]=mul_add(x[L],inv,magic);
        for(int L=0;L<2;L++){
            k[L]=biased[L]-magic;
            kn[L]=_mm512_sub_epi64(_mm512_castpd_si512((__m512d)biased[L]),mb);
            j[L]=_mm512_and_epi64(kn[L],mask);
            q[L]=_mm512_srai_epi64(kn[L],7);
        }
        for(int L=0;L<2;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
        for(int L=0;L<2;L++){
            r[L]=nmul_add(k[L],hi,x[L]);
            r[L]=nmul_add(k[L],mi,r[L]);
            r[L]=nmul_add(k[L],lo,r[L]);
        }
        for(int L=0;L<2;L++) h[L]=mul_add(nq4,r[L],nq3);
        for(int L=0;L<2;L++) h[L]=mul_add(h[L],r[L],nq2);
        for(int L=0;L<2;L++) h[L]=mul_add(h[L],r[L],nq1);
        for(int L=0;L<2;L++) h[L]=mul_add(h[L],r[L],one);
        for(int L=0;L<2;L++) s[L]=h[L]*h[L];
        for(int L=0;L<2;L++) er[L]=mul_add(r[L],s[L],one);
        for(int L=0;L<2;L++) el[L]=mul_add(r[L],s[L],one-er[L]);
        for(int L=0;L<2;L++){
            sb[L]=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));
            scale[L]=Vec8d(_mm512_castsi512_pd(sb[L]));
        }
        for(int L=0;L<2;L++){
            ph[L]=er[L]*scale[L];
            y[L]=mul_add(el[L],scale[L],ph[L]);
            y[L].store(out+i+8*L);
        }
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
