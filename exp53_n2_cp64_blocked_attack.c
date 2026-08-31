/* Strong CP64 repair: retain eight independent Q4 chains, but avoid the
   original CP64 setup-liveness spike by preparing two 4-vector microblocks.
   Each microblock is compressed to (r,scale) before the next is opened.
   Then all eight Q4 chains execute stage-wise, followed by two 4-vector drains.
*/
#include "exp53_n2_vmstyle_u4_0381_frozen.c"

#define CPB_FINAL4(B) do { \
    __m512d s0=_mm512_mul_pd(h[B],h[B]),s1=_mm512_mul_pd(h[(B)+1],h[(B)+1]),s2=_mm512_mul_pd(h[(B)+2],h[(B)+2]),s3=_mm512_mul_pd(h[(B)+3],h[(B)+3]); \
    __m512d e0=_mm512_fmadd_pd(r[B],s0,one),e1=_mm512_fmadd_pd(r[(B)+1],s1,one),e2=_mm512_fmadd_pd(r[(B)+2],s2,one),e3=_mm512_fmadd_pd(r[(B)+3],s3,one); \
    __m512d l0=_mm512_fmadd_pd(r[B],s0,_mm512_sub_pd(one,e0)); \
    __m512d l1=_mm512_fmadd_pd(r[(B)+1],s1,_mm512_sub_pd(one,e1)); \
    __m512d l2=_mm512_fmadd_pd(r[(B)+2],s2,_mm512_sub_pd(one,e2)); \
    __m512d l3=_mm512_fmadd_pd(r[(B)+3],s3,_mm512_sub_pd(one,e3)); \
    __m512d p0=_mm512_mul_pd(e0,scale[B]),p1=_mm512_mul_pd(e1,scale[(B)+1]),p2=_mm512_mul_pd(e2,scale[(B)+2]),p3=_mm512_mul_pd(e3,scale[(B)+3]); \
    __m512d y0=_mm512_fmadd_pd(l0,scale[B],p0),y1=_mm512_fmadd_pd(l1,scale[(B)+1],p1),y2=_mm512_fmadd_pd(l2,scale[(B)+2],p2),y3=_mm512_fmadd_pd(l3,scale[(B)+3],p3); \
    _mm512_storeu_pd(out+i+8*(B),y0);_mm512_storeu_pd(out+i+8*((B)+1),y1);_mm512_storeu_pd(out+i+8*((B)+2),y2);_mm512_storeu_pd(out+i+8*((B)+3),y3); \
} while(0)

__attribute__((target("avx512f,avx512dq,fma"),noinline))
void exp53_n2_cp64_blocked_attack(double *restrict out,const double *restrict in,size_t n)
{
    const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI),mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO),magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2F_Q1),nq2=_mm512_set1_pd(N2F_Q2),nq3=_mm512_set1_pd(N2F_Q3),nq4=_mm512_set1_pd(N2F_Q4);
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127);
    size_t i=0;
    for(;i+64<=n;i+=64){
        __m512d r[8],scale[8],h[8];

        /* Microblock A: four vectors, compressed before B opens. */
        {
            __m512d x[4],biased[4],k[4]; __m512i kn[4],j[4],q[4],tb[4];
            for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+8*L);
            for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
            for(int L=0;L<4;L++){k[L]=_mm512_sub_pd(biased[L],magic);kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);j[L]=_mm512_and_epi64(kn[L],mask);q[L]=_mm512_srai_epi64(kn[L],7);}
            for(int L=0;L<4;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
            for(int L=0;L<4;L++) r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            for(int L=0;L<4;L++) r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]);
            for(int L=0;L<4;L++) r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]);
            for(int L=0;L<4;L++){__m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));scale[L]=_mm512_castsi512_pd(sb);}
        }

        /* Microblock B: same proven four-vector setup. */
        {
            __m512d x[4],biased[4],k[4]; __m512i kn[4],j[4],q[4],tb[4];
            for(int L=0;L<4;L++) x[L]=_mm512_loadu_pd(in+i+32+8*L);
            for(int L=0;L<4;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic);
            for(int L=0;L<4;L++){k[L]=_mm512_sub_pd(biased[L],magic);kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb);j[L]=_mm512_and_epi64(kn[L],mask);q[L]=_mm512_srai_epi64(kn[L],7);}
            for(int L=0;L<4;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8);
            for(int L=0;L<4;L++) r[4+L]=_mm512_fnmadd_pd(k[L],hi,x[L]);
            for(int L=0;L<4;L++) r[4+L]=_mm512_fnmadd_pd(k[L],mi,r[4+L]);
            for(int L=0;L<4;L++) r[4+L]=_mm512_fnmadd_pd(k[L],lo,r[4+L]);
            for(int L=0;L<4;L++){__m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52));scale[4+L]=_mm512_castsi512_pd(sb);}
        }

        /* Prevent any table access from being sunk beyond the 8-chain wave. */
        __asm__ __volatile__("" ::: "memory");

        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1);
        for(int L=0;L<8;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one);

        CPB_FINAL4(0);
        CPB_FINAL4(4);
    }
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i);
}
