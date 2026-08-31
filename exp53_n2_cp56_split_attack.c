/* Targeted repair of CP56: keep seven independent arithmetic chains but avoid
   the compiler's observed very-late last gathers.  Two schedules are tested:
     SPLIT52: first 5 anchors early, last 2 immediately after Q4.
     SPLIT61: first 6 anchors early, last 1 immediately after Q4.
   The late gather(s) are forced before early-group finalization using compiler
   memory barriers, allowing hardware OoO execution to hide their latency under
   ER-low/reconstruction of the early group.
*/
#include "exp53_n2_cp_wip_sweep.c"

#define FINAL1(B) do { \
    __m512d _s0=_mm512_mul_pd(h[B],h[B]); \
    __m512d _e0=_mm512_fmadd_pd(r[B],_s0,one); \
    __m512d _l0=_mm512_fmadd_pd(r[B],_s0,_mm512_sub_pd(one,_e0)); \
    __m512d _p0=_mm512_mul_pd(_e0,scale[B]); \
    __m512d _y0=_mm512_fmadd_pd(_l0,scale[B],_p0); \
    _mm512_storeu_pd(out+i+8*(B),_y0); \
} while(0)

#define FINAL2(B) do { \
    __m512d _s0=_mm512_mul_pd(h[B],h[B]), _s1=_mm512_mul_pd(h[(B)+1],h[(B)+1]); \
    __m512d _e0=_mm512_fmadd_pd(r[B],_s0,one), _e1=_mm512_fmadd_pd(r[(B)+1],_s1,one); \
    __m512d _l0=_mm512_fmadd_pd(r[B],_s0,_mm512_sub_pd(one,_e0)); \
    __m512d _l1=_mm512_fmadd_pd(r[(B)+1],_s1,_mm512_sub_pd(one,_e1)); \
    __m512d _p0=_mm512_mul_pd(_e0,scale[B]), _p1=_mm512_mul_pd(_e1,scale[(B)+1]); \
    __m512d _y0=_mm512_fmadd_pd(_l0,scale[B],_p0), _y1=_mm512_fmadd_pd(_l1,scale[(B)+1],_p1); \
    _mm512_storeu_pd(out+i+8*(B),_y0); _mm512_storeu_pd(out+i+8*((B)+1),_y1); \
} while(0)

#define FINAL4(B) do { \
    __m512d _s0=_mm512_mul_pd(h[B],h[B]), _s1=_mm512_mul_pd(h[(B)+1],h[(B)+1]); \
    __m512d _s2=_mm512_mul_pd(h[(B)+2],h[(B)+2]), _s3=_mm512_mul_pd(h[(B)+3],h[(B)+3]); \
    __m512d _e0=_mm512_fmadd_pd(r[B],_s0,one), _e1=_mm512_fmadd_pd(r[(B)+1],_s1,one); \
    __m512d _e2=_mm512_fmadd_pd(r[(B)+2],_s2,one), _e3=_mm512_fmadd_pd(r[(B)+3],_s3,one); \
    __m512d _l0=_mm512_fmadd_pd(r[B],_s0,_mm512_sub_pd(one,_e0)); \
    __m512d _l1=_mm512_fmadd_pd(r[(B)+1],_s1,_mm512_sub_pd(one,_e1)); \
    __m512d _l2=_mm512_fmadd_pd(r[(B)+2],_s2,_mm512_sub_pd(one,_e2)); \
    __m512d _l3=_mm512_fmadd_pd(r[(B)+3],_s3,_mm512_sub_pd(one,_e3)); \
    __m512d _p0=_mm512_mul_pd(_e0,scale[B]), _p1=_mm512_mul_pd(_e1,scale[(B)+1]); \
    __m512d _p2=_mm512_mul_pd(_e2,scale[(B)+2]), _p3=_mm512_mul_pd(_e3,scale[(B)+3]); \
    __m512d _y0=_mm512_fmadd_pd(_l0,scale[B],_p0), _y1=_mm512_fmadd_pd(_l1,scale[(B)+1],_p1); \
    __m512d _y2=_mm512_fmadd_pd(_l2,scale[(B)+2],_p2), _y3=_mm512_fmadd_pd(_l3,scale[(B)+3],_p3); \
    _mm512_storeu_pd(out+i+8*(B),_y0); _mm512_storeu_pd(out+i+8*((B)+1),_y1); \
    _mm512_storeu_pd(out+i+8*((B)+2),_y2); _mm512_storeu_pd(out+i+8*((B)+3),_y3); \
} while(0)

#define DEFINE_SPLIT56(NAME,EARLY,EARLY_TAIL,LATE_FINAL) \
__attribute__((target("avx512f,avx512dq,fma"),noinline)) \
void NAME(double *restrict out,const double *restrict in,size_t n) \
{ \
    const __m512d inv=_mm512_set1_pd(N2F_INV128),hi=_mm512_set1_pd(N2F_L128_HI),mi=_mm512_set1_pd(N2F_L128_MI),lo=_mm512_set1_pd(N2F_L128_LO),magic=_mm512_set1_pd(N2F_MAGIC),one=_mm512_set1_pd(1.0),nq1=_mm512_set1_pd(N2F_Q1),nq2=_mm512_set1_pd(N2F_Q2),nq3=_mm512_set1_pd(N2F_Q3),nq4=_mm512_set1_pd(N2F_Q4); \
    const __m512i mb=_mm512_set1_epi64((long long)N2F_MAGIC_BITS),mask=_mm512_set1_epi64(127); \
    size_t i=0; \
    for(;i+56<=n;i+=56){ \
        __m512d x[7],biased[7],k[7],r[7],scale[7],h[7]; \
        __m512i kn[7],j[7],q[7],tb[7]; \
        for(int L=0;L<7;L++) x[L]=_mm512_loadu_pd(in+i+8*L); \
        for(int L=0;L<7;L++) biased[L]=_mm512_fmadd_pd(x[L],inv,magic); \
        for(int L=0;L<7;L++){ k[L]=_mm512_sub_pd(biased[L],magic); kn[L]=_mm512_sub_epi64(_mm512_castpd_si512(biased[L]),mb); j[L]=_mm512_and_epi64(kn[L],mask); q[L]=_mm512_srai_epi64(kn[L],7); } \
        for(int L=0;L<(EARLY);L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
        for(int L=0;L<7;L++) r[L]=_mm512_fnmadd_pd(k[L],hi,x[L]); \
        for(int L=0;L<7;L++) r[L]=_mm512_fnmadd_pd(k[L],mi,r[L]); \
        for(int L=0;L<7;L++) r[L]=_mm512_fnmadd_pd(k[L],lo,r[L]); \
        for(int L=0;L<(EARLY);L++){ __m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); scale[L]=_mm512_castsi512_pd(sb); } \
        for(int L=0;L<7;L++) h[L]=_mm512_fmadd_pd(nq4,r[L],nq3); \
        for(int L=0;L<7;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq2); \
        for(int L=0;L<7;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],nq1); \
        for(int L=0;L<7;L++) h[L]=_mm512_fmadd_pd(h[L],r[L],one); \
        __asm__ __volatile__("" ::: "memory"); \
        for(int L=(EARLY);L<7;L++) tb[L]=_mm512_i64gather_epi64(j[L],(const long long*)N2_FROZEN_TAB128,8); \
        __asm__ __volatile__("" ::: "memory"); \
        FINAL4(0); \
        EARLY_TAIL; \
        for(int L=(EARLY);L<7;L++){ __m512i sb=_mm512_add_epi64(tb[L],_mm512_slli_epi64(q[L],52)); scale[L]=_mm512_castsi512_pd(sb); } \
        LATE_FINAL; \
    } \
    if(i<n) exp53_n2_vmstyle_u4_0381_frozen(out+i,in+i,n-i); \
}

DEFINE_SPLIT56(exp53_n2_cp56_split52,5,FINAL1(4),FINAL2(5))
DEFINE_SPLIT56(exp53_n2_cp56_split61,6,FINAL2(4),FINAL1(6))
