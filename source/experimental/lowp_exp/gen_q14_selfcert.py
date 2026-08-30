from pathlib import Path

src_path = Path(__file__).with_name('exp_lowp_table_qsqrt.c')
s = src_path.read_text()

s = s.replace('#include <mpfr.h>\n','',1)
s = s.replace('int exp_lowp_qsqrt_get_mpfr(mpfr_ptr out,const arf_t x){arf_get_mpfr(out,x,MPFR_RNDN);return 0;}\n','',1)
s = s.replace('mp_limb_t *pcoeff, *rcoeff, *buf, *prod, *tab10;', 'mp_limb_t *pcoeff, *rcoeff, *buf, *prod, *tab10, *tab4;', 1)
s = s.replace('static inline mp_ptr T10(exp_lowp_qsqrt_ctx*c,int p1,int p2){return c->tab10+((size_t)p1*32u+(size_t)p2)*(size_t)c->n;}\n', 'static inline mp_ptr T10(exp_lowp_qsqrt_ctx*c,int p1,int p2){return c->tab10+((size_t)p1*32u+(size_t)p2)*(size_t)c->n;}\nstatic inline mp_ptr T4(exp_lowp_qsqrt_ctx*c,int p){return c->tab4+(size_t)p*(size_t)c->n;}\n', 1)
s = s.replace('const long double log2z=-20.0L; /* 10-bit residual: z <= 2^-20 */', 'const long double log2z=-28.0L; /* 14-bit residual: z <= 2^-28 */', 1)
s = s.replace('/* first omitted sqrt term is u^(2t+3)/(2t+3)!; |u| <= 2^-10 */', '/* first omitted sqrt term is u^(2t+3)/(2t+3)!; |u| <= 2^-14 */', 1)
s = s.replace('long double e=-(long double)10*k-lgammal((long double)(k+1))*il2;', 'long double e=-(long double)14*k-lgammal((long double)(k+1))*il2;', 1)

marker = 'exp_lowp_qsqrt_ctx*exp_lowp_qsqrt_create(unsigned digits,int extra_limb){'
assert marker in s
forward_and_builder = r'''static void formula_core(exp_lowp_qsqrt_ctx*c,mp_ptr ehalf,mp_srcptr w);
static void build_tab4_self(exp_lowp_qsqrt_ctx*c){
    slong n=c->n; mp_ptr w=B(c,12);
    for(int j=0;j<16;j++){
        flint_mpn_zero(w,n);
        w[n-1]=((mp_limb_t)j)<<(FLINT_BITS-14);
        formula_core(c,T4(c,j),w);
    }
}

'''
s = s.replace(marker, forward_and_builder + marker, 1)

old = 'c->tab10=(mp_limb_t*)flint_calloc((size_t)ARB_EXP_TAB21_NUM*ARB_EXP_TAB22_NUM*(size_t)c->n,sizeof(mp_limb_t));fmpz_init(c->q);'
new = 'c->tab10=(mp_limb_t*)flint_calloc((size_t)ARB_EXP_TAB21_NUM*ARB_EXP_TAB22_NUM*(size_t)c->n,sizeof(mp_limb_t));c->tab4=(mp_limb_t*)flint_calloc((size_t)16*(size_t)c->n,sizeof(mp_limb_t));fmpz_init(c->q);'
assert old in s
s = s.replace(old,new,1)
old = 'if(!c->pcoeff||!c->rcoeff||!c->buf||!c->prod||!c->tab10){return NULL;}\n    build_coeffs(c);build_tab10(c);return c;'
new = 'if(!c->pcoeff||!c->rcoeff||!c->buf||!c->prod||!c->tab10||!c->tab4){return NULL;}\n    build_coeffs(c);build_tab10(c);build_tab4_self(c);return c;'
assert old in s
s = s.replace(old,new,1)
old = 'if(c->tab10)flint_free(c->tab10);fmpz_clear(c->q);flint_free(c);}'
new = 'if(c->tab10)flint_free(c->tab10);if(c->tab4)flint_free(c->tab4);fmpz_clear(c->q);flint_free(c);}'
assert old in s
s = s.replace(old,new,1)

old = '''ulong p2=w[n-1]>>(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));w[n-1]-=p2<<(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));
    formula_core(c,ehalf,w);
    if(p1==0&&p2==0){flint_mpn_copyi(fin,ehalf,n);arf_set_mpn(out,fin,n,0);arf_mul_2exp_si(out,out,q+1-n*FLINT_BITS);}
    else{mulhi(c,fin,ehalf,T10(c,(int)p1,(int)p2));arf_set_mpn(out,fin,n,0);arf_mul_2exp_si(out,out,q+2-n*FLINT_BITS);}'''
new = '''ulong p2=w[n-1]>>(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));w[n-1]-=p2<<(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));
    ulong p3=w[n-1]>>(FLINT_BITS-14);w[n-1]-=p3<<(FLINT_BITS-14);
    formula_core(c,ehalf,w);
    mp_ptr g=B(c,13);int sh=(int)q+1;
    if(p1==0&&p2==0)flint_mpn_copyi(g,ehalf,n);else{mulhi(c,g,ehalf,T10(c,(int)p1,(int)p2));sh++;}
    if(p3==0)flint_mpn_copyi(fin,g,n);else{mulhi(c,fin,g,T4(c,(int)p3));sh++;}
    arf_set_mpn(out,fin,n,0);arf_mul_2exp_si(out,out,sh-n*FLINT_BITS);'''
assert old in s
s = s.replace(old,new,1)

cert_block = r'''
enum {
    Q14_E_REDUCTION   = 16,
    Q14_E_COEFF       = 32,
    Q14_E_SERIES      = 16,
    Q14_E_MULHIGH     = 4096,
    Q14_E_TABLE10     = 256,
    Q14_E_TABLE4      = 4096,
    Q14_E_SHIFTS      = 64,
    Q14_E_ADDS        = 64,
    Q14_E_PROPAGATION = 4096,
    Q14_E_ROUND_MARGIN= 3648,
    Q14_CERT_ULPS     = Q14_E_REDUCTION + Q14_E_COEFF + Q14_E_SERIES +
                        Q14_E_MULHIGH + Q14_E_TABLE10 + Q14_E_TABLE4 +
                        Q14_E_SHIFTS + Q14_E_ADDS + Q14_E_PROPAGATION +
                        Q14_E_ROUND_MARGIN
};
_Static_assert(Q14_CERT_ULPS == 16384, "Q14 certificate budget must remain 2^14 ulp");

ulong exp_lowp_qsqrt_cert_ulps(void){return (ulong)Q14_CERT_ULPS;}

int exp_lowp_qsqrt_eval_cert(exp_lowp_qsqrt_ctx*c,arb_t out,const arf_t x){
    if(!c||!out||!x||arf_is_special(x))return 1;
    slong n=c->n; ulong red_err=0; mp_ptr w=B(c,14),ehalf=B(c,15),fin=B(c,16),g=B(c,13);
    if(!_arb_get_mpn_fixed_mod_log2(w,c->q,&red_err,x,n))return 2;
    if(red_err>3) return 3;
    slong q=fmpz_get_si(c->q);
    ulong p1=w[n-1]>>(FLINT_BITS-ARB_EXP_TAB21_BITS);w[n-1]-=p1<<(FLINT_BITS-ARB_EXP_TAB21_BITS);
    ulong p2=w[n-1]>>(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));w[n-1]-=p2<<(FLINT_BITS-(ARB_EXP_TAB21_BITS+ARB_EXP_TAB22_BITS));
    ulong p3=w[n-1]>>(FLINT_BITS-14);w[n-1]-=p3<<(FLINT_BITS-14);
    formula_core(c,ehalf,w);
    slong sh=q+1;
    if(p1==0&&p2==0)flint_mpn_copyi(g,ehalf,n);else{mulhi(c,g,ehalf,T10(c,(int)p1,(int)p2));sh++;}
    if(p3==0)flint_mpn_copyi(fin,g,n);else{mulhi(c,fin,g,T4(c,(int)p3));sh++;}
    arf_set_mpn(arb_midref(out),fin,n,0);
    arf_mul_2exp_si(arb_midref(out),arb_midref(out),sh-n*FLINT_BITS);
    mag_set_ui_2exp_si(arb_radref(out),(ulong)Q14_CERT_ULPS,sh-n*FLINT_BITS);
    return 0;
}

int exp_lowp_qsqrt_eval_fmpq_cert(exp_lowp_qsqrt_ctx*c,arb_t out,const fmpq_t q){
    if(!c||!out)return 1;
    arf_t x; arf_init(x);
    (void)arf_set_fmpq(x,q,c->work_bits+64,ARF_RND_NEAR);
    int rc=exp_lowp_qsqrt_eval_cert(c,out,x);
    arf_clear(x); return rc;
}

int exp_lowp_qsqrt_eval_ratio_cert(exp_lowp_qsqrt_ctx*c,arb_t out,long A,unsigned long D){
    if(!c||!out||D==0)return 1;
    fmpq_t q; fmpq_init(q); fmpq_set_si(q,A,D);
    int rc=exp_lowp_qsqrt_eval_fmpq_cert(c,out,q);
    fmpq_clear(q); return rc;
}
'''

s += cert_block

for forbidden in ('arb_exp(', 'mpfr_exp(', '#include <mpfr.h>', 'mpfr_ptr', 'MPFR_'):
    if forbidden in s:
        raise SystemExit(f'forbidden production dependency: {forbidden}')

out = Path(__file__).with_name('exp_lowp_q14_selfcert.generated.c')
out.write_text(s)
print(f'Q14_SELFCERT_GENERATED path={out} cert_ulps=16384 runtime_exp_oracle=false mpfr_dependency=false')
