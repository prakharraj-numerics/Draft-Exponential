from pathlib import Path

p = Path(__file__).with_name('exp_lowp_q14_selfcert.generated.c')
s = p.read_text()

old = '''    mp_limb_t *pcoeff, *rcoeff, *buf, *prod, *tab10, *tab4;\n    fmpz_t q;\n} exp_lowp_qsqrt_ctx;'''
new = '''    mp_limb_t *pcoeff, *rcoeff, *buf, *prod, *tab10, *tab4;\n    fmpz_t q;\n    arf_t input_scratch; /* reusable exact-rational conversion target */\n} exp_lowp_qsqrt_ctx;'''
assert old in s
s = s.replace(old, new, 1)

old = 'c->tab4=(mp_limb_t*)flint_calloc((size_t)16*(size_t)c->n,sizeof(mp_limb_t));fmpz_init(c->q);'
new = 'c->tab4=(mp_limb_t*)flint_calloc((size_t)16*(size_t)c->n,sizeof(mp_limb_t));fmpz_init(c->q);arf_init(c->input_scratch);'
assert old in s
s = s.replace(old, new, 1)

old = 'if(c->tab10)flint_free(c->tab10);if(c->tab4)flint_free(c->tab4);fmpz_clear(c->q);flint_free(c);}'
new = 'if(c->tab10)flint_free(c->tab10);if(c->tab4)flint_free(c->tab4);arf_clear(c->input_scratch);fmpz_clear(c->q);flint_free(c);}'
assert old in s
s = s.replace(old, new, 1)

old = '''int exp_lowp_qsqrt_eval_fmpq_cert(exp_lowp_qsqrt_ctx*c,arb_t out,const fmpq_t q){\n    if(!c||!out)return 1;\n    arf_t x; arf_init(x);\n    (void)arf_set_fmpq(x,q,c->work_bits+64,ARF_RND_NEAR);\n    int rc=exp_lowp_qsqrt_eval_cert(c,out,x);\n    arf_clear(x); return rc;\n}'''
new = '''int exp_lowp_qsqrt_eval_fmpq_cert(exp_lowp_qsqrt_ctx*c,arb_t out,const fmpq_t q){\n    if(!c||!out)return 1;\n    /* Context already owns mutable reduction scratch, so reuse one arf here\n       instead of allocate/free on every exact-rational call.  Keep +64 guard\n       bits unchanged: this is a pure frontend-allocation optimization. */\n    (void)arf_set_fmpq(c->input_scratch,q,c->work_bits+64,ARF_RND_NEAR);\n    return exp_lowp_qsqrt_eval_cert(c,out,c->input_scratch);\n}'''
assert old in s
s = s.replace(old, new, 1)

p.write_text(s)
print('Q14_FRONTEND_REUSE_ARF applied=true precision_unchanged=true certificate_unchanged=true')
