#pragma once
#include "experiments/exp53_park_inactive_helpers_v3_frozen.hpp"

/* Temporary benchmark-only candidate: same frozen EXP53 math/body choices, but no helper threads. */
class Exp53NoHelperAllTemp {
public:
    explicit Exp53NoHelperAllTemp(long = 1) {}
    void run(double* out, const double* in, size_t n, long = 1) {
        if (n <= 100) exp53_vcl_u2z_0100_frozen(out, in, n);
        else if (n < 1400) exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        else if (n <= 3000) exp53_park_hwy_ns::kernel(out, in, n);
        else exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
    }
    void run_streaming_write_once(double* out, const double* in, size_t n, long = 1) {
        if (n <= 100) exp53_vcl_u2z_0100_frozen(out, in, n);
        else if (n <= 3000) exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        else if (n >= 15000 && n <= 65000) exp53_n2_vmstyle_u4_0381_frozen(out, in, n);
        else exp53_n2_vmstyle_u4_0381_nt_sfence(out, in, n);
    }
    bool helper_exists() const { return false; }
};
