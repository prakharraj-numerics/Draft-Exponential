#pragma once

/*
  Resource-aware policy layer for the frozen EXP53 production dispatcher.

  This wrapper deliberately does not modify production/ or any frozen kernel.
  It selects the worker budget passed to Exp53BatchProductionExecutor:

    LowestLatency:
      Preserve the canonical production routing and allow two workers.

    LowestCompute:
      Use a single worker. This minimizes aggregate CPU consumption even when
      a two-worker call would have lower wall latency.

    Balanced:
      Use one worker below 50,000 elements and two workers at or above 50,000.
      The threshold is based on exact Xeon 6973P-C compute-efficiency run
      33616730210: the two-worker path consumed about 80% more aggregate CPU
      at 3.5K, 37% more at 15K, 26% more at 50K, and only 2-3% more at 1M-2M.
      At n<=100 the frozen VCL route is unchanged and ignores the worker budget.

  Mathematical machinery, constants, accuracy, store policy, and production
  size boundaries remain owned by the canonical production executor.
*/

#include <cstddef>
#include "production/exp53_batch_production.hpp"

enum class Exp53ResourcePolicy {
    LowestLatency = 0,
    Balanced = 1,
    LowestCompute = 2
};

class Exp53ResourceAwareExecutor {
public:
    static constexpr size_t kBalancedParallelMinN = 50000;

    explicit Exp53ResourceAwareExecutor(long max_workers = 2)
        : max_workers_(max_workers > 1 ? 2L : 1L),
          production_(max_workers_) {}

    long workers_for(size_t n, Exp53ResourcePolicy policy) const noexcept {
        if (max_workers_ <= 1) return 1;
        switch (policy) {
            case Exp53ResourcePolicy::LowestLatency:
                return 2;
            case Exp53ResourcePolicy::LowestCompute:
                return 1;
            case Exp53ResourcePolicy::Balanced:
                return n >= kBalancedParallelMinN ? 2 : 1;
        }
        return 1;
    }

    void run(double* out, const double* in, size_t n,
             Exp53ResourcePolicy policy = Exp53ResourcePolicy::Balanced) {
        production_.run(out, in, n, workers_for(n, policy));
    }

    void run_streaming_write_once(
        double* out, const double* in, size_t n,
        Exp53ResourcePolicy policy = Exp53ResourcePolicy::Balanced) {
        production_.run_streaming_write_once(out, in, n, workers_for(n, policy));
    }

    void run(double* out, const double* in, size_t n,
             Exp53OutputPolicy output_policy,
             Exp53ResourcePolicy resource_policy = Exp53ResourcePolicy::Balanced) {
        production_.run(out, in, n, output_policy, workers_for(n, resource_policy));
    }

private:
    long max_workers_;
    Exp53BatchProductionExecutor production_;
};
