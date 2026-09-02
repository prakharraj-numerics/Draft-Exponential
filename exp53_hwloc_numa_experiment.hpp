#pragma once

/*
  Experimental hwloc + libnuma placement layer.

  The canonical production dispatcher and frozen kernels are not modified.
  Production still owns its established CPU0/CPU2 affinity. This layer:
    1. inspects CPU0/CPU2 through hwloc,
    2. verifies both CPUs belong to the same NUMA node through libnuma,
    3. provides NUMA-local batch-buffer allocation for that node,
    4. delegates computation unchanged to Exp53BatchProductionExecutor.

  This file is experimental and is not included by production/.
*/

#include <cstddef>
#include <cstdlib>
#include <hwloc.h>
#include <numa.h>
#include "production/exp53_batch_production.hpp"

class Exp53HwlocNumaExperiment {
public:
    explicit Exp53HwlocNumaExperiment(long workers = 2)
        : production_(workers > 1 ? 2 : 1) {
        if (hwloc_topology_init(&topology_) != 0) return;
        topology_initialized_ = true;
        if (hwloc_topology_load(topology_) != 0) return;
        topology_loaded_ = true;

        pu0_ = hwloc_get_pu_obj_by_os_index(topology_, 0);
        pu2_ = hwloc_get_pu_obj_by_os_index(topology_, 2);
        if (!pu0_ || !pu2_ || numa_available() < 0) return;
        node0_ = numa_node_of_cpu(0);
        node2_ = numa_node_of_cpu(2);
        same_node_ = node0_ >= 0 && node0_ == node2_;
        common_ancestor_ = hwloc_get_common_ancestor_obj(topology_, pu0_, pu2_);
        usable_ = same_node_ && common_ancestor_ != nullptr;
    }

    ~Exp53HwlocNumaExperiment() {
        if (topology_initialized_) hwloc_topology_destroy(topology_);
    }

    Exp53HwlocNumaExperiment(const Exp53HwlocNumaExperiment&) = delete;
    Exp53HwlocNumaExperiment& operator=(const Exp53HwlocNumaExperiment&) = delete;

    bool usable() const noexcept { return usable_; }
    bool same_numa_node() const noexcept { return same_node_; }
    int numa_node() const noexcept { return same_node_ ? node0_ : -1; }
    hwloc_obj_type_t common_ancestor_type() const noexcept {
        return common_ancestor_ ? common_ancestor_->type : HWLOC_OBJ_TYPE_NONE;
    }

    double* allocate(size_t n) const {
        if (!usable_ || n == 0) return nullptr;
        return static_cast<double*>(numa_alloc_onnode(n * sizeof(double), node0_));
    }

    void deallocate(double* p, size_t n) const {
        if (p) numa_free(p, n * sizeof(double));
    }

    void run(double* out, const double* in, size_t n, long workers = 2) {
        production_.run(out, in, n, workers);
    }

private:
    hwloc_topology_t topology_{};
    bool topology_initialized_ = false;
    bool topology_loaded_ = false;
    bool same_node_ = false;
    bool usable_ = false;
    int node0_ = -1;
    int node2_ = -1;
    hwloc_obj_t pu0_ = nullptr;
    hwloc_obj_t pu2_ = nullptr;
    hwloc_obj_t common_ancestor_ = nullptr;
    Exp53BatchProductionExecutor production_;
};
