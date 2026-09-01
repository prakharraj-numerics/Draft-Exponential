# EXP53 Production Snapshot

This directory is the canonical production-only snapshot.

It intentionally captures the last accepted stable production routing immediately before the rejected 101-1400 XNNPACK hybrid was wired into the root dispatcher.

Production dispatcher provenance: commit `cdfedb62ee33dc9f31c3444f4c404269af5d6bf6`.

## Frozen routing

- `n <= 100`: frozen VCL + u2z kernel (`exp53_vcl_u2z_0100_frozen.cpp`)
- `101 <= n <= 1399`: frozen serial VM-style kernel (`exp53_n2_vmstyle_u4_0381_frozen.c`)
- `1400 <= n <= 3000`, workers > 1: frozen synchronized Highway path (`exp53_highway_sync_1400_3000_frozen.hpp`)
- `n > 3000`, workers > 1: frozen custom permanent 2-core dispatcher (`exp53_batch_custom_2core_nt_frozen.hpp`)
- single-worker temporal path: frozen serial VM-style kernel
- streaming/write-once: <=100 VCL temporal; 101-3000 serial temporal; >3000 custom NT, or serial NT for one worker

## Internal files kept here

- `exp53_batch_production.hpp`
- `exp53_batch_custom_2core_nt_frozen.hpp`
- `exp53_highway_sync_1400_3000_frozen.hpp`
- `exp53_highway_sync_1600_3000_constants_frozen.hpp`
- `exp53_vcl_u2z_0100_frozen.cpp`
- `exp53_n2_vmstyle_u4_0381_frozen.c`
- `exp53_n2_vmstyle_u4_0381_nt_sfence.c`
- `exp53_n2_fused_u4_038_frozen.c`

External build dependencies remain VCL for the <=100 kernel and Highway 1.4.0 for the 1400-3000 kernel.

## Important

The root-level `exp53_batch_production.hpp` currently contains the later XNNPACK hybrid routing for 101-1400. That hybrid was subsequently benchmarked on the exact Xeon 6973P-C and rejected. Do not treat the root-level dispatcher as the canonical production snapshot.

Research, benchmark, candidate, workflow, diagnostic, and rejected implementations intentionally remain outside this directory so production code is easy to identify and cannot be confused with experiments.
