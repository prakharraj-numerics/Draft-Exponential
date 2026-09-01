# EXP53 Production Snapshot

This directory is the canonical production-only snapshot.

It intentionally captures the accepted stable production routing, with later changes promoted only after exact Intel Xeon 6973P-C validation.

Original production dispatcher provenance: commit `cdfedb62ee33dc9f31c3444f4c404269af5d6bf6`.

## Frozen routing

- `n <= 100`: frozen VCL + u2z kernel (`exp53_vcl_u2z_0100_frozen.cpp`)
- `101 <= n <= 1399`: frozen serial VM-style kernel (`exp53_n2_vmstyle_u4_0381_frozen.c`)
- `1400 <= n <= 3000`, workers > 1: frozen synchronized Highway path (`exp53_highway_sync_1400_3000_frozen.hpp`)
- `n > 3000`, workers > 1, default/common temporal case: frozen custom permanent 2-core dispatcher (`exp53_batch_custom_2core_nt_frozen.hpp`)
- single-worker temporal path: frozen serial VM-style kernel

### Rare streaming/write-once policy

- `n <= 100`: VCL temporal
- `101 <= n <= 3000`: serial temporal
- `3000 < n < 15000`, workers > 1: custom2 + NT stores
- `15000 <= n <= 65000`, workers > 1: **custom2 temporal stores; NT deliberately removed**
- `n > 65000`, workers > 1: custom2 + NT stores
- `n > 3000`, workers <= 1: serial NT path unchanged

## Rare 15K-65K no-NT validation

Exact-Xeon validation run: `33543929387`, accepted Intel Xeon 6973P-C shard 17.

Benchmark geometry intentionally reproduced the previous rare/write-once medium test:

- approximately 256 MiB aligned rotating output ring
- 6,000,000 output values per timing sample
- no output-ring wrap within a timing sample
- UNIT and MID mixed-sign domains
- sizes 15,000 through 65,000 in 5,000-element steps
- Intel comparator: sequential oneMKL `vmdExp(..., VML_HA)`
- same frozen custom2 runtime for temporal and NT paths; only store policy differs

Results over 22 size/domain cells:

- custom2 temporal beat custom2 NT: **22/22**
- custom2 temporal beat Intel VML_HA: **22/22**
- custom2 NT beat Intel VML_HA: **0/22**
- median `NT / temporal` time ratio: **1.542382**
- median `Intel / temporal` time ratio: **1.169068**
- median `Intel / NT` time ratio: **0.765270**
- Intel cross-screen: maximum observed difference <= 2 ULP, no >2-ULP failures

Therefore the production rare/write-once policy uses temporal custom2 specifically for 15K-65K and retains NT outside that band.

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

The root-level `exp53_batch_production.hpp` contains later research routing and must not be treated as canonical production. Use this `production/` directory.

Research, benchmark, candidate, workflow, diagnostic, and rejected implementations intentionally remain outside this directory so production code is easy to identify and cannot be confused with experiments.
