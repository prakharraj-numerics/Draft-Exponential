# EXP53 faithful n=2 — cross-vector pipeline and arbitrary-batch results

## Scope

The frozen production checkpoint `exp53_n2_vmstyle_u4_0381_frozen` was not modified.
All candidates preserve the same n=2 formula, reduction, TAB128 anchors, fused exponent scaling, and ER-low correction.

Compiler: Intel oneAPI ICX 2026.1.1 (`2026.1.1.20260724`).
Target CPU for accepted measurements: Intel Xeon 6973P-C.

## 1. Explicit cross-vector software pipeline

Run: `33387934083`.
Exact-Xeon shards used for replication: 7 and 28.

Candidates:
- `PIPE3`: reduce+gather | Q4+ER-low | scale+store
- `PIPE4`: reduce+gather | Q4 | ER-low | scale+store

Both stream consecutive 8-wide AVX-512 vectors through staggered stages in a single CPU thread.

Accuracy on each accepted shard:
- 200,000 mixed-domain inputs: PIPE3 max ULP 1, >1 ULP 0, bit differences vs frozen 0
- 200,000 mixed-domain inputs: PIPE4 max ULP 1, >1 ULP 0, bit differences vs frozen 0
- exhaustive batch lengths 1..257: both max ULP 1, >1 ULP 0

Static census:

| Function | instructions | gathers | FMA/FNMA | vector stack spills |
|---|---:|---:|---:|---:|
| Frozen u4 | 181 | 4 | 40 | 0 |
| PIPE3 | 389 | 8 | 81 | 0 |
| PIPE4 | 471 | 9 | 95 | 0 |

Long-array timings, ns/input:

| n | shard | Frozen | PIPE3 | PIPE4 | VML_HA |
|---:|---:|---:|---:|---:|---:|
| 10,079 | 7 | 0.391907054 | 0.399228460 | 0.396615165 | 0.323120280 |
| 12,288 | 7 | 0.386376465 | 0.399240741 | 0.396600085 | 0.322203121 |
| 65,536 | 7 | 0.386862575 | 0.399317358 | 0.397556596 | 0.319777802 |
| 10,079 | 28 | 0.391642832 | 0.400284515 | 0.397282369 | 0.324380191 |
| 12,288 | 28 | 0.389817826 | 0.399644786 | 0.396098562 | 0.321635427 |
| 65,536 | 28 | 0.387036247 | 0.402579031 | 0.398082823 | 0.320553194 |

Verdict: explicit rolling cross-vector software pipelining does **not** improve steady-state throughput. It is consistently slower than the existing u4 frozen schedule. There are no vector stack spills; the loss comes from a larger instruction/control footprint and from breaking the existing four-vector grouped ILP/gather schedule into a less favorable conveyor structure. The current u4 scheme already gives the Xeon out-of-order engine substantial independent work.

## 2. Important arbitrary-batch finding: scalar remainder in frozen path

The frozen VM-style kernel handles 32-wide blocks and delegates the remainder to the previous frozen checkpoint, whose final remainder loop falls back to scalar `exp()`.
This explains the apparent large PIPE3/PIPE4 wins at sizes such as 31, 63, 253, and 511: those wins are primarily elimination of the scalar tail, not a win from cross-batch pipelining.

Candidate `exp53_n2_u4_masktail` therefore keeps the frozen 32-input hot body and replaces the final remainder with faithful AVX-512 8-wide vectors plus a masked final 1..7 lanes.

Accepted exact-Xeon artifacts from run `33388774509`: shards 66, 84, 88 (additional exact-Xeon artifacts also existed for 13 and 15).

Static census for `U4_MASKTAIL` wrapper/tail function:
- 146 instructions
- 2 gather sites
- 0 vector stack spills

Accuracy on shards 66, 84, and 88:
- 200,000 mixed-domain inputs: max ULP 1, >1 ULP 0, bit differences vs frozen 0
- exhaustive batch lengths 1..1025: max ULP 1, >1 ULP 0

Representative clean shard 88 timings, ns/input:

| n | Frozen | PIPE3 | U4_MASKTAIL | VML_HA |
|---:|---:|---:|---:|---:|
| 31 | 2.571087849 | 0.469025806 | **0.469746129** | 1.031452043 |
| 32 | **0.398711563** | 0.432794375 | 0.411529583 | 0.960474062 |
| 33 | **0.465317172** | 0.534930707 | 0.544103838 | 0.989930101 |
| 63 | 1.606631005 | **0.439653280** | 0.447973651 | 0.565029788 |
| 64 | 0.403136823 | 0.430467708 | **0.400229635** | 0.531894792 |
| 65 | **0.428761462** | 0.495140231 | 0.471553538 | 0.629253846 |
| 100 | 0.483059600 | 0.450627350 | **0.436892200** | 0.479262600 |
| 253 | 0.674145277 | **0.407186225** | 0.417398379 | 0.373366285 |
| 511 | 0.533051566 | **0.400558464** | 0.405362573 | 0.351583415 |
| 512 | **0.394533594** | 0.399135693 | 0.395115967 | 0.353627832 |
| 513 | **0.398218567** | 0.405450097 | 0.405516326 | 0.365105750 |
| 10,079 | 0.394227012 | 0.399538732 | **0.384415698** | 0.321837253 |
| 12,288 | 0.385317627 | 0.398290348 | **0.384965993** | 0.322268986 |
| 65,536 | 0.384866132 | 0.399424551 | **0.384497337** | 0.320473116 |

Replication examples:
- shard 66, n=10,079: frozen 0.391947443, masktail 0.384491552
- shard 84, n=10,079: frozen was noisy at 0.426705578, masktail 0.385261789
- shard 66, n=65,536: frozen 0.385396849, masktail 0.385152727
- shard 84, n=65,536: frozen 0.385734088, masktail 0.385047789

Interpretation:
- Long sizes divisible by 32 remain essentially neutral: the frozen u4 hot body is retained.
- Long irregular `n` with a large remainder benefits materially. For n=10,079 (remainder 31), the clean shard-88 result improves from 0.394227012 to 0.384415698 ns/input, about 2.49% lower latency.
- Small irregular batches can improve dramatically because the old scalar tail dominates. At n=31, masked AVX-512 is about 5.47x faster than the frozen path on shard 88.
- A one-element remainder is different: for n=33/65/513 the existing scalar one-value tail is cheaper than spinning up a masked-vector remainder. Therefore the final arbitrary-size production dispatcher should be remainder-aware rather than using one tail policy for all 1..31 remainders.

## Final architectural verdict

1. Do **not** replace the existing 32-value u4 steady-state plant with a rolling single-vector conveyor; it loses throughput on Xeon 6973P-C.
2. Keep the existing four-vector / 32-value hot body and its early four-gather scheduling.
3. For arbitrary-size requests, add a remainder-aware AVX-512 tail so large remainders do not fall to scalar libm.
4. Preserve a tiny scalar path for very small remainders where it benchmarks cheaper (confirmed for remainder 1); map the exact crossover before freezing the production dispatcher.
5. Frozen baseline remains unchanged pending a final remainder-crossover freeze.
