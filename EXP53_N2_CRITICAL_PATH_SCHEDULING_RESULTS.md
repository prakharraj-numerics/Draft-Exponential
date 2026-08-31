# EXP53 faithful n=2 — critical-path scheduling attacks

## Scope

Frozen survivor `exp53_n2_vmstyle_u4_0381_frozen` was not modified.
All candidates preserve the faithful n=2 formula, TAB128 reduction, Q4 Horner order, ER-low repair, and fused exponent scaling.

Target CPU: Intel Xeon 6973P-C.
Compiler: Intel oneAPI ICX 2026.1.1 (`2026.1.1.20260724`).

The scheduling objective was to increase independent AVX-512 arithmetic chains in flight while keeping gathers early and minimizing register slack/spills.

## 1. CP56 / seven-stream attacks

Workflow run: `33397547386`.
Accepted exact-Xeon shard: 11.

Variants:
- `exp53_n2_cp56_attack`
- `exp53_n2_cp56_split52`
- `exp53_n2_cp56_split61`

Static assembly census:

| Function | instructions | gathers | FMA family | mulpd | distinct ZMM | vector stack refs |
|---|---:|---:|---:|---:|---:|---:|
| frozen u4 | 147 | 4 | 44 | 8 | 30 | 0 |
| CP56 | 224 | 7 | 77 | 14 | 32 | 0 |
| SPLIT52 | 226 | 7 | 77 | 14 | 32 | 2 |
| SPLIT61 | 226 | 7 | 77 | 14 | 32 | 0 |

Accuracy on 200,000 mixed-domain inputs for all three CP56 variants:
- max ULP = 1
- >1 ULP = 0
- bit differences vs frozen = 5

Clean small/medium timings, ns/input:

| n | frozen | CP56 | SPLIT52 | SPLIT61 | VML_HA |
|---:|---:|---:|---:|---:|---:|
| 10,079 | 0.390314099 | 0.411997705 | 0.413131756 | 0.397806189 | 0.322115503 |
| 12,288 | 0.382788153 | 0.409775764 | 0.412255392 | 0.397396500 | 0.320709329 |

The 65,536 row on this shard was system-noisy (frozen 0.51999 and VML_HA 0.78411), so it is not used for the performance verdict. At 262,144 all candidates clustered around ~0.535 ns/input and did not establish a meaningful compute-path win.

Verdict: seven streams can be kept spill-free, but the larger body and less favorable gather/arithmetic placement do not beat the frozen four-stream schedule. `SPLIT61` is the least damaging CP56 variant but is still ~3.8% slower at n=12,288.

## 2. CP64 blocked / eight-chain attack

Source: `exp53_n2_cp64_blocked_attack.c`.
The design deliberately prepares two 4-vector microblocks, compresses each to `(r, scale)`, then runs an eight-chain Q4 wave and drains two 4-vector final groups. This was intended to realize eight independent Q4 chains while avoiding the original setup-liveness spike.

Workflow run: `33398147661`.
Accepted exact-Xeon shard: 19.

Static assembly census:

| Function | instructions | gathers | FMA family | mulpd | distinct ZMM | vector stack refs |
|---|---:|---:|---:|---:|---:|---:|
| frozen u4 | 147 | 4 | 44 | 8 | 30 | 0 |
| CP64 blocked | 263 | 8 | 88 | 16 | 32 | 10 |

Accuracy:
- max ULP = 1
- >1 ULP = 0
- bit differences vs frozen = 0

Clean timings, ns/input:

| n | frozen | CP64 blocked | VML_HA | frozen / blocked |
|---:|---:|---:|---:|---:|
| 10,079 | 0.390620212 | 0.403272997 | 0.321720913 | 0.968625 |
| 12,288 | 0.383177586 | 0.394722012 | 0.321454386 | 0.970753 |
| 65,536 | 0.384158275 | 0.396221724 | 0.319446189 | 0.969554 |

At 262,144 both frozen and candidate were in a different/noisier regime; that row is not used to select a compute-path winner.

The eight-chain design fails for a concrete architectural reason: it saturates all 32 architectural ZMM registers and ICX emits 10 vector stack references. The intended latency-hiding gain is more than erased by spill/reload and the larger instruction footprint.

## Final verdict

1. The critical-path hypothesis was tested directly, not merely estimated.
2. Seven live streams are feasible without vector spills, but the tested seven-stream schedules are slower than frozen.
3. Eight live Q4 chains trigger register-pressure spill/reload in the blocked CP64 implementation and are slower by about 3% on clean exact-Xeon rows.
4. The frozen VM-style u4 remains the steady-state survivor.
5. Do not promote any CP56/CP64 candidate.
6. Future scheduling work must either reduce live state per stream below the current `(r, scale, Q4/final temporaries)` requirement or reduce arithmetic/table service demand; simply increasing the number of simultaneously materialized vector chains is not sufficient on this 32-ZMM architecture.
