# EXP53 faithful n=2 — synchronized FastFlow batch benchmark

## Scope

Frozen survivor `exp53_n2_vmstyle_u4_0381_frozen` was not modified.

This experiment tests whether FastFlow can reduce **total batch wall-clock time** by distributing independent chunks across CPU cores while each worker executes the existing frozen AVX-512 kernel unchanged.

For fairness, Intel oneMKL `vmdExp(..., VML_HA)` is also wrapped in the **identical FastFlow executor** and given the same worker counts/scheduling choices. Thus this is not a single-thread OURS vs multicore Intel (or vice versa) comparison.

## Environment

Workflow run: `33399802068`

Exact-Xeon shards used:
- shard 2, job `99513042706`
- shard 7, job `99513042642`

CPU:
- Intel Xeon 6973P-C
- 4 logical CPUs exposed by the GitHub runner
- 2 physical cores
- 2 hardware threads per core
- 1 socket
- 1 NUMA node

Compiler:
- Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow:
- repository `fastflow/fastflow`
- pinned commit `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Compiler flags:
`-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG -qopenmp-simd`

VML link remains sequential inside each worker:
`-lmkl_intel_lp64 -lmkl_sequential -lmkl_core`

## Synchronization / scheduling design

Implementation:
- `exp53_fastflow_batch.hpp`
- `bench_exp53_fastflow.cpp`

The FastFlow executor uses a persistent `ff::ParallelFor` object. Worker creation/destruction is outside measured regions; each measured call includes actual dispatch and completion synchronization.

For repeated short calls the executor enables:
- spin-wait workers
- spin barrier

Static scheduling:
- one balanced contiguous region per active worker
- worker boundaries are aligned to 32 elements so each worker preserves the frozen u4 AVX-512 path
- last worker owns the final remainder
- no dynamic scheduler thread is needed

Dynamic scheduling was also swept with 128, 512, 2048 and 8192-element blocks.

Worker counts tested on the runner:
- 2 workers = the two physical cores
- 4 workers = all four logical CPUs / SMT

The 4-worker mode is consistently poor for this very short AVX-512 kernel. It causes severe synchronization/contention effects and is not a survivor. The useful FastFlow configuration on this runner is two workers, i.e. one worker per physical core.

## Correctness

On both exact-Xeon shards, 200,000 mixed-domain inputs:

- FastFlow static output vs frozen serial: bit differences = 0
- FastFlow dynamic output vs frozen serial: bit differences = 0
- max ULP = 1
- >1 ULP = 0

Therefore FastFlow changes only work distribution/synchronization; it does not alter the mathematical output of the frozen kernel.

## Clean shard 2 — primary result

Median ns/input. `OURS effective` and `VML effective` select the faster of serial and the best FastFlow configuration for that implementation.

| n | OURS serial | OURS effective | OURS config | OURS serial / effective | VML serial | VML effective | VML config | VML effective / OURS effective |
|---:|---:|---:|---|---:|---:|---:|---|---:|
| 100 | 0.484550000 | 0.484550000 | serial | 1.000x | 0.477410000 | 0.477410000 | serial | 0.9853x |
| 253 | 0.677425296 | 0.677425296 | serial | 1.000x | 0.379491700 | 0.379491700 | serial | 0.5602x |
| 10,079 | 0.394928549 | 0.359561484 | static, 2 workers | 1.098x | 0.322888073 | 0.302091257 | static, 2 workers | 0.8402x |
| 12,288 | 0.391135317 | 0.334285857 | static, 2 workers | 1.170x | 0.321621094 | 0.274712540 | static, 2 workers | 0.8218x |
| 65,536 | 0.384712594 | 0.258516656 | static, 2 workers | 1.488x | 0.321975958 | 0.206357362 | static, 2 workers | 0.7982x |
| 262,144 | 0.527478536 | 0.254524740 | dynamic, 2 workers, block 8192 | 2.072x | 0.528121440 | 0.217324066 | static, 2 workers | 0.8538x |
| 1,000,000 | 0.541838750 | 0.274047000 | static, 2 workers | 1.977x | 0.544008500 | 0.271398500 | static, 2 workers | 0.9903x |

Key wall-clock findings from shard 2:

1. `n=100` and `n=253`: FastFlow overhead dominates; serial dispatch is mandatory.
2. `n=10,079`: FastFlow gives OURS ~1.10x speedup.
3. `n=12,288`: FastFlow gives OURS ~1.17x speedup.
4. `n=65,536`: FastFlow gives OURS ~1.49x speedup.
5. `n=262,144`: FastFlow gives OURS ~2.07x speedup.
6. `n=1,000,000`: FastFlow gives OURS ~1.98x speedup.
7. Fairly parallelized VML_HA also benefits. OURS therefore does not claim an artificial multicore win over a sequential Intel comparator.
8. At `n=1,000,000`, OURS = 0.274047 ns/input and fair FastFlow VML_HA = 0.2713985 ns/input: the total-time gap is only about 0.98% on this shard.

## Exact shard 7 — confirmation

Median ns/input:

| n | OURS serial | OURS effective | OURS config | OURS speedup | VML effective | VML / OURS |
|---:|---:|---:|---|---:|---:|---:|
| 100 | 0.545490000 | 0.545490000 | serial | 1.000x | 0.544590000 | 0.9984x |
| 253 | 0.780677470 | 0.780677470 | serial | 1.000x | 0.432828458 | 0.5544x |
| 10,079 | 0.448527091 | 0.356814749 | static, 2 workers | 1.257x | 0.291214557 | 0.8162x |
| 12,288 | 0.441594050 | 0.373174079 | dynamic, 2 workers, block 2048 | 1.183x | 0.279886569 | 0.7500x |
| 65,536 | 0.440738615 | 0.265983707 | static, 2 workers | 1.657x | 0.201720191 | 0.7584x |
| 262,144 | 0.538454183 | 0.271205902 | static, 2 workers | 1.985x | 0.210261536 | 0.7753x |
| 1,000,000 | 0.537629500 | 0.277247000 | dynamic, 2 workers, block 8192 | 1.939x | 0.268438750 | 0.9682x |

Shard 7 has a noisier single-thread regime than shard 2, but independently confirms the important result: two-core FastFlow materially reduces OURS wall-clock time for medium/large batches, and the OURS-vs-VML gap becomes very small at one million elements.

## Scheduling interpretation

The winning pattern is not a fine-grain A->B->C task pipeline. It is coarse enough to amortize synchronization:

`FastFlow worker -> large contiguous chunk -> frozen spill-free AVX-512 u4 kernel`

The frozen intra-core schedule remains unchanged. FastFlow supplies inter-core batch parallelism and completion synchronization.

Dynamic scheduling with tiny blocks is strongly inferior because every element has essentially uniform EXP cost; the scheduler traffic is unnecessary. Static balanced scheduling is the default survivor. Large-block dynamic scheduling can occasionally win at very large n, so it remains an optional tuned path rather than the default.

## Final verdict

FastFlow **does rescue total batch time** on the tested Xeon runner once the batch is sufficiently large.

- Tiny batches: use frozen serial.
- Medium/large batches: use FastFlow with one worker per physical core and balanced 32-aligned chunks.
- Do not blindly use logical-CPU count / SMT for this kernel.
- Do not use tiny dynamic grains.
- Accuracy is unchanged.
- Fair multicore VML_HA remains faster at 10k-262k on these two shards.
- At 1,000,000 inputs, FastFlow brings OURS to within ~1-3% of equally parallelized VML_HA on the two exact-Xeon runs.

The exact crossover between 253 and 10,079 inputs was not measured in this first sweep, so no unsupported production threshold is frozen yet.
