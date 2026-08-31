# EXP53 merged default vs plain Intel VML_HA — 5K to 65K

## Correct benchmark protocol

This report uses the merged production default path:

- `Exp53BatchProductionExecutor::run(...)`
- temporal stores
- persistent FastFlow pool
- 2 workers
- immutable `exp53_n2_vmstyle_u4_0381_frozen` mathematical kernel

Comparator:

- plain Intel oneMKL `vmdExp(..., VML_HA)`
- linked with `-lmkl_sequential`
- no FastFlow wrapper
- `MKL_NUM_THREADS=1`

The two finished stacks are benchmarked in **separate process lifetimes**. This is essential because FastFlow is configured with spin-wait workers; timing plain VML while the FastFlow pool remains alive can steal CPU from VML and is not a fair comparator measurement.

No process-wide `taskset` is used, preserving the validated FastFlow topology.

Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow pin: `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Corrected workflow run: `33410923135`

Corrected benchmark/workflow commit: `6dd081393928ed258c628fc0c5709b4471ffe995`

Each accepted shard runs five independent process medians per method; each process median itself contains 25 timed samples. Process order alternates OURS/VML and VML/OURS.

Accepted exact Xeon artifacts:

- shard 47 — Intel(R) Xeon(R) 6973P-C, 2 physical cores / 4 logical CPUs
- shard 10 — Intel(R) Xeon(R) 6973P-C, 2 physical cores / 4 logical CPUs

The requested `25L` size is interpreted as `25K`, since it occurs in the monotonic sequence 5K, 8K, 15K, 25K, 35K, 50K, 65K.

## Exact shard 47 — median ns/input

| n | OURS merged + FF | Intel VML_HA | Intel / OURS |
|---:|---:|---:|---:|
| 5,000 | 0.427312300 | 0.326384600 | 0.763808x |
| 8,000 | 0.336737800 | 0.323919700 | 0.961934x |
| 15,000 | 0.286242843 | 0.321154855 | 1.121966x |
| 25,000 | 0.250183700 | 0.320402400 | 1.280669x |
| 35,000 | 0.241821053 | 0.319194085 | 1.319960x |
| 50,000 | 0.235280600 | 0.318923500 | 1.355503x |
| 65,000 | 0.232268376 | 0.318585219 | 1.371625x |

## Exact shard 10 — median ns/input

| n | OURS merged + FF | Intel VML_HA | Intel / OURS |
|---:|---:|---:|---:|
| 5,000 | 0.423631500 | 0.326055600 | 0.769668x |
| 8,000 | 0.340026700 | 0.323549500 | 0.951541x |
| 15,000 | 0.275602302 | 0.321347047 | 1.165981x |
| 25,000 | 0.251416500 | 0.320144800 | 1.273364x |
| 35,000 | 0.243648020 | 0.319252832 | 1.310303x |
| 50,000 | 0.234819300 | 0.318750700 | 1.357430x |
| 65,000 | 0.230501559 | 0.318853394 | 1.383303x |

## Two-shard consolidated midpoint

| n | OURS merged + FF | Intel VML_HA | Intel / OURS | verdict |
|---:|---:|---:|---:|:---|
| 5,000 | 0.425472 | 0.326220 | 0.7667x | Intel wins |
| 8,000 | 0.338382 | 0.323735 | 0.9567x | near parity; Intel slight win |
| 15,000 | 0.280923 | 0.321251 | 1.1436x | OURS wins ~14% |
| 25,000 | 0.250800 | 0.320274 | 1.2770x | OURS wins ~28% |
| 35,000 | 0.242735 | 0.319223 | 1.3151x | OURS wins ~32% |
| 50,000 | 0.235050 | 0.318837 | 1.3565x | OURS wins ~36% |
| 65,000 | 0.231385 | 0.318719 | 1.3774x | OURS wins ~38% |

## Conclusion

The clean finished-stack crossover is between 8K and 15K on these exact Xeon 6973P-C shards.

- 5K: FastFlow synchronization overhead is too large; plain Intel wins.
- 8K: essentially crossover territory, with Intel still about 4-5% faster.
- 15K onward: merged OURS+FastFlow wins on both exact shards.
- Advantage grows from roughly 14% at 15K to roughly 38% at 65K.

The earlier same-process comparator result is not used for this conclusion because VML was timed while FastFlow spin-wait workers were still alive.
