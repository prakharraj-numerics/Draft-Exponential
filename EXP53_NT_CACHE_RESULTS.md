# EXP53 faithful n=2 — non-temporal output cache attack

## Scope

Frozen survivors were not modified:
- `exp53_n2_vmstyle_u4_0381_frozen.c`
- `exp53_fastflow_batch_2core_frozen.hpp`

Candidate:
- `exp53_n2_vmstyle_u4_0381_nt_sfence.c`

The candidate preserves the exact frozen mathematical body and operation order. For a 64-byte-aligned output pointer, the only hot-loop policy change is temporal AVX-512 stores -> `_mm512_stream_pd` non-temporal stores, followed by one explicit function-level `_mm_sfence()` after the streaming region. A non-64-byte-aligned output pointer falls back completely to the frozen temporal kernel, preserving exact vector grouping and bit identity.

## Environment

Corrected workflow run: `33404696495`
Head: `8514fbb8ae62a16ec6344a27e931a54ef90340c1`

Accepted exact-Xeon jobs:
- shard 3: job `99529250819`
- shard 8: job `99529250777`

CPU on both:
- Intel Xeon 6973P-C
- 4 logical CPUs exposed
- 2 physical cores
- 2 SMT threads/core
- one socket / one NUMA node

Compiler:
- Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)

FastFlow:
- pinned `d476f66ab924d8d122f54b4b90aee00ef979aea8`

Compiler flags:
`-O3 -xHost -qopt-zmm-usage=high -fp-model=precise -fno-math-errno -DNDEBUG -qopenmp-simd`

Assembly audit on both exact shards confirms four `vmovntpd` instructions in the 32-value NT body. The object-level `sfence` grep reports multiple copies because this translation unit contains inherited/frozen routines and compiler-emitted variants; source semantics contain one explicit fence after each completed NT streaming region.

## Benchmark design

The corrected benchmark deliberately avoids an artificially cache-hot single-output comparison.

It measures:
1. repeated same-output (hot) calls;
2. a rotating output ring sized toward ~256 MB total output footprint;
3. the 2-physical-core FastFlow path over that rotating ring.

For the primary FastFlow comparison, temporal frozen, NT candidate, and VML_HA all use the **same persistent FastFlow worker pool** with the frozen winning synchronization policy:
- 2 workers = 2 physical cores;
- spin-wait workers;
- spin barrier;
- static balanced contiguous 32-element-aligned worker regions.

Temporal FastFlow is measured twice around the other methods; the faster temporal median is used as the conservative temporal reference (`temporal_ff_fair`). This prevents pool placement/warm-up artifacts from being mistaken for an NT win.

VML remains `vmdExp(..., VML_HA)` linked with `-lmkl_sequential` inside each identical FastFlow worker.

## Correctness

Both exact-Xeon shards, 200,000 mixed-domain inputs:
- aligned NT vs frozen bit differences: **0**
- misaligned fallback vs frozen bit differences: **0**
- max ULP: **1**
- >1 ULP: **0**

Thus the accepted NT cache candidate is bit-identical to the frozen implementation for the tested inputs and preserves the existing <=1 ULP screen.

## Primary result — OURS + FastFlow

Median ns/input. Speedup = frozen temporal FastFlow / NT FastFlow.

### Exact shard 3

| n | Frozen temporal + FF | NT + FF | Speedup | VML_HA + FF | VML / NT |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 0.433084 | 0.416924 | 1.039x | 0.390731 | 0.937x |
| 262,144 | 0.280253 | 0.268751 | 1.043x | 0.258307 | 0.961x |
| 524,288 | 0.541138 | 0.562429 | 0.962x | 0.545217 | 0.969x |
| 1,000,000 | 0.291374 | 0.290615 | 1.003x | 0.291844 | 1.004x |
| 2,000,000 | 0.600840 | 0.553036 | **1.086x** | 0.652408 | **1.180x** |
| 4,000,000 | 0.469948 | 0.288563 | **1.629x** | 0.474790 | **1.645x** |
| 8,000,000 | 0.771073 | 0.557036 | **1.384x** | 0.787701 | **1.414x** |
| 16,000,000 | 0.614176 | 0.462245 | **1.329x** | 0.636720 | **1.377x** |

### Exact shard 8

| n | Frozen temporal + FF | NT + FF | Speedup | VML_HA + FF | VML / NT |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 0.449927 | 0.434093 | 1.036x | 0.446422 | 1.028x |
| 262,144 | 0.274185 | 0.254891 | 1.076x | 0.265760 | 1.043x |
| 524,288 | 0.543696 | 0.564391 | 0.963x | 0.549665 | 0.974x |
| 1,000,000 | 0.280280 | 0.291796 | 0.961x | 0.280909 | 0.963x |
| 2,000,000 | 0.659898 | 0.553419 | **1.192x** | 0.649142 | **1.173x** |
| 4,000,000 | 0.475644 | 0.296620 | **1.604x** | 0.464402 | **1.566x** |
| 8,000,000 | 0.795720 | 0.563240 | **1.413x** | 0.819186 | **1.454x** |
| 16,000,000 | 0.626376 | 0.500970 | **1.250x** | 0.594942 | **1.188x** |

Interpretation:
- Below ~1M the cache regimes are non-monotonic; NT can lose, tie, or produce only a small win. It must not replace the temporal path globally.
- At **2M and above**, both exact shards independently show a material NT + FastFlow win over frozen temporal FastFlow.
- At **4M**, the improvement is ~1.60-1.63x over our frozen FastFlow path.
- At 4M, NT + FastFlow also beats equally FastFlow-wrapped VML_HA by ~1.57-1.65x on the two exact shards.
- 8M remains a large win: ~1.38-1.41x over temporal and ~1.41-1.45x over VML_HA.
- 16M remains a real win: ~1.25-1.33x over temporal and ~1.19-1.38x over VML_HA.

## Serial rotating-output evidence

The same qualitative memory-traffic transition appears without FastFlow. Examples:

### shard 3
- 2M: temporal ring 0.718911 vs NT ring 0.563768 -> **1.275x**
- 4M: 0.898114 vs 0.564601 -> **1.591x**
- 8M: 0.977401 vs 0.579572 -> **1.686x**
- 16M: 1.103823 vs 0.879483 -> **1.255x**

### shard 8
- 2M: temporal ring 0.831970 vs NT ring 0.565026 -> **1.472x**
- 4M: 0.929222 vs 0.574178 -> **1.618x**
- 8M: 1.005228 vs 0.697817 -> **1.441x**
- 16M: 1.169038 vs 0.811320 -> **1.441x**

This confirms that the large-batch benefit is a cache/memory-store-policy effect rather than a FastFlow scheduling artifact.

## Production interpretation

The NT candidate should be treated as a **large, write-once/output-streaming mode**, not as a universal replacement.

Conservative dispatch rule supported by both exact shards:
- output not 64-byte aligned -> frozen temporal path;
- n < 2,000,000 -> frozen temporal/FastFlow survivor;
- n >= 2,000,000 and output is write-once / not immediately reread -> NT + FastFlow candidate is justified by this benchmark.

If downstream code immediately consumes the output and benefits from it remaining cache-resident, temporal stores may still be preferable; this benchmark intentionally targets the write-once/streaming batch case.

## Verdict

**Non-temporal output stores are a genuine second systems-level win after FastFlow.**

They do not help universally, but in the large streaming-array regime they remove a major memory-traffic penalty. The strongest clean result is around 4M elements, where OURS + FastFlow + NT is roughly 1.6x faster than OURS + FastFlow temporal and also roughly 1.6x faster than equally parallelized VML_HA on both exact Xeon shards.

Do not modify the existing frozen temporal or FastFlow files. Keep the NT implementation as a separate large-stream candidate until/if a production dispatcher is explicitly frozen.
