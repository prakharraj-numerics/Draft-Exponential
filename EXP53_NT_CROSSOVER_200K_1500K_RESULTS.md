# EXP53 NT crossover — 200K to 1.5M

## Scope

Comparison only:
- OURS + FastFlow temporal (`exp53_n2_vmstyle_u4_0381_frozen`)
- OURS + FastFlow + non-temporal stores (`exp53_n2_vmstyle_u4_0381_nt_sfence`)

Frozen files were not modified.

Workflow run: `33407447529`
Workflow commit: `d82fe27f216d4109b61c6b095497d118f5cb3b2b`
FastFlow pinned: `d476f66ab924d8d122f54b4b90aee00ef979aea8`
Compiler: Intel oneAPI ICX/ICPX 2026.1.1 (`2026.1.1.20260724`)
CPU: Intel Xeon 6973P-C, 2 physical cores / 4 logical CPUs exposed.
Process affinity: `taskset -c 0,2`, i.e. the two distinct physical cores.
FastFlow: one persistent 2-worker pool, spin-wait + spin barrier, balanced static contiguous 32-element-aligned regions.

Three independent exact-Xeon artifacts used below: shards 34, 45, 56.
Assembly audit: four `vmovntpd` stores emitted in the NT 32-value body.
Correctness on all three: bitdiff=0, maxULP=1, gt1=0.

## Important correction to the earlier NT interpretation

The earlier `bench_exp53_nt_cache.cpp` helper computed a nominal 256 MiB target but then capped the output ring at **8 buffers**:

`if(s>8)s=8`

Therefore its actual rotating-output footprint was:
- n=262,144: ~16 MiB
- n=524,288: ~32 MiB
- n=1,000,000: ~64 MiB
- n=2,000,000: ~128 MiB
- n=4,000,000: ~256 MiB

Thus the earlier apparent ~1M-2M NT crossover was not an `n`-only threshold. It was strongly coupled to cache footprint / output reuse distance.

This focused benchmark removes that ambiguity by sizing the rotating output ring to approximately **256 MiB at every tested n** (subject only to integer slot rounding).

## True streaming / write-once-style rotating output

Median ns/input. `speedup = temporal / NT`.

### shard 34

| n | temporal+FF | NT+FF | speedup |
|---:|---:|---:|---:|
| 200,000 | 0.769172 | 0.433117 | **1.776x** |
| 500,000 | 0.772377 | 0.570878 | **1.353x** |
| 800,000 | 0.774501 | 0.570606 | **1.357x** |
| 1,000,000 | 0.773610 | 0.564645 | **1.370x** |
| 1,200,000 | 0.768474 | 0.573131 | **1.341x** |
| 1,500,000 | 0.772305 | 0.564274 | **1.369x** |

### shard 45

| n | temporal+FF | NT+FF | speedup |
|---:|---:|---:|---:|
| 200,000 | 0.760859 | 0.439044 | **1.733x** |
| 500,000 | 0.767380 | 0.567294 | **1.353x** |
| 800,000 | 0.767194 | 0.564260 | **1.360x** |
| 1,000,000 | 0.768203 | 0.562485 | **1.366x** |
| 1,200,000 | 0.768225 | 0.568435 | **1.351x** |
| 1,500,000 | 0.763591 | 0.565765 | **1.350x** |

### shard 56

| n | temporal+FF | NT+FF | speedup |
|---:|---:|---:|---:|
| 200,000 | 0.785424 | 0.444584 | **1.767x** |
| 500,000 | 0.784450 | 0.570284 | **1.376x** |
| 800,000 | 0.789746 | 0.568987 | **1.388x** |
| 1,000,000 | 0.784475 | 0.567681 | **1.382x** |
| 1,200,000 | 0.782993 | 0.568542 | **1.377x** |
| 1,500,000 | 0.790042 | 0.572490 | **1.380x** |

Across all three exact Xeon shards, NT wins every requested size in the cache-cold / long-reuse-distance streaming-output regime.

## Repeated same-output / cache-hot diagnostic

This is intentionally a different workload: the same output buffer is overwritten repeatedly.

Range over shards 34/45/56:

| n | temporal+FF ns/input | NT+FF ns/input | temporal/NT |
|---:|---:|---:|---:|
| 200,000 | 0.510-0.520 | 0.417-0.442 | **1.177-1.229x** |
| 500,000 | 0.545-0.548 | 0.562-0.570 | 0.961-0.971x |
| 800,000 | 0.545-0.549 | 0.563-0.572 | 0.960-0.969x |
| 1,000,000 | 0.546-0.550 | 0.564-0.574 | 0.958-0.968x |
| 1,200,000 | 0.546-0.550 | 0.564-0.574 | 0.958-0.969x |
| 1,500,000 | 0.546-0.550 | 0.565-0.574 | 0.959-0.968x |

Thus NT is not a universal size-only replacement. When the destination cache lines are repeatedly reused, temporal stores can be ~3-4% faster for 500K-1.5M.

## Verdict

The correct dispatch variable is **output reuse/cache residency**, not just batch size.

For a write-once / cache-cold / long-reuse-distance output stream, every tested requested size from 200K through 1.5M materially favors FastFlow + NT, by about 1.34x-1.78x on all three exact Xeon shards.

For repeatedly overwritten cache-hot output, temporal FastFlow remains preferable for 500K-1.5M.

Do not freeze a universal `n >= 1.5M` NT threshold from the old sweep. A production API should either know the output-use policy, or keep temporal as the general default and expose/use NT only when output is known to be streaming/write-once.
