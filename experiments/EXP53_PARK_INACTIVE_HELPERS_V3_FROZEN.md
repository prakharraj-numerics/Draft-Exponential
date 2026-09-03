# EXP53 inactive-helper parking v3 — frozen experimental candidate

Status: **FROZEN EXPERIMENTAL CANDIDATE — NOT PRODUCTION**

Do not modify this frozen file in place. Any further lifecycle/scheduling work must use a new candidate file and new benchmark evidence.

## Frozen implementation

- File: `experiments/exp53_park_inactive_helpers_v3_frozen.hpp`
- Blob SHA: `268daccd77120395fc6ed7a451ed9ec1dbdf7036`
- This blob is byte-for-byte identical to `experiments/exp53_park_inactive_helpers_candidate.hpp` as benchmarked at commit `c5c702ffc1f16ac76d5af33439b573ad0f0655a3`.
- Freeze-copy commit: `3b29336428ef73f98c63d7e98096a2ae030db5c1`

## What is frozen

The numerical EXP53 math and routing are unchanged from production. The experimental change is helper lifecycle only:

- inactive constructed helpers park on a condition variable;
- the selected helper keeps the active spin handoff;
- payload and generation are published before waking a parked helper;
- worker `seen` generation is preserved across park/wake;
- park-to-wake transition is serialized with the CV mutex to avoid lost notifications;
- already-active helper calls remain on the mutex-free hot path.

## Validation evidence

Benchmark workflow: `EXP53 park inactive helpers vs current vs Intel Xeon`

- GitHub Actions run: `33736470047`
- Benchmark head: `c5c702ffc1f16ac76d5af33439b573ad0f0655a3`
- Exact CPU: Intel Xeon 6973P-C with AVX-512
- Exact result artifacts: shards 56 and 94
- Full native wall/CPU sweep: PASS
- Intel SDE instructions/logical-memory sweep: PASS
- No deadlock in v3
- Production files were untouched by the experiment
- Observed comparator disagreement in this benchmark: max 2 ULP vs Intel VML_HA (diagnostic comparator only)

## Promotion policy

This state is intentionally preserved before production promotion. Next step is to benchmark the frozen v3 lifecycle in normal production-style call sequences rather than only the forced-both-helper-warmed stress sequence. Production must not be changed unless that follow-up evidence is clean and promotion is explicitly approved.
