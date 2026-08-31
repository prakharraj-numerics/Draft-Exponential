# EXP53 final frozen production checkpoint — custom permanent 2-core + rare NT

## Status

This is the current production survivor and is frozen.

Do **not** modify:
- `exp53_batch_custom_2core_nt_frozen.hpp`
- underlying frozen mathematical kernel `exp53_n2_vmstyle_u4_0381_frozen.c`

Future experiments, if any, must use new candidate files and benchmark against this checkpoint.

## Production policy

Default/common path:
- custom permanent 2-core dispatcher
- temporal stores
- immutable `exp53_n2_vmstyle_u4_0381_frozen` mathematical kernel

Explicit rare streaming/write-once path:
- same custom permanent 2-core dispatcher
- `exp53_n2_vmstyle_u4_0381_nt_sfence`
- NT mode is selected by caller semantics, never by an automatic batch-size threshold

Active production wrapper:
- `exp53_batch_production.hpp`
- FastFlow is no longer used by the active production path
- the older FastFlow frozen survivor remains untouched as historical evidence

## Frozen runtime

File:
- `exp53_batch_custom_2core_nt_frozen.hpp`

Freeze commit:
- `68ad998a61b937993107f582d335193228cbe581`

Production promotion commit:
- `95c44f45839d7faff0b073299e3a4316d9e96101`

Validated topology:
- Intel(R) Xeon(R) 6973P-C
- 2 physical cores / 4 logical CPUs
- CPU0/1 = physical core0
- CPU2/3 = physical core1
- caller CPU0, permanent helper CPU2

Execution:
- caller computes first 32-element-aligned half
- permanent helper computes second half
- generation atomic handoff
- completion atomic rendezvous
- no scheduler, queue, work stealing, or task allocation
- fewer than two complete 32-element blocks fall back to serial frozen kernel

## Performance evidence before freeze

Final requested boundary comparison:
- workflow run `33412987407`
- exact Xeon shards 6 and 54
- benchmark/workflow commit `40bdaa06fd8f2d34a377fcba6ca9ebc16b785eee`
- report `EXP53_CUSTOM2_VS_MERGED_FF_FINAL_BOUNDARIES.md`

Combined completed sweep covered:
- low: 50, 100, 250, 500, 750, 1K, 3K
- intermediate/medium: 5K through 200K
- large: 500K through 2.5M
- high: 3M, 4M, 5M, 8M

Result:
- no tested batch size gave merged FastFlow a reproducible performance advantage over custom2 on this exact 2-core Xeon target
- custom2 was dramatically faster at low/small sizes
- temporal paths approached parity at large sizes
- custom NT remained better through 5M; 8M NT was effectively parity/noise

Correctness throughout comparison:
- default custom output bit-identical to frozen serial
- streaming NT custom output bit-identical to frozen serial

## Final post-promotion smoke

Workflow:
- `.github/workflows/exp53-custom2-final-frozen-smoke.yml`

Run:
- `33413663985`

Exact Xeon validation shard:
- shard 19
- job `99559075277`
- CPU `Intel(R) Xeon(R) 6973P-C`

Validated head:
- `8228fac9ab2cd01e125322f6d55db545c606616c`

Result:

```text
FINAL_FROZEN_SMOKE PASS default_bitdiff=0 streaming_bitdiff=0 serial_escape_bitdiff=0
```

Smoke sizes:
- 50
- 100
- 250
- 1K
- 3K
- 65K
- 1M

## Freeze rule

`exp53_batch_custom_2core_nt_frozen.hpp` is the immutable final custom runtime survivor. The production wrapper may only be changed in the future if explicitly requested; the frozen survivor itself must remain unchanged.
