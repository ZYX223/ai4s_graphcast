# GraphCast Platform Model Execution Profiles

## Goal

Keep one inference implementation while making the H100 reference path and the
five domestic-platform optimized path explicit, reproducible, and impossible to
mix accidentally.

## Agreed profiles

- `baseline` is valid only for `h100`. It uses concatenated edge projection,
  disables the mesh-to-grid full-grid fast path, keeps configured edge chunk
  sizes unchanged, and uses inline `index_add`.
- `domestic-v1` is valid only for `c550`, `bw3000`, `s5000`, `910c`, and
  `810e`. It uses factorized edge projection, enables the mesh-to-grid
  full-grid fast path, sets both edge chunk sizes to `131072`, and uses inline
  `index_add`.

The profile is independent of inference precision: the same profile applies to
both strict FP32 and BF16 mixed-precision runs. Native and FlagGems pairs on a
domestic platform must use the same `domestic-v1` profile. H100 remains on
`baseline` so its result remains the fixed reference for the H100 × 0.8 target.

## Interface and evidence

The formal runner requires `--platform` and `--model-execution-profile`.
Unsupported platform/profile pairs fail before files, model weights, or device
runtime are loaded. The profile's expected backend is also checked before
FlagGems configuration, vendor imports, or device initialization. The
profile-selected backend is authoritative when `GRAPHCAST_BACKEND` is unset or
`auto`; an explicitly configured different backend is rejected without ambient
backend probing. The
low-level model optimization switches are no longer a formal command-line
interface; profile expansion is the only formal source of those values.

Every report records the selected platform/profile/backend identity, an
explicitly named `configuration_overrides` object, and the separately observed
effective model settings. A null chunk override therefore means “preserve the
config value”; it is not confused with the effective chunk size. No
aggregation-buffer reuse is enabled because the prior ablation measured a
small regression.

## Validation

Unit tests cover every accepted and rejected platform/profile pair, both
precisions, and exact expanded settings. Device validation uses one-step
Native/FlagGems correctness checks followed by three fresh 40-step BF16 runs per
mode and platform. Performance is reported only after finite output, expected
operator coverage, and same-precision correctness pass.
