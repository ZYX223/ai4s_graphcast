# GraphCast BF16 Mixed-Precision Inference Design

## Goal

Add an auditable BF16 inference mode to the existing device-resident GraphCast
rollout without changing the established eager execution model or the seven-op
FlagGems routing contract. The primary execution contract keeps autoregressive
state in FP32, runs the model forward under BF16 autocast, converts the model
prediction to FP32, and performs residual reconstruction and state feedback in
FP32.

The first campaign covers H100 plus C550, BW3000, S5000, Ascend 910C, and 810E.
Training BF16 is outside this design.

## Non-goals

- Do not enable `torch.compile`, CUDA Graph, NPU Graph, or another graph-capture
  mode.
- Do not convert the checkpoint or persistent model parameters to BF16.
- Do not keep the autoregressive state or forcing sequence in BF16.
- Do not change the seven logical operators in
  `configs/flaggems/graphcast_inference.yaml`.
- Do not describe FlagGems dispatch coverage as proof that an operator executed
  in BF16 or on matrix acceleration hardware.

## Precision contract

The public precision modes are `fp32` and `bf16`. Existing `fp32`
behaviour remains byte-for-byte compatible at the interface level and remains
the default.

`bf16` is reported as `mixed_precision` and has the following mandatory
semantics:

1. Checkpoint tensors and persistent model parameters remain FP32.
2. CPU preprocessing, normalization, initial state, forcing tensors, feedback
   scales, and the persistent device-resident autoregressive state remain FP32.
3. Only the complete per-lead model forward is entered under
   `torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16)`.
4. The model prediction is checked for a supported floating dtype and converted
   to FP32 immediately after the forward and before residual reconstruction or
   state mutation.
5. Residual-to-absolute conversion, forcing insertion, state feedback, finite
   checks, final D2H, NetCDF creation, and comparison metrics use FP32 or FP64.
6. Warm-up uses the identical precision boundary but does not modify the formal
   state and is excluded from the measured 40-lead E2E interval.
7. BF16 support is fail-closed. A platform that cannot enter BF16 autocast, does
   not produce a supported prediction, silently performs no BF16-eligible
   operation, or fails the correctness contract produces no performance row.

Autocast is a per-operator policy. It may execute matrix operations in BF16 and
retain sensitive reductions or dtype-preserving operators in FP32. The report
therefore calls this mode mixed precision and records observed dtypes instead
of claiming that the entire model ran in BF16.

## Runtime and reporting

`scripts/inference/run_operational_40step_inference.py` owns a typed precision
policy resolved once after the runtime backend is known. The rollout helper
receives this policy explicitly; it must not read ambient environment variables.

The report schema is extended with:

- requested and effective precision mode;
- parameter, state, forcing, forward-prediction, feedback, and final-output
  dtypes;
- autocast device type, requested dtype, availability, and whether at least one
  BF16-eligible operation was observed;
- the existing FlagGems requested/registered/observed dispatch record;
- per-core-operator observed input and output dtypes when the backend exposes
  them;
- an explicit distinction between dispatch coverage, BF16 execution evidence,
  and hardware matrix-path evidence.

The existing seven-op requirement remains unchanged. C550, BW3000, S5000, and
910C require 7/7 observed FlagGems dispatch. The current 810E production profile
continues to require 6/7 because `addmm` remains excluded; its report must say
`addmm=Native` rather than presenting 7/7.

## Correctness design

Correctness and performance are separate processes and output roots.

For every platform, correctness is evaluated in this order:

1. FP32 Native produces the reference from the exact checkpoint, inputs,
   forcings, statistics, source revision, and 40-lead workload.
2. BF16 Native is compared with FP32 Native to measure the precision change.
3. BF16 FlagGems is compared with BF16 Native to isolate replacement-specific
   error from BF16 quantization error.

Two complementary comparisons are required:

- Same-input per-lead comparison: FP32 Native, BF16 Native, and BF16 FlagGems
  consume the same FP32 state for each lead. This measures local forward error
  without feeding candidate error into the next lead.
- Free 40-lead autoregressive comparison: each execution feeds back its own FP32
  reconstructed prediction. This measures accumulated forecast trajectory
  divergence.

Both comparisons record finite status, max absolute error, RMSE, normalized
maximum error, and normalized RMSE per lead and output channel. The acceptance
envelope is frozen from Native FP32 versus Native BF16 evidence before examining
FlagGems results; FlagGems must not add a statistically material error outside
that frozen envelope. Thresholds must not be tuned after candidate results are
visible.

## Performance and MFU design

Only correctness-passing configurations enter performance measurement.

- Execution is eager on all six platforms.
- Data loading, model construction, FlagGems compilation/cache generation, and
  warm-up are outside the E2E interval.
- The measured interval contains all 40 dependent model leads, FP32 feedback,
  the final D2H, and the final device synchronization.
- Intermediate leads remain device-resident and perform no full-output D2H.
- Native and FlagGems each run in three fresh processes with counterbalanced
  order. The reported E2E value is the median; raw samples remain available.
- Profiling is performed in separate runs and is never included in timing data.

BF16 empirical MFU is:

`(forward_FLOPs_per_lead * 40 / median_E2E_seconds) / measured_BF16_GEMM_roof`

The denominator is the same-device, same-runtime BF16 GEMM roof produced by
`tools/dense_gemm_peak`; vendor theoretical BF16 peak remains a separate field.
The H100 Native eager BF16 MFU is the comparison baseline. A domestic-platform
target, if applied, is 80% of that BF16 baseline and is not inherited from the
FP32 campaign.

## Failure handling

Every run uses a new output directory. Missing BF16 capability, dtype mismatch,
non-finite values, incomplete FlagGems dispatch, correctness failure, unstable
timing, device contention, or missing profiler evidence is retained as a failed
attempt and cannot be promoted into the performance table. There is no silent
fallback from BF16 to FP32 or from FlagGems to Native.

## Test strategy

Unit tests cover precision parsing, FP32 default compatibility, autocast scope,
prediction conversion before feedback, persistent-state dtype, warm-up
exclusion, dtype evidence, unsupported runtime rejection, and report schema.
Model-level tests use a deterministic small graph to compare FP32, BF16 Native,
and BF16 FlagGems data flow without requiring a full 25 km workload.

Device acceptance proceeds H100 Native first, then each domestic Native run,
then each domestic FlagGems run. Full 40-lead performance measurement starts
only after the same-input and free-roll correctness gates pass.
