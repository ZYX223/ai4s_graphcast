# Dense GEMM roof and GraphCast MFU tools

This directory retains the workload and profiler launchers used for the
six-platform dense-GEMM campaign.  The measurement is an achievable runtime
roof for one visible device; it is not a vendor theoretical peak.

## Measurement contract

- Workload: square `C = A @ B`, counted as `2 * n^3` FLOPs.
- Primary size: `n=24576`; diagnostic sizes: 8192, 12288, and 16384.
- Precision modes: strict FP32, fast FP32 (TF32/HF32 or a vendor equivalent),
  and BF16.
- Five calls are excluded for warm-up, followed by 20 synchronized samples.
- Six independent processes use counterbalanced precision order, producing 120
  primary samples for each platform and precision.
- Timing and profiler runs are separate.  An enabled precision option is not
  treated as hardware-path proof without a profiler kernel name or counter.

`torch_dense_gemm.py` is the common workload.  `run_container_gemm.sh` runs one
sealed process on H100, 910C, BW3000, S5000, C550, or 810E.  The platform
profilers retain the hardware-path evidence:

| Platform | Retained profiler entry |
|---|---|
| H100 | common runner; collect the timing-free hardware proof separately with Nsight Compute |
| 910C | `run_910c_matrix_profile.sh`, `run_ascend_dmi.sh` |
| BW3000 | `run_bw3000_mmop_profile.sh` |
| S5000 | `run_s5000_activity_profile.sh` |
| C550 | `run_c550_activity_profile.sh`, `run_c550_tensor_profile.sh` |
| 810E | `run_810e_activity_profile.sh`, `run_810e_tensor_profile.sh` |

The physical Hygon device is BW3000.  The existing remote container is still
named `zyx-graphcast-bw1000-v4`; that string is a legacy environment alias and
must not be used as the hardware model in reports.

`run_platform_campaign.sh` executes the six counterbalanced blocks for one
platform.  Its sixth argument is the explicit remote-access helper path; no
machine-specific helper path is hidden in the project.  Run it once per
platform and aggregate the six completed platform directories with
`aggregate_results.py`.  The full controls and invalidation rules are frozen in
`PROTOCOL.md`.

## Path-matched MFU

`calculate_path_matched_mfu.py` computes:

```text
effective_TFLOP/s = useful_FLOPs_for_40_leads / mean_40_lead_E2E_seconds / 1e12
empirical_path_matched_MFU = effective_TFLOP/s / measured_GEMM_roof_TFLOP/s
```

The calculator rejects a TF32/HF32 execution paired with a strict-FP32 roof,
requires exactly three positive E2E samples per platform, and requires exactly
the six known platforms.  Its output is deliberately labelled
`empirical_path_matched_mfu`; it must not be relabelled as vendor-theoretical
MFU.

The current result can be reproduced from the retained input:

```bash
python3 tools/dense_gemm_peak/calculate_path_matched_mfu.py \
  --input tools/dense_gemm_peak/results/graphcast_path_matched_mfu_20260901.input.json \
  --output /tmp/graphcast_path_matched_mfu.json
```

The H100 run uses TF32 plus ordinary `torch.compile`, so its denominator is the
measured fast-FP32/TF32 roof.  The five domestic FlagGems runs use strict FP32
and therefore use each platform's strict-FP32 roof.  A comparable strict-FP32
H100 value requires a new strict-FP32 40-lead run; it cannot be inferred from
the current TF32 timing.

The input and output JSON preserve the timing and roof evidence paths and their
creation-time SHA-256 hashes.  Raw profiler data remains outside Git because it
is large and platform-generated.
