# Six-platform dense GEMM measurement protocol

## Question

Determine whether the values in the supplied accelerator table are credible as
precision-specific dense matrix throughput.  The experiment estimates an
achievable single-device GEMM roof in the installed application runtime.  It
does not redefine or reverse-engineer a vendor theoretical peak.

## Metrics and labels

- Operation: square GEMM, `C = A @ B`.
- FLOP count: `2 * n^3` per call.
- Primary size: `n = 24576`.
- Diagnostic sizes: `8192`, `12288`, and `16384`.
- Primary estimate: median throughput over all accepted steady samples at the
  primary size, with a deterministic 95% bootstrap confidence interval.
- Secondary estimates: arithmetic mean, standard deviation, median absolute
  deviation, coefficient of variation, and per-sample values.
- `vendor theoretical peak`, `vendor-tool measurement`, and `runtime GEMM
  measurement` remain separate columns.

If the primary size cannot allocate, `n = 16384` is the only predeclared
fallback and the result is labelled `fallback_size`, not silently mixed with
the primary result.

## Precision modes

Each mode runs in a fresh process.

1. `strict_fp32`: FP32 inputs/output; fast FP32 matrix modes disabled and read
   back where the runtime exposes a control.
2. `fast_fp32`: FP32 inputs/output; NVIDIA TF32, Ascend HF32, or the vendor's
   documented equivalent explicitly enabled and read back.  If the runtime has
   no auditable control, the mode is unavailable.
3. `bf16`: BF16 inputs/output with the backend's native accumulation contract
   recorded.  No claim about accumulation hardware is made without profiler or
   vendor-tool evidence.

An enabled flag is request evidence, not physical-path evidence.  A result is
labelled as TF32/HF32/Cube/Tensor-Core only when a vendor tool, profiler trace,
or hardware counter confirms that path.

## Experimental units and order control

- One idle physical accelerator/package per platform.
- Six counterbalanced blocks use all six permutations of the three modes:
  `SFB`, `SBF`, `FSB`, `FBS`, `BSF`, `BFS` where `S=strict_fp32`,
  `F=fast_fp32`, and `B=bf16`.
- Every mode in every block uses a new Python process and a new output file.
- Size order alternates ascending and descending by block.
- No performance profiler runs in a timing process.
- A separate profile run uses the same size and precision after timing is
  complete.

This design prevents a mode from always benefiting from a cold device or
always suffering from the hottest position.

## Timing

- Five excluded warm-up calls per size.
- Twenty steady calls per size.
- Explicit device synchronization immediately before and after every timed
  call.
- Host monotonic wall time is the common cross-platform timer.
- Inputs and output are allocated before warm-up.
- Allocation, random initialization, precision setup, correctness checks,
  compilation, and profiling are outside the measured interval.

## Correctness and execution checks

- Full timed output must be finite.
- A deterministic `256 x 256` case is compared with a CPU FP64 reference.
- The report records max absolute error, RMSE, normalized RMSE, and finiteness.
- Predeclared normalized-RMSE gates:
  - strict FP32: `<= 1e-5`
  - fast FP32: `<= 2e-3`
  - BF16: `<= 2e-2`
- Device identity, visible-device count, runtime versions, input/output dtype,
  requested precision, and read-back precision controls are recorded.

## Environmental controls

- Refuse to start if another compute process occupies the selected device.
  Context-only handles are permitted only when their aggregate allocation is
  at most 64 MiB, device utilization remains zero before the run, and the
  exact process list is preserved.  This predeclared exception applies to the
  S5000 host whose service processes reserve tiny contexts on otherwise idle
  cards; any growth, non-zero background utilization, or timing instability
  invalidates the block.
- Record device status before and after each block: memory use, utilization,
  temperature, power, and clocks when exposed.
- Do not change clocks, power limits, or cooling policy.
- A block is invalid if it overlaps another workload, reports thermal/power
  throttling, changes device identity, has a non-finite output, violates the
  correctness gate, or has an incomplete result set.

## Stability and interpretation

- Report sample count and missing/failed samples.
- Flag a mode as unstable when within-mode steady-time CV exceeds 3% or when
  independent block medians span more than 5% relative to their median.
- Diagnostic sizes must approach a plateau.  If throughput from `16384` to
  `24576` rises by more than 5%, report that the tested primary size has not
  established a roof.
- Measured throughput above a claimed theoretical peak by more than 3% is a
  precision/path/denominator audit failure, not a record result.
- A stable runtime GEMM value may be used as an empirical roofline denominator
  only under that label.  It must not be called theoretical MFU.

## Platform cross-checks

- H100: PyTorch/cuBLAS timing; strict TF32 switch read-back; Nsight/NCU path
  evidence in a separate run.
- Ascend 910C: official `ascend-dmi` for `fp32`, `hf32`, and `bf16`, plus the
  common PyTorch experiment; `npu-smi` proves the package/topology used.
- BW3000 (legacy asset/config alias `bw1000`): common PyTorch experiment and
  existing formal JAX/BLAS campaign; rocBLAS/hipBLASLt trace and MMOP hardware
  counters for physical-path classification.  Reports use the physical model
  returned by `hy-smi --showproductname`; legacy aliases remain only where an
  existing path or serialized record cannot be renamed without invalidating
  its checksum.
- S5000: PyTorch MUSA experiment and MUSA profiler evidence.
- C550: PyTorch MetaX experiment and MetaX profiler/library evidence.
- 810E: PyTorch PPU experiment and PPU profiler evidence.

## Evidence

Every artifact is create-once and checksummed.  The final comparison retains
raw samples, the exact benchmark source hash, launch command, process return
code, runtime/device records, precision read-back, telemetry, and profiler or
vendor-tool evidence.  Failed attempts are retained and never reused as valid
measurements.
