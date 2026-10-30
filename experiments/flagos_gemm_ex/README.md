# FlagOS GEMM_EX RNE candidate — 2026-10-30

This is an isolated extension of the frozen `kernels/xf32_rne` baseline.
The baseline source, pretrained checkpoint and previous outputs stay unchanged.
No custom GEMM or BF16 is introduced; GEMM remains rocBLAS. The backend is
**FlagOS JAX (`fl_only`)**, not a direct FlagGems library integration.

## What changes

`build.py` adds `rocblas_gemm_ex` symbol routing alongside SGEMM. Both entry
points use the same frozen-weight fingerprint, stage precision mask, explicit
runtime enable, RNE scratch buffers, counters and math-mode restore logic.
GEMM_EX preserves C/D pointers and strides, alpha/beta, algorithm, solution
index and flags. The optional existing 79-layer producer-RNE policy is reused
unchanged. Runtime calls outside FP32 storage/compute fail closed.

The generated C++ and library hash are saved in `NATIVE_OUTPUT/build.json`.
Autotuning runs before explicit runtime enable and uses unmodified native
operands; measured execution must pass per-step coverage checks.

## Important distinction: requested mode vs actual arithmetic

The two affected Linear shapes have column-major M=512, K=4 and
N=1,618,745 / 327,660. In the tested DTK library, both SGEMM and GEMM_EX use
FP32 arithmetic at K=4 even when XF32 mode is selected. Adding RNE fixes the
missing operand rounding; it does not turn these native fallback kernels into
XF32 kernels. We retain that rocBLAS choice, as in the baseline.

`native_probe.py` uses one-hot B, so A@B exposes input truncation exactly.
It tests the two real K=4 shapes and K=512, with both default and explicit
solution selection. At K=512 the mode=1 result matches XF32 truncation exactly
and differs from FP32. `adapter_probe.py` verifies explicit RNE separately
with exact oracles, C!=D, beta, transpose, padding and mode restoration.
Never infer actual XF32 execution solely from a mode readback.

## Current machine and commands

Container: `flagos-jax-test`; DTK: `/public/software/compiler/dtk-26.04`.
Use the container default identity and prepend `/opt/hyhal/lib` to its
existing `LD_LIBRARY_PATH`. Physical card 8 was checked free before testing;
recheck availability before reproducing. Do not overwrite colleagues' files.

Current experiment workspace inside the container:
`/tmp/zyx-flagos-rne-gemm-ex-20261030/`.
Frozen dependency snapshot:
`/tmp/zyx-flagos-rne-inference-20261029/baseline/`.
Model assets:
`/work/graphcast-strict-inference-cftime-20261027/data/`.

Build into a **new** directory (the builder refuses to overwrite):

```bash
export PYTHONPATH=/tmp/zyx-flagos-rne-inference-20261029/baseline/src
export NATIVE_OUTPUT=/tmp/zyx-flagos-rne-gemm-ex-20261030/native-new
python build.py
```

`GRAPHCAST_EXPERIMENT` optionally selects the existing fusion policy folder;
the default is the frozen 2026-10-29 experiment. Model launches use the
`run-model.sh` retained with the raw evidence. Run labels must be new:

```bash
bash run-model.sh fl_only single model-single-new
# Only proceed after path/coverage checks pass.
bash run-model.sh fl_only bench model-40-new
```

The launcher currently references the tested `native-r1` artifact, so change
that path explicitly when testing a newly built library. The single run uses
HIPPROF with five warmups outside capture. The benchmark uses three rounds,
each five warmups then input reset and 40 independent autoregressive steps.
Compilation, loading, warmups and output writes are outside rollout timing.

## Evidence and comparison

Host evidence:
`/home/zhangyuxin/zyx/graphcast/outputs/flagos-rne-gemm-ex-20261030/`.
It retains the initial failing replay, isolated candidate, generated source,
build hashes, probes, model launchers, logs and subsequent run results.

Reference: gpu41 H100 TF32 without explicit RNE, 2026-10-29, fixed pretrained
weights/input/statistics. Compare +240h only, six upper-air variables × 13
levels, equal-weight mean of per-field relative MAE; H100 output supplies the
denominator. Z500 is separate. Stage limit is both ≤10%; internal regression
limit is both <5%. This measures implementation consistency, not weather truth.

GPU active union time, synchronized call wall time and rollout wall time are
separate. Single-step GPU timing cannot be presented as measured 40-step GPU
time. MFU uses 28.784948924416 TFLOPs / (card seconds × 170.3936 TFLOP/s), a
single-step GEMM-FLOPs approximation. No formal 60% relative-efficiency claim
is made without the agreed hardware normalization.

Code and raw materials remain local pending the requested final unified sync.
