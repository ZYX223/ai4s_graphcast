# Frozen H100 Native training baseline

This publication preserves the exact source used by the latest three complete
H100 runs (median **9.013270% MFU**) and the historical three runs (median
**10.012132% MFU**). It does not claim that the historical performance has been
reproduced in the latest host conditions. All six runs share the same137-file
source manifest; neither a moving branch nor a dirty worktree identifies a run.

## What is included

- Sequential chunk assembly: split input once and concatenate ordered output
  chunks instead of repeated slice/writeback backward chains.
- Leading-node-axis gather shared across training backends, honoring the existing
  explicit `index_select` override. Bounded grouped gather is opt-in; this recipe
  uses `--edge-gather-group-size 4`. MLP chunk boundaries/reduction order remain.
- `--resident-training-inputs enabled`: upload immutable raw data once **per
  microbatch**, outside checkpoint replay; keep static graph buffers on the sole
  model device. Upload remains inside the measured update. MP=1 is required.
- Both execution policies are recorded and checked during exact checkpoint resume.
- Gradient/value/checkpoint/CLI regression tests accompanying the frozen source.

Model parameterization, weighted loss, full autoregressive feedback gradients,
training schedule, useful-FLOP definition and inference implementation are not
changed. Grouped gather defaults to1 and resident inputs todisabled; existing
platform/chunk/checkpoint defaults are not replaced with the benchmark overrides.

**Not included:** later CPU data-buffer reuse experiments, whole-global-batch GPU
staging (`resident-training-inputs=update`), compile, BF16 or new FlagGems patches.
Shared backend code is not evidence that every device/FlagGems combination has
passed this full38-update configuration; revalidate on each target platform.

## Exact source identity

`training-native-baseline-20261012.source-sha256.json` is the original per-file
manifest entries, relative to the repository root. The archived manifest file
SHA256 is `2b61fa4464e9d468bb7a53d911938ca4709f424a17f18c6df9b47871ba8278f1`.
The committed JSON adds a final newline; its byte SHA256 is
`17201e336228e7f39e7a714900cd421c451d299a8b03bf344b5a305fd4ea4d2c`.
Parsed manifest entries and all137 referenced file bytes are unchanged.
It covers137 Python/JSON/YAML files, including tests and configurations; it is not
a digest of the entire repository, software environment, data or mesh NPY assets.
Those assets and runtime identities are separately recorded in the run archives.

From repository root, verify the tested files without importing Torch:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
manifest = Path('ai4s/graphcast/docs/training-native-baseline-20261012.source-sha256.json')
hashes = json.loads(manifest.read_text())
bad = [name for name, digest in hashes.items()
       if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
if bad:
    raise SystemExit('Source mismatch: ' + ', '.join(bad))
print(f'PASS: {len(hashes)} tested source/config/test files match')
PY
```

## Latest measured results

| Run | Complete updates | Update E2E (s) | Effective TFLOP/s/GPU | MFU |
| --- | ---: | ---: | ---: | ---: |
| off-r1 | 38 | 660.386422 | 44.571232 | 9.004289% |
| off-r2 | 38 | 656.841734 | 44.811763 | 9.052881% |
| off-r3 | 38 | 659.728441 | 44.615685 | 9.013270% |

Median E2E659.728441s; sample CV0.286123% in one shared-host window.
Historical E2E593.910498 /595.202802 /591.062961s used the same source and model
recipe, but a different physical H100 (GPU4 versus current GPU6), host time window,
and NumPy hugepage policy (`1` historically, `0` for these latest runs).
Adjacent unchanged-policy reproduction had already recovered to9.898375% in one
run. These sequential observations do not establish a causal hugepage benefit.

All latest runs completed with finite loss/gradients and matching source hashes;
final checkpoint bytes/internal hashes and finite state were checked. Scalar-loss
relative-L1 differences against the historical reference were0.012543%,0.035626%,
0.055711%. This short-training consistency diagnostic is not long-run convergence
or trained-checkpoint weather-skill validation. Completed-artifact review retained
shared-host/order limitations and the missing separate live-env receipt for run3.

## Fixed workload and MFU

- Native Torch eager; FP32 parameters/activations/loss with TF32 matmul enabled.
  No BF16, compile or FlagGems. One H10080GB HBM3, MP=1/DP=1.
- Global batch2, microbatch1, accumulation2; seed27; identical pretrained model
  initialization and fresh optimizer each repeat. No resumed prefix.
- Six R1 warmup updates +ten R1 +two updates for each R2…R12:38 updates,
  sum(R)=170,340 sample-leads. Warmup updates are included in timing/work.
- Mesh step checkpoints; reentrant rollout checkpoint, including R1;12 segments.
- G2M stage checkpoint enabled, M2G stage checkpoint disabled; concatenated
  Mesh edge projection. Edge chunk131072, Mesh edge524288, grid node16384,
  loss grid65536, gather group4, resident microbatch inputs enabled.
- CPU prefetch depth1, split cache16GiB; no new CPU-buffer reuse policy.
- Existing container `nvcr.io/nvidia/pytorch:25.11-py3`, Python3.12.3,
  Torch2.10.0a0+b558c986e8.nv25.11, CUDA13.0, NumPy1.26.4, SciPy1.16.3.
  Image ID `sha256:06faed719d1bbd31d7053c96d0c94fce4f2b5f92fd95f07d60ec323c9744f118`.
- CPU/memory NUMA1 on the measured host; OMP/OpenBLAS8; expandable segments.
  `NUMPY_MADVISE_HUGEPAGE=0` is set before NumPy import, not a system THP change.

Useful training work is **29434.23636529152 TFLOP**. The agreed H100 denominator
is495 TFLOP/s (dense theoretical TF32), not the strict FP32 or BF16 peak:

`MFU (%) = 100 * 29434.23636529152 / sum(all38_update_E2E_seconds) / 495`

E2E is the sum of update intervals, including exposed CPU preparation/prefetch
wait, H2D, forward/backward, checkpoint recomputation, optimizer and synchronization.
It excludes initialization/data preloading, checkpoint writes and management
between those intervals. Do not call it full process wall time. Do not subtract
data stalls, discard slow updates, double-count overlapped producer work, or add
checkpoint recomputation to the useful-FLOP numerator.

## Reproduction command

First provision the same ERA5 corpus/stats/initial weights and compatible runtime;
data/weights are not distributed in this code commit. Select a free physical GPU
in `GRAPHCAST_GPU`, verify the device-local NUMA topology, and use a fresh output
directory for every repeat. The following uses the measured host's NUMA1.
From `ai4s/graphcast`:

```bash
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
numactl --cpunodebind=1 --membind=1 python scripts/training/run_training.py \
  --platform h100 --device-type cuda --backend native \
  --physical-devices "${GRAPHCAST_GPU:?set an available H100 index or UUID}" \
  --config configs/inference/google_graphcast_operational_25km.yaml \
  --corpus /path/to/era5_2017-01-01_07_train.json --stats /path/to/stats \
  --pretrained-checkpoint /path/to/initial-weights.pt \
  --output-dir /path/to/fresh-run \
  --mp 1 --dp 1 --global-batch-size 2 --micro-batch-size-per-rank 1 --seed 27 \
  --precision fp32 --tensor-core enabled \
  --warmup-updates 6 --one-step-updates 10 --curriculum-updates 22 \
  --max-updates 38 --checkpoint-every 3 \
  --gradient-sync manual --mesh-split-step 16 --mesh-checkpoint-granularity step \
  --rollout-checkpoint-implementation reentrant --single-step-rollout-checkpoint enabled \
  --mesh-edge-input-projection concatenated \
  --edge-chunk-size 131072 --mesh-edge-chunk-size 524288 \
  --grid-node-chunk-size 16384 --loss-grid-chunk-size 65536 \
  --grid2mesh-stage-device primary --grid2mesh-node-stage-device stage \
  --grid2mesh-stage-checkpoint enabled --mesh2grid-stage-device output \
  --mesh2grid-node-stage-device primary --mesh2grid-decoder-stage-device primary \
  --mesh2grid-stage-checkpoint disabled --edge-gather-group-size 4 \
  --resident-training-inputs enabled --split-cache-max-gib 16 --data-prefetch-depth 1 \
  --peak-tflops-per-device 0 --peak-tflops-source unavailable
```

The archived runs used the source engine directly; this public adapter translates
to the same engine workload/options. Add `--dry-run` to inspect without training.
Historical raw runs left the in-process peak unspecified; the MFU values above
were recomputed from completed metrics using495. Do not substitute a compute-only
or arithmetic mean of per-update MFUs for the aggregate E2E formula.

Original evidence archive:
`training-h100-baseline-repro-20261011-7c8BZ0/numpy-hugepage-off-20261011/`.
It retains each contract, source pre/post hashes, metrics/start records, final
checkpoint, process/thermal observations and completed audit. Full absolute paths
remain in that archive; logs, tensors and checkpoints are intentionally not added
to Git. The original frozen source and experiments remain unchanged.

## Publication checks

On2026-10-12, the clean publication worktree matched all137 manifest entries.
CPU regression tests using local Torch2.8.0/NumPy2.2.6 passed421 tests and102
subtests (CUDA hidden). The suite emitted one NumPy binary-compatibility runtime
warning during the data integration test; it did not fail. This is a source
publication check, not a new GPU performance run or a replacement for the
archived Torch2.10/CUDA13 environment. The suite included training baseline,
chunk/checkpoint/gather/resident-input, full-BPTT, gradient accumulation, prefetch
and exact-resume tests.
