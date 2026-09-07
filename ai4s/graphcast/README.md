# GraphCast — AI4S Multi-Platform Weather Prediction

PyTorch-based implementation of Google DeepMind's GraphCast global weather forecasting
model (0.25 deg resolution). The shared model and backend routing cover NVIDIA H100,
Huawei Ascend 910C, Hygon BW3000, T-Head PPU-ZW810E, Moore Threads S5000, and
MetaX C550.

This directory is the canonical GraphCast workspace for training and inference. All
workflows use the same implementation under `src/graphcast_compat`; platform and
FlagGems choices are configuration, not separate model copies.

The formal TF32/HF32 autoregressive training entry is
`scripts/training/run_training.py` (or `train_official.sh`). It keeps FP32
storage, uses MP=4 per DP process, and supports weighted accumulation,
rollout schedules and exact optimizer-boundary resume. Start with
`python scripts/training/run_training.py --dry-run`; see the
[training guide](scripts/training/README.md) for corpus, DP=2 and restore examples.
The current lightweight regression gate is `scripts/training/diagnose_training_loss.py`:
it compares complete, recipe-matched optimizer-step intervals using the project
team's relative loss metric; it is not trained-checkpoint weather-skill acceptance.
Current acceptance is CPU integration and CLI validation only; full-model
accelerator memory, training accuracy, FlagGems backward coverage and MFU remain
unverified. The existing quickstart/short training probes remain available.

## Quick Start

```bash
# One command — auto-detects platform, starts Docker, runs inference
./quickstart

# Or explicitly:
./quickstart infer          # inference (default, ~30 sec)
./quickstart train          # 4-GPU training, 3 steps
./quickstart all            # train then infer
```

The first run downloads data (~1 GB) from GitHub Releases automatically if `data/` is missing.
If Docker is available, quickstart auto-detects your GPU/NPU platform and starts a container.

## Canonical Workflows

| Workflow | Entry point | Purpose |
|----------|-------------|---------|
| Quick start | `./quickstart infer` or `./quickstart train` | Auto-detect platform and container |
| Inference | `bash scripts/infer.sh` | Operational 25 km inference |
| Training | `bash scripts/train.sh` | Operational 25 km model-parallel training |

Future GraphCast measurements should start from these entry points. Temporary model
copies and machine-local runner forks are not supported sources of record.

## quickstart Usage

```
./quickstart [MODE] [OPTIONS...]
```

### Modes

| Command | Action | Docker | Typical time |
|---------|--------|--------|-------------|
| `./quickstart` | **Inference** (default) | Auto | ~30 sec |
| `./quickstart infer` | Inference only | Auto | ~30 sec |
| `./quickstart train` | Training (4-GPU, 3 steps) | Auto | ~1-10 min |
| `./quickstart all` | Train then infer | Auto | ~2-15 min |
| `./quickstart container-only` | Start container, nothing else | Auto | ~5 sec |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--rm` | off | Remove Docker container after execution |
| `--image IMAGE` | platform default | Custom Docker image |
| `--devices a,b,c,d` | auto-detect | Comma-separated GPU/NPU IDs (e.g. `0,1,2,3`) |
| `--steps N` | 3 | Training steps |
| `--flagos-mode MODE` | `off` | `flaggems` to enable FlagGems |
| `--flaggems-ops OPS` | platform profile | Optional comma-separated override |
| `--flaggems-config PATH` | inference/training profile | Operator profile override |
| `--flaggems-variant NAME` | `installed` | `installed`, `official`, or `internal` |
| `--flaggems-src PATH` | - | Explicit FlagGems source root override |
| `--data-source PATH` | — | Path to graphcast-data dir (skip download) |

### Auto-Detection

quickstart probes for `nvidia-smi`, `npu-smi`, `rocm-smi`, or PPU `nvidia-smi`
to identify the four platforms listed below, then selects 4 free devices (lowest IDs
with < 5 GB memory). S5000 and C550 currently use the canonical manual entry points
inside their prepared containers; their model and FlagGems routing remain in this same
repository.

| Platform | Image | Container name |
|----------|-------|---------------|
| NVIDIA H100 | chenwei/graphcast_flaggems5.0.2 | `qs_graphcast_nvidia` |
| Ascend 910C | chenwei/graphcast_flaggems5.3.0rc2 | `qs_graphcast_ascend` |
| Hygon BW3000 | chenwei/graphcast_flaggems5.0.2 | `qs_graphcast_hygon` |
| PPU-ZW810E | chenwei/graphcast_flaggems5.3.0rc2_verified_5op | `qs_graphcast_ppu` |

### Examples

```bash
# Quick environment check
./quickstart

# Inference only, cleanup container after
./quickstart --rm infer

# Just start the container, don't run anything
./quickstart container-only

# Training with specific GPUs
./quickstart train --devices 4,5,6,7 --steps 10

# Enable the platform's validated FlagGems profile
./quickstart train --flagos-mode flaggems

# Compare matched upstream and internal optimized runtimes
bash tools/prepare_flaggems_runtime.sh
./quickstart infer --flagos-mode flaggems --flaggems-variant official
./quickstart infer --flagos-mode flaggems --flaggems-variant internal

# Explicit operators remain available as a debugging override
./quickstart infer --flagos-mode flaggems --flaggems-ops "index,addmm"
```

## Supported Platforms

| Platform | Device | Backend | FlagGems Ops | Status |
|----------|--------|---------|-------------|--------|
| NVIDIA H100 | `cuda` | `nvidia` | addmm, cat, native_layer_norm, index_add_, silu, index, add | ✓ |
| Huawei Ascend 910C | `npu` | `ascend` | addmm, silu, native_layer_norm, cat, index, index_add_, add | ✓ |
| Hygon BW3000 | `cuda` (HIP) | `hygon` | addmm, silu, native_layer_norm, cat, index, index_add_, add | ✓ |
| T-Head PPU-ZW810E | `cuda` (HGGC) | `ptg` | cat, native_layer_norm, index_add_, silu, index, add | ✓ (6/7) |
| Moore Threads S5000 | `musa` | `mthreads` | addmm, silu, native_layer_norm, cat, index, index_add_, add | ✓ |
| MetaX C550 | `cuda` (MACA) | `metax` | addmm, silu, native_layer_norm, cat, index, index_add_, add | ✓ |

`GRAPHCAST_BACKEND` identifies the physical platform only. Whether FlagGems is
enabled is controlled separately by `GRAPHCAST_FLAGOS_MODE`. Legacy names such
as `native_hygon` and `flagos_hygon` remain accepted as aliases.

See `ai4s/docs/multi-platform-guide.md` for accelerator compliance requirements.

## Project Structure

```
ai4s/graphcast/
├── quickstart              # one-click data download + train/infer
├── README.md
├── requirements.txt
├── configs/
│   ├── training/           # 25km and 100km training YAML
│   ├── inference/          # operational 25km inference YAML
│   └── flaggems/           # separate inference/training operator profiles
├── src/graphcast_compat/   # model, backend, data loading, normalization
├── scripts/
│   ├── train.sh            # training entry → training/train_mp.sh
│   ├── infer.sh            # inference entry → inference/run_25km_inference.sh
│   ├── training/           # model parallel, DDP, single-GPU
│   └── inference/          # inference scripts
├── tools/                  # official/internal FlagGems runtime preparation
├── tests/                  # routing, runtime, and model integration tests
└── data/                   # cases, stats, checkpoints, wheelhouse (auto-downloaded)
```

## Manual Training

```bash
# Native (baseline B)
bash scripts/train.sh

# FlagGems (baseline C); operators come from the training profile
GRAPHCAST_BACKEND=ptg \
GRAPHCAST_FLAGOS_MODE=flaggems \
  bash scripts/train.sh
```

## Manual Inference

```bash
# Native PyTorch
bash scripts/infer.sh

# FlagGems with the inference profile
GRAPHCAST_FLAGOS_MODE=flaggems bash scripts/infer.sh
```

## Local Checks

Run the repository-level checks from the repository root:

```bash
python -m pytest ai4s/graphcast/tests
pre-commit run --all-files
```

Hardware performance results are produced on the target accelerator; local checks only
validate source organization, routing, and CPU-safe behavior.

## FlagGems Profiles and Source Selection

Inference and training use independent checked-in profiles:

- `configs/flaggems/graphcast_inference.yaml` controls inference operators;
- `configs/flaggems/graphcast_training.yaml` controls training-forward operators.

The inference profile starts from one platform-independent seven-operator request:
`addmm`, `cat`, `native_layer_norm`, `index_add_`, `silu`, `index`, and `add`.
The only checked-in correctness exclusion is PTG/810E `addmm`; the validated
production route therefore requires 7/7 on C550, BW3000, S5000 and 910C, and
6/7 on 810E with `addmm` left Native. Formal FlagGems runs record implementation
entry logs and refuse to publish unless every effective requested operator was
registered and observed. The training profile does not claim backward coverage.
Backward correctness and performance must be validated separately. A non-empty
`GRAPHCAST_FLAGGEMS_OPS` overrides the selected YAML profile.

The default 810E exclusion above preserves the previously accepted strict-FP32
route. The separately validated TF32 7/7 route requires the RNE-v2 AddMM patch
from `flagos-common:zyx/flaggems-op-prcheck/patches/flaggems/wip-addmm`,
the compatible v5.3.5 runtime, and an explicit seven-op override when invoking
the autoregressive Python entry point:
`GRAPHCAST_FLAGGEMS_OPS=addmm,cat,native_layer_norm,index_add_,silu,index,add`.
Do not infer 7/7 from enabling FlagGems alone; check the observed dispatch report.

### Formal autoregressive inference

Round-1 performance uses the device-resident autoregressive runner rather than
the single-step quickstart path. The default is 40 dependent 6-hour leads, but
`--steps` accepts any positive count when the forcing asset contains exactly
that many frames. Warm-up is synchronized before timing and only the final
complete output is copied to the host.

```bash
GRAPHCAST_BACKEND=metax \
GRAPHCAST_FLAGOS_MODE=flaggems \
GRAPHCAST_FLAGGEMS_CONFIG=configs/flaggems/graphcast_inference.yaml \
GRAPHCAST_FLAGGEMS_SOURCE_ROOT=/path/to/FlagGems \
GRAPHCAST_FLAGGEMS_RECORD_PATH=outputs/logs/flaggems-dispatch.log \
GRAPHCAST_STRICT_FLAGOS=1 \
python scripts/inference/run_operational_40step_inference.py \
  --platform c550 \
  --model-execution-profile domestic-v1 \
  --device cuda:0 \
  --config configs/inference/google_graphcast_operational_25km.yaml \
  --weights /assets/google_graphcast_operational_25km_compat_state.pt \
  --inputs /assets/inputs_official_forcings.nc \
  --rollout-forcings /assets/rollout_forcings_40step.nc \
  --target-template /assets/target_template.nc \
  --stats-dir /assets/stats \
  --output outputs/predictions/lead240.nc \
  --report outputs/reports/round1.json
```

Set `GRAPHCAST_FLAGOS_MODE=off` for the paired Native baseline. A run is not
performance evidence unless its JSON report records the requested step count,
finite output, expected source hashes, and `warmup_excluded_from_e2e=true`.

Use `--precision fp32` for strict FP32, `--precision tf32` for platform
TF32/HF32, or `--precision bf16` for BF16 autocast with FP32 state/feedback.
The same model-side profile remains active in all three modes. On S5000,
the TF32 request sets and checks both `torch.backends.musa.matmul.allow_tf32`
(when available) and `torch.backends.mudnn.allow_tf32`.
`domestic-v1` uses prevalidated static indices on S5000; only FlagGems enters
the operator's trusted-index scope. Native retains its ordinary index_add
implementation, with model-side static-buffer validation before/after rollout.

### Project fixed-baseline inference acceptance (5%)

`scripts/inference/diagnose_inference.py` is the project acceptance entry point
for FP32, TF32/HF32 and BF16. It ports the metric from the colleague handoff
`checkpoint_prediction_validation/diagnostic_package` (2026-06-10). It is **not**
a weather-skill assessment or an operator-level allclose test.

For each of U/V/P/T/Q/W and all 13 pressure levels, compute
`mean(abs(candidate-reference)) / mean(abs(reference)) * 100`, then take the
unweighted arithmetic mean of those 78 percentages. `P` means geopotential.
Both that overall mean and the geopotential 500 hPa percentage must be **strictly
less than 5%**, independently against **both** fixed references (JAX FP32 and
PyTorch GPU). The worst layer is reported, not a separate gate. No latitude
weights, standard-deviation normalization, pointwise MAPE, or averaging over
rollout steps is substituted. Percentages are compared before display rounding.

The supplied NPZ is an external ~550 MB asset, not a Git payload. Its checksum
and the original diagnostic's checksum are pinned in
`configs/inference/diagnostic_baseline_day10.json`. That manifest deliberately
starts as `unverified`: the original package lacks complete initialization and
input/checkpoint lineage. To produce formal acceptance, make a reviewed copy
with `provenance_status: verified` and fill:

- `provenance_evidence`: the reviewed source/conversion evidence;
- `initialization_time`: an ISO-8601 UTC timestamp without a timezone suffix;
- `lead_hours`: the actual NPZ reference lead (currently 240);
- `source.weights_sha256`, `inputs_sha256`, `rollout_forcings_sha256`,
  `target_template_sha256` and `source.normalization_stats`: the **approved
  equivalent PyTorch case's** identities, in the same shape as the runner's
  `report.json.source` (three normalization filenames mapped to `sha256`);
- `units`: the verified unit string for each of the six full variable names.

Do not simply bless the current candidate's source hashes: establish their
relationship to the reference-generation case first. Different encodings or
converted checkpoints need documented equivalence and an approved identity;
the evaluator never silently ignores a hash mismatch. An unverified reference
can produce trial metrics, but its status is `unverified_reference`, not passed.

Score existing runs offline, without allocating a device or rerunning inference:

For **newly rebuilt** references, use the [dual-reference workflow](tools/inference_reference/README.md).
It audits original/converted weights, freezes one case, runs pinned official JAX
strict FP32 and H100 Torch Native eager strict FP32 independently, then qualifies
their agreement before acceptance. New metadata v2 separates provenance from
numerical qualification: a traceable but `unqualified` reference cannot pass a
candidate. Initialization, steps, step hours and selected lead remain configurable.
The historical reference and its unknown-lineage status are not overwritten.

```bash
python scripts/inference/diagnose_inference.py \
  --baseline /assets/graphcast_operational_25km_day10_uvptqw_baselines.npz \
  --baseline-metadata /assets/reviewed-diagnostic-baseline.json \
  --run-dir /results/c550-native-tf32 \
  --run-dir /results/c550-flaggems-tf32 \
  --output-dir /results/c550-tf32-acceptance
```

Each run directory must contain `prediction.nc` and its hash-binding
`report.json`. Repeat `--run-dir` for other platforms/precisions. The evaluator
selects the NPZ's physical lead, validates initialization, coordinates, units,
case identities and finite output, and fails closed on missing/invalid data.
It does not assume `time_index=39` or take the last frame blindly. The current
reference is at 240 h (40 dependent 6 h leads); future baselines may use another
positive horizon without changing the evaluator. The updated runner labels its
final prediction using the actual forcing valid time and rollout horizon,
rather than retaining the reusable target template's time. Historical files
with stale template times must have their provenance/time semantics resolved;
they are not silently relabeled by this evaluator.

Outputs are `summary.json` (terminal, binding the detailed CSV by hash) and
`details.csv` (156 reference/variable/level records per valid run). The summary
includes the four percentages, case/source/runtime/precision metadata and
observed core-op counts expressed as `7/7`, `6/7`, etc. Missing observation
evidence is `unverified`; configured operator lists are not counted as hits.
Operator coverage and 5% acceptance remain separate dimensions. A requested
TF32/HF32 mode is not by itself proof of the matrix instructions actually used.

Exit codes: `0` all runs passed; `1` numerical acceptance/nonfinite failure;
`2` invalid inputs, unverified reference, or publication error. Output
directories must be new. The runner's own `status: passed` describes successful
finite execution, **not** this project acceptance; use the diagnostic summary
for the latter.

To score automatically after a new run, use the existing environment-driven
`tools/bf16_inference/run_bf16_case.sh` (legacy name retained, now accepting
`PRECISION=fp32|tf32|bf16`) with:

```bash
export GRAPHCAST_ACCEPTANCE_BASELINE=/assets/graphcast_operational_25km_day10_uvptqw_baselines.npz
export GRAPHCAST_ACCEPTANCE_METADATA=/assets/reviewed-diagnostic-baseline.json
```

The hook runs after inference, retaining the model `rc` and separate
`diagnostic_rc`, `diagnostic.log`, and `acceptance/` outputs. It propagates a
nonzero diagnostic status. Omitting both variables keeps the existing execution
workflow. Offline scoring does not enter E2E or MFU timing. Each platform's
Native and FlagGems runs retain the same model execution profile. The TF32 entry
sets the backend's math controls, including `torch.backends.musa.matmul` on
TorchMUSA versions exposing it, while keeping persistent tensors FP32.

### Auxiliary drift and weather-skill diagnostics

The existing normalized-drift tools remain for debugging; their historical
thresholds and BF16 campaign summaries do not replace the fixed-baseline gate.
Training loss validation is outside this inference change.

For forecast diagnostics, add `--trajectory-output PATH` to both the Native and
candidate runs. This deliberately copies every physical lead to the host and
marks the run as ineligible for performance/MFU reporting. The existing script
name is retained for compatibility; neither capture nor evaluation requires 40
leads. `--steps=N` accepts any positive `N`, and the runner rejects a forcing
asset whose number of time frames differs from `N`.

Evaluate the resulting trajectories against real targets and a matching
climatology with the WeatherBench-style spherical area weighting:

```bash
python scripts/inference/evaluate_forecast_skill.py \
  --truth /assets/targets.nc \
  --native outputs/native-trajectory.nc \
  --candidate outputs/flaggems-trajectory.nc \
  --climatology /assets/climatology.nc \
  --stddev /assets/stats/stddev_by_level.nc \
  --output-json outputs/forecast-diagnostics.json \
  --output-netcdf outputs/forecast-diagnostics.nc
```

The diagnostic result contains every available lead and pressure level, Native
and candidate MSE/RMSE/MAE/bias, climatology-relative ACC, their deltas, and the
optional Native/candidate normalized drift. It also reports raw implementation
parity (`max_abs`, `mean_abs`, RMSE, relative L2 and allclose); use
`--raw-atol`/`--raw-rtol` to record the intended numerical tolerance without
turning that tolerance into a weather-skill decision. Trajectory files remain
lazy and only one variable/lead/level record is loaded at a time, so memory does
not grow with the complete autoregressive horizon. It validates variables,
coordinates, units and valid times before comparison. Aggregate RMSE is
`sqrt(mean(MSE over initialization times))`; it is not a single-case maximum.
Use multiple initialization times before making a weather-skill claim. Treat
the JSON as the terminal artifact: it binds the metrics NetCDF by SHA-256. The
rollout report similarly binds the prediction and optional trajectory hashes.

The formal runner uses one required platform/profile pair. `h100/baseline`
keeps concatenated edge projection, the configured edge chunks, no full-grid
fast path and inline `index_add`. `c550`, `bw3000`, `s5000`, `910c` and `810e`
must use `domestic-v1`, which selects factorized edge projection, 131072-edge
grid2mesh and mesh2grid chunks, the full-grid mesh2grid fast path and inline
`index_add`. The mapping is identical for FP32 and BF16, and a mismatched pair
is rejected before model or device initialization. Native and FlagGems A/B runs
on one domestic platform use the same profile.

The established inference gather route remains
`GRAPHCAST_GATHER_MODE=advanced_index`; the model profile does not switch it to
`index_select` or replace FlagGems kernels. Retired experiments and low-level
model-option overrides (encoder/node factorization, aggregation-buffer reuse,
static/opaque operator boundaries and graph capture) are intentionally not
accepted by the formal runner. The report records the profile name, platform,
and every expanded model setting.

By default FlagGems mode imports the package installed in the environment. Run
`bash tools/prepare_flaggems_runtime.sh` to fetch the active internal manifest
and prepare matched `official` and `internal` source trees. Select one with
`GRAPHCAST_FLAGGEMS_VARIANT`; the resolved commits are recorded in
`third_party/flaggems-runtime/PROVENANCE.env`.

## Data

quickstart auto-downloads data on first run from:
```
https://github.com/yuk1n4/ai4s_graphcast/releases/download/v1.0-data-25km/graphcast_25km_data.tar.gz
```

Or download manually:
```bash
curl -LO https://github.com/yuk1n4/ai4s_graphcast/releases/download/v1.0-data-25km/graphcast_25km_data.tar.gz
tar -xzf graphcast_25km_data.tar.gz
```

Archive contents (~1 GB):
```
data/
├── cases/operational_25km/   # inputs.nc, targets.nc, forcings.nc
├── stats/                    # normalization statistics
├── checkpoints/              # pretrained weights (137 MB)
└── wheelhouse/               # offline Python packages (Ascend + PPU)
```

## License

Apache 2.0. See repository root `LICENSE`.
