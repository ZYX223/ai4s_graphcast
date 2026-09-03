# GraphCast — AI4S Multi-Platform Weather Prediction

PyTorch-based implementation of Google DeepMind's GraphCast global weather forecasting
model (0.25 deg resolution). The shared model and backend routing cover NVIDIA H100,
Huawei Ascend 910C, Hygon BW3000, T-Head PPU-ZW810E, Moore Threads S5000, and
MetaX C550.

This directory is the canonical GraphCast workspace for training and inference. All
workflows use the same implementation under `src/graphcast_compat`; platform and
FlagGems choices are configuration, not separate model copies.

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
