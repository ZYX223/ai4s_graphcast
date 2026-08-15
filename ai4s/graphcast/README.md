# GraphCast — AI4S Multi-Platform Weather Prediction

PyTorch-based implementation of Google DeepMind's GraphCast global weather forecasting
model (0.25 deg resolution). Trained and validated on NVIDIA H100, Huawei Ascend 910C,
Hygon BW1000, and T-Head PPU-ZW810E.

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
to identify the platform, then selects 4 free GPUs (lowest IDs with < 5 GB memory).
The correct Docker image and flags are chosen automatically.

| Platform | Image | Container name |
|----------|-------|---------------|
| NVIDIA H100 | chenwei/graphcast_flaggems5.0.2 | `qs_graphcast_nvidia` |
| Ascend 910C | chenwei/graphcast_flaggems5.3.0rc2 | `qs_graphcast_ascend` |
| Hygon BW1000 | chenwei/graphcast_flaggems5.0.2 | `qs_graphcast_hygon` |
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
| NVIDIA H100 | `cuda` | `nvidia` | installed profile defaults to silu, add | ✓ |
| Huawei Ascend 910C | `npu` | `ascend` | addmm, silu, layer_norm, cat, index, add | ✓ |
| Hygon BW1000 | `cuda` (HIP) | `hygon` | addmm, silu, layer_norm, index | ✓ |
| T-Head PPU-ZW810E | `cuda` (HGGC) | `ptg` | addmm, silu, layer_norm, index, add | ✓ |

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

## FlagGems Profiles and Source Selection

Inference and training use independent checked-in profiles:

- `configs/flaggems/graphcast_inference.yaml` controls inference operators;
- `configs/flaggems/graphcast_training.yaml` controls training-forward operators.

The training profile does not claim backward coverage. Backward correctness and
performance must be validated separately. A non-empty
`GRAPHCAST_FLAGGEMS_OPS` overrides the selected YAML profile.

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
