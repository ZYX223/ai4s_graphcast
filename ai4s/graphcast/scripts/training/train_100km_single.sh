#!/usr/bin/env bash
# Direct entry for Google Small 1-degree / ~100km PyTorch-compatible training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash run_100km_train.sh

Optional environment variables:
  ENV_SH=path/to/env.sh                  # optional conda/module init script
  CONDA_ENV=name_or_prefix               # optional conda environment
  PYTHON_BIN=python                      # python executable
  DEVICE=auto                            # auto, cuda, cuda:0, musa, musa:0, npu, npu:0, cpu
  GRAPHCAST_BACKEND=auto                 # native/flagos profiles for supported platforms
  STEPS=344                              # training steps; 344 matches bundled JAX baseline coverage
  LR=0.000001                            # SGD learning rate
  COMPUTE_GRAD=1                         # set 0 for forward/loss-only run
  AMP=0                                  # keep 0 for FP32/no-bf16 diagnostics
  AMP_DTYPE=bfloat16                     # bfloat16 or float16 when AMP=1
  TRAIN_SCOPE=all                        # all or decoder-grid
  ACTIVATION_CHECKPOINTING=0             # set 1 to checkpoint graph processor steps
  MESH2GRID_EDGE_CHUNK_SIZE=             # optional training-time chunk-size override
  GRAPHCAST_FLAGOS_MODE=off              # off or flaggems
  GRAPHCAST_FLAGGEMS_CONFIG=configs/flaggems/graphcast_training.yaml
  GRAPHCAST_FLAGGEMS_OPS=                 # optional comma-separated override
  GRAPHCAST_FLAGGEMS_VARIANT=installed    # installed, official, or internal
  FLAGGEMS_RUNTIME_DEST=third_party/flaggems-runtime
  GRAPHCAST_FLAGGEMS_SOURCE_ROOT=        # optional FlagGems source root
  GRAPHCAST_STRICT_FLAGOS=0              # fail if FlagOS setup fails
  RUN_NAME=google_small_1deg_train_344step_fp32
  RUN_DIR=outputs/training/runs/100km/$RUN_NAME
  LOSS_MEAN_STEPS=344                    # loss summary window
  SAVE_CHECKPOINT=                       # optional checkpoint path; empty disables checkpoint save
  SKIP_CHECKPOINT_VALIDATION=1           # only used when SAVE_CHECKPOINT is set
  SMOKE=0                                # crop case for entry self-check only

The default path uses the bundled Google Small 1-degree / ~100km NetCDF case:
  data/cases/operational_100km/{inputs.nc,targets.nc,forcings.nc}
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 2
fi

if [[ -n "${ENV_SH:-}" ]]; then
  source "${ENV_SH}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV was set, but conda is unavailable. Set ENV_SH or use PYTHON_BIN." >&2
    exit 2
  fi
  conda activate "${CONDA_ENV}"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

if [[ -n "${CONDA_PREFIX_OVERRIDE:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${GRAPHCAST_FLAGGEMS_SOURCE_ROOT:-}" ]]; then
  if [[ ! -f "${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}/src/flag_gems/__init__.py" ]]; then
    echo "invalid GRAPHCAST_FLAGGEMS_SOURCE_ROOT: ${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}" >&2
    exit 2
  fi
  export PYTHONPATH="${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}/src:${PYTHONPATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
GRAPHCAST_BACKEND="${GRAPHCAST_BACKEND:-auto}"
STEPS="${STEPS:-344}"
LR="${LR:-0.000001}"
COMPUTE_GRAD="${COMPUTE_GRAD:-1}"
AMP="${AMP:-0}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
TRAIN_SCOPE="${TRAIN_SCOPE:-all}"
SMOKE="${SMOKE:-0}"
LOSS_MEAN_STEPS="${LOSS_MEAN_STEPS:-344}"
SKIP_CHECKPOINT_VALIDATION="${SKIP_CHECKPOINT_VALIDATION:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
MESH2GRID_EDGE_CHUNK_SIZE="${MESH2GRID_EDGE_CHUNK_SIZE:-}"
GRAPHCAST_FLAGOS_MODE="${GRAPHCAST_FLAGOS_MODE:-off}"
GRAPHCAST_FLAGGEMS_CONFIG="${GRAPHCAST_FLAGGEMS_CONFIG:-${SCRIPT_DIR}/configs/flaggems/graphcast_training.yaml}"
GRAPHCAST_FLAGGEMS_OPS="${GRAPHCAST_FLAGGEMS_OPS:-}"
GRAPHCAST_FLAGGEMS_VARIANT="${GRAPHCAST_FLAGGEMS_VARIANT:-installed}"
FLAGGEMS_RUNTIME_DEST="${FLAGGEMS_RUNTIME_DEST:-${SCRIPT_DIR}/third_party/flaggems-runtime}"
GRAPHCAST_STRICT_FLAGOS="${GRAPHCAST_STRICT_FLAGOS:-0}"

export GRAPHCAST_BACKEND GRAPHCAST_FLAGOS_MODE GRAPHCAST_FLAGGEMS_CONFIG GRAPHCAST_FLAGGEMS_OPS
export GRAPHCAST_FLAGGEMS_VARIANT FLAGGEMS_RUNTIME_DEST
export GRAPHCAST_FLAGGEMS_SOURCE_ROOT GRAPHCAST_STRICT_FLAGOS

CONFIG="${CONFIG:-configs/training/google_graphcast_100km.yaml}"
WEIGHTS="${WEIGHTS:-data/checkpoints/google_graphcast_small_compat_state.pt}"
CASE_DIR="${CASE_DIR:-data/cases/operational_100km}"
INPUTS="${INPUTS:-${CASE_DIR}/inputs.nc}"
TARGETS="${TARGETS:-${CASE_DIR}/targets.nc}"
FORCINGS="${FORCINGS:-${CASE_DIR}/forcings.nc}"
STATS_DIR="${STATS_DIR:-data/stats}"
RUN_NAME="${RUN_NAME:-operational_100km_train_${STEPS}step_fp32}"
RUN_DIR="${RUN_DIR:-outputs/training/runs/100km/${RUN_NAME}}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-}"
VALIDATION_PREDICTION_OUTPUT="${VALIDATION_PREDICTION_OUTPUT:-${RUN_DIR}/validation_prediction.nc}"

for required_file in "${CONFIG}" "${WEIGHTS}" "${INPUTS}" "${TARGETS}" "${FORCINGS}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required file: ${required_file}" >&2
    exit 2
  fi
done

mkdir -p outputs/training/runs/100km outputs/logs "${RUN_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/training/lib/train_single.py
  --config "${CONFIG}"
  --weights "${WEIGHTS}"
  --inputs "${INPUTS}"
  --targets "${TARGETS}"
  --forcings "${FORCINGS}"
  --stats-dir "${STATS_DIR}"
  --device "${DEVICE}"
  --steps "${STEPS}"
  --lr "${LR}"
  --amp-dtype "${AMP_DTYPE}"
  --train-scope "${TRAIN_SCOPE}"
  --run-dir "${RUN_DIR}"
)

if [[ -n "${SAVE_CHECKPOINT}" ]]; then
  mkdir -p "$(dirname "${SAVE_CHECKPOINT}")"
  cmd+=(--save-checkpoint "${SAVE_CHECKPOINT}")
  cmd+=(--validation-prediction-output "${VALIDATION_PREDICTION_OUTPUT}")
fi
if [[ "${COMPUTE_GRAD}" == "1" ]]; then
  cmd+=(--compute-grad)
fi
if [[ "${AMP}" == "1" ]]; then
  cmd+=(--amp)
fi
if [[ "${SKIP_CHECKPOINT_VALIDATION}" == "1" ]]; then
  cmd+=(--skip-checkpoint-validation)
fi
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--activation-checkpointing)
fi
if [[ -n "${MESH2GRID_EDGE_CHUNK_SIZE}" ]]; then
  cmd+=(--mesh2grid-edge-chunk-size "${MESH2GRID_EDGE_CHUNK_SIZE}")
fi
if [[ "${SMOKE}" == "1" ]]; then
  cmd+=(--lat-count 3 --lon-count 4 --mesh-size 1)
fi

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'backend env: GRAPHCAST_BACKEND=%q GRAPHCAST_FLAGOS_MODE=%q GRAPHCAST_FLAGGEMS_CONFIG=%q GRAPHCAST_FLAGGEMS_OPS=%q GRAPHCAST_FLAGGEMS_VARIANT=%q GRAPHCAST_STRICT_FLAGOS=%q\n' \
  "${GRAPHCAST_BACKEND}" \
  "${GRAPHCAST_FLAGOS_MODE}" \
  "${GRAPHCAST_FLAGGEMS_CONFIG}" \
  "${GRAPHCAST_FLAGGEMS_OPS}" \
  "${GRAPHCAST_FLAGGEMS_VARIANT}" \
  "${GRAPHCAST_STRICT_FLAGOS}"

"${cmd[@]}"

"${PYTHON_BIN}" scripts/training/lib/summarize.py \
  --metrics "${RUN_DIR}/metrics.jsonl" \
  --first-n "${LOSS_MEAN_STEPS}" \
  --json-output "${RUN_DIR}/loss_summary.json" \
  --report "${RUN_DIR}/loss_summary.md"

echo "run_dir=${RUN_DIR}"
echo "metrics=${RUN_DIR}/metrics.jsonl"
echo "loss_summary=${RUN_DIR}/loss_summary.md"
if [[ -n "${SAVE_CHECKPOINT}" ]]; then
  echo "checkpoint=${SAVE_CHECKPOINT}"
fi
