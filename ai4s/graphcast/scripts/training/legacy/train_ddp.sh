#!/usr/bin/env bash
# Multi-GPU DDP training entry for GraphCast 25km / 100km.
# Uses torch.distributed.run with configurable distributed backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/training/legacy/train_ddp.sh    # 25km case (default)
  bash scripts/training/legacy/train_ddp.sh    # or run 100km via overriding vars

Optional environment variables:
  PYTHON_BIN=python                      # python executable
  CONDA_PREFIX_OVERRIDE=                 # conda env prefix for LD_LIBRARY_PATH
  DEVICE=cuda                            # cuda, npu, musa, cpu
  NPROC_PER_NODE=4                       # number of processes per node
  DISTRIBUTED_BACKEND=nccl               # nccl (CUDA), hccl (Ascend), or gloo (CPU fallback)
  STEPS=344                              # training steps
  LR=0.000001                            # SGD learning rate
  TRAIN_SCOPE=all                        # all or decoder-grid
  ACTIVATION_CHECKPOINTING=0             # set 1 to checkpoint graph processor steps
  MESH2GRID_EDGE_CHUNK_SIZE=             # optional training-time chunk-size override
  GRAPHCAST_BACKEND=auto                 # native/flagos backend profile
  GRAPHCAST_FLAGOS_MODE=off              # off or flaggems
  GRAPHCAST_FLAGGEMS_CONFIG=configs/flaggems/graphcast_training.yaml
  GRAPHCAST_FLAGGEMS_OPS=                 # optional comma-separated override
  GRAPHCAST_FLAGGEMS_VARIANT=installed    # installed, official, or internal
  FLAGGEMS_RUNTIME_DEST=third_party/flaggems-runtime
  GRAPHCAST_FLAGGEMS_SOURCE_ROOT=        # optional FlagGems source root
  GRAPHCAST_STRICT_FLAGOS=0              # fail if FlagOS setup fails
  CONFIG=configs/training/...            # model config
  WEIGHTS=data/checkpoints/...        # model weights
  CASE_DIR=cases/...                     # NetCDF case directory
  RUN_NAME=                              # output run name
  RUN_DIR=                               # output run directory
  LOSS_MEAN_STEPS=344                    # loss summary window
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
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEVICE="${DEVICE:-cuda}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
STEPS="${STEPS:-344}"
LR="${LR:-0.000001}"
TRAIN_SCOPE="${TRAIN_SCOPE:-all}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
MESH2GRID_EDGE_CHUNK_SIZE="${MESH2GRID_EDGE_CHUNK_SIZE:-}"
LOSS_MEAN_STEPS="${LOSS_MEAN_STEPS:-344}"
GRAPHCAST_BACKEND="${GRAPHCAST_BACKEND:-auto}"
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
RUN_NAME="${RUN_NAME:-graphcast_ddp_${NPROC_PER_NODE}gpu_${STEPS}step_fp32}"
RUN_DIR="${RUN_DIR:-outputs/training/runs/ddp/${RUN_NAME}}"

for required_file in "${CONFIG}" "${WEIGHTS}" "${INPUTS}" "${TARGETS}" "${FORCINGS}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing required file: ${required_file}" >&2
    exit 2
  fi
done

mkdir -p outputs/training/runs/ddp outputs/logs "${RUN_DIR}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node "${NPROC_PER_NODE}"
  scripts/training/legacy/lib/train_single.py
  --config "${CONFIG}"
  --weights "${WEIGHTS}"
  --inputs "${INPUTS}"
  --targets "${TARGETS}"
  --forcings "${FORCINGS}"
  --stats-dir "${STATS_DIR}"
  --device "${DEVICE}"
  --distributed-backend "${DISTRIBUTED_BACKEND}"
  --steps "${STEPS}"
  --lr "${LR}"
  --compute-grad
  --amp-dtype bfloat16
  --train-scope "${TRAIN_SCOPE}"
  --run-dir "${RUN_DIR}"
)

if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--activation-checkpointing)
fi
if [[ -n "${MESH2GRID_EDGE_CHUNK_SIZE}" ]]; then
  cmd+=(--mesh2grid-edge-chunk-size "${MESH2GRID_EDGE_CHUNK_SIZE}")
fi

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'backend env: GRAPHCAST_BACKEND=%q GRAPHCAST_FLAGOS_MODE=%q GRAPHCAST_FLAGGEMS_CONFIG=%q GRAPHCAST_FLAGGEMS_OPS=%q GRAPHCAST_FLAGGEMS_VARIANT=%q\n' \
  "${GRAPHCAST_BACKEND}" \
  "${GRAPHCAST_FLAGOS_MODE}" \
  "${GRAPHCAST_FLAGGEMS_CONFIG}" \
  "${GRAPHCAST_FLAGGEMS_OPS}" \
  "${GRAPHCAST_FLAGGEMS_VARIANT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"

"${PYTHON_BIN}" scripts/training/legacy/lib/summarize.py \
  --metrics "${RUN_DIR}/metrics.jsonl" \
  --first-n "${LOSS_MEAN_STEPS}" \
  --json-output "${RUN_DIR}/loss_summary.json" \
  --report "${RUN_DIR}/loss_summary.md"

echo "run_dir=${RUN_DIR}"
echo "metrics=${RUN_DIR}/metrics.jsonl"
echo "loss_summary=${RUN_DIR}/loss_summary.md"
