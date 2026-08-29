#!/usr/bin/env bash
# Direct entry for operational 25km PyTorch-compatible GraphCast inference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash run_25km_inference.sh <inputs.nc> <target_template.nc> <forcings.nc>

Optional environment variables:
  ENV_SH=path/to/env.sh                 # optional conda/module init script
  CONDA_ENV=name_or_prefix              # optional conda environment
  PYTHON_BIN=python                     # python executable
  DEVICE=auto                           # auto, cuda, cuda:0, musa, musa:0, npu, npu:0, cpu
  GRAPHCAST_BACKEND=auto                # physical platform: NVIDIA, Hygon, PTG, MUSA, MetaX, or Ascend
  GRAPHCAST_FLAGOS_MODE=off             # off, flaggems, flagtree
  GRAPHCAST_FLAGGEMS_CONFIG=configs/flaggems/graphcast_inference.yaml
  GRAPHCAST_FLAGGEMS_OPS=               # optional comma-separated override
  GRAPHCAST_FLAGGEMS_VARIANT=installed  # installed, official, or internal
  FLAGGEMS_RUNTIME_DEST=third_party/flaggems-runtime
  GRAPHCAST_FLAGGEMS_SOURCE_ROOT=/path/to/FlagGems
  GRAPHCAST_STRICT_FLAGOS=0             # fail if requested FlagOS layer cannot start
  GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL=0
  OUTPUT=outputs/predictions/pred.nc
  REPORT=outputs/reports/report.md

If no NetCDF inputs are provided, the script runs a tiny synthetic self-check.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
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

mkdir -p outputs/predictions outputs/reports outputs/logs

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
GRAPHCAST_BACKEND="${GRAPHCAST_BACKEND:-auto}"
GRAPHCAST_FLAGOS_MODE="${GRAPHCAST_FLAGOS_MODE:-off}"
GRAPHCAST_FLAGGEMS_CONFIG="${GRAPHCAST_FLAGGEMS_CONFIG:-${SCRIPT_DIR}/configs/flaggems/graphcast_inference.yaml}"
GRAPHCAST_FLAGGEMS_OPS="${GRAPHCAST_FLAGGEMS_OPS:-}"
GRAPHCAST_FLAGGEMS_VARIANT="${GRAPHCAST_FLAGGEMS_VARIANT:-installed}"
FLAGGEMS_RUNTIME_DEST="${FLAGGEMS_RUNTIME_DEST:-${SCRIPT_DIR}/third_party/flaggems-runtime}"
GRAPHCAST_FLAGGEMS_SOURCE_ROOT="${GRAPHCAST_FLAGGEMS_SOURCE_ROOT:-}"
GRAPHCAST_STRICT_FLAGOS="${GRAPHCAST_STRICT_FLAGOS:-0}"
GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL="${GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL:-0}"
if [[ -n "${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}" ]]; then
  if [[ ! -f "${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}/src/flag_gems/__init__.py" ]]; then
    echo "invalid GRAPHCAST_FLAGGEMS_SOURCE_ROOT: ${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}" >&2
    exit 2
  fi
  export PYTHONPATH="${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}/src:${PYTHONPATH:-}"
fi
export GRAPHCAST_BACKEND GRAPHCAST_FLAGOS_MODE GRAPHCAST_FLAGGEMS_CONFIG GRAPHCAST_FLAGGEMS_OPS
export GRAPHCAST_FLAGGEMS_VARIANT FLAGGEMS_RUNTIME_DEST
export GRAPHCAST_FLAGGEMS_SOURCE_ROOT GRAPHCAST_STRICT_FLAGOS
CONFIG="${CONFIG:-configs/inference/google_graphcast_operational_25km.yaml}"
WEIGHTS="${WEIGHTS:-data/checkpoints/google_graphcast_operational_25km_compat_state.pt}"
STATS_DIR="${STATS_DIR:-stats}"
OUTPUT="${OUTPUT:-outputs/predictions/operational_25km_prediction.nc}"
REPORT="${REPORT:-outputs/reports/operational_25km_inference.md}"

cmd=(
  "${PYTHON_BIN}" scripts/inference/run_google_small_inference.py
  --device "${DEVICE}"
  --config "${CONFIG}"
  --weights "${WEIGHTS}"
  --stats-dir "${STATS_DIR}"
  --output "${OUTPUT}"
  --report "${REPORT}"
)

if [[ "$#" -eq 3 ]]; then
  cmd+=(--inputs "$1" --target-template "$2" --forcings "$3")
elif [[ "$#" -eq 0 ]]; then
  cmd+=(--synthetic-grid tiny)
else
  usage >&2
  exit 2
fi

case "${GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL,,}" in
  1|true|yes|on) cmd+=(--fuse-post-layer-norm-residual) ;;
esac

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'backend env: GRAPHCAST_BACKEND=%q GRAPHCAST_FLAGOS_MODE=%q GRAPHCAST_FLAGGEMS_CONFIG=%q GRAPHCAST_FLAGGEMS_OPS=%q GRAPHCAST_FLAGGEMS_VARIANT=%q GRAPHCAST_FLAGGEMS_SOURCE_ROOT=%q GRAPHCAST_STRICT_FLAGOS=%q\n' \
  "${GRAPHCAST_BACKEND}" \
  "${GRAPHCAST_FLAGOS_MODE}" \
  "${GRAPHCAST_FLAGGEMS_CONFIG}" \
  "${GRAPHCAST_FLAGGEMS_OPS}" \
  "${GRAPHCAST_FLAGGEMS_VARIANT}" \
  "${GRAPHCAST_FLAGGEMS_SOURCE_ROOT}" \
  "${GRAPHCAST_STRICT_FLAGOS}"
printf 'fusion env: GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL=%q\n' \
  "${GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL}"
"${cmd[@]}"
