#!/usr/bin/env bash
set -Eeuo pipefail

: "${PROJECT_ROOT:?}"
: "${FLAGGEMS_ROOT:?}"
: "${ASSET_WEIGHTS:?}"
: "${ASSET_INPUTS:?}"
: "${ASSET_FORCINGS:?}"
: "${ASSET_TARGET:?}"
: "${ASSET_STATS:?}"
: "${PLATFORM:?}"
: "${MODEL_EXECUTION_PROFILE:?}"
: "${BACKEND:?}"
: "${DEVICE:?}"
: "${MODE:?}"
: "${PRECISION:?}"
: "${STEPS:?}"
: "${RUN_NAME:?}"
: "${OUTPUT_ROOT:?}"
: "${PYTHON_BIN:?}"

case "${MODE}" in
  native|flaggems) ;;
  *) echo "unsupported MODE=${MODE}" >&2; exit 2 ;;
esac
case "${PRECISION}" in
  fp32|tf32|bf16) ;;
  *) echo "unsupported PRECISION=${PRECISION}" >&2; exit 2 ;;
esac
if ! [[ "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${GRAPHCAST_ACCEPTANCE_BASELINE:-}${GRAPHCAST_ACCEPTANCE_METADATA:-}" ]]; then
  : "${GRAPHCAST_ACCEPTANCE_BASELINE:?}"
  : "${GRAPHCAST_ACCEPTANCE_METADATA:?}"
fi

RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_OUTPUT}" ]]; then
  echo "refusing to reuse output directory: ${RUN_OUTPUT}" >&2
  exit 2
fi
mkdir -p "${RUN_OUTPUT}"

export PYTHONNOUSERSITE=1
if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/src:${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
fi
export GRAPHCAST_BACKEND="${BACKEND}"
export GRAPHCAST_FLAGGEMS_CONFIG="${PROJECT_ROOT}/configs/flaggems/graphcast_inference.yaml"
export GRAPHCAST_FLAGGEMS_SOURCE_ROOT="${FLAGGEMS_ROOT}"
export GRAPHCAST_STRICT_FLAGOS=1
export GRAPHCAST_FLAGGEMS_RECORD_PATH="${RUN_OUTPUT}/flag_gems_record.log"
unset GRAPHCAST_FLAGGEMS_OPS
cache_root="${RUN_OUTPUT}/cache"
if [[ -n "${EXECUTABLE_CACHE_PARENT:-}" ]]; then
  cache_root="${EXECUTABLE_CACHE_PARENT}/gc-bf16-${RUN_NAME}-$$"
  test ! -e "${cache_root}"
  trap 'rm -r "${cache_root}"' EXIT
fi
export TRITON_CACHE_DIR="${cache_root}/triton"
export FLAGGEMS_CACHE_DIR="${cache_root}/flaggems"
export XDG_CACHE_HOME="${cache_root}/xdg"
if [[ "${BACKEND}" == "ascend" || "${BACKEND}" == "ptg" ]]; then
  export TMPDIR="/tmp/gc-bf16-${RUN_NAME}-$$"
else
  export TMPDIR="${RUN_OUTPUT}/cache/tmp"
fi
mkdir -p "${TRITON_CACHE_DIR}" "${FLAGGEMS_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

if [[ "${MODE}" == "flaggems" ]]; then
  export GRAPHCAST_FLAGOS_MODE=flaggems
else
  export GRAPHCAST_FLAGOS_MODE=off
fi

arguments=(
  --config "${PROJECT_ROOT}/configs/inference/google_graphcast_operational_25km.yaml"
  --weights "${ASSET_WEIGHTS}"
  --inputs "${ASSET_INPUTS}"
  --rollout-forcings "${ASSET_FORCINGS}"
  --target-template "${ASSET_TARGET}"
  --stats-dir "${ASSET_STATS}"
  --output "${RUN_OUTPUT}/prediction.nc"
  --report "${RUN_OUTPUT}/report.json"
  --platform "${PLATFORM}"
  --model-execution-profile "${MODEL_EXECUTION_PROFILE}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --steps "${STEPS}"
  --warmup-calls 1
)
trajectory_artifact="${RUN_OUTPUT}/trajectory.nc"
if [[ "${DIAGNOSTIC_TRAJECTORY:-0}" == "1" ]]; then
  if [[ -n "${GRAPHCAST_TRAJECTORY_REFERENCE:-}" ]]; then
    : "${GRAPHCAST_TRAJECTORY_DRIVER:?}"
    : "${GRAPHCAST_TRAJECTORY_STDDEV:?}"
    trajectory_artifact="${RUN_OUTPUT}/trajectory_diagnostics.json"
  fi
  arguments+=(--trajectory-output "${trajectory_artifact}" --step-hours 6)
fi

sha256sum \
  "${PROJECT_ROOT}/scripts/inference/run_operational_40step_inference.py" \
  "${PROJECT_ROOT}/configs/flaggems/graphcast_inference.yaml" \
  "${PROJECT_ROOT}/src/graphcast_compat/inference_precision.py" \
  "${PROJECT_ROOT}/src/graphcast_compat/google_small.py" \
  "${FLAGGEMS_ROOT}/src/flag_gems/__init__.py" \
  > "${RUN_OUTPUT}/source_sha256.txt"

runner="${PROJECT_ROOT}/scripts/inference/run_operational_40step_inference.py"
if [[ -n "${GRAPHCAST_TRAJECTORY_REFERENCE:-}" ]]; then
  runner="${GRAPHCAST_TRAJECTORY_DRIVER}"
fi
set +e
"${PYTHON_BIN}" "${runner}" \
  "${arguments[@]}" > "${RUN_OUTPUT}/console.log" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${RUN_OUTPUT}/rc"
if [[ "${status}" -eq 0 ]]; then
  outputs=("${RUN_OUTPUT}/report.json" "${RUN_OUTPUT}/prediction.nc")
  if [[ "${DIAGNOSTIC_TRAJECTORY:-0}" == "1" ]]; then
    outputs+=("${trajectory_artifact}")
  fi
  sha256sum "${outputs[@]}" > "${RUN_OUTPUT}/output_sha256.txt"
fi
if [[ "${status}" -eq 0 && -n "${GRAPHCAST_ACCEPTANCE_BASELINE:-}" ]]; then
  # Offline evaluation is outside the runner's E2E timer. Keep execution and
  # acceptance return codes separate; finite output alone is not acceptance.
  set +e
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/inference/diagnose_inference.py" \
    --run-dir "${RUN_OUTPUT}" \
    --baseline "${GRAPHCAST_ACCEPTANCE_BASELINE}" \
    --baseline-metadata "${GRAPHCAST_ACCEPTANCE_METADATA}" \
    --output-dir "${RUN_OUTPUT}/acceptance" > "${RUN_OUTPUT}/diagnostic.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "${status}" > "${RUN_OUTPUT}/diagnostic_rc"
fi
exit "${status}"
