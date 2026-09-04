#!/usr/bin/env bash
set -Eeuo pipefail

: "${CASE_RUNNER:?}"
: "${PLATFORM:?}"
: "${OUTPUT_ROOT:?}"
: "${STEPS:?}"
: "${GRAPHCAST_TRAJECTORY_DRIVER:?}"
: "${ASSET_STATS:?}"

reference="${OUTPUT_ROOT}/${PLATFORM}-native-fp32-${STEPS}step-trajectory/trajectory.nc"
test -f "${reference}"
export GRAPHCAST_TRAJECTORY_REFERENCE="${reference}"
export GRAPHCAST_TRAJECTORY_STDDEV="${ASSET_STATS}/stddev_by_level.nc"

if [[ "${RUN_NATIVE:-1}" == "1" ]]; then
  DIAGNOSTIC_TRAJECTORY=1 MODE=native PRECISION=bf16 \
    RUN_NAME="${PLATFORM}-native-bf16-${STEPS}step-streaming-diagnostics" \
    bash "${CASE_RUNNER}"
fi
if [[ "${RUN_FLAGGEMS:-1}" == "1" ]]; then
  DIAGNOSTIC_TRAJECTORY=1 MODE=flaggems PRECISION=bf16 \
    RUN_NAME="${PLATFORM}-flaggems-bf16-${STEPS}step-streaming-diagnostics" \
    bash "${CASE_RUNNER}"
fi
