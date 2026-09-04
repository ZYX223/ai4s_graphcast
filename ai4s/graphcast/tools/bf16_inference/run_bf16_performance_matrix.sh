#!/usr/bin/env bash
set -Eeuo pipefail

: "${CASE_RUNNER:?}"
: "${PLATFORM:?}"
: "${OUTPUT_ROOT:?}"
: "${STEPS:?}"

if [[ "${RUN_NATIVE:-1}" == "1" ]]; then
for repetition in 1 2 3; do
  DIAGNOSTIC_TRAJECTORY=0 MODE=native PRECISION=bf16 \
    RUN_NAME="${PLATFORM}-native-bf16-${STEPS}step-r${repetition}" \
    bash "${CASE_RUNNER}"
done
fi
if [[ "${RUN_FLAGGEMS:-1}" == "1" ]]; then
  for repetition in 1 2 3; do
    DIAGNOSTIC_TRAJECTORY=0 MODE=flaggems PRECISION=bf16 \
      RUN_NAME="${PLATFORM}-flaggems-bf16-${STEPS}step-r${repetition}" \
      bash "${CASE_RUNNER}"
  done
fi
