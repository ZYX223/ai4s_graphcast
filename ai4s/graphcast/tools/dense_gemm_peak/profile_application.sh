#!/usr/bin/env bash
set -euo pipefail

profile_python="${PROFILE_PYTHON:?PROFILE_PYTHON is required}"
profile_source="${PROFILE_SOURCE:?PROFILE_SOURCE is required}"
profile_mode="${PROFILE_MODE:?PROFILE_MODE is required}"
profile_device="${PROFILE_DEVICE:?PROFILE_DEVICE is required}"
profile_size="${PROFILE_SIZE:?PROFILE_SIZE is required}"
profile_warmup="${PROFILE_WARMUP:?PROFILE_WARMUP is required}"
profile_iterations="${PROFILE_ITERATIONS:?PROFILE_ITERATIONS is required}"
profile_output="${PROFILE_OUTPUT:?PROFILE_OUTPUT is required}"

case "${profile_mode}" in
  strict_fp32|fast_fp32|bf16) ;;
  *)
    echo "unsupported PROFILE_MODE: ${profile_mode}" >&2
    exit 2
    ;;
esac

[[ "${profile_size}" =~ ^[0-9]+$ ]] && (( profile_size >= 256 ))
[[ "${profile_warmup}" =~ ^[0-9]+$ ]] && (( profile_warmup >= 1 ))
[[ "${profile_iterations}" =~ ^[0-9]+$ ]] && (( profile_iterations >= 3 ))
test -x "${profile_python}"
test -f "${profile_source}"
test ! -e "${profile_output}"

exec "${profile_python}" "${profile_source}" \
  --mode "${profile_mode}" \
  --device "${profile_device}" \
  --sizes "${profile_size}" \
  --warmup "${profile_warmup}" \
  --iterations "${profile_iterations}" \
  --output "${profile_output}"
