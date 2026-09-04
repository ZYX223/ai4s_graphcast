#!/usr/bin/env bash
set -Eeuo pipefail

: "${GENERIC_CASE_RUNNER:?}"
: "${OUTPUT_ROOT:?}"
: "${RUN_NAME:?}"
: "${PLATFORM:?}"

publication_root="${OUTPUT_ROOT}"
local_parent="${LOCAL_STAGING_PARENT:-/dev/shm/graphcast-bf16-inference}"
local_root="${local_parent}/${PLATFORM}"
local_run="${local_root}/${RUN_NAME}"
published_run="${publication_root}/${RUN_NAME}"
executable_cache_parent="${EXECUTABLE_CACHE_PARENT:-/tmp}"
test ! -e "${local_run}"
test ! -e "${published_run}"
mkdir -p "${local_root}"

available_bytes=$(df --output=avail -B1 "${local_root}" | tail -1)
if [[ "${DIAGNOSTIC_TRAJECTORY:-0}" == "1" && -z "${GRAPHCAST_TRAJECTORY_REFERENCE:-}" ]]; then
  required_bytes=$((12 * 1024 * 1024 * 1024))
else
  required_bytes=$((1 * 1024 * 1024 * 1024))
fi
if (( available_bytes < required_bytes )); then
  echo "insufficient local staging space: ${available_bytes} < ${required_bytes}" >&2
  exit 2
fi

status=0
OUTPUT_ROOT="${local_root}" EXECUTABLE_CACHE_PARENT="${executable_cache_parent}" bash "${GENERIC_CASE_RUNNER}" || status=$?
test -d "${local_run}"
printf 'staged_path=%s\npublished_path=%s\n' "${local_run}" "${published_run}" > "${local_run}/relocation.txt"
(
  cd "${local_run}"
  find . -type f ! -name transfer_sha256.txt -print0 | sort -z | xargs -0 sha256sum
) > "${local_run}/transfer_sha256.txt"

mkdir -p "${published_run}"
tar -C "${local_run}" -cf - . | tar -C "${published_run}" -xf -
(
  cd "${published_run}"
  sha256sum -c transfer_sha256.txt
)
rm -r "${local_run}"
exit "${status}"
