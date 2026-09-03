#!/usr/bin/env bash
set -euo pipefail

remote_alias="${1:?remote alias is required}"
platform="${2:?platform is required}"
device_id="${3:?physical device id is required}"
campaign_id="${4:?campaign id is required}"
local_root="${5:?local evidence root is required}"
remote_tool="${6:?remote access tool is required}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_file="${script_dir}/torch_dense_gemm.py"
runner_file="${script_dir}/run_container_gemm.sh"
source_sha256="$(sha256sum "${source_file}" | cut -d' ' -f1)"
runner_sha256="$(sha256sum "${runner_file}" | cut -d' ' -f1)"
protocol_sha256="$(sha256sum "${script_dir}/PROTOCOL.md" | cut -d' ' -f1)"

test -x "${remote_tool}"
test ! -e "${local_root}/${platform}"
mkdir -p "${local_root}/${platform}"
printf 'campaign_id=%s\nplatform=%s\nremote_alias=%s\ndevice_id=%s\nsource_sha256=%s\nrunner_sha256=%s\nprotocol_sha256=%s\n' \
  "${campaign_id}" "${platform}" "${remote_alias}" "${device_id}" \
  "${source_sha256}" "${runner_sha256}" "${protocol_sha256}" \
  >"${local_root}/${platform}/campaign.env"

mode_orders=(
  'strict_fp32 fast_fp32 bf16'
  'strict_fp32 bf16 fast_fp32'
  'fast_fp32 strict_fp32 bf16'
  'fast_fp32 bf16 strict_fp32'
  'bf16 strict_fp32 fast_fp32'
  'bf16 fast_fp32 strict_fp32'
)
ascending=(8192 12288 16384 24576)
descending=(24576 16384 12288 8192)

for block in 0 1 2 3 4 5; do
  if (( block % 2 == 0 )); then
    sizes=("${ascending[@]}")
  else
    sizes=("${descending[@]}")
  fi
  read -r -a modes <<<"${mode_orders[block]}"
  for position in 0 1 2; do
    mode="${modes[position]}"
    run_id="${campaign_id}-${platform}-b${block}-p${position}-${mode}"
    remote_stage="/tmp/${run_id}-stage"
    remote_evidence="/tmp/${run_id}-evidence"
    local_block="${local_root}/${platform}/block-${block}"
    local_evidence="${local_block}/${run_id}"
    mkdir -p "${local_block}"

    bash "${remote_tool}" ssh "${remote_alias}" \
      "test ! -e '${remote_stage}' && test ! -e '${remote_evidence}' && mkdir '${remote_stage}'"
    bash "${remote_tool}" upload "${remote_alias}" \
      "${source_file}" "${remote_stage}/torch_dense_gemm.py"
    bash "${remote_tool}" upload "${remote_alias}" \
      "${runner_file}" "${remote_stage}/run_container_gemm.sh"
    remote_source_sha256="$(bash "${remote_tool}" ssh "${remote_alias}" \
      "sha256sum '${remote_stage}/torch_dense_gemm.py' | cut -d' ' -f1")"
    remote_runner_sha256="$(bash "${remote_tool}" ssh "${remote_alias}" \
      "sha256sum '${remote_stage}/run_container_gemm.sh' | cut -d' ' -f1")"
    test "${remote_source_sha256}" = "${source_sha256}"
    test "${remote_runner_sha256}" = "${runner_sha256}"

    # The JumpServer wrapper does not reliably propagate the remote process
    # status, so the create-once rc file is the authoritative completion gate.
    bash "${remote_tool}" ssh "${remote_alias}" \
      "bash '${remote_stage}/run_container_gemm.sh' '${platform}' '${run_id}' '${source_sha256}' '${mode}' '${device_id}' 5 20 ${sizes[*]}" \
      >"${local_block}/${run_id}.remote.stdout" \
      2>"${local_block}/${run_id}.remote.stderr" || true
    remote_rc="$(bash "${remote_tool}" ssh "${remote_alias}" "cat '${remote_stage}/rc'")"
    test "${remote_rc}" = 0

    test ! -e "${local_evidence}"
    RECURSIVE=1 bash "${remote_tool}" download "${remote_alias}" \
      "${remote_evidence}" "${local_evidence}"
    (
      cd "${local_evidence}"
      sha256sum -c SHA256SUMS
      test "$(python3 -c 'import json; print(json.load(open("result.json"))["source"]["sha256"])')" = \
        "${source_sha256}"
    ) >"${local_block}/${run_id}.verify.log"
  done
done

(
  cd "${local_root}/${platform}"
  find . -type f ! -name PLATFORM_SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >PLATFORM_SHA256SUMS
)
