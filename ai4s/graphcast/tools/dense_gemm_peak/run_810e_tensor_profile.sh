#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_wrapper_sha256="${4:?profile wrapper SHA-256 is required}"

host_parent="${PROFILE_HOST_PARENT:-/tmp}"
host_stage="${host_parent}/${run_id}-stage"
host_evidence="${host_parent}/${run_id}-evidence"
container="lms_dsv4"
container_parent="${PROFILE_CONTAINER_PARENT:-/tmp}"
container_stage="${container_parent}/${run_id}-stage"
container_evidence="${container_parent}/${run_id}-evidence"
shared_transport="${PROFILE_SHARED_TRANSPORT:-0}"
python_bin="/usr/local/bin/python3"
acu_bin="/usr/local/PPU_SDK/asight/bin/acu"
ppu_smi_bin="/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"

test -d "${host_stage}"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/profile_application.sh" | cut -d' ' -f1)" = \
  "${expected_wrapper_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

docker exec "${container}" "${ppu_smi_bin}" \
  >"${host_evidence}/processes-before.txt" 2>&1
grep -q "| ${device_id}  PPU-ZW810E" "${host_evidence}/processes-before.txt"
docker inspect "${container}" >"${host_evidence}/container-inspect.json"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" test ! -e "${container_evidence}"
docker exec "${container}" mkdir "${container_stage}" "${container_evidence}"
for name in torch_dense_gemm.py profile_application.sh; do
  if [[ "${shared_transport}" == "1" ]]; then
    docker exec "${container}" cp "${host_stage}/${name}" \
      "${container_stage}/${name}"
  else
    docker cp "${host_stage}/${name}" \
      "${container}:${container_stage}/${name}"
  fi
done
docker exec "${container}" chmod 0555 "${container_stage}/profile_application.sh"

for mode in fast_fp32 bf16; do
  if [[ "${mode}" == fast_fp32 ]]; then
    metrics='cu__inst_executed_pipe_tensor_tf32.sum,cu__ops_pipe_tensor_tf32.sum.per_cycle_elapsed,cu__inst_executed_pipe_tensor_bf16.sum'
  else
    metrics='cu__inst_executed_pipe_tensor_bf16.sum,cu__ops_pipe_tensor_bf16.sum.per_cycle_elapsed,cu__inst_executed_pipe_tensor_tf32.sum'
  fi
  profile_output="${container_evidence}/${mode}-result.json"
  report_output="${container_evidence}/${mode}.asysrep"
  csv_output="${container_evidence}/${mode}.csv"

  printf '%q ' docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=cuda:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" "${acu_bin}" \
    --devices 0 \
    --check-exit-code yes \
    --kill no \
    --replay-mode kernel \
    --kernel-name 'regex:.*([Gg][Ee][Mm][Mm]|[Mm]at[Mm]ul).*' \
    --launch-count 8 \
    --metrics "${metrics}" \
    --csv-file "${csv_output}" \
    --export "${report_output}" \
    --force-overwrite \
    /usr/bin/bash "${container_stage}/profile_application.sh" \
    >"${host_evidence}/${mode}-command.sh"
  printf '\n' >>"${host_evidence}/${mode}-command.sh"

  docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=cuda:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" "${acu_bin}" \
    --devices 0 \
    --check-exit-code yes \
    --kill no \
    --replay-mode kernel \
    --kernel-name 'regex:.*([Gg][Ee][Mm][Mm]|[Mm]at[Mm]ul).*' \
    --launch-count 8 \
    --metrics "${metrics}" \
    --csv-file "${csv_output}" \
    --export "${report_output}" \
    --force-overwrite \
    /usr/bin/bash "${container_stage}/profile_application.sh" \
    >"${host_evidence}/${mode}.stdout" \
    2>"${host_evidence}/${mode}.stderr"
  docker exec "${container}" test -f "${profile_output}"
  docker exec "${container}" test -s "${csv_output}"
done

if [[ "${shared_transport}" == "1" ]]; then
  docker exec "${container}" cp -a "${container_evidence}/." \
    "${host_evidence}/"
else
  docker cp "${container}:${container_evidence}/." "${host_evidence}/"
fi
docker exec "${container}" "${ppu_smi_bin}" \
  >"${host_evidence}/processes-after.txt" 2>&1
printf 'run_id=%s\ndevice_id=%s\nsource_sha256=%s\nwrapper_sha256=%s\n' \
  "${run_id}" "${device_id}" "${expected_source_sha256}" \
  "${expected_wrapper_sha256}" >"${host_evidence}/identity.env"
(
  cd "${host_evidence}"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >SHA256SUMS
)
