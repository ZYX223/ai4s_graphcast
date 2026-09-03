#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_wrapper_sha256="${4:?profile wrapper SHA-256 is required}"

host_stage="/tmp/${run_id}-stage"
host_evidence="/tmp/${run_id}-evidence"
container="zym-graphcast-910c"
container_stage="/tmp/${run_id}-stage"
container_evidence="/tmp/${run_id}-evidence"
python_bin="/usr/local/python3.11.14/bin/python3"
msprof_bin="/usr/local/Ascend/cann-8.5.0/bin/msprof"

test -d "${host_stage}"
test -f "${host_stage}/torch_dense_gemm.py"
test -f "${host_stage}/profile_application.sh"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/profile_application.sh" | cut -d' ' -f1)" = \
  "${expected_wrapper_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

npu-smi info >"${host_evidence}/device-before.txt"
docker inspect "${container}" >"${host_evidence}/container-inspect.json"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" test ! -e "${container_evidence}"
docker exec "${container}" mkdir "${container_stage}" "${container_evidence}"
docker cp "${host_stage}/torch_dense_gemm.py" \
  "${container}:${container_stage}/torch_dense_gemm.py"
docker cp "${host_stage}/profile_application.sh" \
  "${container}:${container_stage}/profile_application.sh"
docker exec "${container}" chmod 0555 "${container_stage}/profile_application.sh"

for mode in fast_fp32 bf16; do
  profile_output="${container_evidence}/${mode}-result.json"
  profile_root="${container_evidence}/${mode}-msprof"
  command_file="${host_evidence}/${mode}-command.sh"
  printf '%q ' docker exec \
    -e "ASCEND_RT_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=npu:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" "${msprof_bin}" op \
    "--application=${container_stage}/profile_application.sh" \
    "--output=${profile_root}" \
    '--aic-metrics=ArithmeticUtilization,PipeUtilization,BasicInfo' \
    '--launch-count=200' '--warm-up=0' >"${command_file}"
  printf '\n' >>"${command_file}"

  docker exec \
    -e "ASCEND_RT_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=npu:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" "${msprof_bin}" op \
    "--application=${container_stage}/profile_application.sh" \
    "--output=${profile_root}" \
    '--aic-metrics=ArithmeticUtilization,PipeUtilization,BasicInfo' \
    '--launch-count=200' '--warm-up=0' \
    >"${host_evidence}/${mode}-msprof.stdout" \
    2>"${host_evidence}/${mode}-msprof.stderr"
  docker exec "${container}" test -f "${profile_output}"
done

docker cp "${container}:${container_evidence}/." "${host_evidence}/"
npu-smi info >"${host_evidence}/device-after.txt"
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
