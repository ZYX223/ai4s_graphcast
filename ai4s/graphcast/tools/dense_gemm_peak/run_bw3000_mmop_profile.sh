#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_wrapper_sha256="${4:?profile wrapper SHA-256 is required}"
expected_metrics_sha256="${5:?metrics SHA-256 is required}"

host_stage="/tmp/${run_id}-stage"
host_evidence="/tmp/${run_id}-evidence"
container="zyx-graphcast-bw1000-v4"
container_stage="/tmp/${run_id}-stage"
container_evidence="/tmp/${run_id}-evidence"
python_bin="/usr/bin/python3"
hipprof_bin="/opt/dtk/bin/hipprof"

test -d "${host_stage}"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/profile_application.sh" | cut -d' ' -f1)" = \
  "${expected_wrapper_sha256}"
test "$(sha256sum "${host_stage}/bw3000_mmop_metrics.txt" | cut -d' ' -f1)" = \
  "${expected_metrics_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

render_node="/dev/dri/renderD$((128 + device_id))"
test -e "${render_node}"
if fuser "${render_node}" >"${host_evidence}/processes-before.txt" 2>&1; then
  printf 'device is busy: %s\n' "${render_node}" >&2
  exit 1
fi
docker inspect "${container}" >"${host_evidence}/container-inspect.json"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" test ! -e "${container_evidence}"
docker exec "${container}" mkdir "${container_stage}" "${container_evidence}"
for name in torch_dense_gemm.py profile_application.sh bw3000_mmop_metrics.txt; do
  docker cp "${host_stage}/${name}" "${container}:${container_stage}/${name}"
done
docker exec "${container}" chmod 0555 "${container_stage}/profile_application.sh"

for mode in fast_fp32 bf16; do
  profile_output="${container_evidence}/${mode}-result.json"
  output_prefix="${container_evidence}/${mode}-mmop"
  printf '%q ' docker exec \
    -e "HIP_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=cuda:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" bash -lc \
    "set +u; source /opt/dtk/env.sh >/dev/null 2>&1; exec ${hipprof_bin} --pmc --pmc-type 3 -i ${container_stage}/bw3000_mmop_metrics.txt -o ${output_prefix} ${container_stage}/profile_application.sh" \
    >"${host_evidence}/${mode}-command.sh"
  printf '\n' >>"${host_evidence}/${mode}-command.sh"

  docker exec \
    -e "HIP_VISIBLE_DEVICES=${device_id}" \
    -e "PROFILE_PYTHON=${python_bin}" \
    -e "PROFILE_SOURCE=${container_stage}/torch_dense_gemm.py" \
    -e "PROFILE_MODE=${mode}" \
    -e 'PROFILE_DEVICE=cuda:0' \
    -e 'PROFILE_SIZE=8192' \
    -e 'PROFILE_WARMUP=1' \
    -e 'PROFILE_ITERATIONS=3' \
    -e "PROFILE_OUTPUT=${profile_output}" \
    "${container}" bash -lc \
    "set +u; source /opt/dtk/env.sh >/dev/null 2>&1; exec ${hipprof_bin} --pmc --pmc-type 3 -i ${container_stage}/bw3000_mmop_metrics.txt -o ${output_prefix} ${container_stage}/profile_application.sh" \
    >"${host_evidence}/${mode}.stdout" \
    2>"${host_evidence}/${mode}.stderr"
  docker exec "${container}" test -f "${profile_output}"
done

docker cp "${container}:${container_evidence}/." "${host_evidence}/"
if fuser "${render_node}" >"${host_evidence}/processes-after.txt" 2>&1; then
  printf 'device remained busy after profiling: %s\n' "${render_node}" >&2
  exit 1
fi
printf 'run_id=%s\ndevice_id=%s\nsource_sha256=%s\nwrapper_sha256=%s\nmetrics_sha256=%s\n' \
  "${run_id}" "${device_id}" "${expected_source_sha256}" \
  "${expected_wrapper_sha256}" "${expected_metrics_sha256}" \
  >"${host_evidence}/identity.env"
(
  cd "${host_evidence}"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >SHA256SUMS
)
