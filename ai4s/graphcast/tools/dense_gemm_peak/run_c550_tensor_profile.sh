#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_wrapper_sha256="${4:?profile wrapper SHA-256 is required}"
expected_config_sha256="${5:?MCPTI config SHA-256 is required}"
expected_range_sha256="${6:?runtime range SHA-256 is required}"

host_stage="/tmp/${run_id}-stage"
host_evidence="/tmp/${run_id}-evidence"
container="zym-graphcast-c550"
container_stage="/tmp/${run_id}-stage"
container_evidence="/tmp/${run_id}-evidence"
python_bin="/opt/conda/bin/python3"

test -d "${host_stage}"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/profile_application.sh" | cut -d' ' -f1)" = \
  "${expected_wrapper_sha256}"
test "$(sha256sum "${host_stage}/mcProfiler.json" | cut -d' ' -f1)" = \
  "${expected_config_sha256}"
test "$(sha256sum "${host_stage}/runtime_profiler_range.py" | cut -d' ' -f1)" = \
  "${expected_range_sha256}"
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
for name in torch_dense_gemm.py profile_application.sh mcProfiler.json \
  runtime_profiler_range.py; do
  docker cp "${host_stage}/${name}" "${container}:${container_stage}/${name}"
done
docker exec "${container}" chmod 0555 "${container_stage}/profile_application.sh"

for mode in fast_fp32 bf16; do
  profile_output="${container_evidence}/${mode}/${mode}-result.json"
  docker exec "${container}" mkdir "${container_evidence}/${mode}"
  docker exec "${container}" cp "${container_stage}/mcProfiler.json" \
    "${container_evidence}/${mode}/mcProfiler.json"

  printf '%q ' docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    -e "MCTX_TARGET_PROFILE_PATH=${container_evidence}/${mode}" \
    -e 'LD_PRELOAD=/opt/maca/lib/libmcpti.so' \
    -w "${container_evidence}/${mode}" \
    "${container}" "${python_bin}" \
    "${container_stage}/runtime_profiler_range.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --runtime-library /opt/maca/lib/libmcruntime.so \
    --start-symbol mcProfilerStart --stop-symbol mcProfilerStop -- \
    --mode "${mode}" --device cuda:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${profile_output}" \
    >"${host_evidence}/${mode}-command.sh"
  printf '\n' >>"${host_evidence}/${mode}-command.sh"

  docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    -e "MCTX_TARGET_PROFILE_PATH=${container_evidence}/${mode}" \
    -e 'LD_PRELOAD=/opt/maca/lib/libmcpti.so' \
    -w "${container_evidence}/${mode}" \
    "${container}" "${python_bin}" \
    "${container_stage}/runtime_profiler_range.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --runtime-library /opt/maca/lib/libmcruntime.so \
    --start-symbol mcProfilerStart --stop-symbol mcProfilerStop -- \
    --mode "${mode}" --device cuda:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${profile_output}" \
    >"${host_evidence}/${mode}.stdout" \
    2>"${host_evidence}/${mode}.stderr"
  docker exec "${container}" test -f "${profile_output}"
  docker exec "${container}" test -s \
    "${container_evidence}/${mode}/tdump.txt"
done

docker cp "${container}:${container_evidence}/." "${host_evidence}/"
if fuser "${render_node}" >"${host_evidence}/processes-after.txt" 2>&1; then
  printf 'device remained busy after profiling: %s\n' "${render_node}" >&2
  exit 1
fi
printf 'run_id=%s\ndevice_id=%s\nsource_sha256=%s\nwrapper_sha256=%s\nconfig_sha256=%s\nrange_sha256=%s\n' \
  "${run_id}" "${device_id}" "${expected_source_sha256}" \
  "${expected_wrapper_sha256}" "${expected_config_sha256}" \
  "${expected_range_sha256}" \
  >"${host_evidence}/identity.env"
(
  cd "${host_evidence}"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >SHA256SUMS
)
