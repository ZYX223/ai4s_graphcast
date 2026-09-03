#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_profiler_sha256="${4:?activity profiler SHA-256 is required}"

shared_parent="${PROFILE_SHARED_PARENT:-/mnt/cpfs/graphcast-profiling-staging}"
host_stage="${shared_parent}/${run_id}-stage"
host_evidence="${shared_parent}/${run_id}-evidence"
container="lms_dsv4"
container_parent="${PROFILE_CONTAINER_PARENT:-/dev/shm}"
container_stage="${container_parent}/${run_id}-stage"
container_evidence="${container_parent}/${run_id}-evidence"
python_bin="/usr/local/bin/python3"
ppu_smi_bin="/usr/local/PPU_SDK/ppu-smi/bin/ppu-smi"

test -d "${host_stage}"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/torch_activity_profile.py" | cut -d' ' -f1)" = \
  "${expected_profiler_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

docker exec "${container}" "${ppu_smi_bin}" \
  >"${host_evidence}/device-before.txt" 2>&1
python3 - "${host_evidence}/device-before.txt" "${device_id}" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
device = re.escape(sys.argv[2])
match = re.search(
    rf"\|\s*{device}\s+PPU-ZW810E.*?\n"
    rf"\|.*?\|\s*(\d+)MiB\s*/.*?\|\s*(\d+)%",
    text,
)
if match is None:
    raise SystemExit("cannot parse PPU identity and utilization")
used_mib, utilization = map(int, match.groups())
if utilization != 0 or used_mib > 128:
    raise SystemExit(
        f"device is not idle: used={used_mib} MiB utilization={utilization}%"
    )
PY

docker inspect "${container}" >"${host_evidence}/container-inspect.json"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" test ! -e "${container_evidence}"
docker exec "${container}" mkdir "${container_stage}" "${container_evidence}"
for name in torch_dense_gemm.py torch_activity_profile.py; do
  docker exec "${container}" cp "${host_stage}/${name}" \
    "${container_stage}/${name}"
done

for mode in fast_fp32 bf16; do
  output="${container_evidence}/${mode}-result.json"
  trace="${container_evidence}/${mode}-trace.json"
  summary="${container_evidence}/${mode}-activity.json"
  printf '%q ' docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    "${container}" "${python_bin}" \
    "${container_stage}/torch_activity_profile.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --activity cuda --trace "${trace}" --summary "${summary}" -- \
    --mode "${mode}" --device cuda:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${output}" \
    >"${host_evidence}/${mode}-command.sh"
  printf '\n' >>"${host_evidence}/${mode}-command.sh"

  docker exec \
    -e "CUDA_VISIBLE_DEVICES=${device_id}" \
    "${container}" "${python_bin}" \
    "${container_stage}/torch_activity_profile.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --activity cuda --trace "${trace}" --summary "${summary}" -- \
    --mode "${mode}" --device cuda:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${output}" \
    >"${host_evidence}/${mode}.stdout" \
    2>"${host_evidence}/${mode}.stderr"
  docker exec "${container}" test -s "${output}"
  docker exec "${container}" test -s "${trace}"
  docker exec "${container}" test -s "${summary}"
done

docker exec "${container}" cp -a "${container_evidence}/." \
  "${host_evidence}/"
docker exec "${container}" "${ppu_smi_bin}" \
  >"${host_evidence}/device-after.txt" 2>&1
printf 'run_id=%s\ndevice_id=%s\nsource_sha256=%s\nprofiler_sha256=%s\n' \
  "${run_id}" "${device_id}" "${expected_source_sha256}" \
  "${expected_profiler_sha256}" >"${host_evidence}/identity.env"
(
  cd "${host_evidence}"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >SHA256SUMS
)
