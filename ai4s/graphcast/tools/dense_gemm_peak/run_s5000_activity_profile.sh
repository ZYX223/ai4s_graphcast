#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?run id is required}"
device_id="${2:?device id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
expected_profiler_sha256="${4:?activity profiler SHA-256 is required}"

host_stage="/tmp/${run_id}-stage"
host_evidence="/tmp/${run_id}-evidence"
container="zym-graphcast-s550"
container_stage="/tmp/${run_id}-stage"
container_evidence="/tmp/${run_id}-evidence"
python_bin="/usr/bin/python3"

test -d "${host_stage}"
test "$(sha256sum "${host_stage}/torch_dense_gemm.py" | cut -d' ' -f1)" = \
  "${expected_source_sha256}"
test "$(sha256sum "${host_stage}/torch_activity_profile.py" | cut -d' ' -f1)" = \
  "${expected_profiler_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

mthreads-gmi -q -i "${device_id}" --json \
  >"${host_evidence}/device-before.json"
python3 - "${host_evidence}/device-before.json" "${device_id}" <<'PY'
import json
import pathlib
import re
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())["GPU"]
if len(record) != 1 or record[0]["Index"] != sys.argv[2]:
    raise SystemExit("device identity mismatch")
gpu = record[0]
if gpu["Product Name"] != "MTT S5000":
    raise SystemExit("unexpected product")
if gpu["Utilization"]["Gpu"] != "0%":
    raise SystemExit("device is not idle")
used = int(re.fullmatch(r"(\d+)MiB", gpu["FB Memory Usage"]["Used"]).group(1))
if used > 128:
    raise SystemExit(f"device has too much allocated memory: {used} MiB")
PY

docker inspect "${container}" >"${host_evidence}/container-inspect.json"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" test ! -e "${container_evidence}"
docker exec "${container}" mkdir "${container_stage}" "${container_evidence}"
for name in torch_dense_gemm.py torch_activity_profile.py; do
  docker cp "${host_stage}/${name}" "${container}:${container_stage}/${name}"
done

for mode in fast_fp32 bf16; do
  output="${container_evidence}/${mode}-result.json"
  trace="${container_evidence}/${mode}-trace.json"
  summary="${container_evidence}/${mode}-activity.json"
  printf '%q ' docker exec \
    -e "MUSA_VISIBLE_DEVICES=${device_id}" \
    "${container}" "${python_bin}" \
    "${container_stage}/torch_activity_profile.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --activity privateuse1 \
    --trace "${trace}" --summary "${summary}" -- \
    --mode "${mode}" --device musa:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${output}" \
    >"${host_evidence}/${mode}-command.sh"
  printf '\n' >>"${host_evidence}/${mode}-command.sh"

  docker exec \
    -e "MUSA_VISIBLE_DEVICES=${device_id}" \
    "${container}" "${python_bin}" \
    "${container_stage}/torch_activity_profile.py" \
    --source "${container_stage}/torch_dense_gemm.py" \
    --expected-source-sha256 "${expected_source_sha256}" \
    --activity privateuse1 \
    --trace "${trace}" --summary "${summary}" -- \
    --mode "${mode}" --device musa:0 --sizes 8192 \
    --warmup 1 --iterations 3 --output "${output}" \
    >"${host_evidence}/${mode}.stdout" \
    2>"${host_evidence}/${mode}.stderr"
  docker exec "${container}" test -s "${output}"
  docker exec "${container}" test -s "${trace}"
  docker exec "${container}" test -s "${summary}"
done

docker cp "${container}:${container_evidence}/." "${host_evidence}/"
mthreads-gmi -q -i "${device_id}" --json \
  >"${host_evidence}/device-after.json"
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
