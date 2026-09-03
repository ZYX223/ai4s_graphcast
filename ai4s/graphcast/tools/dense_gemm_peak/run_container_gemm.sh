#!/usr/bin/env bash
set -euo pipefail

platform="${1:?platform is required}"
run_id="${2:?run id is required}"
expected_source_sha256="${3:?source SHA-256 is required}"
mode="${4:?mode is required}"
device_id="${5:?physical device id is required}"
warmup="${6:?warmup is required}"
iterations="${7:?iterations is required}"
shift 7
(( $# > 0 )) || { echo "at least one size is required" >&2; exit 2; }
sizes=("$@")

host_stage="/tmp/${run_id}-stage"
host_evidence="/tmp/${run_id}-evidence"
host_source="${host_stage}/torch_dense_gemm.py"
container_stage="/tmp/${run_id}-stage"
container_result="${container_stage}/result.json"

test -f "${host_source}"
test "$(sha256sum "${host_source}" | cut -d' ' -f1)" = "${expected_source_sha256}"
test ! -e "${host_evidence}"
mkdir "${host_evidence}"

case "${platform}" in
  h100)
    container=zym-graphcast-h100
    python_bin=/usr/bin/python3
    visibility_name=CUDA_VISIBLE_DEVICES
    device=cuda:0
    nvidia-smi -q -i "${device_id}" >"${host_evidence}/device-before.txt"
    telemetry_after=(nvidia-smi -q -i "${device_id}")
    ;;
  910c)
    container=zym-graphcast-910c
    python_bin=/usr/local/python3.11.14/bin/python3
    visibility_name=ASCEND_RT_VISIBLE_DEVICES
    device=npu:0
    npu-smi info >"${host_evidence}/device-before.txt"
    telemetry_after=(npu-smi info)
    ;;
  bw3000)
    container=zyx-graphcast-bw1000-v4
    python_bin=/usr/bin/python3
    visibility_name=HIP_VISIBLE_DEVICES
    device=cuda:0
    /opt/rocm/bin/rocm-smi --showallinfo >"${host_evidence}/device-before.txt" 2>&1 || true
    telemetry_after=(/opt/rocm/bin/rocm-smi --showallinfo)
    ;;
  s5000)
    container=zym-graphcast-s550
    python_bin=/usr/bin/python3
    visibility_name=MUSA_VISIBLE_DEVICES
    device=musa:0
    mthreads-gmi -q -i "${device_id}" >"${host_evidence}/device-before.txt" 2>&1 || \
      mthreads-gmi >"${host_evidence}/device-before.txt" 2>&1
    telemetry_after=(mthreads-gmi -q -i "${device_id}")
    ;;
  c550)
    container=zym-graphcast-c550
    python_bin=/opt/conda/bin/python3
    visibility_name=CUDA_VISIBLE_DEVICES
    device=cuda:0
    mx-smi -q -i "${device_id}" >"${host_evidence}/device-before.txt" 2>&1 || \
      mx-smi >"${host_evidence}/device-before.txt" 2>&1
    telemetry_after=(mx-smi -q -i "${device_id}")
    ;;
  810e)
    container=zym-graphcast-ptg-inference-fbbe81e-229
    python_bin=/usr/local/bin/python3
    visibility_name=CUDA_VISIBLE_DEVICES
    device=cuda:0
    docker exec "${container}" /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi \
      >"${host_evidence}/device-before.txt" 2>&1
    telemetry_after=(docker exec "${container}" /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi)
    ;;
  *)
    echo "unsupported platform: ${platform}" >&2
    exit 2
    ;;
esac

case "${platform}" in
  h100)
    nvidia-smi -i "${device_id}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits \
      >"${host_evidence}/processes-before.txt"
    test ! -s "${host_evidence}/processes-before.txt"
    ;;
  910c)
    awk -v card="${device_id}" '
      /Process id/ { in_processes=1; next }
      in_processes && $1 == "|" && $2 == card && $3 ~ /^[01]$/ { print }
    ' "${host_evidence}/device-before.txt" \
      >"${host_evidence}/processes-before.txt"
    test ! -s "${host_evidence}/processes-before.txt"
    ;;
  bw3000)
    docker exec "${container}" bash -lc \
      'set +u; source /opt/dtk/env.sh >/dev/null 2>&1; /opt/dtk/bin/rocm-smi --showpids' \
      >"${host_evidence}/processes-before.txt" 2>&1 || true
    grep -q 'No KFD PIDs currently running' \
      "${host_evidence}/processes-before.txt"
    ;;
  s5000)
    # This host has context-only handles (<64 MiB aggregate on the selected
    # card) from a service using other cards.  Preserve the full process list;
    # stability/telemetry gates decide whether it remained non-competing.
    mthreads-gmi >"${host_evidence}/processes-before.txt" 2>&1
    ;;
  c550)
    mx-smi >"${host_evidence}/processes-before.full.txt" 2>&1
    awk -v gpu="${device_id}" '
      /^\| Process:/ { in_processes=1; next }
      in_processes && $1 == "|" && $2 == gpu { print }
    ' "${host_evidence}/processes-before.full.txt" \
      >"${host_evidence}/processes-before.txt"
    test ! -s "${host_evidence}/processes-before.txt"
    ;;
  810e)
    cp "${host_evidence}/device-before.txt" \
      "${host_evidence}/processes-before.txt"
    grep -q 'No running processes found' \
      "${host_evidence}/processes-before.txt"
    ;;
esac

test "$(docker inspect -f '{{.State.Running}}' "${container}")" = true
docker inspect -f \
  'Name={{.Name}} Image={{.Config.Image}} Id={{.Id}} Devices={{json .HostConfig.Devices}}' \
  "${container}" >"${host_evidence}/container-identity.txt"
docker exec "${container}" test ! -e "${container_stage}"
docker exec "${container}" mkdir "${container_stage}"
docker cp "${host_source}" "${container}:${container_stage}/torch_dense_gemm.py"
container_sha="$(docker exec "${container}" sha256sum "${container_stage}/torch_dense_gemm.py" | cut -d' ' -f1)"
test "${container_sha}" = "${expected_source_sha256}"

printf '%q ' docker exec \
  -e "${visibility_name}=${device_id}" \
  "${container}" "${python_bin}" "${container_stage}/torch_dense_gemm.py" \
  --mode "${mode}" --device "${device}" --sizes "${sizes[@]}" \
  --warmup "${warmup}" --iterations "${iterations}" \
  --output "${container_result}" >"${host_evidence}/command.sh"
printf '\n' >>"${host_evidence}/command.sh"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
if [[ "${platform}" == bw3000 ]]; then
  docker exec \
    -e "${visibility_name}=${device_id}" \
    "${container}" bash -lc \
    "set -eo pipefail; set +u; source /opt/dtk/env.sh; set -u; exec '${python_bin}' '${container_stage}/torch_dense_gemm.py' --mode '${mode}' --device '${device}' --sizes ${sizes[*]} --warmup '${warmup}' --iterations '${iterations}' --output '${container_result}'" \
    >"${host_evidence}/stdout.txt" 2>"${host_evidence}/stderr.txt"
else
  docker exec \
    -e "${visibility_name}=${device_id}" \
    "${container}" "${python_bin}" "${container_stage}/torch_dense_gemm.py" \
    --mode "${mode}" --device "${device}" --sizes "${sizes[@]}" \
    --warmup "${warmup}" --iterations "${iterations}" \
    --output "${container_result}" \
    >"${host_evidence}/stdout.txt" 2>"${host_evidence}/stderr.txt"
fi
rc=$?
set -e
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
"${telemetry_after[@]}" >"${host_evidence}/device-after.txt" 2>&1
set -e

case "${platform}" in
  h100)
    nvidia-smi -i "${device_id}" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits \
      >"${host_evidence}/processes-after.txt"
    [[ ! -s "${host_evidence}/processes-after.txt" ]] || rc=90
    ;;
  910c)
    awk -v card="${device_id}" '
      /Process id/ { in_processes=1; next }
      in_processes && $1 == "|" && $2 == card && $3 ~ /^[01]$/ { print }
    ' "${host_evidence}/device-after.txt" \
      >"${host_evidence}/processes-after.txt"
    [[ ! -s "${host_evidence}/processes-after.txt" ]] || rc=90
    ;;
  bw3000)
    docker exec "${container}" bash -lc \
      'set +u; source /opt/dtk/env.sh >/dev/null 2>&1; /opt/dtk/bin/rocm-smi --showpids' \
      >"${host_evidence}/processes-after.txt" 2>&1 || true
    grep -q 'No KFD PIDs currently running' \
      "${host_evidence}/processes-after.txt" || rc=90
    ;;
  s5000)
    mthreads-gmi >"${host_evidence}/processes-after.txt" 2>&1
    ;;
  c550)
    mx-smi >"${host_evidence}/processes-after.full.txt" 2>&1
    awk -v gpu="${device_id}" '
      /^\| Process:/ { in_processes=1; next }
      in_processes && $1 == "|" && $2 == gpu { print }
    ' "${host_evidence}/processes-after.full.txt" \
      >"${host_evidence}/processes-after.txt"
    [[ ! -s "${host_evidence}/processes-after.txt" ]] || rc=90
    ;;
  810e)
    cp "${host_evidence}/device-after.txt" \
      "${host_evidence}/processes-after.txt"
    grep -q 'No running processes found' \
      "${host_evidence}/processes-after.txt" || rc=90
    ;;
esac

if docker exec "${container}" test -f "${container_result}"; then
  docker cp "${container}:${container_result}" "${host_evidence}/result.json"
fi
printf 'platform=%s\nrun_id=%s\nmode=%s\ndevice_id=%s\nsource_sha256=%s\nstarted_at=%s\nfinished_at=%s\nrc=%s\n' \
  "${platform}" "${run_id}" "${mode}" "${device_id}" \
  "${expected_source_sha256}" "${started_at}" "${finished_at}" "${rc}" \
  >"${host_evidence}/run.env"
(
  cd "${host_evidence}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' \
    | sort -z \
    | while IFS= read -r -d '' name; do
        sha256sum "${name}"
      done >SHA256SUMS
)
printf '%s\n' "${rc}" >"${host_stage}/rc"
exit "${rc}"
