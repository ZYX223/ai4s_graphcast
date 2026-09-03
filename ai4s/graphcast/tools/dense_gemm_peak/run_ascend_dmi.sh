#!/usr/bin/env bash
set -euo pipefail

output_root="${1:?output root is required}"
device_id="${2:?device id is required}"
tool=/usr/local/Ascend/toolbox/latest/Ascend-DMI/bin/ascend-dmi

test ! -e "${output_root}"
mkdir -p "${output_root}"

"${tool}" --version >"${output_root}/version.txt" 2>&1 || true
"${tool}" -f -h >"${output_root}/help.txt" 2>&1
npu-smi info >"${output_root}/device-before.txt"

for mode in fp32 hf32 bf16; do
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  "${tool}" -f -d "${device_id}" -t "${mode}" --fmt json -q \
    >"${output_root}/${mode}.json" \
    2>"${output_root}/${mode}.stderr"
  rc=$?
  set -e
  printf '%s\t%s\t%s\n' "${mode}" "${rc}" "${started_at}" \
    >>"${output_root}/status.tsv"
  test "${rc}" -eq 0
done

npu-smi info >"${output_root}/device-after.txt"
find "${output_root}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${output_root}/SHA256SUMS"
