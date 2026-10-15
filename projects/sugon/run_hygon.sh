#!/usr/bin/env bash
# Run the checked-out GraphCast code against the pinned local Hygon runtime.
set -Eeuo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/outputs/sugon-$(date -u +%Y%m%dT%H%M%SZ)"}
device=${2:-9}
mode=${3:-inference}
[[ "$mode" == inference || "$mode" == operators ]] || { echo 'Mode must be inference or operators' >&2; exit 2; }
[[ "$device" =~ ^[0-9]+$ ]] || { echo 'Device must be a physical card index' >&2; exit 2; }
expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$repo/projects/sugon/runtime-lock.json")
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$repo/projects/sugon/runtime-lock.json")
[[ $(docker image inspect --format '{{.Id}}' "$image") == "$expected" ]]
mkdir -p "$(dirname "$out")"
mkdir "$out" # Never overwrite an existing run.
out=$(cd "$out" && pwd)
exec > >(tee "$out/launch.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/launch.rc"' EXIT
/opt/hyhal/bin/hy-smi -d "$device" --showuse --showmeminfo vram --showbus --json > "$out/device-before.json"
python3 - "$out/device-before.json" "$device" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))['card' + sys.argv[2]]
assert float(d['HCU use (%)']) == 0, 'Selected card is busy'
assert float(d['vram Total Used Memory (MiB)']) < 256, 'Selected card has allocated memory'
PY
# hy-smi/ROCR indices differ from Linux DRM card indices on this machine.
pci=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["card"+sys.argv[2]]["PCI Bus"].split()[0].lower())' "$out/device-before.json" "$device")
[[ "$pci" =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]$ ]]
git -C "$repo" rev-parse HEAD > "$out/checkout-commit.txt"
git -C "$repo" status --porcelain > "$out/checkout-status.txt"
printf '%s\n' "$expected" > "$out/image-id.txt"
docker run --rm --network=none --device=/dev/kfd --device=/dev/dri \
  --ipc=host --cpus=16 --memory=48g \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /opt/hyhal:/opt/hyhal:ro -v "$repo:/workspace:ro" -v "$out:/outputs" \
  -e HOME=/outputs -e ROCR_VISIBLE_DEVICES="$device" -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -e GRAPHCAST_EXPECTED_PCI="$pci" -e GRAPHCAST_OUTPUT_UID="$(id -u)" -e GRAPHCAST_OUTPUT_GID="$(id -g)" \
  -e GRAPHCAST_RUN_MODE="$mode" \
  --entrypoint bash "$expected" -c '
    cp /workspace/projects/sugon/run_in_container.sh /outputs/executed-run-in-container.sh
    exec bash /outputs/executed-run-in-container.sh
  '
/opt/hyhal/bin/hy-smi -d "$device" --showuse --showmeminfo vram --json > "$out/device-after.json"
