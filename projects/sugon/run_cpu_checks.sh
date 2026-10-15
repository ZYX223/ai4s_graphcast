#!/usr/bin/env bash
# Isolate the DCU-specific Triton package only in a disposable CPU test container.
set -Eeuo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$repo/projects/sugon/runtime-lock.json")
if [[ $# -eq 0 ]]; then set -- tests; fi
docker run --rm --network=host --cpus=8 --memory=16g --shm-size=2g \
  -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
  -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
  -v "$repo:/workspace:ro" --entrypoint bash "$image" -lc '
    set -e
    source /opt/dtk/env.sh
    mv /usr/local/lib/python3.10/dist-packages/triton /tmp/triton-dcu-only
    mkdir /tmp/checkout
    cp -a /workspace/ai4s /tmp/checkout/
    cd /tmp/checkout/ai4s/graphcast
    export PYTHONPATH=/tmp/checkout/ai4s/graphcast/src
    python -m pytest "$@" -q -p no:cacheprovider --tb=short
  ' bash "$@"
