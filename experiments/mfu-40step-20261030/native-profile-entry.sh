#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/experiment:/repo/src:/repo/third_party/graphcast
export TMPDIR=/output/tmp
exec /opt/graphcast-jax/bin/python3 /measurement/profile40.py \
 --platform hygon --mode rne_xf32 --variant fused --data /data \
 --output /output/run --profile --repeats 3
