#!/usr/bin/env bash
# Invoked by container.sh inside the fixed runtime.
set -euo pipefail
umask 022
export PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export HIPCC=/public/software/compiler/dtk-26.04/bin/hipcc
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src:$PWD/third_party/graphcast"
python - <<'PY'
import importlib.metadata as m
import sys
assert sys.version_info[:2] == (3, 10), sys.version
for name, expected in (("jax", "0.6.0"), ("jaxlib", "0.6.0"), ("dm-haiku", "0.0.14")):
    actual = m.version(name)
    assert actual == expected, (name, actual, expected)
PY
python scripts/validate.py release "$PWD"
python scripts/validate.py code
source_id=$(sha256sum configs/runtime/code-manifest.json)
version="$(basename -- "$PWD")-${source_id:0:12}"
export BUILD_ROOT="/workspace/$version/build"
export CONFIG_PATH="/workspace/$version/paths.json"
mkdir -p "$BUILD_ROOT" "/workspace/$version/runs"
export RUN_DIR
RUN_DIR=$(mktemp -d "/workspace/$version/runs/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
python - <<'PY'
import json
import os
from pathlib import Path
paths = json.loads(Path("environment/paths.example.json").read_text())
build = Path(os.environ["BUILD_ROOT"])
paths["inference_router"] = str(build / "inference/libxf32_rne_router.so")
paths["training_router"] = str(build / "training/training-router-candidate.so")
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(paths, indent=2) + "\n")
PY
printf '源码: %s\n构建: %s\n本次结果: %s\n' "$PWD" "$BUILD_ROOT" "$RUN_DIR"
if [[ $# -gt 0 ]]; then exec "$@"; fi
exec /bin/bash --noprofile --norc
