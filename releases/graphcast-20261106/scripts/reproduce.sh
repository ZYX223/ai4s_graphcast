#!/usr/bin/env bash
# Complete the documented build, inference and 38-update training in one session.
set -euo pipefail
ulimit -c 0
: "${BUILD_ROOT:?请通过 environment/container.sh 运行}"
: "${CONFIG_PATH:?请通过 environment/container.sh 运行}"
: "${RUN_DIR:?请通过 environment/container.sh 运行}"
exec > >(tee "$RUN_DIR/reproduce.log") 2>&1

selection=$(python -m graphcast_bench.platforms.device "$@")
read -r DEVICE DEVICE_PCI <<< "$selection"
export DEVICE DEVICE_PCI

for kind in inference training; do
  if [[ ! -d $BUILD_ROOT/$kind ]]; then
    python scripts/build_kernels.py --kind "$kind" --hipcc "$HIPCC" \
      --include "$CONDA_PREFIX/include" --library-dir "$CONDA_PREFIX/lib" \
      --output "$BUILD_ROOT/$kind"
  fi
done
python - <<'PY'
import os
from pathlib import Path
from graphcast_bench.platforms.native import verify_native
build = Path(os.environ["BUILD_ROOT"])
verify_native(build / "inference/libxf32_rne_router.so", "inference")
verify_native(build / "training/training-router-candidate.so", "training")
PY

python scripts/inference.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/smoke-inference" --profile single
python scripts/validate.py completion "$RUN_DIR/smoke-inference" single
python scripts/train.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/smoke-training" --rollout 1 --limit 2
python scripts/validate.py completion "$RUN_DIR/smoke-training" training-smoke

python scripts/inference.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
  --output "$RUN_DIR/inference" --profile full40
python scripts/validate.py completion "$RUN_DIR/inference" full40
python scripts/validate.py inference --candidate "$RUN_DIR/inference/run" \
  --reference /data/h100-reference --steps 40 --output "$RUN_DIR/inference-comparison"

for horizon in {1..12}; do
  printf -v tag 'r%02d' "$horizon"
  resume_args=()
  validation_args=()
  if [[ $horizon -eq 1 ]]; then
    updates="$(seq -s, 1 16)"
  else
    printf -v previous 'r%02d' "$((horizon - 1))"
    checkpoint="$RUN_DIR/train-$previous/checkpoint.pkl"
    resume_args=(--resume "$checkpoint")
    validation_args=(--predecessor "$checkpoint")
    first_update=$((2 * horizon + 13))
    updates="$first_update,$((first_update + 1))"
  fi
  python scripts/train.py --paths "$CONFIG_PATH" --device "$DEVICE" --pci "$DEVICE_PCI" \
    --output "$RUN_DIR/train-$tag" --rollout "$horizon" "${resume_args[@]}"
  python scripts/validate.py completion "$RUN_DIR/train-$tag" training-stage \
    --updates "$updates" --rollout "$horizon" "${validation_args[@]}"
done

comparison_failed=0
for horizon in {1..12}; do
  printf -v tag 'r%02d' "$horizon"
  python scripts/validate.py training-loss "$RUN_DIR/train-$tag" \
    --reference-root /data/h100-reference --output "$RUN_DIR/train-$tag-loss.json" \
    || comparison_failed=1
done
test "$comparison_failed" -eq 0
printf '完整推理、38 次训练更新及精度对照完成。结果：%s\n' "$RUN_DIR"
