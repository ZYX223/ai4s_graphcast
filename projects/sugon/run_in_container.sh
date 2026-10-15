#!/usr/bin/env bash
set -Eeuo pipefail
trap 'chown -R "$GRAPHCAST_OUTPUT_UID:$GRAPHCAST_OUTPUT_GID" /outputs' EXIT
set +u
source /opt/dtk/env.sh
set -u
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export TRITON_CACHE_DIR=/outputs/cache/triton FLAGGEMS_CACHE_DIR=/outputs/cache/flaggems
export AMDGCN_USE_BUFFER_OPS=0
project=/workspace/ai4s/graphcast
export PYTHONPATH="/opt/graphcast/source:/opt/graphcast/source/flaggems/src:$project/src"
export GRAPHCAST_FLAGOS_MODE=flaggems GRAPHCAST_STRICT_FLAGOS=1
export GRAPHCAST_FLAGGEMS_SOURCE_ROOT=/opt/graphcast/source/flaggems
export GRAPHCAST_FLAGGEMS_CONFIG="$project/configs/flaggems/graphcast_inference.yaml"
export GRAPHCAST_FLAGGEMS_OPS=addmm,cat,native_layer_norm,index_add_,silu,index,add
export GRAPHCAST_FLAGGEMS_RECORD_PATH=/outputs/flaggems_dispatch.log
export GRAPHCAST_FUSE_FACTORIZED_ADD_ADD_SILU=1 GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL=1 GRAPHCAST_FUSE_ADDMM_SILU=1
unset GRAPHCAST_FUSE_PACKED_MLP NVIDIA_TF32_OVERRIDE TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
mkdir -p "$TRITON_CACHE_DIR" "$FLAGGEMS_CACHE_DIR"
cd /opt/graphcast/source/flaggems
sha256sum --quiet -c /workspace/projects/sugon/flaggems-snapshot.sha256
sha256sum --quiet -c /workspace/projects/sugon/flaggems-overlay.sha256
cd /opt/graphcast
sha256sum --quiet -c /workspace/projects/sugon/assets.sha256
cd /workspace
cd /workspace/projects/sugon/flaggems-overlay
sha256sum --quiet -c ../flaggems-overlay.sha256
cd /workspace
python /workspace/projects/sugon/probe_runtime.py
if [[ "${GRAPHCAST_RUN_MODE:-inference}" == operators ]]; then
  python -m pytest -q -p no:cacheprovider \
    --junitxml=/outputs/operator-tests.xml \
    /workspace/projects/sugon/flaggems-overlay/tests/test_hygon_graphcast_fusions.py \
    /workspace/projects/sugon/flaggems-overlay/tests/test_hygon_addmm_triton_route.py \
    /workspace/projects/sugon/flaggems-overlay/tests/test_hygon_index_add_trusted.py \
    /workspace/projects/sugon/tests/test_inference_fused_kernel.py
  exit 0
fi
case_dir=/opt/graphcast/case/assets
python "$project/scripts/inference/run_operational_40step_inference.py" \
  --config "$case_dir/config.yaml" --weights "$case_dir/weights.pt" \
  --inputs "$case_dir/inputs.nc" --rollout-forcings "$case_dir/rollout_forcings.nc" \
  --target-template "$case_dir/target_template.nc" --stats-dir "$case_dir/stats" \
  --platform bw3000 --device cuda:0 --model-execution-profile domestic-v1 \
  --precision tf32 --steps 40 --step-hours 6 --warmup-calls 3 --device-count 1 \
  --peak-tflops-per-device 0 --output /outputs/prediction.nc --report /outputs/report.json \
  --fuse-post-layer-norm-residual
python /workspace/projects/sugon/validate_report.py /outputs hygon
python "$project/scripts/inference/diagnose_inference.py" \
  --run-dir /outputs --baseline /opt/graphcast/baseline/baseline.npz \
  --baseline-metadata /opt/graphcast/baseline/baseline.json --output-dir /outputs/acceptance
echo GRAPHCAST_40STEP_AND_REFERENCE_CHECK_PASSED
