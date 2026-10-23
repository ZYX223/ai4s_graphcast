#!/usr/bin/env bash
set -Eeuo pipefail
trap 'chown -R "$OUTPUT_UID:$OUTPUT_GID" /output' EXIT
unset PYTHONPATH LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE
unset JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION
set +u
source /opt/dtk/env.sh
set -u
export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTHONPATH=/experiment:/repo/src:/repo/third_party/graphcast
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
py=/opt/graphcast-jax/bin/python3
action=$1
if [[ "$action" == check ]]; then
  export JAX_PLATFORMS=cpu
  "$py" /experiment/test_rne_ops.py
  "$py" /experiment/test_model_hook.py
  cd /repo
  "$py" -m unittest discover -s tests -v
  exit
fi
if [[ "$EXPERIMENT_VARIANT" == fused ]]; then
  "$py" /experiment/build_candidate.py
else
  "$py" /repo/scripts/build_kernel.py --output /output/native
fi
export GRAPHCAST_ROUTER_LIBRARY=/output/native/libxf32_rne_router.so
export GRAPHCAST_STAGE_WEIGHTS=/repo/kernels/xf32_rne/weights.tsv
export GRAPHCAST_FUSED_LAYERS=/experiment/fused_layers.txt
export GRAPHCAST_STAGE_LOG=/dev/null
export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY:$LD_PRELOAD"
if [[ "$action" == profile ]]; then
  env -u XLA_FLAGS -u PYTHONPATH -u GRAPHCAST_STAGE_LOG -u GRAPHCAST_STAGE_WEIGHTS -u GRAPHCAST_ROUTER_LIBRARY -u GRAPHCAST_FUSED_LAYERS \
    /opt/dtk/bin/hipprof --hip-trace --trace-off --output-type 0 -o /output/trace \
    bash /experiment/profile_entry.sh
else
  "$py" /experiment/entry.py --platform hygon --mode rne_xf32 --variant "$EXPERIMENT_VARIANT" --data /data --output /output/run --repeats "$EXPERIMENT_REPEATS"
fi
"$py" -c 'import json; assert json.load(open("/output/run/report.json"))["status"] == "completed"'
