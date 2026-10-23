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
  cd /repo
  "$py" -m unittest discover -s tests -v
  exit
fi
"$py" /experiment/patch_router.py
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O2 -fPIC -shared \
  -I/repo/kernels/xf32_rne /output/native/xf32_rne_router.cc \
  /repo/kernels/xf32_rne/rne_kernel.hip -o /output/native/libexperiment.so -lcrypto -ldl -pthread
export GRAPHCAST_ROUTER_LIBRARY=/output/native/libexperiment.so
export GRAPHCAST_STAGE_WEIGHTS=/repo/kernels/xf32_rne/weights.tsv
export GRAPHCAST_STAGE_LOG=/dev/null
if [[ "$action" == profile ]]; then export GRAPHCAST_STAGE_LOG=/output/gemm.jsonl; fi
export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY:$LD_PRELOAD"
if [[ "$action" == profile ]]; then
  # DTK hipprof crashes before exec under the full Python/XLA environment.
  # Restore these exact values in the child, after the profiler has started.
  env -u XLA_FLAGS -u PYTHONPATH -u GRAPHCAST_STAGE_LOG -u GRAPHCAST_STAGE_WEIGHTS -u GRAPHCAST_ROUTER_LIBRARY \
    /opt/dtk/bin/hipprof --hip-trace --trace-off --output-type 0 -o /output/trace \
    bash /experiment/profile_entry.sh
else
  "$py" /experiment/test_rne_ops.py
  "$py" /experiment/probe.py
fi
"$py" -c 'import json; assert json.load(open("/output/report.json"))["status"] == "completed"'
