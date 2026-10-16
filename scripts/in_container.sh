#!/usr/bin/env bash
set -Eeuo pipefail
trap 'chown -R "$OUTPUT_UID:$OUTPUT_GID" /output' EXIT
unset PYTHONPATH LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE
unset JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTHONPATH=/repo/src:/repo/third_party/graphcast
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
action=$1
shift
py=$GRAPHCAST_PYTHON
if [[ "$GRAPHCAST_PLATFORM" == hygon ]]; then
  set +u
  source /opt/dtk/env.sh
  set -u
  export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
  export JAX_PLATFORMS=rocm
else
  export JAX_PLATFORMS=cuda
fi
case "$action" in
  check)
    export JAX_PLATFORMS=cpu
    cd /repo
    "$py" -m unittest discover -s tests -v
    exit ;;
  prepare)
    export JAX_PLATFORMS=cpu
    "$py" /repo/scripts/prepare_data.py --case /source-case --baseline /source-reference.npz --output /output/data
    exit ;;
  compare)
    export JAX_PLATFORMS=cpu
    "$py" /repo/scripts/compare.py --candidate /candidate.nc --reference "/reference$GRAPHCAST_REFERENCE_SUFFIX" --data /data --output /output/comparison.json
    exit ;;
esac
if [[ "$GRAPHCAST_PLATFORM" == hygon ]]; then
  "$py" /repo/scripts/build_kernel.py --output /output/native
  export GRAPHCAST_ROUTER_LIBRARY=/output/native/libxf32_rne_router.so
  export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY:$LD_PRELOAD"
  export GRAPHCAST_STAGE_WEIGHTS=/repo/kernels/xf32_rne/weights.tsv
  export GRAPHCAST_STAGE_LOG=/dev/null
fi
case "$action" in
  probe)
    if [[ "$GRAPHCAST_PLATFORM" == hygon ]]; then
      "$py" /repo/scripts/probe_hygon.py /output/probe
    else
      "$py" /repo/scripts/probe_h100.py --output-dir /output/probe
    fi ;;
  benchmark)
    "$py" /repo/scripts/entry.py --platform "$GRAPHCAST_PLATFORM" --data /data --output /output/run "$@" ;;
  profile)
    [[ "$GRAPHCAST_PLATFORM" == h100 ]]
    nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --output=/output/model-trace \
      "$py" /repo/scripts/entry.py --platform h100 --data /data --output /output/run --profile "$@" ;;
  *) exit 2 ;;
esac
