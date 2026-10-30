#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTHONPATH=/repo/src:/repo/third_party/graphcast
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export GRAPHCAST_FROZEN_BENCHMARK=/repo/src/graphcast_bench/benchmark.py
export GRAPHCAST_PROFILE_EVIDENCE=/output
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
 --capture-range=cudaProfilerApi --capture-range-end=repeat:3 --export=sqlite --output=/output/model-trace \
 "$GRAPHCAST_PYTHON" /measure/profile40.py --platform h100 --mode tf32 --data /data --output /output/run --profile --repeats 3
