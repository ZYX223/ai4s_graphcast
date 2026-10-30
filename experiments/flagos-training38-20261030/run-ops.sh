#!/usr/bin/env bash
set -Eeuo pipefail
base=/tmp/zyx-training-memory-analysis-20261030
export LD_LIBRARY_PATH="/opt/hyhal/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$base/deps:/tmp/zyx-training38-experiment:/tmp/zyx-flagos-rne-inference-20261029/baseline/src"
export TRAINING_ROUTER="$base/training-router.so" LD_PRELOAD="$base/training-router.so"
export ROCR_VISIBLE_DEVICES=11 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_COMPILE_WORKERS=8 XLA_FLAGTREE_PERSISTENT_WORKERS=8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
for mode in native fl_only; do
 export XLA_MODE="$mode" TRAINING_GEMM_LOG="/tmp/zyx-training38-operators/gemm-$mode.jsonl"
 export TMPDIR="/tmp/zyx-training38-operators/tmp-$mode"
 mkdir -p "$TMPDIR"
 python /tmp/zyx-training38-experiment/bench_training_ops.py
 done
