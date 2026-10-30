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
ulimit -c 0
export XLA_MODE=fl_only OPERATOR_ROOT=/tmp/zyx-training38-operators-retry
export ONLY_CASES=a03a982301e4fd5a
export TMPDIR="$OPERATOR_ROOT/tmp" TRAINING_GEMM_LOG="$OPERATOR_ROOT/gemm.jsonl"
mkdir -p "$TMPDIR"
ln -s /tmp/zyx-training38-operators/logical "$OPERATOR_ROOT/logical"
ln -s /tmp/zyx-training38-operators/bench-native "$OPERATOR_ROOT/bench-native"
python /tmp/zyx-training38-experiment/bench_training_ops.py
