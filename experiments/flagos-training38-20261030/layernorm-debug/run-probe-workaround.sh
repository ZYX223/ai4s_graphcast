#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
base=/tmp/zyx-training-memory-analysis-20261030
root=/tmp/zyx-training38-layernorm-debug
export PYTHONPATH="$base/deps:/tmp/zyx-training38-experiment:/tmp/zyx-flagos-rne-inference-20261029/baseline/src"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export ROCR_VISIBLE_DEVICES=11 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=rocm JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_ENABLE_X64=false
export XLA_MODE=native XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export LN_OUTPUT="$root/$1"
mkdir -p "$LN_OUTPUT"
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0 --xla_disable_hlo_passes=horizontal-loop-fusion-for-copy --xla_dump_to=$LN_OUTPUT/hlo --xla_dump_hlo_as_text"
python "$root/probe.py"
