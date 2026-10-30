#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
root=/tmp/zyx-flagos-training38-20261030
base=/tmp/zyx-training-memory-analysis-20261030
export TRAINING_OUTPUT="$root/$1"
shift
mkdir -p "$TRAINING_OUTPUT"
export TRAINING_PLATFORM=hygon TRAINING_DATA="$base/data"
export TRAINING_WEEKLY=/work/graphcast-strict-inference-cftime-20261027/data
export TRAINING_ROUTER="$base/training-router.so" TRAINING_GEMM_LOG="$TRAINING_OUTPUT/gemm.jsonl"
export PYTHONPATH="$base/deps:$base/official:$root/hygon-policy:$root/experiment"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$TRAINING_ROUTER"
unset ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES=10 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export PHYSICAL_DEVICE=10 EXPECTED_PCI=0000:99:00.0
export XLA_MODE=fl_only XLA_FLAGTREE_COMPILE_WORKERS=12 XLA_FLAGTREE_PERSISTENT_WORKERS=12
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
# Measured horizon9 peak98.63GiB leaves1.81GiB in the70% pool.
# Horizons11/12 need longer live inputs; reserve73% only for these stages.
case "$(basename "$TRAINING_OUTPUT")" in
  flagos-r11|flagos-r12) export XLA_PYTHON_CLIENT_MEM_FRACTION=0.73 ;;
esac
printf '%s\n' "$XLA_PYTHON_CLIENT_MEM_FRACTION" > "$TRAINING_OUTPUT/jax-memory-fraction.txt"
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_LOG=0
export TMPDIR="$TRAINING_OUTPUT/tmp"
mkdir -p "$TMPDIR"
export TRAINING_REMAT_SHA256=28d4a02d1401d154d77e510cedcbf91df126f3a9beaa3d2a0743684b4da49a86
export MEMORY_COMPILE_ONLY=0 TRAINING_JAX_TRACE=1
export PYTHONFAULTHANDLER=1
bash /tmp/zyx-training-minimal-20261030/run-child.sh python -X faulthandler "$root/entry.py" "$@"
