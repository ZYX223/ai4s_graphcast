#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
umask 077
replay_root=/tmp/zyx-flagos-rne-gemm-ex-20261030
frozen_root=/tmp/zyx-flagos-logical-operators-20261029
replay_mode=${1:?native or fl_only}
[[ $replay_mode == native || $replay_mode == fl_only ]]
replay_run_dir="$replay_root/${2:-$replay_mode}"
mkdir "$replay_run_dir"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE
export PYTHONPATH="$frozen_root/source/src:$frozen_root/source/third_party/graphcast"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export ROCR_VISIBLE_DEVICES=8 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export CPU_INPUTS=1 JAX_PLATFORMS=rocm,cpu JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export XLA_MODE="$replay_mode" XLA_FLAGTREE_COMPILE_WORKERS=8 XLA_FLAGTREE_PERSISTENT_WORKERS=8
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0 --xla_dump_to=$replay_run_dir/hlo --xla_dump_hlo_as_text"
export XLA_FLAGTREE_LOG=0
export GRAPHCAST_STAGE_WEIGHTS="$frozen_root/source/kernels/xf32_rne/weights.tsv"
export GRAPHCAST_STAGE_LOG="$replay_run_dir/gemm.jsonl"
export GRAPHCAST_REPLAY_OUTPUT="$replay_run_dir"
export ROCBLAS_LAYER=1 ROCBLAS_LOG_TRACE_PATH="$replay_run_dir/rocblas-trace.log"
export LD_PRELOAD="$replay_root/native-r1/libxf32_rne_router.so"
python "$replay_root/replay-candidate.py"

