#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
root=/tmp/zyx-mfu-40step-20261030
candidate=/tmp/zyx-flagos-rne-gemm-ex-20261030
frozen=/tmp/zyx-flagos-rne-inference-20261029
out=$root/hygon-profile
mkdir "$out"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONPATH="$candidate/model-experiment:$frozen/baseline/src:$frozen/baseline/third_party/graphcast"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES=9 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GRAPHCAST_PLATFORM=hygon GRAPHCAST_PHYSICAL_DEVICE=9 GRAPHCAST_EXPECTED_PCI=0000:88:00.0
export XLA_MODE=fl_only XLA_FLAGTREE_COMPILE_WORKERS=12 XLA_FLAGTREE_PERSISTENT_WORKERS=12
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TMPDIR="$out/tmp"
mkdir "$TMPDIR"
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_LOG=0
export GRAPHCAST_ROUTER_LIBRARY="$candidate/native-r1/libxf32_rne_router.so"
export GRAPHCAST_STAGE_WEIGHTS="$frozen/baseline/kernels/xf32_rne/weights.tsv"
export GRAPHCAST_FUSED_LAYERS="$candidate/model-experiment/fused_layers.txt"
export GRAPHCAST_STAGE_LOG="$out/gemm.jsonl"
export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY"
export GRAPHCAST_FROZEN_BENCHMARK="$candidate/model-experiment/full_model.py"
export GRAPHCAST_PROFILE_EVIDENCE="$out"
/public/software/compiler/dtk-26.04/bin/hipprof --hip-trace --trace-off --output-type 0 -o "$out/trace" \
 python "$root/profile40.py" --platform hygon --mode rne_xf32 --variant fused \
 --data /work/graphcast-strict-inference-cftime-20261027/data --output "$out/run" --profile --repeats 3
