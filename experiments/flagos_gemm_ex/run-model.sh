#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
root=/tmp/zyx-flagos-rne-gemm-ex-20261030
frozen=/tmp/zyx-flagos-rne-inference-20261029
mode=${1:?native or fl_only}; action=${2:?single or bench}; label=${3:?run label}
[[ $mode == native || $mode == fl_only ]]
[[ $action == single || $action == bench ]]
out=$root/$label
mkdir "$out"
export LD_LIBRARY_PATH="/opt/hyhal/lib:$LD_LIBRARY_PATH"
unset LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONPATH="$root/model-experiment:$frozen/baseline/src:$frozen/baseline/third_party/graphcast"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export ROCR_VISIBLE_DEVICES=8 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export GRAPHCAST_PHYSICAL_DEVICE=8 GRAPHCAST_EXPECTED_PCI=0000:85:00.0
export XLA_MODE=$mode XLA_FLAGTREE_COMPILE_WORKERS=12 XLA_FLAGTREE_PERSISTENT_WORKERS=12
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TMPDIR="$out/tmp"
mkdir "$TMPDIR"
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export XLA_FLAGTREE_LOG=0
if [[ $action == single ]]; then
 export XLA_FLAGTREE_LOG=1
 export XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$out/hlo --xla_dump_hlo_as_text"
fi
export GRAPHCAST_ROUTER_LIBRARY="$root/native-r1/libxf32_rne_router.so"
export GRAPHCAST_STAGE_WEIGHTS="$frozen/baseline/kernels/xf32_rne/weights.tsv"
export GRAPHCAST_FUSED_LAYERS="$root/model-experiment/fused_layers.txt"
export GRAPHCAST_STAGE_LOG="$out/gemm.jsonl"
export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY"
args=()
[[ $action != single ]] || args+=(--profile)
launcher=(python)
if [[ $action == single ]]; then
 launcher=(/public/software/compiler/dtk-26.04/bin/hipprof --hip-trace --trace-off --output-type 0 -o "$out/trace" python)
fi
"${launcher[@]}" "$root/model-experiment/entry.py" --platform hygon --mode rne_xf32 --variant fused --data /work/graphcast-strict-inference-cftime-20261027/data --output "$out/run" --repeats 3 "${args[@]}"
