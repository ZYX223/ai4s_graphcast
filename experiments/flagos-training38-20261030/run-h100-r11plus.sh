#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTHONPATH=/minimal/deps:/minimal/official:/experiment/h100-policy:/experiment/experiment
export TRAINING_PLATFORM=h100 TRAINING_OUTPUT=/output TRAINING_WEEKLY=/weekly TRAINING_DATA=/training-data
export PHYSICAL_DEVICE=2 EXPECTED_PCI=0000:3a:00.0
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.96
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
"$GRAPHCAST_PYTHON" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["jax","jaxlib","optax","chex","dm-haiku","numpy","xarray"]})'
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
 --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --output=/output/train-trace \
 "$GRAPHCAST_PYTHON" /experiment/experiment/train.py "$@"
