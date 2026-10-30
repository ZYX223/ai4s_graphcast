#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
base=/tmp/zyx-training-memory-analysis-20261030
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$base/deps:/tmp/zyx-training38-experiment:/tmp/zyx-flagos-rne-inference-20261029/baseline/src"
export TRAINING_ROUTER="$base/training-router.so" LD_PRELOAD="$base/training-router.so"
export ROCR_VISIBLE_DEVICES="${LN_CARD:-11}" HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0 --xla_disable_hlo_passes=horizontal-loop-fusion-for-copy'
export XLA_FLAGTREE_COMPILE_WORKERS=8 XLA_FLAGTREE_PERSISTENT_WORKERS=8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export XLA_MODE="$1" OPERATOR_ROOT=/tmp/zyx-training38-layernorm-debug/bench-idle
python /tmp/zyx-training38-layernorm-debug/resource_gate.py "$ROCR_VISIBLE_DEVICES"
# Derive the exact ten captured LayerNorm IDs, including all backward shapes.
export ONLY_CASES="$(python -c 'from graphcast_bench.logical_bench import Inventory; from operator_replay import training_cases; from pathlib import Path; c,_=training_cases(Inventory(Path("/tmp/zyx-training38-operators/logical"))); ids=[x["id"] for x in c if x["operator"]=="layer_norm"]; assert len(ids)==10; print(",".join(ids))')"
export TMPDIR="$OPERATOR_ROOT/tmp-$XLA_MODE" TRAINING_GEMM_LOG="$OPERATOR_ROOT/gemm-$XLA_MODE.jsonl"
mkdir -p "$TMPDIR"
ln -sfn /tmp/zyx-training38-operators/logical "$OPERATOR_ROOT/logical"
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["OPERATOR_ROOT"]); (p/("config-"+os.environ["XLA_MODE"]+".json")).write_text(json.dumps({k:os.environ[k] for k in ["XLA_MODE","XLA_FLAGS","ONLY_CASES","ROCR_VISIBLE_DEVICES","XLA_PYTHON_CLIENT_MEM_FRACTION","JAX_DEFAULT_MATMUL_PRECISION"]},indent=2)+"\n")'
python /tmp/zyx-training38-layernorm-debug/resource_gate.py "$ROCR_VISIBLE_DEVICES"
python /tmp/zyx-training38-experiment/bench_training_ops.py
