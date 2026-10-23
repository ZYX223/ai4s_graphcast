#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/experiment:/repo/src:/repo/third_party/graphcast
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export GRAPHCAST_ROUTER_LIBRARY=/output/native/libexperiment.so
export GRAPHCAST_STAGE_WEIGHTS=/repo/kernels/xf32_rne/weights.tsv
export GRAPHCAST_STAGE_LOG=/output/gemm.jsonl
exec /opt/graphcast-jax/bin/python3 /experiment/probe.py --profile
