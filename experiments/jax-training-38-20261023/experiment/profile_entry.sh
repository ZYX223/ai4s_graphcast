#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONPATH=/experiment:/official:/dependencies/python
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
exec /opt/graphcast-jax/bin/python3 /experiment/train.py "$@"
