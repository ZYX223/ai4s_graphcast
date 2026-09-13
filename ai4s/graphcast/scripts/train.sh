#!/usr/bin/env bash
# Formal GraphCast training; arguments match scripts/training/run_training.py.
# Legacy SGD probe options must not silently become a different training recipe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for legacy_name in DEVICES STEPS LR OPTIMIZER TRAIN_SCOPE ACTIVATION_CHECKPOINTING \
    GRID2MESH_NODE_CHUNK_SIZE MESH2GRID_EDGE_CHUNK_SIZE MESH2GRID_NODE_CHUNK_SIZE \
    MESH2GRID_DECODER_CHUNK_SIZE MESH2GRID_GRID_PARTITIONS SAVE_CHECKPOINT \
    RUN_NAME REPORT CHECKPOINT GRAPHCAST_BACKEND GRAPHCAST_FLAGOS_MODE \
    GRAPHCAST_FLAGGEMS_CONFIG GRAPHCAST_FLAGGEMS_OPS GRAPHCAST_FLAGGEMS_VARIANT; do
    if [[ -n "${!legacy_name:-}" ]]; then
        printf 'Legacy training variable %s is set. Unset legacy variables and use the formal CLI, or run scripts/training/legacy/train_mp.sh. See scripts/training/README.md.\n' "$legacy_name" >&2
        exit 2
    fi
done
exec bash "$ROOT/scripts/training/train_pipeline.sh" "$@"
