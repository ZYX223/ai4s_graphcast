#!/usr/bin/env bash
# GraphCast 25km training entry point (default: 4-GPU model parallel)
# Usage: bash scripts/train.sh
# Env vars: DEVICES, STEPS, GRAPHCAST_FLAGOS_MODE, GRAPHCAST_FLAGGEMS_CONFIG
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/training/train_mp.sh" "$@"
