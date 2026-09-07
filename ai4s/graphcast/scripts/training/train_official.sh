#!/usr/bin/env bash
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# One process owns four model-parallel cards. DP=2 is launched explicitly with
# torchrun (see README); this wrapper does not alter visibility or environments.
exec python "${TRAINING_SCRIPT_DIR}/run_training.py" "$@"
