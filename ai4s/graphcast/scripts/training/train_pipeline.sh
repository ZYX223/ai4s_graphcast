#!/usr/bin/env bash
# Public platform-aware adapter; policies live in training_platform.py.
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${TRAINING_SCRIPT_DIR}/run_training.py" "$@"
