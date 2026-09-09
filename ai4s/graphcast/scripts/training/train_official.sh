#!/usr/bin/env bash
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Both public launchers share the canonical epoch runner and platform policy.
exec bash "${TRAINING_SCRIPT_DIR}/train_pipeline.sh" "$@"
