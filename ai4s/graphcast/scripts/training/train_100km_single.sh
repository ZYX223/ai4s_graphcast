#!/usr/bin/env bash
# Compatibility entry for the legacy probe; does not select formal training.
set -euo pipefail
TRAINING_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' 'Deprecated probe path; use scripts/training/legacy/train_100km_single.sh. Formal training: scripts/train.sh.' >&2
exec bash "${TRAINING_SCRIPT_DIR}/legacy/train_100km_single.sh" "$@"
