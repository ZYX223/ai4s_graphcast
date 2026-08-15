#!/usr/bin/env bash
# Fetch and compose matched official/internal FlagGems source trees.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_flaggems_runtime.py" "$@"
