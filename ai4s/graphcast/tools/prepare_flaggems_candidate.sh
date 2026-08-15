#!/usr/bin/env bash
# Compatibility entrypoint for the remote runtime preparation workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "note: prepare_flaggems_candidate.sh now prepares official and internal runtimes" >&2
exec "${SCRIPT_DIR}/prepare_flaggems_runtime.sh" "$@"
