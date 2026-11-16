#!/usr/bin/env bash
# The host uses stdlib Python; JAX and the model load inside the delivery container.
# exec preserves exit status and signal handling; disable bytecode writes to keep sources unchanged.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/delivery.py" "$@"
