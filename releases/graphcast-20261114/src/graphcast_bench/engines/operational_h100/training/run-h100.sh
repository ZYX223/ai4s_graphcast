#!/usr/bin/env bash
set -Eeuo pipefail
# Python/JAX/native environment is supplied by platforms/environment.py.
"$GRAPHCAST_PYTHON" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["jax","jaxlib","optax","chex","dm-haiku","numpy","xarray"]})'
if [ "${MEMORY_COMPILE_ONLY:-0}" = 1 ]; then
 exec "$GRAPHCAST_PYTHON" /candidate/entry.py "$@"
fi
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
 --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite --output=/output/train-trace \
 "$GRAPHCAST_PYTHON" /candidate/entry.py "$@"
