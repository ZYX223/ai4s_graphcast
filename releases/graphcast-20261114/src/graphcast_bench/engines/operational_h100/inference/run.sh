#!/usr/bin/env bash
set -Eeuo pipefail
# Python/JAX/native environment is supplied by platforms/environment.py.
case "$GRAPHCAST_VARIANT" in
  best|fastvar|stable_ln|shifted_ln|shifted_mesh) ;;
  *) exit 2 ;;
esac
"$GRAPHCAST_PYTHON" -m pip freeze > /output/packages.txt
nsys --version > /output/nsys-version.txt
if [[ "${GRAPHCAST_MESH_PROBE:-0}" == 1 ]]; then
 "$GRAPHCAST_PYTHON" /experiment/probe_mesh.py
 exit 0
fi
if [[ "${GRAPHCAST_OPERATOR_PROBE:-0}" == 1 ]]; then
 "$GRAPHCAST_PYTHON" /experiment/probe_layernorm.py
 exit 0
fi
if [[ "${GRAPHCAST_SINGLE_STEP:-0}" == 1 ]]; then
 nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --export=sqlite --output=/output/model-trace \
  "$GRAPHCAST_PYTHON" /experiment/smoke_single_step.py --runtime-context /output/inference-context.json --platform h100 --mode tf32 --data /data --output /output/run --profile
 "$GRAPHCAST_PYTHON" /experiment/analyze_single_step.py --profile-dir /output
 exit 0
fi
nsys profile --trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none \
 --capture-range=cudaProfilerApi --capture-range-end="repeat:$GRAPHCAST_REPEATS" --export=sqlite --output=/output/model-trace \
 "$GRAPHCAST_PYTHON" /experiment/profile40.py --runtime-context /output/inference-context.json --platform h100 --mode tf32 --data /data --output /output/run --profile --repeats "$GRAPHCAST_REPEATS"
if [[ "${GRAPHCAST_GENERIC_INFERENCE:-0}" == 1 ]]; then
 # The delivery adapter validates expected custom counts, traces and saved outputs.
 exit 0
fi
"$GRAPHCAST_PYTHON" /experiment/analyze.py --profile-dir /output --output /output/timing.json
"$GRAPHCAST_PYTHON" /experiment/compare.py --candidate /output/run --reference /frozen-reference --case /repo/configs/case.json --output /output/accuracy-h100
