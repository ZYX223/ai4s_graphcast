#!/usr/bin/env bash
set -Eeuo pipefail
trap 'chown -R "$OUTPUT_UID:$OUTPUT_GID" /output' EXIT
set +u
source /opt/dtk/env.sh
set -u
unset ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE GRAPHCAST_ROUTER_LIBRARY
export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONPATH=/experiment:/repo/src:/repo/third_party/graphcast:/dependencies/python
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_PLATFORMS=rocm XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
py=/opt/graphcast-jax/bin/python3
if [[ "$1" == compare ]]; then
  export JAX_PLATFORMS=cpu
  "$py" /experiment/compare_results.py
  exit
fi
mkdir /output/native
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O2 -fPIC -shared /experiment/training_router.cc /rne/rne_kernel.hip -I/rne -o /output/native/libtraining_rne.so -ldl -pthread
"$py" - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['/experiment/training_router.cc','/rne/rne_kernel.hip','/rne/rne_kernel.h','/output/native/libtraining_rne.so']
Path('/output/native/build.json').write_text(json.dumps(dict(sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},compiler=subprocess.check_output(['/opt/dtk/bin/hipcc','--version'],text=True)),indent=2))
PY
export TRAINING_ROUTER=/output/native/libtraining_rne.so
export TRAINING_GEMM_LOG=/output/gemm.jsonl
export LD_PRELOAD="$TRAINING_ROUTER:$LD_PRELOAD"
"$py" /experiment/entry.py --mode "$1"
