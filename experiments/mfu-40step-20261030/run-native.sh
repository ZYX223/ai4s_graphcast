#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
unset PYTHONPATH LD_PRELOAD ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE XLA_MODE XLA_FLAGTREE_LOG
unset JAX_DUMP_IR_TO XLA_PYTHON_CLIENT_ALLOCATOR XLA_PYTHON_CLIENT_MEM_FRACTION
set +u
source /opt/dtk/env.sh
set -u
export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTHONPATH=/experiment:/repo/src:/repo/third_party/graphcast
export JAX_ENABLE_X64=false JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32 JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0'
export GRAPHCAST_PLATFORM=hygon
export GRAPHCAST_FROZEN_BENCHMARK=/experiment/full_model.py
export GRAPHCAST_PROFILE_EVIDENCE=/output
export TMPDIR=/output/tmp
mkdir "$TMPDIR"
py=/opt/graphcast-jax/bin/python3
"$py" - <<'PY'
import ctypes, hashlib, json, os, sys
import importlib.metadata as md
from pathlib import Path
import jax, jax.numpy as jnp
assert md.version('jax') == '0.6.0+das.opt1.dtk2604'
assert len(jax.devices('gpu')) == 1
driver=ctypes.CDLL('libamdhip64.so'); bus=ctypes.create_string_buffer(64)
assert driver.hipDeviceGetPCIBusId(bus,len(bus),0)==0
assert bus.value.decode().lower()==os.environ['GRAPHCAST_EXPECTED_PCI']
x=jnp.ones((512,512),jnp.float32)
y=jax.jit(lambda a:a@a)(x);y.block_until_ready()
assert float(y[0,0])==512
raw=dict(device=str(jax.devices()),pci=bus.value.decode(),python=sys.version,
    versions={n:md.version(n) for n in ['jax','jaxlib','jax-rocm60-plugin','jax-rocm60-pjrt','dm-haiku','numpy','xarray']},
    dtk=Path('/opt/dtk').resolve().name,
    library_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/opt/dtk/lib/librocblas.so'),Path('/opt/dtk/lib/libamdhip64.so'),Path('/baseline-native/libxf32_rne_router.so')]})
Path('/output/environment-probe.json').write_text(json.dumps(raw,indent=2)+'\n')
print(json.dumps(raw),flush=True)
PY
export GRAPHCAST_ROUTER_LIBRARY=/baseline-native/libxf32_rne_router.so
export GRAPHCAST_STAGE_WEIGHTS=/repo/kernels/xf32_rne/weights.tsv
export GRAPHCAST_FUSED_LAYERS=/experiment/fused_layers.txt
export GRAPHCAST_STAGE_LOG=/output/gemm.jsonl
export LD_PRELOAD="$GRAPHCAST_ROUTER_LIBRARY:$LD_PRELOAD"
# The archived profiler crashes with inherited Python/temp paths before launch.
# Keep its startup environment isolated, then restore those paths in the child.
env -u TMPDIR -u PYTHONPATH /opt/dtk/bin/hipprof --hip-trace --trace-off --output-type 0 -o /output/trace \
 bash /measurement/native-profile-entry.sh
"$py" -c 'import json; assert json.load(open("/output/run/report.json"))["status"] == "completed"'
