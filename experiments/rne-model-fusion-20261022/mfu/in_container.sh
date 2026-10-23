#!/usr/bin/env bash
set -Eeuo pipefail
trap 'chown -R "$OUTPUT_UID:$OUTPUT_GID" /output' EXIT
set +u
source /opt/dtk/env.sh
set -u
unset ROCBLAS_MATH_MODE NVIDIA_TF32_OVERRIDE
export LD_PRELOAD=/opt/mpi/lib/libmpi.so:/usr/lib/x86_64-linux-gnu/libsqlite3.so.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 JAX_PLATFORMS=rocm XLA_PYTHON_CLIENT_PREALLOCATE=false
/opt/dtk/bin/hipcc --offload-arch=gfx938 -O2 /repo/kernels/operator_bench/gemm_bench.hip /repo/kernels/xf32_rne/rne_kernel.hip -o /output/gemm_bench -lrocblas
for n in 8192 16384 24576; do
  /opt/hyhal/bin/hy-smi -d "$GRAPHCAST_PHYSICAL_DEVICE" --showclocks --showuse --showmeminfo vram --json > "/output/device-before-$n.json"
  /output/gemm_bench "$n" "$n" "$n" 3 20 > "/output/gemm-$n.json"
  /opt/hyhal/bin/hy-smi -d "$GRAPHCAST_PHYSICAL_DEVICE" --showclocks --showuse --showmeminfo vram --json > "/output/device-after-$n.json"
  cat "/output/gemm-$n.json"
done
