"""Independent CUDA environment and actual TF32 numerical-effect probe."""

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--output-dir", type=Path, required=True)
args = p.parse_args()
args.output_dir.mkdir(exist_ok=False)
devices = jax.devices("gpu")
assert len(devices) == 1 and "H100" in devices[0].device_kind
assert os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
assert not jax.config.jax_enable_x64


def rne10(x):
    bits = x.view(np.uint32).copy()
    bits = (bits + np.uint32(4095) + ((bits >> 13) & 1)) & np.uint32(0xFFFFE000)
    return bits.view(np.float32)


rows = []
rng = np.random.default_rng(20261016)
for m in (512, 4096):
    a = rng.normal(size=(m, 512)).astype(np.float32)
    b = rng.normal(size=(512, 512)).astype(np.float32)
    output = {}
    for mode in ("highest", "tensorfloat32"):
        fn = jax.jit(lambda a, b: jnp.matmul(a, b, precision=mode))
        da, db = jax.device_put(a), jax.device_put(b)
        result = fn(da, db)
        jax.block_until_ready(result)
        output[mode] = np.asarray(result)
        (args.output_dir / f"{m}-{mode}-hlo.txt").write_text(fn.lower(da, db).compile().as_text())
    exact = a.astype(np.float64) @ b.astype(np.float64)
    rounded = rne10(a).astype(np.float64) @ rne10(b).astype(np.float64)
    xf = output["tensorfloat32"].astype(np.float64)
    fp = output["highest"].astype(np.float64)
    assert np.isfinite(xf).all() and np.isfinite(fp).all()
    row = {
        "shape_mkn": [m, 512, 512],
        "highest_mae_vs_exact": float(np.mean(np.abs(fp - exact))),
        "tf32_mae_vs_exact": float(np.mean(np.abs(xf - exact))),
        "tf32_mae_vs_rne10_reference": float(np.mean(np.abs(xf - rounded))),
        "tf32_mae_vs_highest": float(np.mean(np.abs(xf - fp))),
        "identical_to_highest": bool(np.array_equal(xf, fp)),
    }
    row["matches_rne10_better_than_full_fp32"] = (
        row["tf32_mae_vs_rne10_reference"] < row["tf32_mae_vs_exact"] / 20
    )
    np.savez(args.output_dir / f"{m}-outputs.npz", **output)
    rows.append(row)

gradient = jax.jit(jax.grad(lambda x: jnp.sum(x * x)))(
    jnp.array([1.0, 2.0, -3.0], dtype=jnp.float32)
)
np.testing.assert_array_equal(np.asarray(gradient), np.array([2.0, 4.0, -6.0], dtype=np.float32))
packages = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "dm-haiku",
    "numpy",
    "xarray",
    "netCDF4",
    "scipy",
    "pandas",
    "chex",
    "jraph",
    "trimesh",
    "rtree",
)
report = {
    "devices": [str(x) for x in devices],
    "device_kind": devices[0].device_kind,
    "versions": {name: metadata.version(name) for name in packages},
    "xla_flags": os.environ.get("XLA_FLAGS"),
    "rows": rows,
    "gradient_passed": True,
    "actual_tf32_numerical_effect": all(
        x["matches_rne10_better_than_full_fp32"] and not x["identical_to_highest"] for x in rows
    ),
    "nvidia_smi": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ),
}
report["status"] = "passed" if report["actual_tf32_numerical_effect"] else "failed"
(args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
assert report["actual_tf32_numerical_effect"], "TF32 requested but numerical path not established"
