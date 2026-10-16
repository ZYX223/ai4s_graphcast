"""Copy only JAX assets and extract the pinned official FP32 reference."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from graphcast_bench.assets import ROOT, read_json, sha256, verify_files, write_json
from graphcast_bench.metrics import VARIABLES, final_frame, load_reference, validate_case_grid_units

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--case", type=Path, required=True)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
case = read_json(ROOT / "configs/case.json")
verify_files(a.case, case["assets"])
if sha256(a.baseline) != case["source_reference_bundle_sha256"]:
    raise ValueError("The source reference is not the qualified weekly baseline")
a.output.mkdir(parents=True, exist_ok=False)
for relative in case["assets"]:
    dst = a.output / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.case / relative, dst)
verify_files(a.output, case["assets"])
with np.load(a.baseline, allow_pickle=False) as old:
    values = {name: old["jax_fp32_no_bf16__" + name] for name in VARIABLES}
    values.update({name: old[name] for name in ("lat", "lon", "level")})
    np.savez(a.output / "reference.npz", **values)
reference = load_reference(a.output / "reference.npz", case)
validate_case_grid_units(reference, case)
final_frame(reference, case["initialization_time"], case["lead_hours"])
write_json(
    a.output / "manifest.json",
    {
        "schema": "graphcast.jax_case.v1",
        "case_config_sha256": sha256(ROOT / "configs/case.json"),
        "source_reference_bundle_sha256": case["source_reference_bundle_sha256"],
        "reference_sha256": sha256(a.output / "reference.npz"),
        "reference": "official_jax_fp32",
        "reference_values_changed": False,
    },
)
print("Prepared and verified JAX-only assets:", a.output)
