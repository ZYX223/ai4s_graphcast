"""Compare final forecasts, optionally against the prepared official FP32 reference."""

import argparse
from pathlib import Path

import xarray as xr

from graphcast_bench.assets import ROOT, load_case, read_json, sha256, write_json
from graphcast_bench.metrics import compare, load_reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--data", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
candidate = xr.load_dataset(a.candidate, decode_timedelta=True)
if a.reference.suffix == ".npz":
    if a.data is None:
        p.error("A prepared --data directory is required for the FP32 reference")
    case, prepared = load_case(a.data)
    if sha256(a.reference) != prepared["reference_sha256"]:
        raise ValueError("Reference digest mismatch")
    reference = load_reference(a.reference, case)
    reference_kind = "official_jax_fp32"
else:
    case = read_json(ROOT / "configs/case.json")
    reference = xr.load_dataset(a.reference, decode_timedelta=True)
    reference_kind = "provided_prediction"
result = compare(candidate, reference, case=case)
result.update(
    reference=reference_kind,
    candidate_sha256=sha256(a.candidate),
    reference_sha256=sha256(a.reference),
)
write_json(a.output, result)
print({k: v for k, v in result.items() if k != "layers"})
