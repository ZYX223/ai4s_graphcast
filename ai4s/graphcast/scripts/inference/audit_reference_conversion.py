#!/usr/bin/env python3
"""Verify the existing Haiku-to-Torch checkpoint mapping without changing weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.inference.inference_reference import write_json  # noqa: E402
from scripts.inference.diagnose_inference import sha256  # noqa: E402


def audit_conversion(jax_weights: Path, torch_weights: Path) -> dict:
    import torch

    before = [sha256(path) for path in (jax_weights, torch_weights)]
    state = torch.load(torch_weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("expected the project's bare Torch state dictionary")
    suffixes = {"w": "weight", "b": "bias", "scale": "weight", "offset": "bias"}
    mapping = {}
    with np.load(jax_weights, allow_pickle=False) as archive:
        for key in archive.files:
            if not key.startswith("params:"):
                continue
            _, module, parameter = key.split(":")
            if parameter not in suffixes:
                raise ValueError(f"unknown Haiku parameter kind: {key}")
            target = ".".join([part for part in module.split("/") if not part.startswith("~")]
                              + [suffixes[parameter]])
            if target in mapping or target not in state:
                raise ValueError(f"ambiguous or missing Torch parameter: {target}")
            value, converted = archive[key], state[target]
            if (not isinstance(converted, torch.Tensor) or converted.dtype != torch.float32
                    or value.dtype != np.float32 or not np.isfinite(value).all()):
                raise ValueError(f"parameter is not finite FP32: {target}")
            expected = value.T if parameter == "w" else value
            if not np.array_equal(expected, converted.numpy()):
                raise ValueError(f"parameter values differ after layout conversion: {target}")
            mapping[target] = {"jax_key": key, "transpose": parameter == "w", "shape": list(expected.shape),
                               "array_sha256": hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest()}
    if not mapping or set(mapping) != set(state):
        raise ValueError("parameter sets differ; partial conversion cannot be certified")
    if before != [sha256(path) for path in (jax_weights, torch_weights)]:
        raise ValueError("checkpoint changed during conversion audit")
    return {"schema": "graphcast.checkpoint_conversion_audit.v1", "status": "passed",
            "jax_weights_sha256": before[0], "torch_weights_sha256": before[1],
            "converter_sha256": sha256(Path(__file__)), "parameter_count": len(mapping), "mapping": mapping,
            "description": "Reproduced the named Haiku module/parameter mapping and linear-weight transpose; all FP32 values exact. This audits existing weights, not recovery of the historical converter."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-weights", type=Path, required=True)
    parser.add_argument("--torch-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = audit_conversion(args.jax_weights, args.torch_weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(f"exact parameter correspondence: {result['parameter_count']}/{result['parameter_count']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"conversion audit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
