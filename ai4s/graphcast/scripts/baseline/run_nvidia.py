#!/usr/bin/env python3
"""Reproduce H100 Native TF32 baselines using the unchanged model runners."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/baseline/h100_native_tf32.json"
STATS = ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")
RESET_VARIABLES = {
    "PYTHONPATH", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "FLAGOS_REPLACE", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
}
GPU_PROBE = """
import json, sys, torch
from importlib.metadata import version, PackageNotFoundError
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')
if torch.version.hip is not None:
    raise RuntimeError('This recipe requires NVIDIA CUDA')
if torch.cuda.device_count() != 1:
    raise RuntimeError('Exactly one visible GPU is required')
p = torch.cuda.get_device_properties(0)
if 'H100 80GB HBM3' not in p.name or p.total_memory <= 75 * 2**30:
    raise RuntimeError(
        'The fixed 495 TFLOP/s denominator is only for the recorded H100 80GB HBM3; '
        'do not reuse this recipe for another GPU: ' + p.name)
packages = {}
for name in ('numpy', 'scipy', 'pandas', 'xarray', 'PyYAML', 'trimesh',
             'rtree', 'netCDF4', 'h5py', 'cftime'):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = 'not-installed'
print(json.dumps(dict(device=p.name, device_uuid=str(getattr(p, 'uuid', 'unknown')),
                     total_memory=p.total_memory, python=sys.version,
                     torch=torch.__version__, cuda=torch.version.cuda, packages=packages)))
"""


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    for mode in ("train", "infer"):
        sub = modes.add_parser(mode)
        sub.add_argument("--gpu", required=True, help="One CUDA index or full GPU UUID")
        sub.add_argument("--weights", type=Path, required=True)
        sub.add_argument("--stats", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
        sub.add_argument("--numa-node", type=int, help="Optional verified GPU-local NUMA node")
        sub.add_argument("--dry-run", action="store_true", help="No Torch import, data reads or GPU work")
        if mode == "train":
            sub.add_argument("--corpus", type=Path, required=True)
        else:
            sub.add_argument("--inputs", type=Path, required=True)
            sub.add_argument("--rollout-forcings", type=Path, required=True)
            sub.add_argument("--target-template", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=40, help="Positive autoregressive lead count")
    return p


def source_hashes(recipe):
    """Keep the published training/model bytes fixed, with paths mapped to root."""
    expected = json.loads((ROOT / recipe["source_manifest"]).read_text())
    result = {}
    for name, digest in expected.items():
        relative = name.removeprefix("ai4s/graphcast/")
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("Frozen source differs: " + relative)
        result[relative] = actual
    return result


def build_plan(args, environ):
    if not re.fullmatch(r"[0-9]+|GPU-[0-9a-fA-F-]+", args.gpu):
        raise ValueError("--gpu must select one index or GPU UUID, not a card list")
    if args.numa_node is not None and args.numa_node < 0:
        raise ValueError("--numa-node must be nonnegative")
    if "RANK" in environ or "LOCAL_RANK" in environ or "WORLD_SIZE" in environ:
        raise ValueError("Run this entry directly, not inside an existing torchrun job")
    recipe = json.loads(RECIPE.read_text())
    files = source_hashes(recipe)
    section = recipe[args.mode]
    values = dict(section["arguments"])
    assets = {"weights": args.weights.resolve(), "stats": args.stats.resolve()}
    output = args.output_dir.resolve()
    values["config"] = str(ROOT / recipe["model_config"])
    if args.mode == "train":
        assets["corpus"] = args.corpus.resolve()
        values.update(corpus=str(assets["corpus"]), stats=str(assets["stats"]))
        values["pretrained-checkpoint"] = str(assets["weights"])
        values["output-dir"] = str(output)
    else:
        if args.steps < 1:
            raise ValueError("--steps must be positive")
        for key in ("inputs", "rollout_forcings", "target_template"):
            assets[key] = getattr(args, key).resolve()
            values[key.replace("_", "-")] = str(assets[key])
        values.update(weights=str(assets["weights"]), steps=args.steps)
        values["stats-dir"] = str(assets["stats"])
        values["output"] = str(output / "prediction.nc")
        values["report"] = str(output / "report.json")
    cleared = sorted(key for key in environ if key.startswith(("GRAPHCAST_", "FLAGGEMS_"))
                     or key in RESET_VARIABLES)
    env = {key: value for key, value in environ.items() if key not in cleared}
    fixed_env = dict(section["environment"], CUDA_VISIBLE_DEVICES=args.gpu,
                     GRAPHCAST_BACKEND="nvidia", GRAPHCAST_FLAGOS_MODE="off",
                     FLAGOS_REPLACE="0", PYTHONPATH=str(ROOT / "src"))
    env.update(fixed_env)
    command = [sys.executable, str(ROOT / section["entry"])]
    command += [item for key, value in values.items() for item in ("--" + key, str(value))]
    if args.numa_node is not None:
        command = ["numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}"] + command
    plan = dict(schema="graphcast.nvidia_baseline_launch.v1", mode=args.mode,
                source_base=recipe["source_commit"], source_sha256=files,
                recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
                launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                command=command, environment_overrides=fixed_env,
                cleared_environment_names=cleared, output_dir=str(output),
                assets={key: str(path) for key, path in assets.items()},
                peak_tflops_per_device=recipe["peak_tflops_per_device"],
                peak_basis=recipe["peak_basis"], execution="native_eager")
    return plan, env


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan, env = build_plan(args, os.environ)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        output = Path(plan["output_dir"])
        if output.exists():
            raise ValueError("Use a fresh --output-dir; refusing to overwrite " + str(output))
        for name, value in plan["assets"].items():
            paths = [Path(value) / item for item in STATS] if name == "stats" else [Path(value)]
            for path in paths:
                if not path.is_file():
                    raise ValueError("Missing required asset: " + str(path))
        # Separate preflight process: CUDA initialization does not enter E2E.
        probe = subprocess.run([sys.executable, "-c", GPU_PROBE], env=env,
                               text=True, capture_output=True, check=True, timeout=60)
        plan["hardware"] = json.loads(probe.stdout)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "baseline-launch.json").open("x") as stream:
            json.dump(plan, stream, indent=2)
        result = subprocess.run(plan["command"], env=env, cwd=ROOT, check=False)
        unchanged = source_hashes(json.loads(RECIPE.read_text())) == plan["source_sha256"]
        with (output / "baseline-exit.json").open("x") as stream:
            json.dump(dict(returncode=result.returncode, source_unchanged=unchanged,
                           diagnostic_status="not_run"), stream, indent=2)
        return result.returncode if unchanged else 2
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            p.error("GPU preflight failed: " + (error.stderr or str(error)))
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
