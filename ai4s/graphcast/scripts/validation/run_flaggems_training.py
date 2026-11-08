#!/usr/bin/env python3
"""Validate training routing without changing the frozen Native model/recipe.

One backend per fresh process. Default: first three updates of the unchanged
38-update schedule. Use --max-updates 38 for the complete curriculum. Logged
calls/profiling runs are diagnostic only and must not supply E2E/MFU results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/flaggems/graphcast_training_full.yaml"
RUNTIME_LOCK = ROOT / "configs/flaggems/training_runtime.json"
CORRECTION_LOCK = ROOT / "configs/flaggems/training_runtime_v540_corrections.json"


def runtime_lock_path(selection=None):
    selection = (os.environ.get("GRAPHCAST_TRAINING_FLAGGEMS_RUNTIME", "official")
                 if selection is None else selection)
    if selection not in ("official", "corrections"):
        raise ValueError("unknown FlagGems v5.4.0 runtime selection")
    return RUNTIME_LOCK if selection == "official" else CORRECTION_LOCK


def load_runtime_lock(selection=None):
    path = runtime_lock_path(selection)
    lock = json.loads(path.read_text())
    kind = "official" if path == RUNTIME_LOCK else "approved_v540_corrections"
    if not isinstance(lock, dict) or lock.get("source_kind", "official") != kind:
        raise ValueError("runtime selection does not match its trusted source kind")
    return lock


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", required=True, choices=("h100", "ascend910c", "hygon", "metax", "mthreads", "thead"))
    p.add_argument("--backend", required=True, choices=("native", "flaggems"))
    p.add_argument("--gpu", required=True,
                   help="One device index in the container runtime; record host mapping separately")
    for name in ("corpus", "stats", "weights", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--flaggems-source", type=Path, help="Explicit pinned v5.4.0 source/build tree")
    p.add_argument("--flaggems-runtime", choices=("official", "corrections"), default="official",
                   help="Original v5.4.0 or explicitly recorded user-approved v5.4.0 corrections")
    p.add_argument("--max-updates", type=int, default=3)
    p.add_argument("--observe", action="store_true", help="Logged FlagGems calls, phase-separated")
    p.add_argument("--profile", action="store_true", help="Capture update 3 only (1 wait + 1 warmup)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight-only", action="store_true", help="Validate source/import/registration in the container without training")
    return p


def build_plan(args, environ):
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise ValueError("select one container-runtime GPU index")
    if not 1 <= args.max_updates <= 38:
        raise ValueError("max-updates must be between 1 and 38")
    if args.profile and args.max_updates < 3:
        raise ValueError("profiling requires at least 3 updates")
    if args.observe and args.backend != "flaggems":
        raise ValueError("the logged-call observer requires FlagGems")
    if args.backend == "flaggems" and not args.flaggems_source:
        raise ValueError("FlagGems requires an explicit --flaggems-source")
    if any(k in environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise ValueError("run this launcher outside torchrun")
    baseline = load_script("_fixed_training_recipe", ROOT / "scripts/baseline/run_nvidia.py")
    recipe = json.loads(baseline.RECIPE.read_text())
    source_hashes = baseline.source_hashes(recipe)
    launcher = load_script("_public_training_entry", ROOT / recipe["train"]["entry"])
    values = dict(recipe["train"]["arguments"])
    values.pop("device-type")  # platform policy chooses cuda/npu/musa
    values.update(platform=args.platform, backend=args.backend, **{
        "physical-devices": args.gpu, "max-updates": args.max_updates,
        "corpus": str(args.corpus.resolve()), "stats": str(args.stats.resolve()),
        "pretrained-checkpoint": str(args.weights.resolve()),
        "output-dir": str(args.output_dir.resolve() / "run"),
        "flaggems-config": str(PROFILE), "flaggems-variant": "installed",
    })
    argv = [item for k, v in values.items() for item in ("--" + k, str(v))]
    contract = launcher.resolve_contract(launcher.parser().parse_args(argv))
    cleared = {k for k in environ if k.startswith(("GRAPHCAST_", "FLAGGEMS_")) or k in {
        "GEMS_VENDOR", "FLAGOS_REPLACE", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES", "MUSA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING", "ASCEND_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_NPU_ALLOC_CONF", "PYTORCH_MUSA_ALLOC_CONF",
    }}
    env = {k: v for k, v in environ.items() if k not in cleared}
    env.update({k: v for k, v in recipe["train"]["environment"].items() if not k.startswith("PYTORCH_")})
    env.update(contract["environment"])
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for key, subdir in (("FLAGGEMS_CACHE_DIR", "flaggems"), ("TRITON_CACHE_DIR", "triton"), ("XDG_CACHE_HOME", "xdg")):
        env[key] = str(args.output_dir.resolve() / "cache" / subdir)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
    if args.backend == "flaggems":
        env["GRAPHCAST_TRAINING_FLAGGEMS_RUNTIME"] = args.flaggems_runtime
        env["GRAPHCAST_FLAGGEMS_SOURCE_ROOT"] = str(args.flaggems_source.resolve())
        env["GEMS_VENDOR"] = {"h100": "nvidia", "ascend910c": "ascend", "hygon": "hygon",
                              "metax": "metax", "mthreads": "mthreads", "thead": "thead"}[args.platform]
    command = list(contract["launch_command"])
    if args.observe:
        index = command.index(str(ROOT / "scripts/training/run_official_epoch_training.py"))
        command[index:index + 1] = [str(ROOT / "scripts/validation/trace_training_release.py"),
                                  "--trace-dir", str(args.output_dir.resolve() / "calls"), "--"]
    if args.profile:
        command += ["--profile-hotspots", "--profile-wait", "1", "--profile-warmup", "1",
                    "--profile-active", "1", "--profile-repeat", "1", "--profile-top-k", "1000"]
    copy_routing = None
    if args.platform == "ascend910c" and args.backend == "flaggems":
        ops = yaml.safe_load(PROFILE.read_text())["ops"]
        env["GRAPHCAST_FLAGGEMS_OPS"] = ",".join(op for op in ops if op != "copy_")
        env["GRAPHCAST_TRAINING_NATIVE_COPY_IO"] = "1"
        copy_routing = dict(same_device="flaggems_direct_via_dispatch_mode",
                            cross_device="native_io", cross_device_counts_as_flaggems=False,
                            global_registration=False, correctness="unverified")
    layernorm_routing = None
    if args.backend == "flaggems":
        index = next(i for i, item in enumerate(command) if item.endswith(
            ("run_official_epoch_training.py", "trace_training_release.py")))
        wrapper = [str(ROOT / "scripts/validation/run_training_with_native_io.py"),
                   "--layernorm-report", str(args.output_dir.resolve() / "layernorm-routing.json")]
        if copy_routing is not None:
            wrapper += ["--copy-report", str(args.output_dir.resolve() / "copy-routing.json")]
        command[index:index] = wrapper
        layernorm_routing = dict(route="model_layernorm_non_normalized_prefix_reshape",
                                parameters_unchanged=True, normalized_shape_unchanged=True,
                                official_source_unchanged=True, correctness="unverified")
    plan = dict(schema="graphcast.training_integration.v1", contract=contract, command=command,
                device_selection_scope="container_runtime_index_host_mapping_requires_external_evidence",
                copy_routing=copy_routing, layernorm_routing=layernorm_routing,
                flaggems_runtime_selection=args.flaggems_runtime,
                flaggems_runtime=load_runtime_lock(args.flaggems_runtime) if args.backend == "flaggems" else None,
                integration_source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), PROFILE, runtime_lock_path(args.flaggems_runtime),
                              ROOT / "src/graphcast_compat/training_copy_routing.py",
                              ROOT / "src/graphcast_compat/training_layernorm_routing.py",
                              ROOT / "scripts/validation/trace_training_release.py",
                              ROOT / "scripts/validation/run_training_with_native_io.py")},
                frozen_source_sha256=source_hashes, recipe_sha256=hashlib.sha256(baseline.RECIPE.read_bytes()).hexdigest(),
                profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                timing_candidate=args.max_updates == 38 and not (args.observe or args.profile),
                performance_eligible=False,  # Requires a separate successful paired diagnostic.
                correctness="unverified", precision_evidence="requested_tf32_hf32_not_instruction_proof",
                environment_overrides={k: v for k, v in env.items() if environ.get(k) != v},
                cleared_environment_names=sorted(cleared))
    return plan, env


def runtime_hashes(path):
    files = sorted((path / "src/flag_gems").rglob("*"))
    result = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")
              and p.name != "_version.py"}
    if not result:
        raise ValueError("FlagGems source has no runtime files: " + str(path))
    for name in ("pyproject.toml", "setup.py"):
        result[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    return result


def validate_runtime_hashes(hashes, lock, *, baseline=None):
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != lock["runtime_manifest_sha256"] or len(hashes) != lock["runtime_files"]:
        raise ValueError("FlagGems source must match unmodified official " + lock["tag"]
                         + " or the selected recorded corrections; unrecorded overlays are not accepted")
    kind = lock.get("source_kind", "official")
    baseline = load_runtime_lock("official") if baseline is None else baseline
    if kind == "official":
        if lock != baseline:
            raise ValueError("self-declared official runtime differs from the trusted v5.4.0 pin")
        return
    if kind != "approved_v540_corrections":
        raise ValueError("unknown FlagGems source kind")
    if (any(lock.get(key) != baseline[key] for key in ("version", "tag", "commit"))
            or lock.get("overlays") != []
            or lock.get("baseline_runtime_manifest_sha256") != baseline["runtime_manifest_sha256"]):
        raise ValueError("corrections must derive from the pinned official v5.4.0 baseline")
    is_sha = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    changes = lock.get("changes")
    if not isinstance(changes, dict) or not changes or not is_sha(lock.get("patch_sha256")):
        raise ValueError("corrections require a nonempty recorded diff")
    reconstructed = dict(hashes)
    for name, change in changes.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or not (name.startswith("src/flag_gems/") or name in ("pyproject.toml", "setup.py"))
                or not isinstance(change, dict)):
            raise ValueError("invalid correction path/entry")
        before, after = change.get("before_sha256"), change.get("after_sha256")
        if (before == after or before is not None and not is_sha(before)
                or after is not None and not is_sha(after) or hashes.get(name) != after):
            raise ValueError("correction hash does not match the selected source: " + name)
        if before is None:
            reconstructed.pop(name, None)
        else:
            reconstructed[name] = before
    original_digest = hashlib.sha256(json.dumps(
        reconstructed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (len(reconstructed) != baseline["runtime_files"]
            or original_digest != baseline["runtime_manifest_sha256"]):
        raise ValueError("correction diff cannot reconstruct the complete official baseline")


def verify_runtime(path, lock=None):
    lock = load_runtime_lock() if lock is None else lock
    hashes = runtime_hashes(path)
    validate_runtime_hashes(hashes, lock)
    return hashes


def require_container():
    """Device runs are container-only; dry-runs remain portable."""
    markers = [p for p in (Path('/.dockerenv'), Path('/run/.containerenv')) if p.exists()]
    if not markers:
        raise ValueError('Device tests must run inside a Docker/Podman container, not on the host')
    return dict(markers=[str(p) for p in markers], hostname=Path('/etc/hostname').read_text().strip())


PREFLIGHT = """
import json, os, torch
from graphcast_compat.accelerator import accelerator_module, configure_float32_math
from graphcast_compat.training_backend import enable_network_replacements_from_env
m = accelerator_module()
assert m.is_available() and m.device_count() == 1, 'Expose exactly one available device'
free, total = m.mem_get_info(0)
assert free > 50 * 2**30 and total - free < 6 * 2**30, 'Device is busy or has insufficient free memory'
mode = configure_float32_math(True)
device = torch.device(os.environ['GRAPHCAST_ACCELERATOR_TYPE'] + ':0')
r = enable_network_replacements_from_env(device=device)
gems_version = None
if r.enabled:
    import flag_gems
    gems_version = flag_gems.__version__
    assert gems_version == '5.4.0', 'Build the official v5.4.0 version metadata before running'
    assert r.metadata['versions']['flag_gems'] == '5.4.0', 'Official build package metadata must be visible'
copy_routing = None
if os.environ.get('GRAPHCAST_TRAINING_NATIVE_COPY_IO') == '1':
    from graphcast_compat.training_copy_routing import from_environment
    copy_routing = from_environment().metadata()
print(json.dumps(dict(torch=torch.__version__, device=m.get_device_name(0),
    memory_free=free, memory_total=total, requested_precision=mode,
    replacement=r.metadata, flag_gems_version=gems_version, copy_routing=copy_routing,
    coverage='registration_and_routing_only_not_actual_calls')))
"""


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        plan, env = build_plan(args, os.environ)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        plan['container'] = require_container()
        if not args.preflight_only:
            for asset in (args.corpus, args.weights, *(args.stats / f for f in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"))):
                if not asset.is_file():
                    raise ValueError("Missing input: " + str(asset))
        if args.backend == "flaggems":
            plan["flaggems_source_sha256"] = verify_runtime(
                args.flaggems_source.resolve(), plan["flaggems_runtime"])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for key in ("FLAGGEMS_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "launch.json").write_text(json.dumps(plan, indent=2))
        probe = subprocess.run([sys.executable, "-c", PREFLIGHT], env=env, capture_output=True,
                               text=True, timeout=180)
        (args.output_dir / "preflight.json").write_text(json.dumps(dict(returncode=probe.returncode,
            stdout=probe.stdout, stderr=probe.stderr), indent=2))
        if probe.returncode:
            print(probe.stderr, file=sys.stderr)
            return probe.returncode
        if args.preflight_only:
            print(probe.stdout)
            return 0
        with (args.output_dir / "training.log").open("x") as log:
            completed = subprocess.run(plan["command"], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        after, _ = build_plan(args, os.environ)
        unchanged = after["frozen_source_sha256"] == plan["frozen_source_sha256"]
        unchanged &= after["integration_source_sha256"] == plan["integration_source_sha256"]
        if args.backend == "flaggems":
            unchanged &= runtime_hashes(args.flaggems_source.resolve()) == plan["flaggems_source_sha256"]
        (args.output_dir / "exit.json").write_text(json.dumps(dict(returncode=completed.returncode,
            source_unchanged=unchanged, correctness="requires_paired_diagnostic"), indent=2))
        return completed.returncode if unchanged else 2
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
