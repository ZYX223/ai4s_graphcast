"""Build an inspectable worker command before importing JAX or loading a router.

Run these commands inside the pinned FlagOS runtime. Physical device selection,
assets and native libraries are supplied by the caller, never discovered in /tmp.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from graphcast_bench.assets import ROOT, read_json, write_json
from graphcast_bench.validation.provenance import verify_maintained


def absolute_path(value, label):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(label + " must be an absolute runtime path")
    return str(path)


def make_plan(
    kind, config, paths, output, device, pci, rollout=1, resume=None, limit=38, profile="none"
):
    if kind not in ("training", "inference"):
        raise ValueError("Unknown workload")
    if (config["schema"], config["platform"], config["backend"], config["precision"]) != (
        "graphcast.runtime.v1",
        "hygon",
        "fl_only",
        "rne_xf32",
    ):
        raise ValueError("This entry supports the frozen BW1100 RNE+XF32 configuration")
    if (
        type(device) is not int
        or device < 0
        or not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci)
    ):
        raise ValueError("Explicit physical device index and PCI bus ID required")
    if profile not in ("none", "single", "full40"):
        raise ValueError("Unknown profile action")
    output = absolute_path(output, "output")
    required = ["assets", kind + "_graph", kind + "_router"]
    if kind == "training":
        required.append("training_data")
        if not 1 <= rollout <= 12 or not 1 <= limit <= 38:
            raise ValueError("Expected horizon 1..12 and update limit 1..38")
        if rollout > 1 and not resume:
            raise ValueError("Later horizons require the previous completed checkpoint")
    for key in required:
        absolute_path(paths[key], key)
    if resume:
        resume = absolute_path(resume, "resume")
    cfg = config[kind]
    graph = paths[kind + "_graph"]
    library = paths[kind + "_router"]
    env = {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/graphcast")]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONFAULTHANDLER": "1",
        "OPENBLAS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
        "ROCR_VISIBLE_DEVICES": str(device),
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "XLA_MODE": "fl_only",
        "JAX_PLATFORMS": "rocm",
        "JAX_ENABLE_X64": "false",
        "JAX_DEFAULT_MATMUL_PRECISION": "tensorfloat32",
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer= --xla_gpu_graph_level=0",
        "XLA_FLAGTREE_COMPILE_WORKERS": "12",
        "XLA_FLAGTREE_PERSISTENT_WORKERS": "12",
        "XLA_FLAGTREE_LOG": "0",
        "LD_PRELOAD": library,
        "TMPDIR": str(Path(output) / "tmp"),
        "GRAPHCAST_REFERENCE_GRAPH": graph,
        "GRAPHCAST_REFERENCE_GRAPH_SHA256": cfg["reference_graph_sha256"],
        "GRAPHCAST_REFERENCE_GRAPH_AUDIT": str(Path(output) / "reference-graph-audit.jsonl"),
        "GRAPHCAST_RUNTIME_CONFIG": json.dumps(config),
    }
    if paths.get("library_dirs"):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [absolute_path(p, "library_dirs") for p in paths["library_dirs"]]
            + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])
        )
    command = [sys.executable, "-X", "faulthandler", "-m", "graphcast_bench." + kind + ".entry"]
    if kind == "training":
        if (
            cfg["schedule"] != "training38"
            or cfg["candidate"] != "mesh-group8-m2g-save-stats-vjp-finalmesh-g2m"
            or cfg["rne_policy"] != "warmup-verify"
        ):
            raise ValueError("Unvalidated training schedule or optimization policy")
        fraction = cfg["horizon11_memory_fraction"] if rollout == 11 else cfg["memory_fraction"]
        env.update(
            {
                "TRAINING_PLATFORM": "hygon",
                "PHYSICAL_DEVICE": str(device),
                "EXPECTED_PCI": pci.lower(),
                "TRAINING_OUTPUT": output,
                "TRAINING_ASSETS": paths["assets"],
                "TRAINING_DATA": paths["training_data"],
                "TRAINING_ROUTER": library,
                "TRAINING_GEMM_LOG": str(Path(output) / "gemm.jsonl"),
                "TRAINING_CANDIDATE": cfg["candidate"],
                "RNE_PRE_ROUNDED": cfg["rne_policy"],
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": str(fraction),
                "TRAINING_JAX_TRACE": "1",
                "MEMORY_COMPILE_ONLY": "0",
            }
        )
        command += ["--mode", "rne_xf32", "--rollout", str(rollout), "--limit", str(limit)]
        if resume:
            command += ["--resume", resume]
    else:
        if (cfg["steps"], cfg["repeats"], cfg["warmup_steps"]) != (40, 3, 5):
            raise ValueError("Expected three independent 40-step rollouts with five warmup steps")
        env.update(
            {
                "GRAPHCAST_PHYSICAL_DEVICE": str(device),
                "GRAPHCAST_EXPECTED_PCI": pci.lower(),
                "GRAPHCAST_ROUTER_LIBRARY": library,
                "GRAPHCAST_STAGE_WEIGHTS": str(ROOT / "kernels/xf32_rne/weights.tsv"),
                "GRAPHCAST_FUSED_LAYERS": str(ROOT / "configs/runtime/inference/fused_layers.txt"),
                "GRAPHCAST_STAGE_LOG": str(Path(output) / "gemm.jsonl"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": str(cfg["memory_fraction"]),
            }
        )
        command += [
            "--platform",
            "hygon",
            "--mode",
            "rne_xf32",
            "--variant",
            "fused",
            "--data",
            paths["assets"],
            "--output",
            str(Path(output) / "run"),
            "--steps",
            "1" if profile == "single" else "40",
            "--repeats",
            "1" if profile == "single" else "3",
        ]
        if profile != "none":
            command.append("--profile")
    return {
        "kind": kind,
        "command": command,
        "environment": env,
        "required_paths": {key: paths[key] for key in required},
        "output": output,
        "resume": resume,
    }


def run_plan(plan):
    """Preflight and launch one clean process; record the actual exit status."""
    verify_maintained()
    for key, path in plan["required_paths"].items():
        if not Path(path).exists():
            raise FileNotFoundError(key + ": " + path)
    from graphcast_bench.platforms.native import verify_native

    verify_native(Path(plan["environment"]["LD_PRELOAD"]), plan["kind"])
    if plan["resume"]:
        from graphcast_bench.training.resume import validate_resume

        validate_resume(plan["resume"])
    out = Path(plan["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "tmp").mkdir()
    write_json(out / "launch.json", plan)
    env = dict(os.environ)
    for key in (
        "ROCBLAS_MATH_MODE",
        "NVIDIA_TF32_OVERRIDE",
        "JAX_DUMP_IR_TO",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "TRAINING_FP32_REFERENCE",
    ):
        env.pop(key, None)
    env.update(plan["environment"])
    result = subprocess.run(plan["command"], env=env, cwd=ROOT)
    (out / "python-exit-code.txt").write_text(str(result.returncode) + "\n")
    return result.returncode


def main(kind):
    parser = argparse.ArgumentParser(
        description="BW1100 " + kind + "; run inside the pinned runtime"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/runtime/bw1100.json")
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--pci", required=True)
    parser.add_argument("--dry-run", action="store_true")
    if kind == "training":
        parser.add_argument("--rollout", type=int, required=True)
        parser.add_argument("--resume")
        parser.add_argument("--limit", type=int, default=38)
    else:
        parser.add_argument("--profile", choices=("none", "single", "full40"), default="none")
    args = vars(parser.parse_args())
    config, paths = read_json(args.pop("config")), read_json(args.pop("paths"))
    dry_run = args.pop("dry_run")
    plan = make_plan(kind, config, paths, **args)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return run_plan(plan)
