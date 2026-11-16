"""Build an inspectable worker command before importing JAX or loading a router.

Run these commands inside the pinned FlagOS runtime. Physical device selection,
assets and native libraries are supplied by the caller, never discovered in /tmp.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from graphcast_bench.assets import ROOT, read_json, write_json
from graphcast_bench.validation.provenance import verify_maintained
from graphcast_bench.delivery.options import resolve_options, TRAINING_DEFAULTS, positive_integer
from graphcast_bench.platforms.context import RuntimeContext
from graphcast_bench.platforms.environment import bw_environment, clean_environment


def absolute_path(value, label):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(label + " must be an absolute runtime path")
    return str(path)


def make_plan(
    kind, config, paths, output, device, pci, rollout=1, resume=None, limit=38, profile="none",
    steps=None, warmup_steps=None, repeats=None, warmup_scope=None,
    warmup_iters=TRAINING_DEFAULTS["warmup_iters"],
    save_every=TRAINING_DEFAULTS["save_every"],
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
    if profile not in ("none", "single", "full40", "custom"):
        raise ValueError("Unknown profile action")
    output = absolute_path(output, "output")
    required = ["assets", kind + "_graph", kind + "_router"]
    if kind == "training":
        positive_integer("warmup_iters", warmup_iters)
        if save_every is not None:
            positive_integer("save_every", save_every)
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
    context = RuntimeContext(kind, "bw1100", output, dict(index=device, pci=pci),
                             paths, dict(config, rollout=rollout))
    env = bw_environment(context, ROOT)
    command = [sys.executable, "-X", "faulthandler", "-m", "graphcast_bench." + kind + ".entry",
               "--runtime-context", str(Path(output) / "runtime-context.json")]
    if kind == "training":
        if (
            cfg["schedule"] != "training38"
            or cfg["candidate"] != "mesh-group8-m2g-save-stats-vjp-finalmesh-g2m"
            or cfg["rne_policy"] != "warmup-verify"
        ):
            raise ValueError("Unvalidated training schedule or optimization policy")
        command += ["--mode", "rne_xf32", "--rollout", str(rollout), "--limit", str(limit),
                    "--warmup-iters", str(warmup_iters)]
        if save_every is not None:
            command += ["--save-every", str(save_every)]
        if resume:
            command += ["--resume", resume]
    else:
        values = resolve_options("infer", "bw1100", dict(
            steps=steps, warmup_steps=warmup_steps, repeats=repeats,
            warmup_scope=warmup_scope))["inference"]
        if profile in ("single", "full40"):
            if any(value is not None for value in (steps, warmup_steps, repeats, warmup_scope)):
                raise ValueError("Fixed profile rejects custom controls; use --profile custom")
            values = resolve_options(profile, "bw1100")["inference"]
        if (cfg["steps"], cfg["repeats"], cfg["warmup_steps"]) != (40, 3, 5):
            raise ValueError("Expected three independent 40-step rollouts with five warmup steps")
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
            str(values["steps"]),
            "--repeats",
            str(values["repeats"]),
            "--warmup-steps", str(values["warmup_steps"]),
            "--warmup-scope", values["warmup_scope"],
        ]
        if profile != "none":
            command.append("--profile")
    return {
        "kind": kind,
        "runtime_context": context.to_dict(),
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
    RuntimeContext.from_dict(plan["runtime_context"]).write(out / "runtime-context.json")
    env = clean_environment(updates=plan["environment"])
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
        parser.add_argument("--warmup-iters", type=int, default=TRAINING_DEFAULTS["warmup_iters"])
        parser.add_argument("--save-every", type=int, default=TRAINING_DEFAULTS["save_every"])
    else:
        parser.add_argument("--profile", choices=("none", "single", "full40", "custom"), default="none")
        parser.add_argument("--steps", type=int)
        parser.add_argument("--warmup-steps", type=int)
        parser.add_argument("--repeats", type=int)
        parser.add_argument("--warmup-scope", choices=("once", "per-repeat"))
    args = vars(parser.parse_args())
    config, paths = read_json(args.pop("config")), read_json(args.pop("paths"))
    dry_run = args.pop("dry_run")
    plan = make_plan(kind, config, paths, **args)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    return run_plan(plan)
