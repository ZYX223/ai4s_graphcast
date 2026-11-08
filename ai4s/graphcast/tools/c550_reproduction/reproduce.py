"""Prepare and execute isolated C550 reproductions; preparation is CPU-only.

The numerical harnesses are frozen under harness/. This module binds paths and
device placement, never rewrites the model, optimizer, schedule or evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODES = ("full38", "short8", "recovery", "evaluation")
ENTRIES = {"full38": "run_full38.py", "short8": "run_sparse.py",
           "recovery": "run_recovery.py", "evaluation": "evaluation_worker.py"}
LIMITS = {"full38": 7500, "short8": 1800, "recovery": 2100, "evaluation": 43350}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render(value, variables):
    """Replace only declared placeholders, including input-manifest path keys."""
    if isinstance(value, dict):
        return {render(k, variables): render(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([A-Z_0-9]+)\}", value)
    if match and match[1] in variables:
        return variables[match[1]]
    for name, replacement in variables.items():
        value = value.replace("${" + name + "}", str(replacement))
    if "${" in value:
        raise ValueError("unbound placeholder: " + value)
    return value


def validate_settings(settings, job):
    keys = ("model_root", "flaggems_root", "train_data_root", "initial_checkpoint",
            "cache_seed", "cp36", "cp38", "parent_run", "resume_audit", "validation_data_root",
            "python", "runtime_library")
    for key in keys:
        if not Path(settings[key]).is_absolute():
            raise ValueError(key + " must be an absolute path")
    for path in settings["dependency_paths"] + settings["python_site_paths"]:
        if not Path(path).is_absolute():
            raise ValueError("dependency paths must be absolute")
    if not re.fullmatch(r"\d+", settings["visible_device"]):
        raise ValueError("exactly one visible device is required")
    if type(settings["physical_gpu"]) is not int or settings["physical_gpu"] < 0:
        raise ValueError("physical_gpu must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", settings["pci_bus"]):
        raise ValueError("use full PCI domain:bus:device.function")
    # Outputs must never be an input tree or contain an input tree.
    for key in keys:
        path = Path(settings[key]).resolve()
        if job == path or job in path.parents or path in job.parents:
            raise ValueError("output overlaps input: " + key)


def prepare(mode, settings, job):
    if mode not in MODES:
        raise ValueError(mode)
    job = Path(job).resolve()
    if job.exists():
        raise FileExistsError(job)
    validate_settings(settings, job)
    values = dict(MODEL=settings["model_root"], GEMS=settings["flaggems_root"],
                  TRAIN_DATA=settings["train_data_root"], INITIAL=settings["initial_checkpoint"],
                  CACHE=settings["cache_seed"], CP36=settings["cp36"], CP38=settings["cp38"],
                  PARENT_RUN=settings["parent_run"], RESUME_AUDIT=settings["resume_audit"],
                  VALIDATION_DATA=settings["validation_data_root"], PYTHON=settings["python"],
                  DEPENDENCIES=":".join(settings["dependency_paths"]),
                  MASK=settings["visible_device"], PHYSICAL=settings["physical_gpu"],
                  PCI=settings["pci_bus"].lower(), OUTPUT=str(job / "candidate"))
    # Fully render in memory before creating the new directory.
    templates = {p.name: render(read(p), values) for p in (ROOT / "templates" / mode).glob("*.json")}
    job.mkdir(parents=True, exist_ok=False)
    for source in (ROOT / "harness" / mode).glob("*.py"):
        shutil.copyfile(source, job / source.name)
    for name in ("reproduce.py", "runtime_gate.py", "evaluation_worker.py", "bootstrap.py", "task_tree.py"):
        shutil.copyfile(ROOT / name, job / name)
    for name, data in templates.items():
        dump(job / name, data)
    shutil.copyfile(ROOT / "locks/reference.json", job / "reference.json")
    shutil.copyfile(ROOT / "locks/flaggems-accepted.json", job / "flaggems-accepted.json")
    if mode == "evaluation":
        # A real resource record, produced after check, is still needed to run.
        dump(job / "config.json", {"key": "c550", "model": settings["model_root"]})
    meta = dict(mode=mode, settings=settings, device=dict(physical_gpu=values["PHYSICAL"],
                pci_bus=values["PCI"], visible_device=values["MASK"],
                runtime_library=settings["runtime_library"]),
                timeout_seconds=LIMITS[mode], gpu_work_started=False,
                packaging_hardware_validated=False)
    dump(job / "job.json", meta)
    manifest = {p.name: sha(p) for p in sorted(job.iterdir()) if p.is_file()}
    dump(job / "package-manifest.json", manifest)
    return job


def verify_job(job):
    job = Path(job).resolve()
    manifest = read(job / "package-manifest.json")
    for name, expected in manifest.items():
        if Path(name).name != name or sha(job / name) != expected:
            raise ValueError("job package changed: " + name)
    return sha(job / "package-manifest.json")


def validate_gate(gate, device, digest, now=None):
    now = time.time() if now is None else now
    checks = [gate.get("allowed") is True, gate.get("platform") == "c550",
              bool(gate.get("owner_user_id")), gate.get("healthy") is True,
              gate.get("process_count") == 0, gate.get("utilization_percent") == 0,
              0 <= gate.get("memory_used_mib", -1) <= 1024,
              0 <= now - gate.get("epoch", 0) <= 120,
              gate.get("physical_gpu") == device["physical_gpu"],
              gate.get("pci_bus", "").lower() == device["pci_bus"].lower(),
              bool(gate.get("device_uuid")), gate.get("container_mapping_verified") is True,
              gate.get("environment_verified") is True, gate.get("job_digest") == digest,
              bool(re.fullmatch(r"[a-f0-9]{64}", gate.get("raw_evidence_sha256", "")))]
    if not all(checks):
        raise ValueError("missing, stale, occupied, unhealthy or mismatched resource attestation")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources(job):
    """Read-only CPU checks; do not load Torch or query any accelerator."""
    meta = read(job / "job.json")
    s = meta["settings"]
    reference = read(job / "reference.json")
    for section in ("frozen_source_sha256", "integration_source_sha256"):
        for name, expected in reference[section].items():
            if sha(Path(s["model_root"]) / name) != expected:
                raise ValueError("model source drift: " + name)
    source = Path(s["flaggems_root"])
    accepted = read(job / "flaggems-accepted.json")
    if sha(source / "retest-source-manifest.json") != sha(job / "flaggems-accepted.json"):
        raise ValueError("FlagGems manifest drift")
    for name, expected in accepted.items():
        if sha(source / name) != expected:
            raise ValueError("FlagGems source drift: " + name)
    # The mesh is an input asset, not one of the 144 Python/config files.
    mesh = Path(s['model_root']) / 'src/graphcast_compat/assets/google_mesh6_vertices.npy'
    if sha(mesh) != 'aa2462b34f8811cf11006eea1462ed558652cc7c4ce9b3f62ca4786c8814e40a':
        raise ValueError('fixed mesh asset drift')
    if meta["mode"] == "evaluation":
        module = load_module(job / "evaluate_remaining_cp38.py", "preflight_evaluation")
        return module.preflight(read(job / "spec.json"))
    verify_training_data(job, s)
    template = read(job / ("plan-template.json" if meta["mode"] == "short8" else "candidate-template.json"))
    for path, expected in template["input_file_sha256"].items():
        if sha(path) != expected:
            raise ValueError("input drift: " + path)
    if not Path(s["cache_seed"]).is_dir():
        raise ValueError("explicit cache seed directory missing (an empty new directory is allowed)")
    if meta["mode"] == "recovery":
        resume = read(job / "config.json")["resume_configuration"]
        if sha(s["cp36"]) != resume["resume_sha256"]:
            raise ValueError("CP36 identity changed")
        audit = read(s["resume_audit"])
        if audit["checkpoint_sha256"] != resume["resume_sha256"] or not audit["hash_matches"]:
            raise ValueError("CP36 CPU audit missing or mismatched")
        validate_resume_paths(audit, s)
        if not (Path(s["parent_run"]) / "metrics").is_dir():
            raise ValueError("continuous-run metric prefix is required")
    return dict(source_files=len(accepted), model_files=144, gpu_work_started=False)


def validate_resume_paths(audit, settings):
    model, data, gems = settings['model_root'], settings['train_data_root'], settings['flaggems_root']
    expected = dict(config=model + '/configs/inference/google_graphcast_operational_25km.yaml',
                    corpus_index=data + '/corpora/era5_2017-01-01_07_train.json', stats_dir=data + '/stats',
                    flaggems_source_root=gems,
                    flaggems_ops_config=model + '/configs/flaggems/graphcast_training_full.yaml',
                    flaggems_module_origin=gems + '/src/flag_gems/__init__.py')
    if audit.get('required_resume_paths') != expected:
        raise ValueError('CP36 path contract differs; use the recorded internal mounts and a current cp36_probe audit')


def verify_training_data(job, settings):
    path = Path(settings['train_data_root']) / 'official_era5/era5_2017-01-01_07_graphcast_operational_pretrain.nc'
    expected = 'f277accdd54a8d652b2357035fde107fc80f7cdcfceb71d0ba155ed92b3c02a3'
    st = path.stat()
    if st.st_size != 5896612497:
        raise ValueError('training corpus size changed')
    receipt_path = Path(settings.get('training_data_receipt') or job / 'training-data-verified.json')
    if receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt.get('sha256'), receipt.get('bytes'), receipt.get('mtime_ns')) != (expected, st.st_size, st.st_mtime_ns):
            raise ValueError('training data changed after verification')
        return
    if settings.get('training_data_receipt'):
        raise FileNotFoundError(receipt_path)
    # A new deployment without a receipt requires one streaming hash. Later
    # checks reuse this receipt+stat, never read the complete array into memory.
    if sha(path) != expected:
        raise ValueError('training corpus content changed')
    after = path.stat()
    if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('training corpus changed during verification')
    dump(receipt_path, dict(path=str(path), sha256=expected, bytes=st.st_size, mtime_ns=st.st_mtime_ns))


def check(job):
    job = Path(job).resolve()
    digest = verify_job(job)
    receipt = dict(job_digest=digest, checks=verify_sources(job), epoch=time.time(),
                   gpu_work_started=False, acceptance=False)
    dump(job / "cpu-preflight.json", receipt)
    return receipt


def run(job, gate_file, raw_evidence):
    """Run only on explicit invocation, after CPU checks and a fresh host gate."""
    import fcntl
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise ValueError("assertion-eliding Python modes are forbidden")
    job = Path(job).resolve()
    meta = read(job / "job.json")
    lock = (job / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("launcher.started", "launcher-exit.json", "candidate"):
        if (job / name).exists():
            raise FileExistsError("job already attempted; preserve it and prepare a new job")
    # Repeat integrity/source checks on run: an old preflight is never permission.
    digest = verify_job(job)
    verify_sources(job)
    gate = read(gate_file)
    validate_gate(gate, meta["device"], digest)
    if sha(raw_evidence) != gate["raw_evidence_sha256"]:
        raise ValueError("host resource evidence changed")
    dump(job / "resource-gate.json", gate)
    shutil.copyfile(raw_evidence, job / "resource-evidence.txt")
    with (job / "launcher.started").open("x") as f:
        f.write(str(time.time()))
    mode = meta["mode"]
    command = [meta["settings"]["python"], "-S", str(job / "bootstrap.py"), str(job / ENTRIES[mode])]
    if mode != "evaluation":
        command += ["candidate"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    # -O disables assertions in the frozen accepted harness: explicitly reject it.
    if env.get("PYTHONOPTIMIZE"):
        raise ValueError("PYTHONOPTIMIZE is forbidden")
    process = None
    tree = None
    rc, timed_out = 1, False
    started = time.time()
    try:
        with (job / "console.log").open("x") as log:
            process = subprocess.Popen(command, cwd=job, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            from task_tree import TaskTree
            tree = TaskTree(process)
            try:
                rc = tree.wait(meta["timeout_seconds"])
                # Some runtimes create separate worker sessions. A successful
                # parent exit does not authorize leaving its descendants behind.
                if tree.observe():
                    tree.stop()
            except subprocess.TimeoutExpired:
                timed_out = True
                tree.stop()
                rc = 124
    except BaseException:
        rc = 1
        if process is not None:
            (tree.stop() if tree is not None else stop_group(process))
        raise
    finally:
        dump(job / "launcher-exit.json", dict(returncode=rc, timed_out=timed_out,
             started_epoch=started, ended_epoch=time.time(), command=command,
             automatic_acceptance=False))
    return rc


def stop_group(process):
    from task_tree import TaskTree
    TaskTree(process).stop()


def training_command(plan, root):
    """Use explicit -S import paths in both the torch launcher and its worker."""
    expected = ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', '1']
    if plan['command'][1:6] != expected:
        raise ValueError('unexpected distributed command')
    bootstrap = [plan['command'][0], '-S', str(root / 'bootstrap.py')]
    prefix = [*bootstrap, '--module', 'torch.distributed.run', '--standalone', '--nproc-per-node', '1']
    worker = ['--no-python', *bootstrap, str(root / 'timing_worker.py'), '--plan',
              str(Path(plan['output_dir']) / 'launch.json')]
    return prefix, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("mode", choices=MODES)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p = subs.add_parser("check")
    p.add_argument("--job", type=Path, required=True)
    p = subs.add_parser("run")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--resource-evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        job = prepare(args.mode, read(args.settings), args.job)
        print(json.dumps(dict(job=str(job), job_digest=verify_job(job), gpu_work_started=False)))
    elif args.action == "check":
        print(json.dumps(check(args.job)))
    else:
        raise SystemExit(run(args.job, args.gate, args.resource_evidence))


if __name__ == "__main__":
    if sys.flags.optimize:
        raise RuntimeError("assertion-eliding Python modes are forbidden")
    main()
