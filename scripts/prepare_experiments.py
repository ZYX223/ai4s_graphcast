#!/usr/bin/env python3
"""Stage measured experiment sources and external inputs without running a GPU job."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ("rne-producer-fusion-20261022", "rne-model-fusion-20261022")
TRAINING = (
    "jax-training-38-20261023",
    "jax-trained-inference-40-20261023",
    "jax-trained-inference-diagnosis-20261023",
    "jax-first-update-replay-20261023",
)


def verify_archive(repo=ROOT):
    manifest = json.loads((repo / "experiments/PROVENANCE.json").read_text())
    for row in manifest["files"]:
        path = (repo / row["path"]).resolve()
        if repo.resolve() not in path.parents:
            raise ValueError("Archive path escapes repository: " + row["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Archive SHA256 mismatch: " + row["path"])
    return len(manifest["files"])


def prepare(assets, output, suite="all", reuse_training=False, repo=ROOT):
    assets, output, repo = assets.resolve(), output.resolve(), repo.resolve()
    if output.exists():
        raise FileExistsError("Output must be a new directory: " + str(output))
    if repo == output or repo in output.parents:
        raise ValueError("Choose a work directory outside the code repository")
    verify_archive(repo)
    inference = suite in ("inference", "all")
    training = suite in ("training", "all")
    if reuse_training and not training:
        raise ValueError("--reuse-training-results requires training or all")
    links = {
        "jax-reorganization-20261016/prepared/data": assets
        / "jax-reorganization-20261016/prepared/data",
    }
    if inference:
        for name in ("capture-r3/run", "bench-full-r2/logical"):
            links["logical-operators-20261022/" + name] = (
                assets / "logical-operators-20261022" / name
            )
    if training:
        for name in ("dependencies", "data"):
            links["jax-single-card-training-20261022/" + name] = (
                assets / "jax-single-card-training-20261022" / name
            )
    training_data = assets / "jax-training-38-20261023/data"
    if training:
        for name in ("source.json", "hres-20220101-025-13-12.nc"):
            if not (training_data / name).is_file():
                raise FileNotFoundError("Missing training input: " + str(training_data / name))
    for name, source in links.items():
        if not source.is_dir():
            raise FileNotFoundError("Missing external asset for " + name + ": " + str(source))
    historical = []
    if reuse_training:
        # Copy small reports, hardlink large immutable payloads. These paths are
        # bind-mounted read-only by the inference/replay launchers. Absolute
        # symlinks inside /training would not resolve inside the container.
        for mode in ("fp32", "rne_xf32"):
            for horizon in (1, 12):
                folder = "%s-r%02d" % (mode, horizon)
                names = ["report.json", "checkpoint.pkl"]
                if mode == "fp32" and horizon == 1:
                    names += ["gradient-01.npy", "gradient-02.npy"]
                for name in names:
                    rel = Path("jax-training-38-20261023") / folder / name
                    source = assets / rel
                    if not source.is_file():
                        raise FileNotFoundError("Missing historical training asset: " + str(source))
                    historical.append((rel, source))
        ancestor = output.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if any(source.stat().st_dev != ancestor.stat().st_dev for _, source in historical):
            raise ValueError("Reusing checkpoints requires work and assets on the same filesystem")

    output.mkdir(parents=True)
    names = (INFERENCE if inference else ()) + (TRAINING if training else ())
    for name in names:
        shutil.copytree(
            str(repo / "experiments" / name),
            str(output / name),
            ignore=shutil.ignore_patterns("__pycache__", "data"),
        )
    shared = output / "shared"
    shutil.copytree(str(repo / "experiments/shared"), str(shared))
    # A real copy is used for each Docker mount root; no container needs to
    # resolve a host-only absolute symlink within a mounted directory.
    for name in INFERENCE:
        if name in names or name == "rne-model-fusion-20261022":
            destination = output / name / "baseline-source"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(shared / "baseline"), str(destination))
    if training:
        destination = output / "jax-single-card-training-20261022/official"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(shared / "training-official"), str(destination))
        curriculum = output / "jax-training-38-20261023/run_curriculum.py"
        text = curriculum.read_text()
        if text.count("device=12\n") != 1:
            raise ValueError("Unexpected archived curriculum launcher")
        curriculum.write_text(
            text.replace("device=12\n", "device=int(os.environ.get('GRAPHCAST_DEVICE','12'))\n")
        )
    for relative, source in links.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)
    if training:
        data_destination = output / "jax-training-38-20261023/data"
        data_destination.mkdir()
        for name in ("source.json", "hres-20220101-025-13-12.nc"):
            source, destination = training_data / name, data_destination / name
            if source.suffix == ".nc" and source.stat().st_dev == output.stat().st_dev:
                os.link(str(source), str(destination))
            else:
                shutil.copy2(str(source), str(destination))
    for relative, source in historical:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in (".pkl", ".npy"):
            os.link(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))
    record = {
        "suite": suite,
        "assets_root": str(assets),
        "historical_training_reused": reuse_training,
        "external_inputs": {name: str(path) for name, path in links.items()},
        "training_data_source": str(training_data) if training else None,
        "reused_training_files": [str(rel) for rel, _ in historical],
        "adaptation": "Staged curriculum reads GRAPHCAST_DEVICE (default 12); numerical code unchanged.",
        "source_sha256": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in names
            for path in (output / name).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".sh", ".cc", ".hip", ".h")
            and "data" not in path.parts
        },
    }
    (output / "workspace.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--suite", choices=("inference", "training", "all"), default="all")
    parser.add_argument("--reuse-training-results", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print("Verified %d archived files" % verify_archive())
        return
    if args.assets_root is None or args.output is None:
        parser.error("--assets-root and --output are required for workspace preparation")
    prepare(args.assets_root, args.output, args.suite, args.reuse_training_results)
    print("Prepared workspace: " + str(args.output.resolve()))


if __name__ == "__main__":
    main()
