#!/usr/bin/env python3
"""Check a delivery independently of older releases; never starts a GPU.

Hashes all source files and material manifests. Material payloads are checked
for readability here; the run entry performs their full SHA check before GPU use.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphcast_bench.delivery.catalog import checked_path, read, sha256


def verify(release, platform, plans=False):
    release = Path(release).resolve()
    bundle = release.parent.parent
    if release.parent.name != "releases":
        raise ValueError("Expected <bundle>/releases/<version>")
    record = read(release / "release.json")
    if record.get("schema") != "graphcast.delivery_release.v2" or not record.get("files"):
        raise ValueError("Invalid delivery release manifest")
    for name, expected in record["files"].items():
        if sha256(checked_path(release, name)) != expected:
            raise ValueError("Source SHA256 mismatch: " + name)
    actual = {str(p.relative_to(release)) for p in release.rglob("*")
              if p.is_file() or p.is_symlink()}
    if actual != set(record["files"]) | {"release.json"}:
        raise ValueError("Missing or extra release files")

    manifests, files, omitted = 0, 0, []
    for relative, expected in read(release / "configs/artifacts.lock.json")["artifacts"].items():
        artifact = checked_path(bundle, relative)
        environment = relative.startswith("environments/")
        if (environment and not relative.startswith("environments/" + platform + "/")
                and not (artifact / "manifest.json").exists()):
            omitted.append(relative)
            continue
        if sha256(artifact / "manifest.json") != expected:
            raise ValueError("Artifact manifest SHA256 mismatch: " + relative)
        manifests += 1
        if environment:
            continue  # No expensive environment payload scan in this lightweight check.
        material = read(artifact / "manifest.json")
        for name, destination in material.get("symlinks", {}).items():
            path = artifact / name
            checked_path(bundle, str(Path(relative) / name))
            if not path.is_symlink() or os.readlink(str(path)) != destination:
                raise ValueError("Material symlink mismatch: " + relative + "/" + name)
        for name in material["files"]:
            with checked_path(bundle, str(Path(relative) / name)).open("rb") as stream:
                stream.read(1)
            files += 1

    run_plans = []
    if plans:
        for case_path in sorted((release / "configs/cases").glob("*.json")):
            case = read(case_path)
            for device in ("bw1100", "h100"):
                for workload in case["workloads"]:
                    output = subprocess.check_output([
                        "bash", str(release / "scripts/run.sh"), "--platform", device,
                        "--case", case["id"], "--workload", workload, "--plan",
                    ])
                    run_plans.append(json.loads(output))
    return dict(status="passed", release=record["version"],
                release_sha256=sha256(release / "release.json"),
                source_payload_files=len(record["files"]),
                material_manifests_verified=manifests, material_files_readable=files,
                omitted_other_platform_environments=omitted, plans=run_plans,
                material_payload_rehashed=False, gpu_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT)
    parser.add_argument("--platform", choices=("bw1100", "h100"), default="bw1100")
    parser.add_argument("--plans", action="store_true", help="also inspect all run plans, without GPU")
    args = parser.parse_args()
    print(json.dumps(verify(args.release, args.platform, args.plans), indent=2))


if __name__ == "__main__":
    main()
