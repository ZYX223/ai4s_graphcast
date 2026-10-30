"""Portable asset and upstream-source identity checks (no accelerator import)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_COMMIT = "08cf73625c9d12bd9aaa038868bcb2fe488f2a22"
STATS = ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_files(root, files):
    root = Path(root).resolve()
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Asset path escapes root: " + relative)
        if sha256(path) != expected:
            raise ValueError("SHA256 mismatch: " + str(path))


def verify_source():
    folder = ROOT / "third_party/graphcast"
    pin = read_json(folder / "UPSTREAM.json")
    if pin["commit"] != OFFICIAL_COMMIT:
        raise ValueError("Unexpected official GraphCast version")
    verify_files(folder, pin["files"])
    actual = {str(p.relative_to(folder)) for p in (folder / "graphcast").glob("*.py")}
    if actual != {x for x in pin["files"] if x.endswith(".py")}:
        raise ValueError("Unexpected upstream Python module")
    return pin


def load_case(root):
    root = Path(root)
    case = read_json(ROOT / "configs/case.json")
    prepared = read_json(root / "manifest.json")
    if prepared.get("schema") != "graphcast.jax_case.v1":
        raise ValueError("Prepare the portable JAX case first")
    if prepared.get("case_config_sha256") != sha256(ROOT / "configs/case.json"):
        raise ValueError("Prepared case differs from pinned configuration")
    verify_files(root, case["assets"])
    verify_files(root, {"reference.npz": prepared["reference_sha256"]})
    return case, prepared


def code_snapshot():
    files = []
    for folder in ("src", "scripts", "configs", "kernels", "third_party"):
        files.extend(
            p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
