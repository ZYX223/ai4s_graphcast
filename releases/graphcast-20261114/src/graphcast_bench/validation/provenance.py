"""Verify maintained code and byte-identical upstream before installing hooks."""

from pathlib import Path

from graphcast_bench.assets import ROOT, read_json, sha256, verify_files, verify_source


def verify_module(module):
    path = Path(module.__file__).resolve()
    relative = str(path.relative_to(ROOT))
    if relative.startswith("third_party/graphcast/"):
        verify_source()
    else:
        files = read_json(ROOT / "configs/runtime/code-manifest.json")["files"]
        if relative not in files:
            raise ValueError("Unrecorded maintained dependency: " + relative)
        verify_files(ROOT, {relative: files[relative]})
    return sha256(path)


def verify_maintained():
    verify_source()
    manifest = read_json(ROOT / "configs/runtime/code-manifest.json")
    verify_files(ROOT, manifest["files"])
    return manifest
