"""Check source-delivery paths and file hashes without changing the package."""

from pathlib import Path

from graphcast_bench.assets import read_json, sha256


def checked_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Release path escapes root: " + relative)
    # Host-side verification also runs with Python 3.6 on the delivery server.
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError("Release path escapes root: " + relative)
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("Symlinks are not release sources: " + relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_release(root):
    root = Path(root)
    manifest = read_json(root / "release.json")
    schema = manifest.get("schema")
    if schema not in ("graphcast.release.v1", "graphcast.delivery_release.v2"):
        raise ValueError("Unknown release schema")
    expected = manifest["files"]
    if not isinstance(expected, dict) or (schema == "graphcast.delivery_release.v2" and not expected):
        raise ValueError("Invalid release file manifest")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(expected) | {"release.json"}:
        raise ValueError("Missing or extra release files")
    for name, digest in expected.items():
        if sha256(checked_path(root, name)) != digest:
            raise ValueError("Release checksum mismatch: " + name)
    if schema == "graphcast.release.v1":
        return dict(status="verified", file_count=len(expected), evidence=manifest["evidence"])
    # This command verifies source integrity, not environment/data payloads or
    # GPU execution. verify_delivery.py additionally checks material manifests.
    return dict(
        status="verified", file_count=len(expected), schema=schema,
        version=manifest["version"], scope="source files and SHA256 only",
        gpu_execution=False,
    )
