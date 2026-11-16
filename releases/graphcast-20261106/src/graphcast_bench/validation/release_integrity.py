"""Check source-delivery paths and file hashes without changing the package."""

from pathlib import Path

from graphcast_bench.assets import read_json, sha256


def checked_path(root, relative):
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Release path escapes root: " + relative)
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("Symlinks are not release sources: " + relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_release(root):
    root = Path(root)
    manifest = read_json(root / "release.json")
    if manifest.get("schema") != "graphcast.release.v1":
        raise ValueError("Unknown release schema")
    expected = manifest["files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(expected) | {"release.json"}:
        raise ValueError("Missing or extra release files")
    for name, digest in expected.items():
        if sha256(checked_path(root, name)) != digest:
            raise ValueError("Release checksum mismatch: " + name)
    return dict(status="verified", file_count=len(expected), evidence=manifest["evidence"])
