#!/usr/bin/env python3
"""Fetch and compose matched official/internal FlagGems source trees."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

DEFAULT_COMMON_REMOTE = "https://github.com/flagos-adaptation-lab/flagos-common.git"
DEFAULT_COMMON_BRANCH = "zyx/flaggems-op-prcheck"
MANIFEST_RELATIVE_PATH = Path("patches/flaggems/runtime-candidate.yaml")
SUPPORTED_RUNTIME_ROOTS = {"src", "conf"}


class PreparationError(RuntimeError):
    """Raised when a runtime cannot be prepared without ambiguity."""


def _run_git(*args: str, cwd: Path | None = None) -> str:
    command = ["git"]
    if cwd is not None:
        command.extend(("-C", str(cwd)))
    command.extend(args)
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "git command failed").strip()
        raise PreparationError(f"{' '.join(command)}: {detail}") from exc
    return result.stdout.strip()


def sync_repository(
    remote: str,
    branch: str,
    destination: Path,
    *,
    offline: bool,
    refresh: bool,
) -> str:
    """Clone or update one managed cache and return its resolved commit."""

    git_dir = destination / ".git"
    if destination.exists() and not git_dir.exists():
        if not refresh:
            raise PreparationError(f"cache is not a Git checkout: {destination}")
        shutil.rmtree(destination)

    if not destination.exists():
        if offline:
            raise PreparationError(f"offline cache is missing: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            branch,
            remote,
            str(destination),
        )
    elif not offline:
        _run_git("remote", "set-url", "origin", remote, cwd=destination)
        _run_git("fetch", "--depth", "1", "origin", branch, cwd=destination)
        _run_git("checkout", "--detach", "FETCH_HEAD", cwd=destination)
        _run_git("reset", "--hard", "FETCH_HEAD", cwd=destination)
        _run_git("clean", "-fdx", cwd=destination)

    return _run_git("rev-parse", "HEAD", cwd=destination)


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _safe_relative_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PreparationError(f"{context} is not a safe relative path: {value!r}")
    return path.as_posix()


def load_runtime_manifest(common_root: Path) -> dict[str, Any]:
    """Load and validate the active runtime manifest from flagos-common."""

    path = common_root / MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise PreparationError(f"runtime manifest is missing: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PreparationError(f"invalid runtime manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise PreparationError(f"runtime manifest {path} must use version 1")

    upstream = data.get("upstream")
    if not isinstance(upstream, dict):
        raise PreparationError("runtime manifest upstream must be a mapping")
    repository = _require_string(upstream, "repository", "upstream")
    branch = _require_string(upstream, "branch", "upstream")

    overlays = data.get("overlays")
    if not isinstance(overlays, list) or not all(isinstance(item, str) for item in overlays):
        raise PreparationError("runtime manifest overlays must be a list of names")
    normalized_overlays = []
    for name in overlays:
        safe_name = _safe_relative_path(name, "overlay")
        if "/" in safe_name or safe_name.startswith("wip-"):
            raise PreparationError(f"runtime overlay is not eligible: {name}")
        normalized_overlays.append(safe_name)
    if len(set(normalized_overlays)) != len(normalized_overlays):
        raise PreparationError("runtime manifest contains duplicate overlays")

    roots = data.get("runtime_roots")
    if not isinstance(roots, list) or not roots:
        raise PreparationError("runtime_roots must be a non-empty list")
    normalized_roots = []
    for root in roots:
        if not isinstance(root, str) or root not in SUPPORTED_RUNTIME_ROOTS:
            raise PreparationError(f"unsupported runtime root: {root!r}")
        normalized_roots.append(root)

    allowed = data.get("allowed_overwrites", [])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise PreparationError("allowed_overwrites must be a list of paths")
    normalized_allowed = {_safe_relative_path(item, "allowed overwrite") for item in allowed}

    return {
        "repository": repository,
        "branch": branch,
        "overlays": normalized_overlays,
        "runtime_roots": normalized_roots,
        "allowed_overwrites": normalized_allowed,
    }


def _resolve_manifest_file(
    common_root: Path,
    overlay_root: Path,
    manifest_name: str,
) -> tuple[Path, str]:
    normalized = _safe_relative_path(manifest_name.lstrip("*"), "manifest path")
    candidates = (
        (common_root / normalized, normalized),
        (overlay_root / normalized, f"{overlay_root.name}/{normalized}"),
    )
    for candidate, display in candidates:
        if candidate.is_file():
            return candidate, display
    raise PreparationError(
        f"manifest entry is missing for overlay {overlay_root.name}: {manifest_name}"
    )


def verify_overlay_manifest(common_root: Path, overlay_name: str) -> set[str]:
    """Verify all overlay hashes and return overlay-relative manifest paths."""

    overlay_root = common_root / "patches" / "flaggems" / overlay_name
    files_root = overlay_root / "files"
    manifest = overlay_root / "MANIFEST.sha256"
    if not files_root.is_dir() or not manifest.is_file():
        raise PreparationError(f"overlay {overlay_name} is missing files/ or MANIFEST.sha256")

    verified: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise PreparationError(f"invalid checksum entry in {manifest}:{line_number}")
        expected, manifest_name = fields
        source, display = _resolve_manifest_file(common_root, overlay_root, manifest_name.strip())
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected.lower():
            raise PreparationError(f"checksum mismatch for overlay {overlay_name}: {display}")
        try:
            overlay_relative = source.relative_to(overlay_root).as_posix()
        except ValueError:
            prefix = f"patches/flaggems/{overlay_name}/"
            common_relative = source.relative_to(common_root).as_posix()
            if not common_relative.startswith(prefix):
                raise PreparationError(
                    f"manifest entry escapes overlay {overlay_name}: {common_relative}"
                ) from None
            overlay_relative = common_relative[len(prefix) :]
        verified.add(overlay_relative)
    if not verified:
        raise PreparationError(f"overlay manifest is empty: {overlay_name}")
    return verified


def _copy_upstream(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git"),
    )


def _apply_overlays(
    common_root: Path,
    internal_root: Path,
    manifest: dict[str, Any],
) -> None:
    owners: dict[str, str] = {}
    allowed = manifest["allowed_overwrites"]
    for overlay_name in manifest["overlays"]:
        overlay_root = common_root / "patches" / "flaggems" / overlay_name
        files_root = overlay_root / "files"
        verified = verify_overlay_manifest(common_root, overlay_name)
        for runtime_root in manifest["runtime_roots"]:
            source_root = files_root / runtime_root
            if not source_root.exists():
                continue
            for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
                relative = source.relative_to(files_root).as_posix()
                manifest_relative = f"files/{relative}"
                if manifest_relative not in verified:
                    raise PreparationError(
                        f"runtime file is not covered by {overlay_name} manifest: {relative}"
                    )
                previous = owners.get(relative)
                if previous is not None and relative not in allowed:
                    raise PreparationError(
                        f"undeclared overlay overwrite: {relative} ({previous} -> {overlay_name})"
                    )
                destination = internal_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                owners[relative] = overlay_name


def _write_provenance(
    output_root: Path,
    runtime_destination: Path,
    *,
    common_remote: str,
    common_branch: str,
    common_commit: str,
    upstream_remote: str,
    upstream_branch: str,
    upstream_commit: str,
    overlays: list[str],
) -> None:
    values = {
        "FLAGOS_COMMON_REPOSITORY": common_remote,
        "FLAGOS_COMMON_BRANCH": common_branch,
        "FLAGOS_COMMON_RESOLVED_COMMIT": common_commit,
        "FLAGGEMS_UPSTREAM_REPOSITORY": upstream_remote,
        "FLAGGEMS_UPSTREAM_BRANCH": upstream_branch,
        "FLAGGEMS_UPSTREAM_RESOLVED_COMMIT": upstream_commit,
        "FLAGGEMS_OVERLAYS": ",".join(overlays),
        "FLAGGEMS_OFFICIAL_SOURCE_ROOT": str(
            (runtime_destination / "official" / "FlagGems").resolve()
        ),
        "FLAGGEMS_INTERNAL_SOURCE_ROOT": str(
            (runtime_destination / "internal" / "FlagGems").resolve()
        ),
    }
    content = "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
    (output_root / "PROVENANCE.env").write_text(content, encoding="utf-8")


def _replace_destination(staged: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if had_destination and backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def compose_runtime(
    common_root: Path,
    upstream_root: Path,
    destination: Path,
    manifest: dict[str, Any],
    *,
    common_remote: str,
    common_branch: str,
    common_commit: str,
    upstream_commit: str,
) -> None:
    """Build both trees in a sibling directory and replace them together."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staged-", dir=destination.parent))
    try:
        official = staged / "official" / "FlagGems"
        internal = staged / "internal" / "FlagGems"
        _copy_upstream(upstream_root, official)
        shutil.copytree(official, internal)
        _apply_overlays(common_root, internal, manifest)
        for root in (official, internal):
            if not (root / "src" / "flag_gems" / "__init__.py").is_file():
                raise PreparationError(f"FlagGems source tree is incomplete: {root}")
        _write_provenance(
            staged,
            destination,
            common_remote=common_remote,
            common_branch=common_branch,
            common_commit=common_commit,
            upstream_remote=manifest["repository"],
            upstream_branch=manifest["branch"],
            upstream_commit=upstream_commit,
            overlays=manifest["overlays"],
        )
        _replace_destination(staged, destination)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def prepare_runtime(
    *,
    common_remote: str,
    common_branch: str,
    cache_root: Path,
    destination: Path,
    offline: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Synchronize sources, compose both variants, and return provenance data."""

    if offline and refresh:
        raise PreparationError("--offline and --refresh cannot be used together")
    cache_root = Path(cache_root)
    destination = Path(destination)
    common_cache = cache_root / "flagos-common"
    upstream_cache = cache_root / "FlagGems"

    common_commit = sync_repository(
        common_remote,
        common_branch,
        common_cache,
        offline=offline,
        refresh=refresh,
    )
    manifest = load_runtime_manifest(common_cache)
    upstream_commit = sync_repository(
        manifest["repository"],
        manifest["branch"],
        upstream_cache,
        offline=offline,
        refresh=refresh,
    )
    compose_runtime(
        common_cache,
        upstream_cache,
        destination,
        manifest,
        common_remote=common_remote,
        common_branch=common_branch,
        common_commit=common_commit,
        upstream_commit=upstream_commit,
    )
    return {
        "common_commit": common_commit,
        "upstream_commit": upstream_commit,
        "overlays": list(manifest["overlays"]),
        "destination": str(destination.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_cache = Path(
        os.environ.get(
            "FLAGGEMS_CACHE_ROOT",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "flagos",
        )
    )
    default_destination = Path(
        os.environ.get(
            "FLAGGEMS_RUNTIME_DEST",
            project_root / "third_party" / "flaggems-runtime",
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true", help="rebuild damaged caches")
    mode.add_argument("--offline", action="store_true", help="reuse caches without network")
    parser.add_argument(
        "--common-remote",
        default=os.environ.get("FLAGOS_COMMON_REMOTE", DEFAULT_COMMON_REMOTE),
    )
    parser.add_argument(
        "--common-branch",
        default=os.environ.get("FLAGOS_COMMON_BRANCH", DEFAULT_COMMON_BRANCH),
    )
    parser.add_argument("--cache-root", type=Path, default=default_cache)
    parser.add_argument("--destination", type=Path, default=default_destination)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = prepare_runtime(
            common_remote=args.common_remote,
            common_branch=args.common_branch,
            cache_root=args.cache_root,
            destination=args.destination,
            offline=args.offline,
            refresh=args.refresh,
        )
    except PreparationError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(f"FlagGems runtimes prepared at {result['destination']}")
    print(f"flagos-common commit: {result['common_commit']}")
    print(f"FlagGems upstream commit: {result['upstream_commit']}")
    print(f"internal overlays: {','.join(result['overlays'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
