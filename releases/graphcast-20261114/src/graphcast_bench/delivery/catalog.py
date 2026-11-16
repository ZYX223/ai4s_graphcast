"""Resolve versioned cases and validate their immutable material bindings.

This module uses only Python 3.6+ stdlib so a recipient can inspect a release
before loading either accelerator runtime.
"""

import hashlib
import json
import os
from pathlib import Path

from .options import resolve_options
from graphcast_bench.progress import emit


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Expected a relative delivery path: " + str(relative))
    resolved = (root / part).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("Material link escapes delivery root: " + str(relative))
    return resolved


def verify_artifact(bundle, relative, expected=None, cache_dir=None, force=False):
    """Verify material identity; optionally reuse hashes for unchanged file identities."""
    from . import verification_cache

    root = checked_path(bundle, relative)
    manifest_path = root / "manifest.json"
    manifest_sha = sha256(manifest_path)
    if expected and manifest_sha != expected:
        raise ValueError("Artifact manifest SHA256 mismatch: " + relative)
    manifest = read(manifest_path)
    if manifest["schema"] != "graphcast.artifact.v1" or not manifest["files"]:
        raise ValueError("Invalid artifact manifest: " + relative)
    for name, destination in manifest.get("symlinks", {}).items():
        joined = str(Path(relative) / name)
        checked_path(bundle, joined)
        link = Path(bundle) / joined
        if not link.is_symlink() or os.readlink(str(link)) != destination or not link.exists():
            raise ValueError("Artifact symlink differs from manifest: " + joined)
    identity = dict(schema='graphcast.verification_cache.v1', root=str(root), manifest_sha256=manifest_sha)
    record_path = verification_cache.cache_path(cache_dir, root) if cache_dir is not None else None
    verified = verification_cache.load(record_path, identity) if record_path is not None and not force else {}
    # Retire previous on-disk evidence even for forced checks. If this check
    # fails, a later invocation must not reuse evidence it has just disproved.
    if record_path is not None and not verification_cache.invalidate(record_path):
        verified = {}
    cache, next_files = {}, {}
    reused, hashed = 0, 0
    for name, digest in manifest["files"].items():
        # Always check containment: a renamed parent can preserve file metadata
        # while redirecting the path outside the delivery root.
        path = checked_path(bundle, str(Path(relative) / name))
        # Reuse content evidence only while identity, size, mtime and ctime match.
        if name in verified:
            fingerprint = verification_cache.fingerprint(path)
            if verified[name] == fingerprint:
                next_files[name] = fingerprint
                reused += 1
                continue
        fingerprint = verification_cache.fingerprint(path)
        key = tuple(fingerprint)
        if key not in cache:
            cache[key] = sha256(path)
        if cache[key] != digest:
            raise ValueError("Artifact file SHA256 mismatch: " + relative + "/" + name +
                             "; restore the delivered file, then retry (--verify-full forces a full check)")
        if verification_cache.fingerprint(path) != fingerprint:
            raise ValueError("Artifact changed during verification; retry after updates finish: " + relative + "/" + name)
        next_files[name] = fingerprint
        hashed += 1
    if record_path is not None:
        verification_cache.save(record_path, identity, next_files)
        emit('Materials', '{}; reused {} unchanged files; rehashed {} files'.format(relative, reused, hashed))
    return manifest


def resolve_plan(release, platform, case_id, workload, overrides=None):
    release = Path(release)
    if platform not in ("bw1100", "h100"):
        raise ValueError("Unknown platform")
    if "/" in case_id or ".." in case_id:
        raise ValueError("Invalid case")
    case = read(release / "configs/cases" / (case_id + ".json"))
    options = resolve_options(workload, platform, overrides)
    recipe = options["recipe"]
    work = read(release / "configs/workloads" / (recipe + ".json"))
    if recipe not in case["workloads"]:
        raise ValueError(case_id + ": " + workload + " is not released; full 38-update training cannot be replaced by a short probe")
    training = work["kind"] == "training"
    engine = case["training_engine"] if training else case["inference_engine"]
    engine = case.get("workload_engines", {}).get(recipe, engine)
    if engine == "operational":
        engine += "_" + platform
    artifacts = dict(case["artifacts"])
    artifacts.update(case.get("workload_artifacts", {}).get(recipe, {}))
    artifacts["environment"] = "environments/" + case["environments"][platform]
    return dict(
        schema="graphcast.run_plan.v1",
        case=case_id,
        platform=platform,
        workload=options["workload"],
        run_options=options,
        kind=work["kind"],
        engine=engine,
        cards=case["training_cards"] if training else 1,
        artifacts=artifacts,
        steps=options.get("inference", {}).get("steps"),
        horizons=work.get("horizons", []),
        validation="Numerical sources frozen; new delivery GPU execution pending",
    )
