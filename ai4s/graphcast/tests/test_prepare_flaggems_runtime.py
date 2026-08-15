from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

GRAPHCAST_ROOT = Path(__file__).resolve().parents[1]
if str(GRAPHCAST_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_ROOT))

from tools.prepare_flaggems_runtime import PreparationError, prepare_runtime  # noqa: E402

COMMON_BRANCH = "zyx/flaggems-op-prcheck"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(path: Path, branch: str) -> None:
    path.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch", branch, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(path, "config", "user.name", "Runtime Test")
    _git(path, "config", "user.email", "runtime-test@example.com")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_overlay(common: Path, name: str, files: dict[str, str]) -> None:
    overlay = common / "patches" / "flaggems" / name
    for relative, content in files.items():
        _write(overlay / "files" / relative, content)
    lines = [
        f"{_sha256(overlay / 'files' / relative)}  files/{relative}" for relative in sorted(files)
    ]
    _write(overlay / "MANIFEST.sha256", "\n".join(lines) + "\n")


def _write_manifest(
    common: Path,
    upstream: Path,
    overlays: list[str],
    *,
    allowed_overwrites: list[str] | None = None,
) -> None:
    manifest = {
        "version": 1,
        "scope": "graphcast_inference",
        "upstream": {
            "repository": upstream.resolve().as_uri(),
            "branch": "master",
        },
        "overlays": overlays,
        "runtime_roots": ["src", "conf"],
        "allowed_overwrites": allowed_overwrites or [],
    }
    _write(
        common / "patches" / "flaggems" / "runtime-candidate.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
    )


@pytest.fixture
def runtime_repositories(tmp_path: Path) -> dict[str, Path]:
    upstream = tmp_path / "upstream"
    _init_repo(upstream, "master")
    _write(upstream / "src" / "flag_gems" / "__init__.py", "VERSION = 'official-v1'\n")
    _write(upstream / "src" / "flag_gems" / "ops" / "base.py", "BASE = 1\n")
    _write(upstream / "conf" / "operators.yaml", "operators: []\n")
    _write(upstream / "tests" / "upstream.txt", "upstream test\n")
    _commit(upstream, "initial upstream")

    common = tmp_path / "common"
    _init_repo(common, COMMON_BRANCH)
    _write_overlay(
        common,
        "op-a",
        {
            "src/flag_gems/ops/a.py": "A = 1\n",
            "tests/test_a.py": "raise AssertionError('must not be copied')\n",
        },
    )
    _write_manifest(common, upstream, ["op-a"])
    _commit(common, "add runtime overlay")

    return {
        "upstream": upstream,
        "common": common,
        "cache": tmp_path / "cache",
        "dest": tmp_path / "runtime",
    }


def _prepare(repositories: dict[str, Path], **kwargs):
    return prepare_runtime(
        common_remote=repositories["common"].resolve().as_uri(),
        common_branch=COMMON_BRANCH,
        cache_root=repositories["cache"],
        destination=repositories["dest"],
        **kwargs,
    )


def test_first_prepare_builds_matched_official_and_internal_trees(
    runtime_repositories: dict[str, Path],
) -> None:
    result = _prepare(runtime_repositories)
    destination = runtime_repositories["dest"]
    official = destination / "official" / "FlagGems"
    internal = destination / "internal" / "FlagGems"

    assert (official / "src/flag_gems/ops/base.py").is_file()
    assert not (official / "src/flag_gems/ops/a.py").exists()
    assert (internal / "src/flag_gems/ops/a.py").read_text() == "A = 1\n"
    assert not (internal / "tests/test_a.py").exists()
    assert (internal / "tests/upstream.txt").is_file()
    assert result["upstream_commit"] == _git(runtime_repositories["upstream"], "rev-parse", "HEAD")
    assert result["overlays"] == ["op-a"]

    provenance = (destination / "PROVENANCE.env").read_text()
    assert f"FLAGGEMS_UPSTREAM_RESOLVED_COMMIT={result['upstream_commit']}" in provenance
    assert f"FLAGOS_COMMON_RESOLVED_COMMIT={result['common_commit']}" in provenance
    assert "FLAGGEMS_OVERLAYS=op-a" in provenance
    assert f"FLAGGEMS_OFFICIAL_SOURCE_ROOT={destination.resolve()}/official/FlagGems" in provenance
    assert f"FLAGGEMS_INTERNAL_SOURCE_ROOT={destination.resolve()}/internal/FlagGems" in provenance


def test_online_prepare_fetches_new_remote_commits(
    runtime_repositories: dict[str, Path],
) -> None:
    first = _prepare(runtime_repositories)
    upstream = runtime_repositories["upstream"]
    common = runtime_repositories["common"]

    _write(upstream / "src" / "flag_gems" / "ops" / "base.py", "BASE = 2\n")
    upstream_commit = _commit(upstream, "update upstream")
    _write_overlay(common, "op-a", {"src/flag_gems/ops/a.py": "A = 2\n"})
    common_commit = _commit(common, "update overlay")

    second = _prepare(runtime_repositories)

    assert second["upstream_commit"] == upstream_commit != first["upstream_commit"]
    assert second["common_commit"] == common_commit != first["common_commit"]
    assert (
        runtime_repositories["dest"] / "official/FlagGems/src/flag_gems/ops/base.py"
    ).read_text() == "BASE = 2\n"
    assert (
        runtime_repositories["dest"] / "internal/FlagGems/src/flag_gems/ops/a.py"
    ).read_text() == "A = 2\n"


def test_offline_prepare_reuses_existing_cache(
    runtime_repositories: dict[str, Path],
) -> None:
    first = _prepare(runtime_repositories)
    shutil.rmtree(runtime_repositories["common"])
    shutil.rmtree(runtime_repositories["upstream"])

    second = _prepare(runtime_repositories, offline=True)

    assert second["common_commit"] == first["common_commit"]
    assert second["upstream_commit"] == first["upstream_commit"]


def test_offline_prepare_requires_existing_cache(tmp_path: Path) -> None:
    with pytest.raises(PreparationError, match="offline cache is missing"):
        prepare_runtime(
            common_remote="file:///missing/common",
            common_branch=COMMON_BRANCH,
            cache_root=tmp_path / "cache",
            destination=tmp_path / "runtime",
            offline=True,
        )


def test_manifest_failure_preserves_previous_runtime(
    runtime_repositories: dict[str, Path],
) -> None:
    _prepare(runtime_repositories)
    destination = runtime_repositories["dest"]
    previous_provenance = (destination / "PROVENANCE.env").read_bytes()
    previous_internal = (destination / "internal/FlagGems/src/flag_gems/ops/a.py").read_bytes()

    manifest = runtime_repositories["common"] / "patches/flaggems/op-a/MANIFEST.sha256"
    manifest.write_text("0" * 64 + "  files/src/flag_gems/ops/a.py\n")
    _commit(runtime_repositories["common"], "damage manifest")

    with pytest.raises(PreparationError, match="checksum mismatch"):
        _prepare(runtime_repositories)

    assert (destination / "PROVENANCE.env").read_bytes() == previous_provenance
    assert (
        destination / "internal/FlagGems/src/flag_gems/ops/a.py"
    ).read_bytes() == previous_internal


def test_unknown_overlay_fails_before_replacing_runtime(
    runtime_repositories: dict[str, Path],
) -> None:
    _prepare(runtime_repositories)
    destination = runtime_repositories["dest"]
    previous = (destination / "PROVENANCE.env").read_bytes()
    _write_manifest(
        runtime_repositories["common"],
        runtime_repositories["upstream"],
        ["missing-overlay"],
    )
    _commit(runtime_repositories["common"], "select missing overlay")

    with pytest.raises(PreparationError, match="missing-overlay"):
        _prepare(runtime_repositories)

    assert (destination / "PROVENANCE.env").read_bytes() == previous


def test_duplicate_runtime_file_requires_explicit_allowance(
    runtime_repositories: dict[str, Path],
) -> None:
    common = runtime_repositories["common"]
    upstream = runtime_repositories["upstream"]
    duplicate = "src/flag_gems/ops/a.py"
    _write_overlay(common, "op-b", {duplicate: "A = 2\n"})
    _write_manifest(common, upstream, ["op-a", "op-b"])
    _commit(common, "add conflicting overlay")

    with pytest.raises(PreparationError, match="undeclared overlay overwrite"):
        _prepare(runtime_repositories)

    _write_manifest(
        common,
        upstream,
        ["op-a", "op-b"],
        allowed_overwrites=[duplicate],
    )
    _commit(common, "allow overlay overwrite")
    _prepare(runtime_repositories)

    assert (runtime_repositories["dest"] / "internal/FlagGems" / duplicate).read_text() == "A = 2\n"


def test_refresh_rebuilds_a_damaged_cache(
    runtime_repositories: dict[str, Path],
) -> None:
    _prepare(runtime_repositories)
    shutil.rmtree(runtime_repositories["cache"] / "flagos-common" / ".git")

    with pytest.raises(PreparationError, match="not a Git checkout"):
        _prepare(runtime_repositories)

    result = _prepare(runtime_repositories, refresh=True)

    assert result["common_commit"] == _git(runtime_repositories["common"], "rev-parse", "HEAD")
