"""Container-only path adapters and native builds, independent of model math."""

import json
import os
import subprocess
import sys
from pathlib import Path

from .catalog import read, sha256


def alias(target, source):
    if not Path("/.dockerenv").exists() or os.environ.get("GRAPHCAST_DELIVERY_LOCKED") != "1":
        raise RuntimeError("Runtime aliases require the locked delivery container")
    source, target = Path(source).resolve(), Path(target)
    if not any(source == Path(p) or Path(p) in source.parents for p in ("/bundle", "/work")):
        raise ValueError("Alias source must be delivery material or owned work")
    if not source.exists():
        raise ValueError("Alias source missing: " + str(source))
    if target.is_symlink():
        old = target.resolve()
        if not any(old == Path(p) or Path(p) in old.parents for p in ("/bundle", "/work")):
            raise ValueError("Refusing to replace an unrelated container alias: " + str(target))
        target.unlink()
    elif target.exists():
        raise ValueError("Alias target is an existing real path: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def build_router(release, build, kind="training"):
    release, build = Path(release), Path(build)
    build.mkdir(parents=True, exist_ok=True)
    compiler = "/public/software/compiler/dtk-26.04/bin/hipcc"
    if kind != "training_two":
        from graphcast_bench.platforms.native import verify_native

        name = "libxf32_rne_router.so" if kind == "inference" else "training-router-candidate.so"
        target = build / kind / name
        if not target.exists():
            subprocess.check_call(
                [
                    sys.executable,
                    str(release / "scripts/build_kernels.py"),
                    "--kind",
                    kind,
                    "--hipcc",
                    compiler,
                    "--include",
                    sys.prefix + "/include",
                    "--library-dir",
                    sys.prefix + "/lib",
                    "--output",
                    str(target.parent),
                ]
            )
        verify_native(target, kind)
        return str(target)
    root = release / "kernels/training_rne_two"
    sources = {p.name: sha256(p) for p in root.iterdir() if p.suffix in (".cc", ".hip", ".h")}
    target = build / "training_two" / "training-router-two-gpu.so"
    if target.exists():
        record = read(target.parent / "build.json")
        if record["sources"] != sources or record["library_sha256"] != sha256(target):
            raise ValueError("Cached two-device router differs from source/build record")
        return str(target)
    target.parent.mkdir(exist_ok=False)
    cmd = [
        compiler,
        "--offload-arch=gfx938",
        "-O2",
        "-fPIC",
        "-shared",
        "-I" + str(root),
        "-I" + sys.prefix + "/include",
        "-L" + sys.prefix + "/lib",
    ]
    cmd += [str(root / n) for n in sorted(sources) if n.endswith((".cc", ".hip"))]
    cmd += ["-o", str(target), "-ldl", "-pthread"]
    subprocess.check_call(cmd)
    (target.parent / "build.json").write_text(
        json.dumps(dict(sources=sources, command=cmd, library_sha256=sha256(target)), indent=2)
        + "\n"
    )
    return str(target)


def run(context):
    """Dispatch the host-selected engine from the second entry inside the container.

    Material paths use /bundle; output/build paths use /work. Reuse the resolved
    plan; the platform executor creates stages, launches workers, and validates results.
    """
    if not Path("/.dockerenv").exists() or os.environ.get("GRAPHCAST_DELIVERY_LOCKED") != "1":
        raise RuntimeError("Use the host delivery entrypoint")
    # Host sampler writes into stage directories created by the root runtime.
    # The enclosing host work directory is private to this recipient account.
    # Group write access lets the sampler follow new stages while the worker runs;
    # the host cleanup later restores output ownership to the recipient UID/GID.
    os.setgid(context["owner_gid"])
    os.umask(0o002)
    if context["engine"] == "operational_bw1100":
        from .operational_bw1100 import run as execute
    elif context["engine"] == "operational_h100":
        from .operational_h100 import run as execute
    elif context["engine"] == "full38":
        from .full38 import run as execute
    else:
        from .shapes import run as execute
    return execute(context)
