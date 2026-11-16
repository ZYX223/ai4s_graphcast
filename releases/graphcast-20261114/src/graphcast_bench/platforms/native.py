"""Build and verify the inference and training routers independently."""

import argparse
import json
import subprocess
from pathlib import Path

from graphcast_bench.assets import ROOT, read_json, sha256, verify_files, write_json


def sources(kind):
    if kind == "training":
        return [
            "kernels/training_rne/" + name
            for name in (
                "training_router_candidate.cc",
                "pre_rounded_check.hip",
                "rne_kernel.hip",
                "rne_kernel.h",
            )
        ]
    if kind == "inference":
        return (
            ["kernels/inference_rne/xf32_rne_router.cc"]
            + [
                "kernels/xf32_rne/" + name
                for name in ("rne_kernel.hip", "rne_kernel.h", "weights.tsv", "weights.json")
            ]
            + [
                "configs/runtime/inference/policy.json",
                "configs/runtime/inference/fused_layers.txt",
            ]
        )
    raise ValueError("Unknown native router kind")


def verify_native(library, kind):
    build = read_json(library.parent / "build.json")
    if build.get("schema") != "graphcast.native.v1" or build["kind"] != kind:
        raise ValueError("Use a recorded native build of the correct kind")
    if set(build["managed_sources"]) != set(sources(kind)):
        raise ValueError("Incomplete native source manifest")
    verify_files(ROOT, build["managed_sources"])
    if sha256(library) != build["library_sha256"]:
        raise ValueError("Native library differs from its build manifest")
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("inference", "training"), required=True)
    parser.add_argument("--hipcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include", type=Path, action="append", default=[])
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inputs = sources(args.kind)
    name = "libxf32_rne_router.so" if args.kind == "inference" else "training-router-candidate.so"
    output = args.output.resolve()
    library = output / name
    command = [str(args.hipcc), "--offload-arch=gfx938", "-O2", "-fPIC", "-shared"]
    includes = args.include + [
        ROOT / ("kernels/xf32_rne" if args.kind == "inference" else "kernels/training_rne")
    ]
    command += ["-I" + str(p) for p in includes] + ["-L" + str(p) for p in args.library_dir]
    command += [str(ROOT / p) for p in inputs if Path(p).suffix in (".cc", ".hip")]
    command += ["-o", str(library), "-lcrypto", "-ldl", "-pthread"]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    hashes = {p: sha256(ROOT / p) for p in inputs}
    subprocess.run(command, check=True)
    verify_files(ROOT, hashes)
    build = dict(
        schema="graphcast.native.v1",
        kind=args.kind,
        command=command,
        compiler=subprocess.check_output([str(args.hipcc), "--version"], text=True),
        library_sha256=sha256(library),
        managed_sources=hashes,
    )
    if args.kind == "inference":
        # Router's legacy checks still cover the common RNE kernel and 84 weight identities.
        build["sources"] = {
            Path(p).name: value for p, value in hashes.items() if p.startswith("kernels/xf32_rne/")
        }
    write_json(output / "build.json", build)
    verify_native(library, args.kind)
