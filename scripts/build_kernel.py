"""Build the tested RNE interposer and record exact compiler/source/library identity."""

import argparse
import subprocess
from pathlib import Path

from graphcast_bench.assets import ROOT, sha256, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--arch", default="gfx938")
a = p.parse_args()
if a.arch != "gfx938":
    p.error("Only the tested gfx938 target is supported")
a.output.mkdir(parents=True, exist_ok=False)
source = ROOT / "kernels/xf32_rne"
library = a.output / "libxf32_rne_router.so"
command = [
    "/opt/dtk/bin/hipcc",
    "--offload-arch=" + a.arch,
    "-O2",
    "-fPIC",
    "-shared",
    str(source / "xf32_rne_router.cc"),
    str(source / "rne_kernel.hip"),
    "-o",
    str(library),
    "-lcrypto",
    "-ldl",
    "-pthread",
]
subprocess.run(command, check=True)
write_json(
    a.output / "build.json",
    {
        "command": command,
        "library_sha256": sha256(library),
        "sources": {
            f: sha256(source / f)
            for f in (
                "rne_kernel.h",
                "rne_kernel.hip",
                "xf32_rne_router.cc",
                "weights.tsv",
                "weights.json",
            )
        },
        "compiler": subprocess.check_output(["/opt/dtk/bin/hipcc", "--version"], text=True),
    },
)
