"""Audit the unchanged Hygon per-GEMM mode/rounding interposer."""

import ctypes as C
import os
from pathlib import Path

from .assets import ROOT, read_json, sha256


class Router:
    def __init__(self):
        library = Path(os.environ["GRAPHCAST_ROUTER_LIBRARY"])
        build = read_json(library.parent / "build.json")
        if sha256(library) != build["library_sha256"]:
            raise ValueError("RNE library differs from build record")
        for name, digest in build["sources"].items():
            if sha256(ROOT / "kernels/xf32_rne" / name) != digest:
                raise ValueError("RNE source differs from built library: " + name)
        self.library = C.CDLL(str(library))
        self.library.graphcast_stage_select.argtypes = [C.c_uint64, C.c_char_p]
        self.library.graphcast_round_select.argtypes = [C.c_uint64]
        for name in (
            "graphcast_stage_calls",
            "graphcast_round_calls",
            "graphcast_stage_unknown",
            "graphcast_stage_count",
        ):
            getattr(self.library, name).restype = C.c_uint64
        self.library.graphcast_stage_count.argtypes = [C.c_uint]
        self.expected = {i: 0 for i in range(9)}
        for row in read_json(ROOT / "kernels/xf32_rne/weights.json")["layers"]:
            self.expected[row["bit"]] += 1
        if sum(self.expected.values()) != 84:
            raise ValueError("Expected the frozen 84 runtime GEMMs")
        self.build = build

    def select(self, mode, label):
        mask, rnd = {"fp32": (0, 0), "xf32": (511, 0), "rne_xf32": (511, 511)}[mode]
        self.library.graphcast_stage_select(mask, label.encode())
        self.library.graphcast_round_select(rnd)

    def snapshot(self):
        return (
            self.library.graphcast_stage_calls(),
            self.library.graphcast_round_calls(),
            [self.library.graphcast_stage_count(i) for i in self.expected],
        )

    def verify(self, before, mode):
        after = self.snapshot()
        expected_round = 84 if mode == "rne_xf32" else 0
        if after[0] - before[0] != 84 or after[1] - before[1] != expected_round:
            raise RuntimeError("Runtime GEMM/rounding call count differs from contract")
        if self.library.graphcast_stage_unknown() != 0:
            raise RuntimeError("Unknown runtime weight")
        for i, count in self.expected.items():
            if after[2][i] - before[2][i] != count:
                raise RuntimeError("Runtime layer distribution changed")
        return {"native_calls": 84, "rounded_calls": expected_round}
