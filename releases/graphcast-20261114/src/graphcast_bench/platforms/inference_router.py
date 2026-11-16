import ctypes as C
from pathlib import Path

from graphcast_bench.assets import read_json
from graphcast_bench.router import Router


class CandidateRouter(Router):
    def __init__(self, library):
        super().__init__(library)
        from graphcast_bench.assets import ROOT
        from graphcast_bench.platforms.native import verify_native

        verify_native(Path(library), "inference")
        folder = ROOT / "configs/runtime/inference"
        policy = read_json(folder / "policy.json")["layers"]
        if {row["module"] for row in policy} != set(
            (folder / "fused_layers.txt").read_text().splitlines()
        ):
            raise ValueError("JAX/native policy mismatch")
        self.library.graphcast_fused_activation_calls.restype = C.c_uint64

    def snapshot(self):
        return (*super().snapshot(), self.library.graphcast_fused_activation_calls())

    def verify(self, before, mode):
        result = super().verify(before, mode)
        skipped = self.snapshot()[3] - before[3]
        if mode != "rne_xf32" or skipped != 79:
            raise RuntimeError("Expected exactly 79 activation RNE skips")
        return dict(result, fused_activation_calls=skipped)
