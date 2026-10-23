import ctypes as C
import os
from pathlib import Path
from graphcast_bench.router import Router
from graphcast_bench.assets import read_json,sha256

class CandidateRouter(Router):
    def __init__(self):
        super().__init__()
        folder=Path(os.environ['GRAPHCAST_ROUTER_LIBRARY']).parent
        if sha256(folder/'xf32_rne_router.cc')!=self.build['patched_source_sha256']:
            raise ValueError('Candidate source differs from build')
        for name,key in [('policy.json','policy_sha256'),('fused_layers.txt','fused_layers_sha256')]:
            if sha256(Path('/experiment')/name)!=self.build[key]:
                raise ValueError('Candidate policy differs from build')
        policy=read_json(Path('/experiment/policy.json'))['layers']
        if {r['module'] for r in policy}!=set(Path('/experiment/fused_layers.txt').read_text().splitlines()):
            raise ValueError('JAX/native policy mismatch')
        self.library.graphcast_fused_activation_calls.restype=C.c_uint64

    def snapshot(self):
        return (*super().snapshot(),self.library.graphcast_fused_activation_calls())

    def verify(self,before,mode):
        result=super().verify(before,mode)
        skipped=self.snapshot()[3]-before[3]
        if mode!='rne_xf32' or skipped!=79:
            raise RuntimeError('Expected exactly 79 activation RNE skips')
        return dict(result,fused_activation_calls=skipped)
