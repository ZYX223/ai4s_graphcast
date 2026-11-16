"""Install unchanged H100 numerical hooks at explicit variant boundaries."""
import importlib
import os
from pathlib import Path
import sys

from graphcast_bench.assets import ROOT


class HookPolicy:
    def __init__(self, context):
        self.single = context.config['recipe'] == 'single'
        engine = ROOT / 'src/graphcast_bench/engines/operational_h100/inference'
        if context.path('inference_engine').resolve() != engine.resolve():
            raise ValueError('Inference hook directory differs from the delivery')
        # Legacy hook files import siblings by name; confine this adapter to the worker.
        sys.path.insert(0,str(engine))
        os.environ['GRAPHCAST_VARIANT'] = 'shifted_mesh'
        self.regular3 = importlib.import_module('regular3_hooks')
        self.candidate = importlib.import_module('candidate_hooks')
        self.layernorm = importlib.import_module('shifted_ln_hooks')
        self.mesh = importlib.import_module('mesh_gather_hooks')
        for module in (self.regular3,self.candidate,self.layernorm,self.mesh):
            if Path(module.__file__).resolve().parent != engine.resolve():
                raise ValueError('Imported an unrelated H100 hook module')

    def prepare_round(self, round_id):
        if round_id == 1:
            if self.single:
                self.regular3.install()
            else:
                self.candidate.install()
            return True
        if self.single and round_id == 2:
            self.candidate.install(include_regular3=False)
            return True
        return False

    def save_audits(self, output):
        errors = []
        # Attempt every independent audit even if an earlier hook never traced.
        for module,name in ((self.regular3,'regular3-audit.json'),
                            (self.layernorm,'aggregation-audit.json'),
                            (self.mesh,'mesh-gather-audit.json')):
            try:
                module.save_audit(output / name)
            except Exception as error:
                errors.append(name + ': ' + repr(error))
        if errors:
            raise RuntimeError('; '.join(errors))
