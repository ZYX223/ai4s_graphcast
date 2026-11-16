"""Compose tested shifted LayerNorm and exact mesh gather, retaining regular3."""
import regular3_hooks,shifted_ln_hooks,mesh_gather_hooks

def install(include_regular3=True):
    if include_regular3:regular3_hooks.install()
    shifted_ln_hooks.install(include_regular3=False)
    mesh_gather_hooks.install()

def save_audit(path):
    shifted_ln_hooks.save_audit(path)
    mesh_gather_hooks.save_audit(path.with_name('mesh-gather-audit.json'))
