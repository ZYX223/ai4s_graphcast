"""Select independently frozen candidate by explicit launch variant."""
import os

def module():
    if os.environ['GRAPHCAST_VARIANT']=='fastvar':
        import fastvar_hooks
        return fastvar_hooks
    if os.environ['GRAPHCAST_VARIANT']=='stable_ln':
        import stable_ln_hooks
        return stable_ln_hooks
    if os.environ['GRAPHCAST_VARIANT']=='shifted_ln':
        import shifted_ln_hooks
        return shifted_ln_hooks
    if os.environ['GRAPHCAST_VARIANT']=='shifted_mesh':
        import combined_hooks
        return combined_hooks
    raise ValueError('Unsupported candidate variant')

def install(include_regular3=True):return module().install(include_regular3=include_regular3)
def save_audit(path):return module().save_audit(path)
