"""Round only the audited Linear input edges; never mutate shared tensors."""
import haiku as hk
import jax.numpy as jnp
from rne_ops import rne


class RneHook:
    def __init__(self, targets):
        self.targets = targets
        self.seen = set()

    def __call__(self, next_f, args, kwargs, context):
        module = context.module
        if (not isinstance(module, hk.Linear) or context.method_name != '__call__'
                or module.module_name not in self.targets):
            return next_f(*args, **kwargs)
        name = module.module_name
        value = args[0] if args else kwargs['inputs']
        if list(value.shape) != self.targets[name]['input_shape']:
            raise ValueError('Frozen Linear input shape changed: ' + name)
        if value.dtype != jnp.float32:
            raise TypeError('Expected FP32 Linear input: ' + name)
        self.seen.add(name)
        if args:
            return next_f(rne(value), *args[1:], **kwargs)
        return next_f(**dict(kwargs, inputs=rne(value)))

    def verify_traced(self):
        if self.seen != set(self.targets):
            raise ValueError('RNE target layers not traced: ' + str(sorted(set(self.targets) - self.seen)))
