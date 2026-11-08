"""Private-process gradient clipping routes to frozen public FlagGems functions.

This is project integration, not an upstream foreach implementation. Only no-grad
nonempty contiguous FP32 CUDA tensors enter the new numerical paths. Other
schemas, dtypes, layouts and autograd paths retain native dispatch. No model
names, dimensions, values or global Torch dispatcher registrations are changed.
"""
from contextlib import nullcontext
import functools
import math


def eligible(x, grad_enabled, allow_bool=False):
    dtypes = ("torch.float32", "torch.bool") if allow_bool else ("torch.float32",)
    return (not grad_enabled and str(getattr(x, "dtype", None)) in dtypes
            and str(getattr(x, "device", "")).split(":")[0] == "cuda"
            and str(getattr(x, "layout", None)) == "torch.strided"
            and not getattr(x, "requires_grad", True) and x.numel() > 0
            and x.is_contiguous())


def select_route(name, args, kwargs, grad_enabled):
    if not args:
        return None
    get = lambda index, key, default: args[index] if len(args) > index else kwargs.get(key, default)
    dtype = get(2, "dtype", None) if name == "aten._foreach_norm.Scalar" else kwargs.get("dtype")
    if name == "aten._foreach_norm.Scalar":
        xs = args[0]
        if (isinstance(xs, (list, tuple)) and xs and get(1, "ord", 2) == 2
                and (dtype is None or str(dtype) == "torch.float32")
                and all(eligible(x, grad_enabled) for x in xs)
                and len({str(x.device) for x in xs}) == 1):
            return "foreach_norm"
        return None
    if not eligible(args[0], grad_enabled, allow_bool=name == "aten.logical_or.default"):
        return None
    if name == "aten.linalg_vector_norm.default":
        if (get(1, "ord", 2) == 2 and get(2, "dim", None) is None
                and not get(3, "keepdim", False)
                and (dtype is None or str(dtype) == "torch.float32")):
            return "vector_norm"
    elif name in ("aten.isinf.default", "aten.reciprocal.default"):
        return name.split(".")[1]
    elif name == "aten.logical_or.default":
        if (len(args) == 2 and eligible(args[1], grad_enabled, allow_bool=True)
                and str(args[0].device) == str(args[1].device)):
            return "logical_or"
    elif name == "aten.clamp.default":
        lo, hi = get(1, "min", None), get(2, "max", None)
        if ((lo is not None or hi is not None)
                and all(v is None or (isinstance(v, (float, int)) and math.isfinite(v)) for v in (lo, hi))):
            return "clamp"
    return None


def clamp_preserve_nan(gems, value, minimum, maximum):
    # Frozen backend clamp loses NaNs. Compose existing public operations rather
    # than relaxing correctness or changing the shared/public kernel source.
    clipped = gems.clamp(value, mini=minimum, maxi=maximum)
    return gems.where_self(gems.isnan(value), value, clipped)


class ClipRouting:
    def __init__(self, torch, gems, annotation=None):
        from torch.utils._python_dispatch import TorchDispatchMode
        self.torch, self.gems, self.annotation = torch, gems, annotation
        self.calls = {}
        self.fallback_calls = {}
        routing = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kw = kwargs or {}
                route = select_route(str(func), args, kw, torch.is_grad_enabled())
                if route is None:
                    name = str(func)
                    routing.fallback_calls[name] = routing.fallback_calls.get(name, 0) + 1
                    return func(*args, **kw)
                name = str(func)
                scope = routing.annotation(name) if routing.annotation else nullcontext()
                with scope:
                    if route == "foreach_norm":
                        value = [gems.vector_norm(x, ord=2, dim=None, keepdim=False,
                                                  dtype=torch.float32) for x in args[0]]
                    elif route == "vector_norm":
                        value = gems.vector_norm(args[0], ord=2, dim=None, keepdim=False, dtype=torch.float32)
                    elif route == "clamp":
                        lo = args[1] if len(args) > 1 else kw.get("min")
                        hi = args[2] if len(args) > 2 else kw.get("max")
                        value = clamp_preserve_nan(gems, args[0], lo, hi)
                    else:
                        value = getattr(gems, route)(*args, **kw)
                routing.calls[name] = routing.calls.get(name, 0) + 1
                return value

        self.Mode = Mode
        self.original = None

    def mode(self):
        return self.Mode()

    def install(self):
        if self.original is not None:
            raise RuntimeError("duplicate clip routing installation")
        self.original = self.torch.nn.utils.clip_grad_norm_
        original, routing = self.original, self

        @functools.wraps(original)
        def clipped(*args, **kwargs):
            with routing.mode():
                return original(*args, **kwargs)

        self.wrapper = clipped
        self.torch.nn.utils.clip_grad_norm_ = clipped
        return self

    def close(self):
        if self.original is not None:
            if self.torch.nn.utils.clip_grad_norm_ is not self.wrapper:
                raise RuntimeError("clip function binding changed unexpectedly")
            self.torch.nn.utils.clip_grad_norm_ = self.original
            self.original = None

    def record(self):
        return dict(kind="project scoped composition using existing FlagGems public functions",
                    scope="torch.nn.utils.clip_grad_norm_ only", calls=self.calls,
                    native_fallback_calls=self.fallback_calls,
                    public_kernels_modified=False, shared_environment_modified=False)
