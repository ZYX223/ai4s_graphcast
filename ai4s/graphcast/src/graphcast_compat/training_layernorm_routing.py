"""Scoped model-side LayerNorm input layout for official FlagGems v5.4.0.

The official backward treats the first input dimension as the row count.
GraphCast's standard LayerNorm modules normalize the trailing feature axes;
flatten their non-normalized prefix before dispatch and restore output shape.
The operator registrations, normalization axes and parameters are unchanged.
This optional helper is not imported by the frozen Native baseline.
"""
from collections import Counter
import math
import threading

import torch
from torch import nn


class ModelLayerNormRouting:
    """Adapt standard LayerNorm instances on models built inside this scope.

    Construction is scoped to one model class. Each adapted instance keeps its
    original forward method and parameter objects; custom LayerNorm subclasses
    are left alone. Exiting restores both the constructor and instance methods.
    This context must encompass training, including checkpoint recomputation.
    """

    def __init__(self, model_type=None):
        self.model_type = model_type
        self._active = False
        self._original_constructor = None
        self._constructor = None
        self._forwards = {}
        self._counts = Counter()
        self._lock = threading.Lock()

    def __enter__(self):
        if self._active:
            raise RuntimeError("LayerNorm routing scope is already active")
        if self.model_type is None:
            from .training_model import GoogleSmallCompatibleModel
            self.model_type = GoogleSmallCompatibleModel
        if not issubclass(self.model_type, nn.Module):
            raise TypeError("LayerNorm routing requires an nn.Module model class")
        self._original_constructor = self.model_type.__init__

        def constructor(model, *args, **kwargs):
            self._original_constructor(model, *args, **kwargs)
            if type(model) is self.model_type:
                self.attach(model)

        self._constructor = constructor
        self.model_type.__init__ = constructor
        self._active = True
        return self

    def attach(self, model):
        """Idempotently attach this scope to a constructed model's LayerNorms."""
        if not self._active:
            raise RuntimeError("LayerNorm routing scope is not active")
        for module in model.modules():
            if not isinstance(module, nn.LayerNorm) or module in self._forwards:
                continue
            if type(module) is not nn.LayerNorm:
                self._counts["skipped_custom_modules"] += 1
                continue
            original = module.forward
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            routed = self._make_forward(module, original)
            module.forward = routed
            self._forwards[module] = (routed, had_instance_forward, instance_forward)
            self._counts["adapted_modules"] += 1

    def _make_forward(self, module, original):
        def forward(input):
            normalized_shape = tuple(module.normalized_shape)
            rank = len(normalized_shape)
            # Preserve the original operator's validation for invalid shapes.
            if (not isinstance(input, torch.Tensor) or not rank
                    or input.ndim < rank
                    or tuple(input.shape[-rank:]) != normalized_shape):
                return original(input)
            if input.numel() == 0:
                raise RuntimeError(
                    "empty LayerNorm input is unsupported by the official v5.4.0 "
                    "training layout adapter"
                )
            original_shape = tuple(input.shape)
            reshaped = input.ndim != rank + 1
            if reshaped:
                rows = math.prod(original_shape[:-rank])
                input = input.reshape(rows, *normalized_shape)
            output = original(input)
            with self._lock:
                self._counts["successful_calls"] += 1
                self._counts["reshaped_calls" if reshaped else "canonical_calls"] += 1
                self._counts["grad_enabled_calls" if torch.is_grad_enabled()
                             else "grad_disabled_calls"] += 1
            return output.reshape(original_shape) if reshaped else output

        return forward

    def __exit__(self, exc_type, exc_value, traceback):
        conflicts = []
        if self.model_type.__init__ is self._constructor:
            self.model_type.__init__ = self._original_constructor
        else:
            conflicts.append("model constructor")
        for module, (routed, had_instance_forward, instance_forward) in self._forwards.items():
            if module.__dict__.get("forward") is not routed:
                conflicts.append("LayerNorm instance forward")
                continue
            if had_instance_forward:
                module.forward = instance_forward
            else:
                del module.forward
        self._forwards.clear()
        self._active = False
        if conflicts:
            self._counts["restoration_conflicts"] += len(conflicts)
            if exc_type is None:
                raise RuntimeError("LayerNorm routing changed outside its scope: "
                                   + ", ".join(conflicts))
        return False

    def metadata(self):
        with self._lock:
            counts = dict(self._counts)
        return dict(route="model_layernorm_non_normalized_prefix_reshape",
                    normalized_shape_unchanged=True, parameters_unchanged=True,
                    operator_registration_unchanged=True,
                    count_kind="module_forward_calls_not_kernel_counts",
                    adapted_modules=counts.get("adapted_modules", 0),
                    skipped_custom_modules=counts.get("skipped_custom_modules", 0),
                    reshaped_calls=counts.get("reshaped_calls", 0), counts=counts)
