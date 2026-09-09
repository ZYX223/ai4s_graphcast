"""Data-parallel process lifecycle for model-parallel GraphCast replicas."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


DEVICE_BUSY_DEFINITION = (
    "mean across participating devices of each device's unioned kernel "
    "intervals inside the matching ProfilerStep"
)
_ACCELERATOR_API: Any | None = None


def _accelerator_api() -> Any:
    """Load the sibling boundary without importing the heavy package root.

    The standalone collective tests deliberately import this file by path.
    Loading the sibling directly preserves that useful test mode and avoids
    executing ``graphcast_compat.__init__`` merely to initialize collectives.
    """

    global _ACCELERATOR_API
    if _ACCELERATOR_API is not None:
        return _ACCELERATOR_API
    path = Path(__file__).with_name("accelerator.py")
    module_name = "_graphcast_accelerator_for_distributed"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load accelerator boundary: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    _ACCELERATOR_API = module
    return module


def _destroy_process_group(group: Any | None = None) -> None:
    """Destroy a process group despite the torch_musa 2.5 Gloo teardown bug.

    That vendor build reports MCCL as available and then unconditionally asks
    every process group for a MUSA backend during destruction.  A CPU-only
    Gloo group has no such backend, so the otherwise completed job exits with
    ``No backend type associated with device type musa``.  Temporarily hiding
    MCCL only while a Gloo group is destroyed lets PyTorch run its normal
    cleanup path; MCCL groups and other backends are left untouched.
    """

    backend = str(dist.get_backend(group))
    distributed_c10d = getattr(dist, "distributed_c10d", None)
    is_mccl_available = getattr(distributed_c10d, "is_mccl_available", None)
    if (
        backend != "gloo"
        or not callable(is_mccl_available)
        or not is_mccl_available()
    ):
        dist.destroy_process_group(group)
        return

    distributed_c10d.is_mccl_available = lambda: False
    try:
        dist.destroy_process_group(group)
    finally:
        distributed_c10d.is_mccl_available = is_mccl_available


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def loss_scale_for_valid_samples(
    *,
    local_valid_count: int,
    global_valid_count: int,
    replicas: int,
) -> float:
    """Scale a local mean so replica averaging yields the global valid mean."""

    replicas = _positive_integer(replicas, name="replicas")
    global_valid_count = _positive_integer(
        global_valid_count,
        name="global_valid_count",
    )
    if (
        isinstance(local_valid_count, bool)
        or not isinstance(local_valid_count, int)
        or local_valid_count < 0
        or local_valid_count > global_valid_count
    ):
        raise ValueError(
            "local_valid_count must be in [0, global_valid_count]"
        )
    return replicas * local_valid_count / global_valid_count


@dataclass
class DistributedContext:
    """One DDP rank owning a disjoint model-parallel device group."""

    rank: int
    local_rank: int
    world_size: int
    dp_replicas: int
    mp_devices_per_replica: int
    device_indices: tuple[int, ...]
    backend: str | None
    initialized: bool
    accelerator_type: str = "cuda"
    _owns_process_group: bool = False
    _object_group: Any | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        dp_replicas: int,
        mp_devices_per_replica: int,
        available_device_count: int | None = None,
        initialize: bool = True,
        backend: str | None = None,
        accelerator_type: str | None = None,
    ) -> "DistributedContext":
        dp_replicas = _positive_integer(
            dp_replicas,
            name="dp_replicas",
        )
        mp_devices_per_replica = _positive_integer(
            mp_devices_per_replica,
            name="mp_devices_per_replica",
        )
        try:
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError(
                "RANK, LOCAL_RANK, and WORLD_SIZE must be integers"
            ) from error
        if world_size != dp_replicas:
            raise ValueError(
                f"WORLD_SIZE={world_size} must equal DP replicas "
                f"{dp_replicas}"
            )
        if rank < 0 or rank >= world_size:
            raise ValueError("RANK is outside WORLD_SIZE")
        if local_rank < 0:
            raise ValueError("LOCAL_RANK must be non-negative")
        required_devices = dp_replicas * mp_devices_per_replica
        resolved_accelerator_type = (
            accelerator_type
            or os.environ.get("GRAPHCAST_ACCELERATOR_TYPE")
            or "cuda"
        ).strip().lower()
        if available_device_count is None:
            accelerator = _accelerator_api()
            resolved_accelerator_type = accelerator.resolve_accelerator_type(
                accelerator_type
            )
            available_device_count = accelerator.accelerator_device_count(
                resolved_accelerator_type
            )
        if (
            isinstance(available_device_count, bool)
            or not isinstance(available_device_count, int)
            or available_device_count < required_devices
        ):
            raise ValueError(
                f"topology requires {required_devices} accelerator devices; "
                f"only {available_device_count} are available"
            )
        start = rank * mp_devices_per_replica
        device_indices = tuple(
            range(start, start + mp_devices_per_replica)
        )

        owns_process_group = False
        resolved_backend = backend
        initialized = dist.is_available() and dist.is_initialized()
        if initialize:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is unavailable")
            if not initialized:
                accelerator = _accelerator_api()
                resolved_backend = resolved_backend or (
                    accelerator.default_process_group_backend(
                        resolved_accelerator_type
                    )
                )
                if (
                    str(resolved_backend) in {"nccl", "mccl", "hccl"}
                    and not accelerator.accelerator_is_available(
                        resolved_accelerator_type
                    )
                ):
                    raise RuntimeError(
                        f"{str(resolved_backend).upper()} topology requires "
                        f"{resolved_accelerator_type.upper()}"
                    )
                if str(resolved_backend) in {"nccl", "mccl", "hccl"}:
                    accelerator.set_accelerator_device(
                        device_indices[0],
                        resolved_accelerator_type,
                    )
                dist.init_process_group(
                    backend=resolved_backend,
                    init_method="env://",
                    rank=rank,
                    world_size=world_size,
                )
                owns_process_group = True
                initialized = True
            else:
                if dist.get_rank() != rank or dist.get_world_size() != world_size:
                    raise RuntimeError(
                        "existing process group disagrees with torchrun "
                        "environment"
                    )
                resolved_backend = dist.get_backend()
        object_group = None
        if initialized and str(resolved_backend) in {"nccl", "mccl", "hccl"}:
            # Accelerator object collectives stage pickled Python payloads
            # through a guessed device. That is fragile here because each
            # process owns four devices rather than one LOCAL_RANK device.
            # Keep tensors on NCCL/MCCL and metadata on a CPU Gloo subgroup.
            object_group = dist.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
            )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            dp_replicas=dp_replicas,
            mp_devices_per_replica=mp_devices_per_replica,
            device_indices=device_indices,
            backend=resolved_backend,
            initialized=initialized,
            accelerator_type=resolved_accelerator_type,
            _owns_process_group=owns_process_group,
            _object_group=object_group,
        )

    @property
    def participating_device_count(self) -> int:
        return self.dp_replicas * self.mp_devices_per_replica

    @property
    def object_collective_backend(self) -> str | None:
        if self._object_group is not None:
            return "gloo"
        return self.backend

    @property
    def control_collective_backend(self) -> str | None:
        """Backend used for lifecycle and checkpoint synchronization."""

        if self._object_group is not None:
            return "gloo"
        return self.backend

    @property
    def primary_device(self) -> torch.device:
        return torch.device(
            self.accelerator_type,
            self.device_indices[0],
        )

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def local_batch_size(self, *, global_batch_size: int) -> int:
        global_batch_size = _positive_integer(
            global_batch_size,
            name="global_batch_size",
        )
        if global_batch_size % self.dp_replicas:
            raise ValueError(
                "global batch size must be divisible by DP replicas"
            )
        return global_batch_size // self.dp_replicas

    def barrier(self) -> None:
        if self.initialized and self.world_size > 1:
            if self._object_group is not None:
                # Lifecycle and checkpoint barriers carry no model tensors.
                # Keep them off the multi-device-per-process NCCL group: a
                # rank may spend materially longer serializing its four-device
                # state, and an unmatched NCCL barrier otherwise occupies the
                # primary GPU until its watchdog aborts the process group.
                dist.barrier(group=self._object_group)
            elif str(self.backend) in {"nccl", "mccl", "hccl"}:
                dist.barrier(device_ids=[self.device_indices[0]])
            else:
                dist.barrier()

    def prewarm_device_collectives(self) -> int:
        """Create one tensor communicator for every model-parallel stage.

        Some vendor runtimes allocate a per-device communicator workspace on
        the first collective. Full-grid GraphCast can leave little memory by
        the time gradient synchronization starts, so creating those unchanged
        communication resources before model construction avoids a late,
        capacity-dependent initialization failure.
        """

        if not self.initialized or self.world_size == 1 or str(self.backend) not in {"nccl", "mccl", "hccl"}:
            return 0
        accelerator = _accelerator_api()
        expected = self.world_size * (self.world_size + 1) / 2.0
        for device_index in self.device_indices:
            accelerator.set_accelerator_device(
                device_index,
                self.accelerator_type,
            )
            probe = torch.tensor(
                float(self.rank + 1),
                dtype=torch.float32,
                device=torch.device(self.accelerator_type, device_index),
            )
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
            accelerator.synchronize_devices(
                (device_index,),
                self.accelerator_type,
            )
            actual = float(probe.item())
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6):
                raise RuntimeError(
                    f"collective prewarm on device {device_index} returned "
                    f"{actual}, expected {expected}"
                )
        accelerator.set_accelerator_device(
            self.device_indices[0],
            self.accelerator_type,
        )
        self.barrier()
        return len(self.device_indices)

    def global_max_seconds(self, local_seconds: float) -> float:
        return _all_reduce_scalar(
            local_seconds,
            operation=dist.ReduceOp.MAX,
            context=self,
        )

    def global_sum(self, value: float) -> float:
        return _all_reduce_scalar(
            value,
            operation=dist.ReduceOp.SUM,
            context=self,
        )

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.initialized or self.world_size == 1:
            return [value]
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            gathered,
            value,
            group=self._object_group,
        )
        return gathered

    def wrap_ddp(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.initialized:
            raise RuntimeError("DDP requires an initialized process group")
        return torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            # The compatibility topology deliberately retains Google
            # mesh2grid mesh-node processor/decoder heads in the state dict,
            # although the forecast loss consumes only the grid-node head.
            # Their stable absence from autograd must be reported to DDP or
            # the reducer fails when the second optimizer iteration starts.
            find_unused_parameters=True,
        )

    def synchronize_gradients(self, module: torch.nn.Module) -> int:
        """Deterministically average every gradient across DP replicas.

        Used only by the explicit manual synchronization path. DDP callers
        rely on its reducer and must not perform this second collective pass.
        """

        if self.world_size == 1:
            return 0
        parameters = [
            (name, parameter)
            for name, parameter in sorted(module.named_parameters())
            if parameter.requires_grad and parameter.grad is not None
        ]
        names = tuple(name for name, _ in parameters)
        if self.initialized and self.world_size > 1:
            names_by_rank = self.gather_objects(names)
            if any(tuple(row) != names for row in names_by_rank):
                raise RuntimeError(
                    "DP replicas produced different gradient parameter sets"
                )
            for _, parameter in parameters:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)
        return len(parameters)

    def close(self, *, synchronize: bool = True) -> None:
        if not synchronize:
            # A failed torchrun worker must exit without waiting for unfinished
            # peer collectives. Process exit aborts its communicators and lets
            # the launcher terminate peers; graceful destruction can deadlock.
            self.initialized = False
            self._owns_process_group = False
            self._object_group = None
            return
        if (
            self._owns_process_group
            and dist.is_available()
            and dist.is_initialized()
        ):
            if os.environ.get(
                "GRAPHCAST_SKIP_PROCESS_GROUP_TEARDOWN", ""
            ).strip().lower() in {"1", "true", "yes", "on"}:
                # Some vendor NCCL-compatible runtimes complete every training,
                # checkpoint and control-group barrier, but block indefinitely
                # while explicitly destroying the tensor process group.  This
                # opt-in is set only by that platform launcher: process exit is
                # then the owner of communicator cleanup.
                self.initialized = False
                self._owns_process_group = False
                self._object_group = None
                return
            try:
                if synchronize:
                    self.barrier()
            finally:
                if self._object_group is not None:
                    _destroy_process_group(self._object_group)
                    self._object_group = None
                _destroy_process_group()
                self.initialized = False
                self._owns_process_group = False

    def __enter__(self) -> "DistributedContext":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        self.close(synchronize=exc_type is None)


def _all_reduce_scalar(
    value: float,
    *,
    operation: dist.ReduceOp,
    context: DistributedContext,
) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("distributed reduction value must be finite")
    if not context.initialized or context.world_size == 1:
        return number
    # Multi-device ranks already own a CPU Gloo control group.  Use it for
    # scalar reductions: the project-pinned torch_musa 2.5 MCCL path returns
    # 1.0 for float64 scalar all-reduces, corrupting loss, FLOPs and timing
    # even though tensor-gradient collectives are correct.  NCCL/MCCL remains
    # the unchanged path for model tensors and gradients.
    control_group = context._object_group
    device = (
        torch.device("cpu")
        if control_group is not None
        else (
            context.primary_device
            if str(context.backend) in {"nccl", "mccl", "hccl"}
            else torch.device("cpu")
        )
    )
    tensor = torch.tensor(number, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=operation, group=control_group)
    return float(tensor.item())


def merge_profiled_steps(
    rank_profiles: Sequence[Mapping[int, Mapping[str, Any]]],
    *,
    expected_devices: Sequence[str | int],
) -> dict[int, dict[str, Any]]:
    """Merge rank-local profiler summaries into one physical-device view."""

    expected = tuple(str(device) for device in expected_devices)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_devices must be non-empty and unique")
    if not rank_profiles:
        return {}
    step_sets = [set(profile) for profile in rank_profiles]
    if any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError("rank profiler outputs contain different steps")

    merged: dict[int, dict[str, Any]] = {}
    for step in sorted(step_sets[0]):
        devices: dict[str, float] = {}
        profile_indices = set()
        profile_walls = []
        busy_definitions = set()
        profile_sources = set()
        for rank_profile in rank_profiles:
            profile = rank_profile[step]
            profile_indices.add(int(profile["profile_step_index"]))
            profile_walls.append(float(profile["profile_step_wall_ms"]))
            busy_definitions.add(
                str(profile.get("device_busy_definition", DEVICE_BUSY_DEFINITION))
            )
            if profile.get("profile_source") is not None:
                profile_sources.add(str(profile["profile_source"]))
            rank_devices = profile["device_busy_us_by_device"]
            if not isinstance(rank_devices, Mapping):
                raise ValueError("rank device busy data must be a mapping")
            for raw_device, raw_busy in rank_devices.items():
                device = str(raw_device)
                if device in devices:
                    raise ValueError(
                        f"duplicate profiled device {device} at step {step}"
                    )
                busy = float(raw_busy)
                if not math.isfinite(busy) or busy < 0:
                    raise ValueError("profiled device busy time is invalid")
                devices[device] = busy
        if len(profile_indices) != 1:
            raise ValueError("rank profiler step indices differ")
        if len(busy_definitions) != 1:
            raise ValueError("rank profiler busy definitions differ")
        if len(profile_sources) > 1:
            raise ValueError("rank profiler sources differ")
        if set(devices) != set(expected):
            missing = sorted(set(expected) - set(devices))
            unexpected = sorted(set(devices) - set(expected))
            raise ValueError(
                "merged profiler devices differ from expected topology: "
                f"missing={missing}, unexpected={unexpected}"
            )
        ordered_devices = {device: devices[device] for device in expected}
        total_busy_us = float(sum(ordered_devices.values()))
        merged[step] = {
            "profile_step_index": profile_indices.pop(),
            "profile_step_wall_ms": max(profile_walls),
            "device_busy_us_by_device": ordered_devices,
            "device_busy_ms": total_busy_us / len(expected) / 1.0e3,
            "total_device_busy_us": total_busy_us,
            "total_device_busy_ms": total_busy_us / 1.0e3,
            "device_busy_definition": busy_definitions.pop(),
            "profile_source": (
                profile_sources.pop() if profile_sources else None
            ),
        }
    return merged
