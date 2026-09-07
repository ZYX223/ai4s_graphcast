"""Data-parallel process lifecycle for model-parallel GraphCast replicas."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import torch
import torch.distributed as dist

from . import training_accelerator


def _destroy_process_group(group: Any | None = None) -> None:
    """Destroy Gloo safely on vendor builds that expose an unrelated backend."""

    backend = str(dist.get_backend(group))
    distributed_c10d = getattr(dist, "distributed_c10d", None)
    is_mccl_available = getattr(distributed_c10d, "is_mccl_available", None)
    if backend != "gloo" or not callable(is_mccl_available) or not is_mccl_available():
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
    global_valid_count = _positive_integer(global_valid_count, name="global_valid_count")
    if (
        isinstance(local_valid_count, bool)
        or not isinstance(local_valid_count, int)
        or local_valid_count < 0
        or local_valid_count > global_valid_count
    ):
        raise ValueError("local_valid_count must be in [0, global_valid_count]")
    return replicas * local_valid_count / global_valid_count


@dataclass
class DistributedContext:
    """One DP rank owning one disjoint model-parallel device group."""

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
        dp_replicas = _positive_integer(dp_replicas, name="dp_replicas")
        mp_devices_per_replica = _positive_integer(
            mp_devices_per_replica,
            name="mp_devices_per_replica",
        )
        try:
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as error:
            raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must be integers") from error
        if world_size != dp_replicas:
            raise ValueError(f"WORLD_SIZE={world_size} must equal DP replicas {dp_replicas}")
        if rank < 0 or rank >= world_size:
            raise ValueError("RANK is outside WORLD_SIZE")
        if local_rank < 0:
            raise ValueError("LOCAL_RANK must be non-negative")

        required_devices = dp_replicas * mp_devices_per_replica
        resolved_accelerator_type = (
            accelerator_type or os.environ.get("GRAPHCAST_ACCELERATOR_TYPE") or "cuda"
        ).strip().lower()
        if available_device_count is None:
            resolved_accelerator_type = training_accelerator.resolve_accelerator_type(
                accelerator_type
            )
            available_device_count = training_accelerator.accelerator_device_count(
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
        device_indices = tuple(range(start, start + mp_devices_per_replica))

        owns_process_group = False
        resolved_backend = backend
        initialized = dist.is_available() and dist.is_initialized()
        if initialize:
            if not dist.is_available():
                raise RuntimeError("torch.distributed is unavailable")
            if not initialized:
                resolved_backend = resolved_backend or (
                    training_accelerator.default_process_group_backend(
                        resolved_accelerator_type
                    )
                )
                if (
                    str(resolved_backend) in {"nccl", "mccl", "hccl"}
                    and not training_accelerator.accelerator_is_available(
                        resolved_accelerator_type
                    )
                ):
                    raise RuntimeError(
                        f"{str(resolved_backend).upper()} topology requires "
                        f"{resolved_accelerator_type.upper()}"
                    )
                if str(resolved_backend) in {"nccl", "mccl", "hccl"}:
                    training_accelerator.set_accelerator_device(
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
                        "existing process group disagrees with torchrun environment"
                    )
                resolved_backend = dist.get_backend()
        object_group = None
        if initialized and str(resolved_backend) in {"nccl", "mccl", "hccl"}:
            object_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
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
    def primary_device(self) -> torch.device:
        return torch.device(self.accelerator_type, self.device_indices[0])

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def local_batch_size(self, *, global_batch_size: int) -> int:
        global_batch_size = _positive_integer(global_batch_size, name="global_batch_size")
        if global_batch_size % self.dp_replicas:
            raise ValueError("global batch size must be divisible by DP replicas")
        return global_batch_size // self.dp_replicas

    def barrier(self) -> None:
        if self.initialized:
            if self._object_group is not None:
                dist.barrier(group=self._object_group)
            elif str(self.backend) in {"nccl", "mccl", "hccl"}:
                dist.barrier(device_ids=[self.device_indices[0]])
            else:
                dist.barrier()

    def prewarm_device_collectives(self) -> int:
        """Create one tensor communicator for every local MP stage device."""

        if not self.initialized or str(self.backend) not in {"nccl", "mccl", "hccl"}:
            return 0
        expected = self.world_size * (self.world_size + 1) / 2.0
        for device_index in self.device_indices:
            training_accelerator.set_accelerator_device(
                device_index,
                self.accelerator_type,
            )
            probe = torch.tensor(
                float(self.rank + 1),
                dtype=torch.float32,
                device=torch.device(self.accelerator_type, device_index),
            )
            dist.all_reduce(probe, op=dist.ReduceOp.SUM)
            training_accelerator.synchronize_devices(
                (device_index,),
                self.accelerator_type,
            )
            actual = float(probe.item())
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-6):
                raise RuntimeError(
                    f"collective prewarm on device {device_index} returned "
                    f"{actual}, expected {expected}"
                )
        training_accelerator.set_accelerator_device(
            self.device_indices[0],
            self.accelerator_type,
        )
        self.barrier()
        return len(self.device_indices)

    def global_max_seconds(self, local_seconds: float) -> float:
        return _all_reduce_scalar(local_seconds, operation=dist.ReduceOp.MAX, context=self)

    def global_sum(self, value: float) -> float:
        return _all_reduce_scalar(value, operation=dist.ReduceOp.SUM, context=self)

    def gather_objects(self, value: Any) -> list[Any]:
        if not self.initialized:
            return [value]
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value, group=self._object_group)
        return gathered

    def synchronize_gradients(self, module: torch.nn.Module) -> int:
        """Deterministically average named gradients across DP-over-MP replicas."""

        parameters = [
            (name, parameter)
            for name, parameter in sorted(module.named_parameters())
            if parameter.requires_grad and parameter.grad is not None
        ]
        names = tuple(name for name, _ in parameters)
        if self.initialized:
            names_by_rank = self.gather_objects(names)
            if any(tuple(row) != names for row in names_by_rank):
                raise RuntimeError("DP replicas produced different gradient parameter sets")
            for _, parameter in parameters:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(self.world_size)
        return len(parameters)

    def close(self, *, synchronize: bool = True) -> None:
        if self._owns_process_group and dist.is_available() and dist.is_initialized():
            if os.environ.get(
                "GRAPHCAST_SKIP_PROCESS_GROUP_TEARDOWN", ""
            ).strip().lower() in {"1", "true", "yes", "on"}:
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
    if not context.initialized:
        return number
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
