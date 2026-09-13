"""Pipeline training execution over the shared GraphCast parameter definitions."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .google_small import (
    GoogleSmallCompatibleModel as BaseModel,
)
from .google_small import (
    GoogleSmallConfig as BaseConfig,
)
from .google_small import (
    GoogleSmallGraphNet as BaseGraphNet,
)
from .google_small import gather_node_features
from .graph import (
    CANONICAL_GRAPH_TOPOLOGY_VERSION,
    GoogleSmallGraph,
    build_google_small_graph,
)
from .precision import PrecisionLinear
from .recompute_flops import record_linear_recompute, tracked_checkpoint

DEFAULT_CONFIG = Path("configs/inference/google_graphcast_small.yaml")


def _gather_pipeline_nodes(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # Ascend's general indexing of a middle axis dominates pipeline training.
    # Reuse the leading-node-axis gather already used by ZYX inference.
    if features.device.type == "npu":
        return gather_node_features(features, indices)
    return features[:, indices, :]


def _synchronize_vendor_pipeline_device(device: torch.device | None) -> None:
    """Fence a specific destination device for vendor pipeline hand-offs."""

    if device is None or device.type not in {"musa", "npu"}:
        return
    if device.type == "musa":
        if not hasattr(torch, "musa"):
            import torch_musa  # noqa: F401
        torch.musa.synchronize(device)
    else:
        if not hasattr(torch, "npu"):
            import torch_npu  # noqa: F401
        torch.npu.synchronize(device)


def _operator_device_guard(device: torch.device):
    """Select the tensor's device for vendor replacement kernels.

    Native ATen dispatch follows each tensor's device.  Current FlagGems
    Triton launchers on MetaX/MUSA also consult the process's current device,
    which differs for the later stages of a one-process multi-device pipeline.
    Ascend's native AsStrided implementation also consults the current NPU;
    always guard NPU execution so a view can never receive an address owned by
    another PP stage.  Keep native CUDA/MUSA execution unchanged and add their
    guards only when replacements are explicitly enabled.
    """

    replacement_enabled = os.environ.get("FLAGOS_REPLACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if device.type == "npu":
        if not hasattr(torch, "npu"):
            import torch_npu  # noqa: F401
        return torch.npu.device(device)
    if not replacement_enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.cuda.device(device)
    if device.type == "musa":
        if not hasattr(torch, "musa"):
            import torch_musa  # noqa: F401
        return torch.musa.device(device)
    return nullcontext()


class _PipelineDeviceCopy(torch.autograd.Function):
    """Copy one PP chunk with an explicit, checkpoint-safe gradient path.

    The pinned Ascend runtime can retain a source-card address in the native
    CopyBackward node and later present it to a destination-card AsStrided
    kernel.  Wrapping the transfer in a custom autograd function makes both
    ownership and the reverse transfer explicit and saves no tensors in
    ``ctx``, which is compatible with the existing non-reentrant checkpoints.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, target_text: str) -> torch.Tensor:
        source = value.device
        target = torch.device(target_text)
        ctx.source_text = str(source)
        if source == target:
            return value
        return _pipeline_device_transfer(value, target, clone=True)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        source = torch.device(ctx.source_text)
        if gradient.device == source:
            return gradient, None
        return _pipeline_device_transfer(gradient, source, clone=True), None


def _pipeline_device_transfer(
    value: torch.Tensor,
    target: torch.device,
    *,
    dtype: torch.dtype | None = None,
    clone: bool = False,
) -> torch.Tensor:
    """Transfer one pipeline value with a destination-NPU safety fence.

    PyTorch already orders ordinary CUDA/HIP/MUSA copies, so their established
    path remains unchanged.  The pinned torch_npu runtime segfaults when the
    full GraphCast checkpoint graph outlives asynchronous cross-device
    allocation ownership.  Fence only the destination NPU after each pipeline
    copy; the launcher retains its independently configurable, synchronous-safe
    global launch default.  ``clone=True`` preserves the
    explicit destination ownership required by the established
    CopyBackward/AsStrided workaround.
    """

    target = torch.device(target)
    if value.device == target and (dtype is None or value.dtype == dtype):
        return value
    if value.device.type != "npu" or target.type != "npu":
        moved = value.to(device=target, dtype=dtype)
        return moved.clone() if clone else moved

    if not hasattr(torch, "npu"):
        import torch_npu  # noqa: F401
    with _operator_device_guard(value.device):
        moved = value.to(device=target, dtype=dtype)
    _synchronize_vendor_pipeline_device(target)
    if not clone:
        return moved
    with _operator_device_guard(target):
        owned = moved.clone()
    _synchronize_vendor_pipeline_device(target)
    return owned


def _pipeline_device_copy(
    value: torch.Tensor,
    target: torch.device,
) -> torch.Tensor:
    if value.device == target:
        return value
    if value.device.type != "npu" and target.type != "npu":
        # Preserve the established CUDA/HIP/MUSA/T-Head autograd path.  The
        # explicit copy node exists solely for the pinned torch_npu address-
        # ownership defect documented above.
        return value.to(target)
    return _PipelineDeviceCopy.apply(value, str(target))


@dataclass(frozen=True)
class GoogleSmallConfig(BaseConfig):
    """Minimal architecture contract for Google GraphCast_small."""

    mesh_edge_chunk_size: int | None = None
    grid_node_chunk_size: int = 131072

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> GoogleSmallConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        model_config = config["model_config"]
        task_config = config.get("task_config", {})
        channel_contract = config["channel_contract"]
        return cls(
            resolution=float(model_config["resolution"]),
            mesh_size=int(model_config["mesh_size"]),
            latent_size=int(model_config["latent_size"]),
            hidden_layers=int(model_config["hidden_layers"]),
            gnn_msg_steps=int(model_config["gnn_msg_steps"]),
            radius_query_fraction_edge_length=float(
                model_config["radius_query_fraction_edge_length"]
            ),
            mesh2grid_edge_normalization_factor=float(
                model_config["mesh2grid_edge_normalization_factor"]
            ),
            mesh_edge_chunk_size=(
                int(model_config["mesh_edge_chunk_size"])
                if model_config.get("mesh_edge_chunk_size") is not None
                else None
            ),
            mesh2grid_edge_chunk_size=(
                int(model_config["mesh2grid_edge_chunk_size"])
                if model_config.get("mesh2grid_edge_chunk_size") is not None
                else None
            ),
            grid_node_chunk_size=int(model_config.get("grid_node_chunk_size", 131072)),
            node_input_dim=int(channel_contract["grid_node_embed_input_channels"]),
            edge_input_dim=int(channel_contract["edge_feature_channels"]),
            output_dim=int(channel_contract["model_output_channels"]),
            input_variables=tuple(task_config.get("input_variables", ())),
            target_variables=tuple(task_config.get("target_variables", ())),
            forcing_variables=tuple(task_config.get("forcing_variables", ())),
            extra_mesh2grid_final_output_heads=(
                dict(model_config.get("extra_mesh2grid_final_output_heads", {})) or None
            ),
        )


class GoogleSmallGraphNet(BaseGraphNet):
    """A minimal PyTorch DeepTypedGraphNet-compatible module."""

    linear_type = PrecisionLinear

    def __init__(self, *, edge_update_chunk_sizes=None, **kwargs):
        kwargs.setdefault("node_update_chunk_sizes", None)
        kwargs.setdefault("decoder_node_chunk_sizes", None)
        super().__init__(edge_update_chunk_sizes=None, **kwargs)
        self.edge_update_chunk_sizes = dict(edge_update_chunk_sizes or {})
        self._decoder_device = None
        self._node_processor_devices = {}

    def configure_decoder_device(self, device: torch.device | None) -> None:
        """Optionally place only final node decoders on another PP stage."""

        self._decoder_device = device
        if device is None:
            return
        for node_type in self.node_output_types:
            getattr(self, f"decoder_nodes_{node_type}_mlp").to(device)

    def configure_node_processor_device(
        self,
        node_type: str,
        device: torch.device | None,
    ) -> None:
        """Optionally place one node-update path on another PP stage.

        This is an execution-only partition.  It does not change the node
        processor modules, parameters, graph, or operation order.
        """

        if node_type not in self.node_types:
            raise ValueError(f"unknown graph node type {node_type!r}")
        if device is None:
            self._node_processor_devices.pop(node_type, None)
            return
        self._node_processor_devices[node_type] = device
        for step in range(self.message_passing_steps):
            getattr(
                self,
                f"processor_nodes_{step}_{node_type}_mlp",
            ).to(device)
            getattr(
                self,
                f"processor_nodes_{step}_{node_type}_layer_norm",
            ).to(device)

    def forward(
        self,
        nodes: dict[str, torch.Tensor],
        edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        checkpoint_steps: bool = False,
        checkpoint_chunks: bool = False,
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Any:
        nodes = dict(nodes)
        edges = dict(edges)
        end = self.message_passing_steps if step_end is None else step_end
        if not 0 <= step_start <= end <= self.message_passing_steps:
            raise ValueError(
                f"invalid message step range [{step_start}, {end}) for "
                f"{self.message_passing_steps} steps"
            )

        if step_start == 0:
            raw_edges: dict = {}
            for edge_type in self.edge_types:
                chunk_size = self.edge_update_chunk_sizes.get(edge_type)
                edge_count = int(edges[edge_type].shape[1])
                if chunk_size is not None and chunk_size > 0 and edge_count > chunk_size:
                    raw_edges[edge_type] = edges[edge_type]
                    edges[edge_type] = None  # encode inline per chunk
                else:
                    edges[edge_type] = self._apply_mlp_with_layer_norm(
                        f"encoder_edges_{edge_type}", edges[edge_type]
                    )

            for node_type in self.node_types:
                base_name = f"encoder_nodes_{node_type}"
                if hasattr(self, f"{base_name}_mlp"):
                    nodes[node_type] = self._apply_mlp_with_layer_norm(base_name, nodes[node_type])
        else:
            raw_edges = {}

        for step in range(step_start, end):
            if checkpoint_steps and torch.is_grad_enabled():
                node_names = tuple(self.node_types)
                edge_names = tuple(self.edge_types)

                def run_step(
                    *flat_tensors, current_step=step, node_names=node_names, edge_names=edge_names
                ):
                    n_count = len(node_names)
                    e_count = len(edge_names)
                    step_nodes = dict(zip(node_names, flat_tensors[:n_count], strict=False))
                    step_edges = {}
                    step_raw = {}
                    for i, et in enumerate(edge_names):
                        val = flat_tensors[n_count + i]
                        step_edges[et] = val if val.shape[1] > 0 else None
                    for i, et in enumerate(edge_names):
                        val = flat_tensors[n_count + e_count + i]
                        step_raw[et] = val if val.numel() > 0 and val.shape[1] > 0 else None
                    out_nodes, out_edges = self._message_step(
                        current_step,
                        step_nodes,
                        step_edges,
                        edge_indices,
                        checkpoint_chunks=checkpoint_chunks,
                        raw_edges=(step_raw if current_step == 0 else {}),
                    )
                    return tuple(out_nodes[name] for name in node_names) + tuple(
                        out_edges[name] for name in edge_names
                    )

                all_tensors = (
                    tuple(nodes[name] for name in node_names)
                    + tuple(
                        (
                            edges.get(name)
                            if edges.get(name) is not None
                            else torch.empty(0, 0, device=nodes[node_names[0]].device)
                        )
                        for name in edge_names
                    )
                    + tuple(
                        (
                            raw_edges.get(name)
                            if raw_edges.get(name) is not None
                            else torch.empty(0, 0, device=nodes[node_names[0]].device)
                        )
                        for name in edge_names
                    )
                )
                flat_out = tracked_checkpoint(run_step, *all_tensors)
                nodes = dict(zip(node_names, flat_out[: len(node_names)], strict=False))
                edges = dict(zip(edge_names, flat_out[len(node_names) :], strict=False))
            else:
                nodes, edges = self._message_step(
                    step,
                    nodes,
                    edges,
                    edge_indices,
                    checkpoint_chunks=checkpoint_chunks,
                    raw_edges=(raw_edges if step == 0 else {}),
                )

        if end == self.message_passing_steps:
            for node_type in self.node_output_types:
                decoder = getattr(self, f"decoder_nodes_{node_type}_mlp")
                decoder_inputs = nodes[node_type]
                node_count = int(decoder_inputs.shape[1])
                chunk_sz = self.node_update_chunk_sizes.get(node_type)
                decoder_device = self._decoder_device or decoder_inputs.device
                cross_device = decoder_inputs.device != decoder_device
                if chunk_sz and node_count > chunk_sz and hasattr(decoder, "linear_1"):
                    out_dim = decoder.linear_1.out_features
                    output = decoder_inputs.new_empty(
                        decoder_inputs.shape[0],
                        node_count,
                        out_dim,
                        device=decoder_device,
                    )
                    # A full 0.25-degree latent has 1,038,240 x 512 FP32
                    # values (about 1.98 GiB).  When processor and decoder
                    # occupy different PP stages, stream the already-defined
                    # node chunks instead of first materializing that complete
                    # latent on the decoder stage.  Tensor values, decoder
                    # parameters and operation order are unchanged.
                    with _operator_device_guard(decoder_device):
                        for start in range(0, node_count, chunk_sz):
                            chunk_end = min(start + chunk_sz, node_count)
                            # FlagGems keeps the current NPU stage selected.
                            # Ascend's AsStrided kernel also consults that
                            # current stage, so create a view under the source
                            # tensor's guard before transferring the chunk.
                            with _operator_device_guard(decoder_inputs.device):
                                decoder_chunk = decoder_inputs[:, start:chunk_end]
                            if cross_device:
                                decoder_chunk = _pipeline_device_copy(
                                    decoder_chunk,
                                    decoder_device,
                                )
                                _synchronize_vendor_pipeline_device(decoder_device)
                            output[:, start:chunk_end] = decoder(decoder_chunk)
                    nodes[node_type] = output
                else:
                    if cross_device:
                        decoder_inputs = _pipeline_device_transfer(
                            decoder_inputs,
                            decoder_device,
                        )
                        _synchronize_vendor_pipeline_device(decoder_device)
                    with _operator_device_guard(decoder_device):
                        nodes[node_type] = decoder(decoder_inputs)

        return nodes, edges

    def _message_step(
        self,
        step: int,
        nodes: dict[str, torch.Tensor],
        edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        checkpoint_chunks: bool = False,
        raw_edges: dict | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if raw_edges is None:
            raw_edges = {}
        previous_nodes = nodes
        previous_edges = edges
        edge_updates = {}
        chunked_edge_outputs: dict[str, torch.Tensor] = {}
        received_by_node_type: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.node_types
        }
        for edge_type in self.edge_types:
            sender_type, receiver_type = self.edge_node_sets[edge_type]
            senders, receivers = edge_indices[edge_type]
            chunk_size = self.edge_update_chunk_sizes.get(edge_type)
            src_edge = previous_edges.get(edge_type)
            if src_edge is None:
                src_edge = (raw_edges or {}).get(edge_type)
            if src_edge is None:
                continue
            edge_count = int(src_edge.shape[1])
            factorized = self.edge_input_projection_mode == "factorized"
            sender_values = previous_nodes[sender_type]
            receiver_values = previous_nodes[receiver_type]
            if factorized:
                # Step-local differentiable projections; never cache across
                # message steps or optimizer updates. Outer step checkpointing
                # replays these along with the rest of the message step.
                sender_values, receiver_values = self._project_edge_nodes(
                    step, edge_type, sender_values, receiver_values,
                )
            if chunk_size is not None and chunk_size > 0 and edge_count > chunk_size:
                # A multi-step Mesh GraphNet needs the residual edge state as
                # input to its next message-passing step.  One-step
                # Grid-to-Mesh/Mesh-to-Grid adapters do not consume that
                # output, so retain their established allocation-saving
                # behaviour.
                preserve_edge_state = self.message_passing_steps > 1
                updated_edge: torch.Tensor | None = None
                aggregate_device = self._node_processor_devices.get(
                    receiver_type,
                    previous_nodes[receiver_type].device,
                )
                aggregated = previous_nodes[receiver_type].new_zeros(
                    previous_nodes[receiver_type].shape[0],
                    previous_nodes[receiver_type].shape[1],
                    previous_nodes[receiver_type].shape[-1],
                    device=aggregate_device,
                )
                for start in range(0, edge_count, chunk_size):
                    chunk_end = min(start + chunk_size, edge_count)
                    chunk_senders = senders[start:chunk_end]
                    chunk_receivers = receivers[start:chunk_end]
                    # Save raw/previous edge views and shared node tensors, not
                    # the much larger encoded/gathered/concatenated features.
                    # Bind metadata now: checkpoint replays after this loop.
                    compute_chunk = partial(
                        self._edge_chunk,
                        step=step,
                        edge_type=edge_type,
                        encode_raw=raw_edges.get(edge_type) is not None,
                        preserve_edge_state=preserve_edge_state,
                        factorized=factorized,
                    )
                    chunk_args = (
                        src_edge[:, start:chunk_end, :],
                        sender_values,
                        receiver_values,
                        chunk_senders,
                        chunk_receivers,
                    )
                    if checkpoint_chunks and torch.is_grad_enabled():
                        edge_delta, residual_chunk = tracked_checkpoint(compute_chunk, *chunk_args)
                    else:
                        edge_delta, residual_chunk = compute_chunk(*chunk_args)
                    if preserve_edge_state:
                        if updated_edge is None:
                            updated_edge = edge_delta.new_empty(
                                edge_delta.shape[0],
                                edge_count,
                                edge_delta.shape[-1],
                            )
                        updated_edge[:, start:chunk_end] = residual_chunk
                    aggregate_delta = edge_delta
                    if aggregate_delta.device != aggregate_device:
                        # When a full-grid node processor is assigned to a
                        # different PP stage, accumulate each already-defined
                        # edge chunk directly on that stage.  This avoids
                        # materializing a second 1,038,240 x 512 FP32 tensor on
                        # the edge stage only to transfer it immediately.  The
                        # same edge deltas, receiver indices and chunk order are
                        # retained, so this changes placement, not GraphNet
                        # mathematics or model size.
                        aggregate_delta = _pipeline_device_transfer(
                            aggregate_delta,
                            aggregate_device,
                        )
                        chunk_receivers = _pipeline_device_transfer(
                            chunk_receivers,
                            aggregate_device,
                        )
                        _synchronize_vendor_pipeline_device(aggregate_device)
                    with _operator_device_guard(aggregate_device):
                        aggregated.index_add_(
                            1,
                            chunk_receivers,
                            aggregate_delta,
                        )
                edge_updates[edge_type] = None
                if updated_edge is not None:
                    chunked_edge_outputs[edge_type] = updated_edge
                received_by_node_type[receiver_type].append(aggregated)
                continue

            if factorized:
                edge_updates[edge_type], _ = self._edge_chunk(
                    previous_edges[edge_type], sender_values, receiver_values,
                    senders, receivers, step=step, edge_type=edge_type,
                    encode_raw=False, preserve_edge_state=False, factorized=True,
                )
                continue
            edge_inputs = torch.cat(
                [
                    previous_edges[edge_type],
                    _gather_pipeline_nodes(previous_nodes[sender_type], senders),
                    _gather_pipeline_nodes(previous_nodes[receiver_type], receivers),
                ],
                dim=-1,
            )
            edge_updates[edge_type] = self._apply_mlp_with_layer_norm(
                f"processor_edges_{step}_{edge_type}", edge_inputs
            )

        updated_nodes = {}
        for node_type in self.node_types:
            received_features = list(received_by_node_type[node_type])
            for edge_type in self.edge_types:
                _, receiver_type = self.edge_node_sets[edge_type]
                if receiver_type != node_type or edge_updates[edge_type] is None:
                    continue
                _, receivers = edge_indices[edge_type]
                received_features.append(
                    aggregate_edges_to_nodes(
                        edge_updates[edge_type],
                        receivers,
                        previous_nodes[node_type].shape[1],
                    )
                )
            chunk_size = self.node_update_chunk_sizes.get(node_type)
            node_count = int(previous_nodes[node_type].shape[1])
            node_device = self._node_processor_devices.get(
                node_type,
                previous_nodes[node_type].device,
            )
            cross_device = previous_nodes[node_type].device != node_device
            if chunk_size and node_count > chunk_size:
                output = previous_nodes[node_type].new_empty(
                    previous_nodes[node_type].shape,
                    device=node_device,
                )
                with _operator_device_guard(node_device):
                    for start in range(0, node_count, chunk_size):
                        chunk_end = min(start + chunk_size, node_count)
                        previous_node_values = previous_nodes[node_type]
                        with _operator_device_guard(previous_node_values.device):
                            previous_chunk = previous_node_values[:, start:chunk_end]
                        received_chunks = []
                        for item in received_features:
                            with _operator_device_guard(item.device):
                                received_chunks.append(item[:, start:chunk_end])
                        if cross_device:
                            previous_chunk = _pipeline_device_copy(
                                previous_chunk,
                                node_device,
                            )
                            transferred_received = []
                            for item in received_chunks:
                                transferred = _pipeline_device_copy(
                                    item,
                                    node_device,
                                )
                                transferred_received.append(transferred)
                            received_chunks = transferred_received
                            _synchronize_vendor_pipeline_device(node_device)
                        chunk_inputs = torch.cat(
                            [previous_chunk] + received_chunks,
                            dim=-1,
                        )
                        chunk_delta = self._apply_checkpointed_mlp(
                            f"processor_nodes_{step}_{node_type}",
                            chunk_inputs,
                            checkpoint_chunks,
                        )
                        output[:, start:chunk_end] = previous_chunk + chunk_delta
                updated_nodes[node_type] = output
            else:
                previous_values = previous_nodes[node_type]
                received_values = received_features
                if cross_device:
                    previous_values = _pipeline_device_transfer(
                        previous_values,
                        node_device,
                    )
                    received_values = [
                        _pipeline_device_transfer(item, node_device) for item in received_values
                    ]
                    _synchronize_vendor_pipeline_device(node_device)
                with _operator_device_guard(node_device):
                    node_inputs = (
                        torch.cat(
                            [previous_values] + received_values,
                            dim=-1,
                        )
                        if received_values
                        else previous_values
                    )
                    node_delta = self._apply_mlp_with_layer_norm(
                        f"processor_nodes_{step}_{node_type}", node_inputs
                    )
                    updated_nodes[node_type] = previous_values + node_delta

        updated_edges = {}
        for edge_type in self.edge_types:
            if edge_type in chunked_edge_outputs:
                updated_edges[edge_type] = chunked_edge_outputs[edge_type]
                continue
            edge_delta = edge_updates[edge_type]
            updated_edges[edge_type] = (
                previous_edges[edge_type]
                if edge_delta is None
                else previous_edges[edge_type] + edge_delta
            )
        return updated_nodes, updated_edges

    def _project_edge_nodes(self, step, edge_type, sender_nodes, receiver_nodes):
        linear = getattr(self, f"processor_edges_{step}_{edge_type}_mlp").linear_0
        sender_width, receiver_width = sender_nodes.shape[-1], receiver_nodes.shape[-1]
        edge_width = linear.in_features - sender_width - receiver_width
        _, sender_weight, receiver_weight = linear.weight.split(
            (edge_width, sender_width, receiver_width), dim=1,
        )
        # This experiment projects colocated Mesh nodes, not full-grid inputs
        # on a different pipeline stage. Do not silently transfer a full grid.
        if sender_nodes.device != linear.weight.device or receiver_nodes.device != linear.weight.device:
            raise ValueError("factorized edge projection requires colocated nodes and weights")
        with _operator_device_guard(linear.weight.device):
            zero_bias = linear.weight.new_zeros(linear.out_features)
            record_linear_recompute(sender_nodes, sender_weight)
            sender_projection = nn.functional.linear(sender_nodes, sender_weight, zero_bias)
            record_linear_recompute(receiver_nodes, receiver_weight)
            receiver_projection = nn.functional.linear(receiver_nodes, receiver_weight, zero_bias)
        return sender_projection, receiver_projection

    def _edge_chunk(
        self,
        edge_chunk: torch.Tensor,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
        sender_indices: torch.Tensor,
        receiver_indices: torch.Tensor,
        *,
        step: int,
        edge_type: str,
        encode_raw: bool,
        preserve_edge_state: bool,
        factorized: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pure chunk replay; accumulation into the shared buffer stays outside."""
        edge_device = edge_chunk.device
        with _operator_device_guard(edge_device):
            encoded_chunk = (
                self._apply_mlp_with_layer_norm(f"encoder_edges_{edge_type}", edge_chunk)
                if encode_raw else edge_chunk
            )
        if sender_indices.device != sender_nodes.device:
            sender_indices = _pipeline_device_transfer(sender_indices, sender_nodes.device)
        with _operator_device_guard(sender_nodes.device):
            sender_chunk = _gather_pipeline_nodes(sender_nodes, sender_indices)
        if receiver_indices.device != receiver_nodes.device:
            receiver_indices = _pipeline_device_transfer(receiver_indices, receiver_nodes.device)
        with _operator_device_guard(receiver_nodes.device):
            receiver_chunk = _gather_pipeline_nodes(receiver_nodes, receiver_indices)
        if sender_chunk.device != edge_device:
            sender_chunk = _pipeline_device_copy(sender_chunk, edge_device)
        if receiver_chunk.device != edge_device:
            receiver_chunk = _pipeline_device_copy(receiver_chunk, edge_device)
        if sender_nodes.device != edge_device or receiver_nodes.device != edge_device:
            _synchronize_vendor_pipeline_device(edge_device)
        with _operator_device_guard(edge_device):
            base_name = f"processor_edges_{step}_{edge_type}"
            if factorized:
                mlp = getattr(self, f"{base_name}_mlp")
                edge_weight = mlp.linear_0.weight[:, :encoded_chunk.shape[-1]]
                record_linear_recompute(encoded_chunk, edge_weight)
                hidden = nn.functional.linear(encoded_chunk, edge_weight, mlp.linear_0.bias)
                hidden = hidden + sender_chunk
                hidden = hidden + receiver_chunk
                edge_delta = getattr(self, f"{base_name}_layer_norm")(
                    mlp.linear_1(mlp.activation(hidden))
                )
            else:
                edge_inputs = torch.cat([encoded_chunk, sender_chunk, receiver_chunk], dim=-1)
                edge_delta = self._apply_mlp_with_layer_norm(base_name, edge_inputs)
            residual_chunk = encoded_chunk + edge_delta if preserve_edge_state else None
        return edge_delta, residual_chunk

    def _apply_checkpointed_mlp(
        self, base_name: str, x: torch.Tensor, enabled: bool
    ) -> torch.Tensor:
        if enabled and torch.is_grad_enabled():
            return tracked_checkpoint(
                lambda value: self._apply_mlp_with_layer_norm(base_name, value),
                x,
            )
        return self._apply_mlp_with_layer_norm(base_name, x)

    def _apply_mlp_with_layer_norm(self, base_name: str, x: torch.Tensor) -> torch.Tensor:
        x = getattr(self, f"{base_name}_mlp")(x)
        return getattr(self, f"{base_name}_layer_norm")(x)


def aggregate_edges_to_nodes(
    edge_features: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    out = edge_features.new_zeros(edge_features.shape[0], num_nodes, edge_features.shape[-1])
    return out.index_add_(1, receivers, edge_features)


class GoogleSmallCompatibleModel(BaseModel):
    """PyTorch model matching Google GraphCast_small params and tensor flow."""

    graphnet_type = GoogleSmallGraphNet

    def __init__(self, config: GoogleSmallConfig | None = None) -> None:
        config = config or GoogleSmallConfig()
        execution_config = replace(
            config,
            grid2mesh_edge_chunk_size=config.mesh2grid_edge_chunk_size,
            grid2mesh_node_chunk_size=config.grid_node_chunk_size,
            mesh2grid_node_chunk_size=config.grid_node_chunk_size,
            mesh2grid_decoder_chunk_size=config.grid_node_chunk_size,
        )
        super().__init__(execution_config)
        self.config = config
        self.mesh_gnn.edge_update_chunk_sizes = (
            {"mesh": config.mesh_edge_chunk_size} if config.mesh_edge_chunk_size else {}
        )
        self.mesh_gnn.node_update_chunk_sizes = {"mesh_nodes": 16384}
        self._primary_device = None
        self._grid2mesh_device = None
        self._mesh_first_device = None
        self._secondary_device = None
        self._output_device = None
        self._mesh2grid_device = None
        self._mesh2grid_node_device = None
        self._mesh_split_step = None
        self._activation_checkpoint = False
        self._mesh_checkpoint_per_step = False
        self._grid2mesh_stage_checkpoint = True
        self._mesh2grid_stage_checkpoint = False

    def configure_activation_checkpoint(self, enabled: bool) -> None:
        self._activation_checkpoint = enabled

    def configure_model_parallel(
        self,
        primary: torch.device,
        secondary: torch.device,
        *,
        split_step: int,
        activation_checkpoint: bool,
        checkpoint_mesh_per_step: bool = False,
    ) -> None:
        if not 0 < split_step < self.config.gnn_msg_steps:
            raise ValueError(
                f"invalid split step {split_step} for {self.config.gnn_msg_steps} message steps"
            )
        if primary == secondary:
            raise ValueError("model-parallel devices must be different")

        self._primary_device = primary
        self._grid2mesh_device = primary
        self._mesh_first_device = primary
        self._secondary_device = secondary
        self._output_device = secondary
        self._mesh2grid_device = secondary
        self._mesh2grid_node_device = secondary
        self._mesh_split_step = split_step
        self._activation_checkpoint = activation_checkpoint
        self._mesh_checkpoint_per_step = checkpoint_mesh_per_step
        self._grid2mesh_stage_checkpoint = True
        self._mesh2grid_stage_checkpoint = False
        self.grid2mesh_gnn.to(primary)
        self.mesh2grid_gnn.to(secondary)
        self.mesh2grid_gnn.configure_node_processor_device(
            "grid_nodes",
            None,
        )
        self.mesh2grid_gnn.configure_decoder_device(secondary)

        for edge_type in self.mesh_gnn.edge_types:
            self._move_mlp_with_layer_norm(self.mesh_gnn, f"encoder_edges_{edge_type}", primary)
        for node_type in self.mesh_gnn.node_types:
            base_name = f"encoder_nodes_{node_type}"
            if hasattr(self.mesh_gnn, f"{base_name}_mlp"):
                self._move_mlp_with_layer_norm(self.mesh_gnn, base_name, primary)
        for step in range(self.config.gnn_msg_steps):
            step_device = primary if step < split_step else secondary
            for edge_type in self.mesh_gnn.edge_types:
                self._move_mlp_with_layer_norm(
                    self.mesh_gnn,
                    f"processor_edges_{step}_{edge_type}",
                    step_device,
                )
            for node_type in self.mesh_gnn.node_types:
                self._move_mlp_with_layer_norm(
                    self.mesh_gnn,
                    f"processor_nodes_{step}_{node_type}",
                    step_device,
                )

    def configure_pipeline_parallel(
        self,
        primary: torch.device,
        secondary: torch.device,
        output: torch.device,
        *,
        split_step: int,
        activation_checkpoint: bool,
    ) -> None:
        if len({primary, secondary, output}) != 3:
            raise ValueError("pipeline devices must be distinct")
        self.configure_model_parallel(
            primary,
            secondary,
            split_step=split_step,
            activation_checkpoint=activation_checkpoint,
        )
        self._output_device = output
        self._mesh2grid_device = output
        self._mesh2grid_node_device = output
        self.mesh2grid_gnn.to(output)
        self.mesh2grid_gnn.configure_node_processor_device(
            "grid_nodes",
            None,
        )
        self.mesh2grid_gnn.configure_decoder_device(output)

    def configure_four_stage_pipeline(
        self,
        primary: torch.device,
        mesh_first: torch.device,
        mesh_second: torch.device,
        output: torch.device,
        *,
        split_step: int,
        activation_checkpoint: bool,
        checkpoint_mesh_per_step: bool = False,
        grid2mesh_stage_device: str = "primary",
        grid2mesh_node_stage_device: str = "stage",
        checkpoint_grid2mesh_stage: bool = True,
        mesh2grid_stage_device: str = "output",
        mesh2grid_node_stage_device: str = "stage",
        mesh2grid_decoder_stage_device: str = "output",
        checkpoint_mesh2grid_stage: bool = False,
    ) -> None:
        """Place four logical stages on one, two or four physical devices."""

        if len({primary, mesh_first, mesh_second, output}) not in {1, 2, 4}:
            raise ValueError("pipeline placement requires one, two or four physical devices")
        if not 0 < split_step <= self.config.gnn_msg_steps:
            raise ValueError(
                f"invalid split step {split_step} for {self.config.gnn_msg_steps} message steps"
            )
        if mesh2grid_stage_device not in {
            "output",
            "primary",
            "mesh_first",
            "mesh_second",
        }:
            raise ValueError(
                "mesh2grid_stage_device must be output, primary, mesh_first, or mesh_second"
            )
        if grid2mesh_stage_device not in {
            "primary",
            "mesh_first",
            "mesh_second",
            "output",
        }:
            raise ValueError(
                "grid2mesh_stage_device must be primary, mesh_first, mesh_second, or output"
            )
        if grid2mesh_node_stage_device not in {
            "stage",
            "primary",
            "mesh_first",
            "mesh_second",
            "output",
        }:
            raise ValueError(
                "grid2mesh_node_stage_device must be stage, primary, "
                "mesh_first, mesh_second, or output"
            )
        if mesh2grid_node_stage_device not in {
            "stage",
            "output",
            "primary",
            "mesh_first",
            "mesh_second",
        }:
            raise ValueError(
                "mesh2grid_node_stage_device must be stage, output, "
                "primary, mesh_first, or mesh_second"
            )
        if mesh2grid_decoder_stage_device not in {
            "output",
            "primary",
            "mesh_first",
            "mesh_second",
        }:
            raise ValueError(
                "mesh2grid_decoder_stage_device must be output, primary, mesh_first, or mesh_second"
            )

        self._primary_device = primary
        self._mesh_first_device = mesh_first
        self._secondary_device = mesh_second
        self._output_device = output
        self._grid2mesh_device = {
            "primary": primary,
            "mesh_first": mesh_first,
            "mesh_second": mesh_second,
            "output": output,
        }[grid2mesh_stage_device]
        grid2mesh_node_device = {
            "stage": self._grid2mesh_device,
            "primary": primary,
            "mesh_first": mesh_first,
            "mesh_second": mesh_second,
            "output": output,
        }[grid2mesh_node_stage_device]
        self._mesh2grid_device = {
            "output": output,
            "primary": primary,
            "mesh_first": mesh_first,
            "mesh_second": mesh_second,
        }[mesh2grid_stage_device]
        mesh2grid_node_device = {
            "stage": self._mesh2grid_device,
            "output": output,
            "primary": primary,
            "mesh_first": mesh_first,
            "mesh_second": mesh_second,
        }[mesh2grid_node_stage_device]
        mesh2grid_decoder_device = {
            "output": output,
            "primary": primary,
            "mesh_first": mesh_first,
            "mesh_second": mesh_second,
        }[mesh2grid_decoder_stage_device]
        self._mesh2grid_node_device = mesh2grid_node_device
        self._mesh_split_step = split_step
        self._activation_checkpoint = activation_checkpoint
        self._mesh_checkpoint_per_step = checkpoint_mesh_per_step
        self._grid2mesh_stage_checkpoint = checkpoint_grid2mesh_stage
        self._mesh2grid_stage_checkpoint = checkpoint_mesh2grid_stage
        self.grid2mesh_gnn.to(self._grid2mesh_device)
        # The full-resolution Grid-to-Mesh grid-node path can be streamed to
        # a less occupied PP stage with the same chunk boundaries and tensor
        # values.  Only placement changes; the graph and parameters do not.
        self.grid2mesh_gnn.configure_node_processor_device(
            "grid_nodes",
            grid2mesh_node_device,
        )
        self.mesh2grid_gnn.to(self._mesh2grid_device)
        # The 0.25-degree grid latent contains 1,038,240 x 512 FP32 values.
        # Keep its node update on a separately selected PP stage when the
        # edge-processing stage cannot hold both full-grid activations and
        # their backward gradients.  The same processor module and chunks are
        # executed in the same order.
        self.mesh2grid_gnn.configure_node_processor_device(
            "grid_nodes",
            mesh2grid_node_device,
        )
        # The decoder is mathematically independent of the PP placement.  A
        # memory-constrained runtime may stream the already-defined grid-node
        # chunks to a less occupied stage; parameters, values and operation
        # order remain unchanged.
        self.mesh2grid_gnn.configure_decoder_device(mesh2grid_decoder_device)

        for edge_type in self.mesh_gnn.edge_types:
            self._move_mlp_with_layer_norm(self.mesh_gnn, f"encoder_edges_{edge_type}", mesh_first)
        for node_type in self.mesh_gnn.node_types:
            base_name = f"encoder_nodes_{node_type}"
            if hasattr(self.mesh_gnn, f"{base_name}_mlp"):
                self._move_mlp_with_layer_norm(self.mesh_gnn, base_name, mesh_first)
        for step in range(self.config.gnn_msg_steps):
            step_device = mesh_first if step < split_step else mesh_second
            for edge_type in self.mesh_gnn.edge_types:
                self._move_mlp_with_layer_norm(
                    self.mesh_gnn,
                    f"processor_edges_{step}_{edge_type}",
                    step_device,
                )
            for node_type in self.mesh_gnn.node_types:
                self._move_mlp_with_layer_norm(
                    self.mesh_gnn,
                    f"processor_nodes_{step}_{node_type}",
                    step_device,
                )

    @staticmethod
    def _run_graphnet_stage(
        graphnet: GoogleSmallGraphNet,
        nodes: dict[str, torch.Tensor],
        edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        checkpoint_stage: bool,
        checkpoint_chunks: bool = False,
        checkpoint_per_step: bool = False,
        step_start: int = 0,
        step_end: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        end = graphnet.message_passing_steps if step_end is None else step_end
        if (
            checkpoint_per_step
            and checkpoint_stage
            and torch.is_grad_enabled()
            and end - step_start > 1
        ):
            # Keep the colleague-provided GraphNet math unchanged, but place one
            # existing tracked checkpoint around each message-passing step.
            # This avoids retaining a multi-step edge-state replay on 64 GiB
            # two-device targets while preserving all mesh-6/latent-512/GNN-16
            # operations and their order.
            step_nodes = nodes
            step_edges = edges
            for current_step in range(step_start, end):
                step_nodes, step_edges = GoogleSmallCompatibleModel._run_graphnet_stage(
                    graphnet,
                    step_nodes,
                    step_edges,
                    edge_indices,
                    checkpoint_stage=True,
                    checkpoint_chunks=checkpoint_chunks,
                    checkpoint_per_step=False,
                    step_start=current_step,
                    step_end=current_step + 1,
                )
            return step_nodes, step_edges

        stage_device = next(iter(edges.values())).device
        if not checkpoint_stage or not torch.is_grad_enabled():
            with _operator_device_guard(stage_device):
                return graphnet(
                    nodes,
                    edges,
                    edge_indices,
                    checkpoint_chunks=checkpoint_chunks,
                    step_start=step_start,
                    step_end=step_end,
                )

        node_names = tuple(graphnet.node_types)
        edge_names = tuple(graphnet.edge_types)

        def run_stage(*flat_tensors):
            node_count = len(node_names)
            stage_nodes = dict(zip(node_names, flat_tensors[:node_count], strict=False))
            stage_edges = dict(zip(edge_names, flat_tensors[node_count:], strict=False))
            with _operator_device_guard(flat_tensors[len(node_names)].device):
                out_nodes, out_edges = graphnet(
                    stage_nodes,
                    stage_edges,
                    edge_indices,
                    step_start=step_start,
                    step_end=step_end,
                    checkpoint_chunks=checkpoint_chunks,
                )
            return tuple(out_nodes[name] for name in node_names) + tuple(
                out_edges[name] for name in edge_names
            )

        flat_out = tracked_checkpoint(
            run_stage,
            *(nodes[name] for name in node_names),
            *(edges[name] for name in edge_names),
        )
        return (
            dict(zip(node_names, flat_out[: len(node_names)], strict=False)),
            dict(zip(edge_names, flat_out[len(node_names) :], strict=False)),
        )

    @staticmethod
    def _move_mlp_with_layer_norm(module: nn.Module, base_name: str, device: torch.device) -> None:
        getattr(module, f"{base_name}_mlp").to(device)
        getattr(module, f"{base_name}_layer_norm").to(device)

    def init_graph(
        self,
        latitudes: Any | None = None,
        longitudes: Any | None = None,
    ) -> GoogleSmallGraph:
        graph = build_google_small_graph(
            resolution=self.config.resolution,
            mesh_size=self.config.mesh_size,
            radius_query_fraction_edge_length=(self.config.radius_query_fraction_edge_length),
            mesh2grid_edge_normalization_factor=(self.config.mesh2grid_edge_normalization_factor),
            latitudes=latitudes,
            longitudes=longitudes,
            canonical=True,
        )
        for name in (
            "mesh_node_latitudes",
            "mesh_node_longitudes",
            "grid_node_latitudes",
            "grid_node_longitudes",
            "grid_node_geo_features",
            "mesh_node_geo_features",
            "grid2mesh_edge_features",
            "grid2mesh_senders",
            "grid2mesh_receivers",
            "mesh_edge_features",
            "mesh_senders",
            "mesh_receivers",
            "mesh2grid_edge_features",
            "mesh2grid_senders",
            "mesh2grid_receivers",
        ):
            self._set_nonpersistent_buffer(name, getattr(graph, name))
        self._graph_initialized = True
        self._num_grid_nodes = graph.num_grid_nodes
        self._num_mesh_nodes = graph.num_mesh_nodes
        return graph

    def estimate_forward_linear_flops(self, batch_size: int) -> int:
        """Count useful forward FLOPs from Linear layers on the initialized graph."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._graph_initialized:
            raise RuntimeError("graph must be initialized before estimating FLOPs")

        def mlp_flops(module: nn.Module, rows: int) -> int:
            weights = sum(
                layer.in_features * layer.out_features
                for layer in module.modules()
                if isinstance(layer, nn.Linear)
            )
            return 2 * batch_size * rows * weights

        def graphnet_flops(
            graphnet: GoogleSmallGraphNet,
            node_rows: dict[str, int],
            edge_rows: dict[str, int],
        ) -> int:
            total = 0
            for edge_type in graphnet.edge_types:
                total += mlp_flops(
                    getattr(graphnet, f"encoder_edges_{edge_type}_mlp"),
                    edge_rows[edge_type],
                )
            for node_type in graphnet.node_types:
                name = f"encoder_nodes_{node_type}_mlp"
                if hasattr(graphnet, name):
                    total += mlp_flops(getattr(graphnet, name), node_rows[node_type])
            for step in range(graphnet.message_passing_steps):
                for edge_type in graphnet.edge_types:
                    total += mlp_flops(
                        getattr(graphnet, f"processor_edges_{step}_{edge_type}_mlp"),
                        edge_rows[edge_type],
                    )
                    if graphnet.edge_input_projection_mode == "factorized":
                        sender, receiver = graphnet.edge_node_sets[edge_type]
                        linear = getattr(graphnet, f"processor_edges_{step}_{edge_type}_mlp").linear_0
                        # Sender/receiver are encoded latent node states. Their
                        # first-layer projections now run per node, not per edge.
                        width = self.config.latent_size
                        total -= 2 * batch_size * linear.out_features * width * (
                            2 * edge_rows[edge_type] - node_rows[sender] - node_rows[receiver]
                        )
                for node_type in graphnet.node_types:
                    total += mlp_flops(
                        getattr(graphnet, f"processor_nodes_{step}_{node_type}_mlp"),
                        node_rows[node_type],
                    )
            for node_type in graphnet.node_output_types:
                total += mlp_flops(
                    getattr(graphnet, f"decoder_nodes_{node_type}_mlp"),
                    node_rows[node_type],
                )
            return total

        grid_rows = int(self.grid_node_geo_features.shape[0])
        mesh_rows = int(self.mesh_node_geo_features.shape[0])
        total = graphnet_flops(
            self.grid2mesh_gnn,
            {"grid_nodes": grid_rows, "mesh_nodes": mesh_rows},
            {"grid2mesh": int(self.grid2mesh_edge_features.shape[0])},
        )
        total += graphnet_flops(
            self.mesh_gnn,
            {"mesh_nodes": mesh_rows},
            {"mesh": int(self.mesh_edge_features.shape[0])},
        )
        total += graphnet_flops(
            self.mesh2grid_gnn,
            {"grid_nodes": grid_rows, "mesh_nodes": mesh_rows},
            {"mesh2grid": int(self.mesh2grid_edge_features.shape[0])},
        )
        return total

    def graph_topology_contract(self) -> dict[str, int | float | str]:
        """Describe the static graph whose tensors drive this model."""

        if not self._graph_initialized:
            raise RuntimeError("graph must be initialized before reporting topology")
        return {
            "schema_version": 1,
            "resolution_degrees": self.config.resolution,
            "mesh_size": self.config.mesh_size,
            "mesh_message_passing_steps": self.config.gnn_msg_steps,
            "grid_nodes": int(self.grid_node_geo_features.shape[0]),
            "mesh_nodes": int(self.mesh_node_geo_features.shape[0]),
            "grid2mesh_edges": int(self.grid2mesh_edge_features.shape[0]),
            "mesh_edges": int(self.mesh_edge_features.shape[0]),
            "mesh2grid_edges": int(self.mesh2grid_edge_features.shape[0]),
            "grid2mesh_neighbor_order": "grid_then_mesh_index_ascending",
            "graph_topology_version": CANONICAL_GRAPH_TOPOLOGY_VERSION,
            "mesh_vertex_source": "pinned_h100_mesh6_asset_prefix",
        }

    def forward(self, grid_node_features: torch.Tensor) -> torch.Tensor:
        """Run a minimal stacked-tensor forward pass.

        Args:
            grid_node_features: Tensor with shape ``[batch, grid_nodes, 183]``.

        Returns:
            Tensor with shape ``[batch, grid_nodes, 83]``.
        """

        if not self._graph_initialized:
            self.init_graph()
        if grid_node_features.ndim != 3:
            raise ValueError("expected [batch, grid_nodes, channels] input")
        if grid_node_features.shape[-1] != self.config.node_input_dim - 3:
            raise ValueError(
                f"expected {self.config.node_input_dim - 3} input channels, "
                f"got {grid_node_features.shape[-1]}"
            )
        if grid_node_features.shape[1] != self.grid_node_geo_features.shape[0]:
            raise ValueError(
                f"input grid_nodes={grid_node_features.shape[1]} does not match "
                f"initialized graph grid_nodes={self.grid_node_geo_features.shape[0]}"
            )

        batch_size = grid_node_features.shape[0]
        device = grid_node_features.device
        if self._primary_device is not None and device != self._primary_device:
            raise ValueError(
                f"model-parallel input must be on {self._primary_device}, got {device}"
            )
        dtype = grid_node_features.dtype
        grid2mesh_device = self._grid2mesh_device or device
        grid2mesh_inputs = grid_node_features
        if grid2mesh_inputs.device != grid2mesh_device:
            # Raw ERA5 inputs do not require gradients.  Moving them before
            # Grid-to-Mesh keeps that complete module and its full-grid latent
            # gradient on the selected PP stage instead of retaining another
            # 1.98 GiB gradient buffer on the input stage.
            grid2mesh_inputs = _pipeline_device_transfer(
                grid2mesh_inputs,
                grid2mesh_device,
            )
            _synchronize_vendor_pipeline_device(grid2mesh_device)

        grid_geo = _pipeline_device_transfer(
            self.grid_node_geo_features,
            grid2mesh_device,
            dtype=dtype,
        )
        mesh_geo = _pipeline_device_transfer(
            self.mesh_node_geo_features,
            grid2mesh_device,
            dtype=dtype,
        )

        grid_nodes = torch.cat(
            [grid2mesh_inputs, grid_geo.unsqueeze(0).expand(batch_size, -1, -1)],
            dim=-1,
        )
        dummy_mesh_inputs = grid2mesh_inputs.new_zeros(
            batch_size,
            mesh_geo.shape[0],
            self.config.node_input_dim - 3,
        )
        mesh_nodes = torch.cat(
            [dummy_mesh_inputs, mesh_geo.unsqueeze(0).expand(batch_size, -1, -1)],
            dim=-1,
        )

        grid2mesh_edges = self._edge_features_for_batch(
            self.grid2mesh_edge_features,
            batch_size,
            grid2mesh_device,
            dtype,
        )
        grid2mesh_nodes, _ = self._run_graphnet_stage(
            self.grid2mesh_gnn,
            {"grid_nodes": grid_nodes, "mesh_nodes": mesh_nodes},
            {"grid2mesh": grid2mesh_edges},
            {
                "grid2mesh": (
                    _pipeline_device_transfer(
                        self.grid2mesh_senders,
                        grid2mesh_device,
                    ),
                    _pipeline_device_transfer(
                        self.grid2mesh_receivers,
                        grid2mesh_device,
                    ),
                )
            },
            checkpoint_stage=(self._activation_checkpoint and self._grid2mesh_stage_checkpoint),
            checkpoint_chunks=self._activation_checkpoint,
        )

        mesh_first_device = self._mesh_first_device or device
        mesh_edges = self._edge_features_for_batch(
            self.mesh_edge_features, batch_size, mesh_first_device, dtype
        )
        mesh_indices = {
            "mesh": (
                _pipeline_device_transfer(
                    self.mesh_senders,
                    mesh_first_device,
                ),
                _pipeline_device_transfer(
                    self.mesh_receivers,
                    mesh_first_device,
                ),
            )
        }
        if self._secondary_device is None:
            with _operator_device_guard(mesh_first_device):
                mesh_nodes_out, _ = self.mesh_gnn(
                    {"mesh_nodes": grid2mesh_nodes["mesh_nodes"]},
                    {"mesh": mesh_edges},
                    mesh_indices,
                    checkpoint_steps=self._activation_checkpoint,
                )
            output_device = device
            mesh2grid_device = output_device
            output_grid_nodes = grid2mesh_nodes["grid_nodes"]
        else:
            if self._mesh_split_step is None:
                raise RuntimeError("model-parallel split step is not configured")
            _synchronize_vendor_pipeline_device(grid2mesh_device)
            first_mesh_nodes = {
                "mesh_nodes": _pipeline_device_transfer(
                    grid2mesh_nodes["mesh_nodes"],
                    mesh_first_device,
                )
            }
            _synchronize_vendor_pipeline_device(mesh_first_device)
            mid_nodes, mid_edges = self._run_graphnet_stage(
                self.mesh_gnn,
                first_mesh_nodes,
                {"mesh": mesh_edges},
                mesh_indices,
                checkpoint_stage=self._activation_checkpoint,
                checkpoint_chunks=self._activation_checkpoint,
                checkpoint_per_step=self._mesh_checkpoint_per_step,
                step_start=0,
                step_end=self._mesh_split_step,
            )
            mesh_device = self._secondary_device
            _synchronize_vendor_pipeline_device(mesh_first_device)
            if self._mesh_split_step == self.config.gnn_msg_steps:
                # All Mesh message-passing steps have already completed on
                # ``mesh_first``.  Mesh-to-Grid consumes only the final Mesh
                # nodes, so transferring the final 327,660-edge latent to the
                # next PP stage would retain roughly 0.63 GiB with no consumer.
                # Dropping that unused output leaves every model operation,
                # consumed tensor and autograd path unchanged.
                del mid_edges
                mesh_nodes_out = {
                    name: _pipeline_device_transfer(value, mesh_device)
                    for name, value in mid_nodes.items()
                }
                _synchronize_vendor_pipeline_device(mesh_device)
            else:
                mid_nodes = {
                    name: _pipeline_device_transfer(value, mesh_device)
                    for name, value in mid_nodes.items()
                }
                mid_edges = {
                    name: _pipeline_device_transfer(value, mesh_device)
                    for name, value in mid_edges.items()
                }
                secondary_indices = {
                    name: (
                        _pipeline_device_transfer(senders, mesh_device),
                        _pipeline_device_transfer(receivers, mesh_device),
                    )
                    for name, (senders, receivers) in mesh_indices.items()
                }
                _synchronize_vendor_pipeline_device(mesh_device)
                mesh_nodes_out, unused_mesh_edges = self._run_graphnet_stage(
                    self.mesh_gnn,
                    mid_nodes,
                    mid_edges,
                    secondary_indices,
                    checkpoint_stage=self._activation_checkpoint,
                    checkpoint_chunks=self._activation_checkpoint,
                    checkpoint_per_step=self._mesh_checkpoint_per_step,
                    step_start=self._mesh_split_step,
                    step_end=None,
                )
                del unused_mesh_edges
            output_device = self._output_device or mesh_device
            mesh2grid_device = self._mesh2grid_device or output_device
            _synchronize_vendor_pipeline_device(mesh_device)
            mesh_nodes_out = {
                name: _pipeline_device_transfer(value, mesh2grid_device)
                for name, value in mesh_nodes_out.items()
            }
            if self._mesh2grid_node_device == mesh2grid_device:
                output_grid_nodes = _pipeline_device_transfer(
                    grid2mesh_nodes["grid_nodes"],
                    mesh2grid_device,
                )
            else:
                # The receiver-grid latent is about 1.98 GiB in FP32.  When
                # edge and node processors intentionally occupy different PP
                # stages, leave the complete latent on its producer and gather
                # only each already-defined edge chunk onto the edge stage.
                # The node processor likewise consumes contiguous chunks on
                # its own stage.  This removes a redundant full-grid copy and
                # its equally large backward buffer without changing any
                # GraphCast edge, weight, value, or operation order.
                output_grid_nodes = grid2mesh_nodes["grid_nodes"]
            _synchronize_vendor_pipeline_device(mesh2grid_device)

        mesh2grid_edges = self._edge_features_for_batch(
            self.mesh2grid_edge_features,
            batch_size,
            mesh2grid_device,
            dtype,
        )
        mesh2grid_nodes, _ = self._run_graphnet_stage(
            self.mesh2grid_gnn,
            {
                "mesh_nodes": mesh_nodes_out["mesh_nodes"],
                "grid_nodes": output_grid_nodes,
            },
            {"mesh2grid": mesh2grid_edges},
            {
                "mesh2grid": (
                    _pipeline_device_transfer(
                        self.mesh2grid_senders,
                        mesh2grid_device,
                    ),
                    _pipeline_device_transfer(
                        self.mesh2grid_receivers,
                        mesh2grid_device,
                    ),
                )
            },
            checkpoint_stage=(self._activation_checkpoint and self._mesh2grid_stage_checkpoint),
            checkpoint_chunks=self._activation_checkpoint,
        )
        return mesh2grid_nodes["grid_nodes"]

    @staticmethod
    def _edge_features_for_batch(
        features: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return (
            _pipeline_device_transfer(
                features,
                device,
                dtype=dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )


def build_google_small_compatible_model(
    config_path: str | Path = DEFAULT_CONFIG,
) -> GoogleSmallCompatibleModel:
    """Build the compatible model from the local YAML contract."""

    return GoogleSmallCompatibleModel(GoogleSmallConfig.from_yaml(config_path))
