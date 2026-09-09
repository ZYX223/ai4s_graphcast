"""PyTorch compatibility model for the Google GraphCast_small checkpoint."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .backend import use_advanced_index_gather
from .graph import GoogleSmallGraph, build_google_small_graph

DEFAULT_CONFIG = Path("configs/inference/google_graphcast_small.yaml")
INDEX_ADD_EXECUTIONS = frozenset({"inline", "prevalidated-static"})


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GoogleSmallConfig:
    """Minimal architecture contract for Google GraphCast_small."""

    resolution: float = 1.0
    mesh_size: int = 5
    latent_size: int = 512
    hidden_layers: int = 1
    gnn_msg_steps: int = 16
    radius_query_fraction_edge_length: float = 0.6
    mesh2grid_edge_normalization_factor: float = 0.6180338738074472
    grid2mesh_node_chunk_size: int | None = None
    grid2mesh_edge_chunk_size: int | None = None
    mesh2grid_edge_chunk_size: int | None = None
    mesh2grid_node_chunk_size: int | None = None
    mesh2grid_decoder_chunk_size: int | None = None
    node_input_dim: int = 186
    edge_input_dim: int = 4
    output_dim: int = 83
    input_variables: tuple[str, ...] = ()
    target_variables: tuple[str, ...] = ()
    forcing_variables: tuple[str, ...] = ()
    extra_mesh2grid_final_output_heads: dict[str, int] | None = None

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
            grid2mesh_node_chunk_size=(
                int(model_config["grid2mesh_node_chunk_size"])
                if model_config.get("grid2mesh_node_chunk_size") is not None
                else None
            ),
            grid2mesh_edge_chunk_size=(
                int(model_config["grid2mesh_edge_chunk_size"])
                if model_config.get("grid2mesh_edge_chunk_size") is not None
                else None
            ),
            mesh2grid_edge_chunk_size=(
                int(model_config["mesh2grid_edge_chunk_size"])
                if model_config.get("mesh2grid_edge_chunk_size") is not None
                else None
            ),
            mesh2grid_node_chunk_size=(
                int(model_config["mesh2grid_node_chunk_size"])
                if model_config.get("mesh2grid_node_chunk_size") is not None
                else None
            ),
            mesh2grid_decoder_chunk_size=(
                int(model_config["mesh2grid_decoder_chunk_size"])
                if model_config.get("mesh2grid_decoder_chunk_size") is not None
                else None
            ),
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


@dataclass(frozen=True)
class StaticIndexAddBufferContract:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    data_ptr: int
    version: int
    numel: int
    lower_bound: int
    upper_value: int
    upper_bound: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "data_ptr": self.data_ptr,
            "version": self.version,
            "numel": self.numel,
            "lower_bound": self.lower_bound,
            "upper_value": self.upper_value,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class StaticIndexAddContract:
    buffers: tuple[StaticIndexAddBufferContract, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "scope": "graphcast_static_receiver_buffers",
            "buffers": [item.to_metadata() for item in self.buffers],
        }


class HaikuStyleMLP(nn.Module):
    """Two-linear-layer MLP matching ``hk.nets.MLP`` for hidden_layers=1."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int,
        hidden_layers: int,
        linear_type: type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        if hidden_layers != 1:
            raise ValueError("Google GraphCast_small uses hidden_layers=1.")

        self.linear_0 = linear_type(in_features, hidden_features)
        self.activation = nn.SiLU()
        self.linear_1 = linear_type(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_1(self.activation(self.linear_0(x)))


class HaikuStyleFinalLinearHead(nn.Module):
    """Final-only output head matching a checkpoint with just ``linear_1``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        linear_type: type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.linear_1 = linear_type(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_1(x)


class MLPWithLayerNorm:
    """Container that registers Haiku-style MLP and LayerNorm as sibling names."""

    def __init__(
        self,
        parent: nn.Module,
        *,
        base_name: str,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
    ) -> None:
        self.mlp_name = f"{base_name}_mlp"
        self.layer_norm_name = f"{base_name}_layer_norm"
        parent.add_module(
            self.mlp_name,
            HaikuStyleMLP(
                in_features,
                out_features,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                linear_type=getattr(parent, "linear_type", nn.Linear),
            ),
        )
        parent.add_module(self.layer_norm_name, nn.LayerNorm(out_features))


class GoogleSmallGraphNet(nn.Module):
    """A minimal PyTorch DeepTypedGraphNet-compatible module."""

    linear_type = nn.Linear

    def __init__(
        self,
        *,
        edge_types: Iterable[str],
        edge_node_sets: dict[str, tuple[str, str]],
        node_types: Iterable[str],
        node_input_dims: dict[str, int] | None,
        edge_input_dims: dict[str, int],
        node_processor_input_dims: dict[str, int],
        edge_processor_input_dims: dict[str, int],
        node_output_dims: dict[str, int] | None,
        extra_final_node_output_dims: dict[str, int] | None,
        edge_update_chunk_sizes: dict[str, int] | None,
        node_update_chunk_sizes: dict[str, int] | None,
        decoder_node_chunk_sizes: dict[str, int] | None,
        message_passing_steps: int,
        config: GoogleSmallConfig,
        activation_checkpointing: bool = False,
        fuse_post_layer_norm_residual: bool = False,
        edge_input_projection_mode: str = "concatenated",
        index_add_execution: str = "inline",
    ) -> None:
        super().__init__()
        if edge_input_projection_mode not in {"concatenated", "factorized"}:
            raise ValueError(
                "edge input projection mode must be 'concatenated' or 'factorized', "
                f"got {edge_input_projection_mode!r}"
            )
        if index_add_execution not in INDEX_ADD_EXECUTIONS:
            raise ValueError(
                "index_add execution must be 'inline' or 'prevalidated-static', "
                f"got {index_add_execution!r}"
            )
        self.edge_types = tuple(edge_types)
        self.edge_node_sets = edge_node_sets
        self.node_types = tuple(node_types)
        self.message_passing_steps = message_passing_steps
        self.node_output_types = tuple(node_output_dims or {})
        self.edge_update_chunk_sizes = dict(edge_update_chunk_sizes or {})
        self.node_update_chunk_sizes = dict(node_update_chunk_sizes or {})
        self.decoder_node_chunk_sizes = dict(decoder_node_chunk_sizes or {})
        self.activation_checkpointing = activation_checkpointing
        self.fuse_post_layer_norm_residual = fuse_post_layer_norm_residual
        self.edge_input_projection_mode = edge_input_projection_mode
        self.index_add_execution = index_add_execution
        self.encoder_device: torch.device | None = None
        self.processor_step_devices: tuple[torch.device | None, ...] = tuple(
            None for _ in range(message_passing_steps)
        )
        self.decoder_device: torch.device | None = None
        if self.edge_update_chunk_sizes and message_passing_steps != 1:
            raise ValueError("chunked edge updates are only implemented for one-step graph nets")

        for edge_type in self.edge_types:
            MLPWithLayerNorm(
                self,
                base_name=f"encoder_edges_{edge_type}",
                in_features=edge_input_dims[edge_type],
                out_features=config.latent_size,
                hidden_features=config.latent_size,
                hidden_layers=config.hidden_layers,
            )

        if node_input_dims:
            for node_type in self.node_types:
                MLPWithLayerNorm(
                    self,
                    base_name=f"encoder_nodes_{node_type}",
                    in_features=node_input_dims[node_type],
                    out_features=config.latent_size,
                    hidden_features=config.latent_size,
                    hidden_layers=config.hidden_layers,
                )

        for step in range(message_passing_steps):
            for edge_type in self.edge_types:
                MLPWithLayerNorm(
                    self,
                    base_name=f"processor_edges_{step}_{edge_type}",
                    in_features=edge_processor_input_dims[edge_type],
                    out_features=config.latent_size,
                    hidden_features=config.latent_size,
                    hidden_layers=config.hidden_layers,
                )
            for node_type in self.node_types:
                MLPWithLayerNorm(
                    self,
                    base_name=f"processor_nodes_{step}_{node_type}",
                    in_features=node_processor_input_dims[node_type],
                    out_features=config.latent_size,
                    hidden_features=config.latent_size,
                    hidden_layers=config.hidden_layers,
                )

        if node_output_dims:
            for node_type, output_dim in node_output_dims.items():
                self.add_module(
                    f"decoder_nodes_{node_type}_mlp",
                    HaikuStyleMLP(
                        config.latent_size,
                        output_dim,
                        hidden_features=config.latent_size,
                        hidden_layers=config.hidden_layers,
                        linear_type=self.linear_type,
                    ),
                )

        if extra_final_node_output_dims:
            for node_type, output_dim in extra_final_node_output_dims.items():
                self.add_module(
                    f"decoder_nodes_{node_type}_mlp",
                    HaikuStyleFinalLinearHead(
                        config.latent_size,
                        output_dim,
                        linear_type=self.linear_type,
                    ),
                )

    def forward(
        self,
        nodes: dict[str, torch.Tensor],
        edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        node_output_slices: dict[str, tuple[int, int]] | None = None,
        encoded_node_types: Iterable[str] = (),
    ) -> Any:
        nodes = dict(nodes)
        edges = dict(edges)
        node_output_slices = dict(node_output_slices or {})
        encoded_node_types = frozenset(encoded_node_types)

        if self.encoder_device is not None:
            nodes = move_tensor_dict(nodes, self.encoder_device)
            edges = move_tensor_dict(edges, self.encoder_device)

        for edge_type in self.edge_types:
            if self._delay_chunked_edge_encoding(edge_type, edges[edge_type]):
                continue
            edges[edge_type] = self._apply_mlp_with_layer_norm(
                f"encoder_edges_{edge_type}", edges[edge_type]
            )

        for node_type in self.node_types:
            base_name = f"encoder_nodes_{node_type}"
            if node_type not in encoded_node_types and hasattr(self, f"{base_name}_mlp"):
                nodes[node_type] = self._apply_mlp_with_layer_norm(base_name, nodes[node_type])

        for step in range(self.message_passing_steps):
            step_device = self.processor_step_devices[step]
            if step_device is not None:
                nodes = move_tensor_dict(nodes, step_device)
                edges = move_tensor_dict(edges, step_device)
                step_edge_indices = move_edge_indices(edge_indices, step_device)
            else:
                step_edge_indices = edge_indices
            if self._use_activation_checkpointing():
                nodes, edges = self._checkpoint_processor_step(
                    step,
                    nodes,
                    edges,
                    step_edge_indices,
                    node_output_slices,
                )
            else:
                nodes, edges = self._processor_step(
                    step,
                    nodes,
                    edges,
                    step_edge_indices,
                    node_output_slices,
                )

        if self.decoder_device is not None:
            nodes = move_tensor_dict(nodes, self.decoder_device)

        for node_type in self.node_output_types:
            nodes[node_type] = self._apply_decoder_node_output(node_type, nodes[node_type])

        return nodes, edges

    def _use_activation_checkpointing(self) -> bool:
        return self.activation_checkpointing and self.training and torch.is_grad_enabled()

    def _delay_chunked_edge_encoding(
        self,
        edge_type: str,
        edge_features: torch.Tensor,
    ) -> bool:
        chunk_size = self.edge_update_chunk_sizes.get(edge_type)
        return (
            self.message_passing_steps == 1
            and chunk_size is not None
            and chunk_size > 0
            and int(edge_features.shape[1]) > chunk_size
        )

    def _checkpoint_processor_step(
        self,
        step: int,
        nodes: dict[str, torch.Tensor],
        edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        node_output_slices: dict[str, tuple[int, int]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        node_count = len(self.node_types)
        flat_inputs = tuple(nodes[name] for name in self.node_types) + tuple(
            edges[name] for name in self.edge_types
        )

        def wrapped_step(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
            step_nodes = {
                node_type: tensors[index] for index, node_type in enumerate(self.node_types)
            }
            step_edges = {
                edge_type: tensors[node_count + index]
                for index, edge_type in enumerate(self.edge_types)
            }
            out_nodes, out_edges = self._processor_step(
                step,
                step_nodes,
                step_edges,
                edge_indices,
                node_output_slices,
            )
            return tuple(out_nodes[name] for name in self.node_types) + tuple(
                out_edges[name] for name in self.edge_types
            )

        flat_outputs = activation_checkpoint(
            wrapped_step,
            *flat_inputs,
            use_reentrant=True,
        )
        out_nodes = {
            node_type: flat_outputs[index] for index, node_type in enumerate(self.node_types)
        }
        out_edges = {
            edge_type: flat_outputs[node_count + index]
            for index, edge_type in enumerate(self.edge_types)
        }
        return out_nodes, out_edges

    def _processor_step(
        self,
        step: int,
        previous_nodes: dict[str, torch.Tensor],
        previous_edges: dict[str, torch.Tensor],
        edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
        node_output_slices: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        node_output_slices = node_output_slices or {}
        edge_updates = {}
        received_by_node_type: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.node_types
        }
        for edge_type in self.edge_types:
            sender_type, receiver_type = self.edge_node_sets[edge_type]
            senders, receivers = edge_indices[edge_type]
            factorized_projection = None
            if self.edge_input_projection_mode == "factorized":
                factorized_projection = self._prepare_factorized_edge_projection(
                    f"processor_edges_{step}_{edge_type}",
                    previous_nodes[sender_type],
                    previous_nodes[receiver_type],
                )
            chunk_size = self.edge_update_chunk_sizes.get(edge_type)
            edge_count = int(previous_edges[edge_type].shape[1])
            if chunk_size is not None and chunk_size > 0 and edge_count > chunk_size:
                aggregated = previous_nodes[receiver_type].new_zeros(
                    previous_nodes[receiver_type].shape[0],
                    previous_nodes[receiver_type].shape[1],
                    previous_nodes[receiver_type].shape[2],
                )
                for start in range(0, edge_count, chunk_size):
                    end = min(start + chunk_size, edge_count)
                    chunk_senders = senders[start:end]
                    chunk_receivers = receivers[start:end]
                    edge_chunk = previous_edges[edge_type][:, start:end, :]
                    if edge_chunk.shape[-1] != previous_nodes[sender_type].shape[-1]:
                        edge_chunk = self._apply_mlp_with_layer_norm(
                            f"encoder_edges_{edge_type}", edge_chunk
                        )
                    if factorized_projection is None:
                        edge_inputs = torch.cat(
                            [
                                edge_chunk,
                                gather_node_features(
                                    previous_nodes[sender_type],
                                    chunk_senders,
                                ),
                                gather_node_features(
                                    previous_nodes[receiver_type],
                                    chunk_receivers,
                                ),
                            ],
                            dim=-1,
                        )
                        edge_delta = self._apply_mlp_with_layer_norm(
                            f"processor_edges_{step}_{edge_type}", edge_inputs
                        )
                    else:
                        edge_inputs = None
                        edge_delta = self._apply_factorized_edge_projection(
                            f"processor_edges_{step}_{edge_type}",
                            edge_chunk,
                            chunk_senders,
                            chunk_receivers,
                            factorized_projection,
                        )
                    execute_index_add_(
                        aggregated,
                        chunk_receivers,
                        edge_delta,
                        execution=self.index_add_execution,
                    )
                    del edge_chunk, edge_inputs, edge_delta
                edge_updates[edge_type] = None
                received_by_node_type[receiver_type].append(aggregated)
                continue

            if factorized_projection is None:
                edge_inputs = torch.cat(
                    [
                        previous_edges[edge_type],
                        gather_node_features(previous_nodes[sender_type], senders),
                        gather_node_features(previous_nodes[receiver_type], receivers),
                    ],
                    dim=-1,
                )
                edge_delta = self._apply_mlp_with_layer_norm(
                    f"processor_edges_{step}_{edge_type}", edge_inputs
                )
            else:
                edge_delta = self._apply_factorized_edge_projection(
                    f"processor_edges_{step}_{edge_type}",
                    previous_edges[edge_type],
                    senders,
                    receivers,
                    factorized_projection,
                )
            edge_updates[edge_type] = edge_delta

        updated_nodes = {}
        for node_type in self.node_types:
            received_features = list(received_by_node_type[node_type])
            for edge_type in self.edge_types:
                _, receiver_type = self.edge_node_sets[edge_type]
                if receiver_type != node_type:
                    continue
                edge_delta = edge_updates[edge_type]
                if edge_delta is None:
                    continue
                _, receivers = edge_indices[edge_type]
                received_features.append(
                    aggregate_edges_to_nodes(
                        edge_delta,
                        receivers,
                        previous_nodes[node_type].shape[1],
                        index_add_execution=self.index_add_execution,
                    )
                )

            chunk_size = self.node_update_chunk_sizes.get(node_type)
            node_count = int(previous_nodes[node_type].shape[1])
            output_slice = node_output_slices.get(node_type)
            if output_slice is None:
                output_start, output_end = 0, node_count
            else:
                output_start, output_end = output_slice
                if output_start < 0 or output_end <= output_start:
                    raise ValueError(f"invalid node output slice for {node_type}: {output_slice}")
                if output_end > node_count:
                    raise ValueError(
                        f"node output slice for {node_type} exceeds {node_count}: {output_slice}"
                    )
            output_count = output_end - output_start
            if chunk_size is not None and chunk_size > 0 and output_count > chunk_size:
                updated_chunks = []
                for start in range(output_start, output_end, chunk_size):
                    end = min(start + chunk_size, output_end)
                    previous_chunk = previous_nodes[node_type][:, start:end, :]
                    if received_features:
                        feature_chunks = [feature[:, start:end, :] for feature in received_features]
                        node_inputs = torch.cat(
                            [previous_chunk] + feature_chunks,
                            dim=-1,
                        )
                        del feature_chunks
                    else:
                        node_inputs = previous_chunk
                    updated_chunks.append(
                        self._apply_mlp_with_layer_norm_residual(
                            f"processor_nodes_{step}_{node_type}",
                            node_inputs,
                            previous_chunk,
                        )
                    )
                    del previous_chunk, node_inputs
                updated_nodes[node_type] = torch.cat(updated_chunks, dim=1)
                del updated_chunks
                continue

            previous_output = previous_nodes[node_type][:, output_start:output_end, :]
            if received_features:
                sliced_received_features = [
                    feature[:, output_start:output_end, :] for feature in received_features
                ]
                node_inputs = torch.cat(
                    [previous_output] + sliced_received_features,
                    dim=-1,
                )
                del sliced_received_features
            else:
                node_inputs = previous_output
            updated_nodes[node_type] = self._apply_mlp_with_layer_norm_residual(
                f"processor_nodes_{step}_{node_type}",
                node_inputs,
                previous_output,
            )

        updated_edges = {}
        for edge_type in self.edge_types:
            edge_delta = edge_updates[edge_type]
            updated_edges[edge_type] = (
                previous_edges[edge_type]
                if edge_delta is None
                else previous_edges[edge_type] + edge_delta
            )
        return updated_nodes, updated_edges

    def _prepare_factorized_edge_projection(
        self,
        base_name: str,
        sender_nodes: torch.Tensor,
        receiver_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlp = getattr(self, f"{base_name}_mlp")
        linear_0 = mlp.linear_0
        sender_features = int(sender_nodes.shape[-1])
        receiver_features = int(receiver_nodes.shape[-1])
        edge_features = int(linear_0.in_features) - sender_features - receiver_features
        if edge_features <= 0:
            raise ValueError(
                f"{base_name} cannot split {linear_0.in_features} input features into "
                f"edge + sender({sender_features}) + receiver({receiver_features})"
            )
        edge_weight, sender_weight, receiver_weight = linear_0.weight.split(
            (edge_features, sender_features, receiver_features),
            dim=1,
        )
        # Keep both node projections on addmm so the factorized path stays
        # inside the existing inference operator contract.  The learned bias
        # is still applied exactly once by the edge projection below.
        zero_bias = linear_0.bias.new_zeros(linear_0.out_features)
        sender_projection = torch.nn.functional.linear(
            sender_nodes,
            sender_weight,
            zero_bias,
        )
        receiver_projection = torch.nn.functional.linear(
            receiver_nodes,
            receiver_weight,
            zero_bias,
        )
        return edge_weight, sender_projection, receiver_projection

    def _apply_factorized_edge_projection(
        self,
        base_name: str,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        projection: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        edge_weight, sender_projection, receiver_projection = projection
        if int(edge_features.shape[-1]) != int(edge_weight.shape[1]):
            raise ValueError(
                f"{base_name} expected {edge_weight.shape[1]} edge features, "
                f"got {edge_features.shape[-1]}"
            )
        mlp = getattr(self, f"{base_name}_mlp")
        hidden = torch.nn.functional.linear(edge_features, edge_weight, mlp.linear_0.bias)
        hidden = hidden + gather_node_features(sender_projection, senders)
        hidden = hidden + gather_node_features(receiver_projection, receivers)
        output = mlp.linear_1(mlp.activation(hidden))
        return getattr(self, f"{base_name}_layer_norm")(output)

    def _apply_mlp_with_layer_norm(self, base_name: str, x: torch.Tensor) -> torch.Tensor:
        x = getattr(self, f"{base_name}_mlp")(x)
        return getattr(self, f"{base_name}_layer_norm")(x)

    def _apply_mlp_with_layer_norm_residual(
        self,
        base_name: str,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        x = getattr(self, f"{base_name}_mlp")(x)
        layer_norm = getattr(self, f"{base_name}_layer_norm")
        if not self.fuse_post_layer_norm_residual:
            return layer_norm(x) + residual

        # Node deltas have one residual consumer. Edge deltas are intentionally
        # excluded because they are also consumed by the message aggregation.
        try:
            import flag_gems
        except ImportError as exc:
            raise RuntimeError("post_layer_norm_residual fusion requires FlagGems") from exc
        fused_op = getattr(flag_gems, "post_layer_norm_residual", None)
        if fused_op is None:
            raise RuntimeError(
                "the selected FlagGems runtime does not provide post_layer_norm_residual"
            )
        return fused_op(
            x,
            residual,
            layer_norm.normalized_shape,
            layer_norm.weight,
            layer_norm.bias,
            layer_norm.eps,
        )

    def _apply_decoder_node_output(
        self,
        node_type: str,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = self.decoder_node_chunk_sizes.get(node_type)
        node_count = int(node_features.shape[1])
        decoder = getattr(self, f"decoder_nodes_{node_type}_mlp")
        if chunk_size is None or chunk_size <= 0 or node_count <= chunk_size:
            return decoder(node_features)

        output_chunks = []
        for start in range(0, node_count, chunk_size):
            end = min(start + chunk_size, node_count)
            output_chunks.append(decoder(node_features[:, start:end, :]))
        return torch.cat(output_chunks, dim=1)

    def set_stage_devices(
        self,
        *,
        encoder_device: str | torch.device | None = None,
        processor_step_devices: Iterable[str | torch.device | None] | None = None,
        decoder_device: str | torch.device | None = None,
    ) -> None:
        self.encoder_device = normalize_optional_device(encoder_device)
        self.decoder_device = normalize_optional_device(decoder_device)
        if processor_step_devices is None:
            self.processor_step_devices = tuple(None for _ in range(self.message_passing_steps))
        else:
            normalized_steps = tuple(
                normalize_optional_device(device) for device in processor_step_devices
            )
            if len(normalized_steps) != self.message_passing_steps:
                raise ValueError("processor_step_devices length must equal message_passing_steps")
            self.processor_step_devices = normalized_steps

        if self.encoder_device is not None:
            for edge_type in self.edge_types:
                self._move_mlp_with_layer_norm(f"encoder_edges_{edge_type}", self.encoder_device)
            for node_type in self.node_types:
                base_name = f"encoder_nodes_{node_type}"
                if hasattr(self, f"{base_name}_mlp"):
                    self._move_mlp_with_layer_norm(base_name, self.encoder_device)

        for step, device in enumerate(self.processor_step_devices):
            if device is None:
                continue
            for edge_type in self.edge_types:
                self._move_mlp_with_layer_norm(f"processor_edges_{step}_{edge_type}", device)
            for node_type in self.node_types:
                self._move_mlp_with_layer_norm(f"processor_nodes_{step}_{node_type}", device)

        if self.decoder_device is not None:
            for name, module in self.named_children():
                if name.startswith("decoder_nodes_") and name.endswith("_mlp"):
                    module.to(self.decoder_device)

    def _move_mlp_with_layer_norm(self, base_name: str, device: torch.device) -> None:
        getattr(self, f"{base_name}_mlp").to(device)
        getattr(self, f"{base_name}_layer_norm").to(device)


def execute_index_add_(
    out: torch.Tensor,
    receivers: torch.Tensor,
    edge_features: torch.Tensor,
    *,
    execution: str,
) -> torch.Tensor:
    if execution in {"inline", "prevalidated-static"}:
        return out.index_add_(1, receivers, edge_features)
    raise ValueError(
        f"index_add execution must be 'inline' or 'prevalidated-static', got {execution!r}"
    )


def aggregate_edges_to_nodes(
    edge_features: torch.Tensor,
    receivers: torch.Tensor,
    num_nodes: int,
    *,
    index_add_execution: str = "inline",
) -> torch.Tensor:
    out = edge_features.new_zeros(
        edge_features.shape[0],
        num_nodes,
        edge_features.shape[-1],
    )
    return execute_index_add_(
        out,
        receivers,
        edge_features,
        execution=index_add_execution,
    )


def gather_node_features(
    node_features: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather node features for graph edges along the node dimension."""

    if use_advanced_index_gather():
        gathered = node_features.transpose(0, 1)[indices].transpose(0, 1)
        return gathered.contiguous()
    return node_features.index_select(1, indices)


def normalize_optional_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def move_tensor_dict(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device=device) for name, tensor in tensors.items()}


def move_edge_indices(
    edge_indices: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (senders.to(device=device), receivers.to(device=device))
        for name, (senders, receivers) in edge_indices.items()
    }


class GoogleSmallCompatibleModel(nn.Module):
    """PyTorch model matching Google GraphCast_small params and tensor flow."""

    graphnet_type = GoogleSmallGraphNet

    def __init__(
        self,
        config: GoogleSmallConfig | None = None,
        *,
        activation_checkpointing: bool = False,
        fuse_post_layer_norm_residual: bool | None = None,
        edge_input_projection_mode: str = "concatenated",
        mesh2grid_full_grid_fast_path: bool = False,
        index_add_execution: str = "inline",
    ) -> None:
        super().__init__()
        if fuse_post_layer_norm_residual is None:
            fuse_post_layer_norm_residual = _environment_flag(
                "GRAPHCAST_FUSE_POST_LAYER_NORM_RESIDUAL"
            )
        self.config = config or GoogleSmallConfig()
        self.activation_checkpointing = activation_checkpointing
        self.fuse_post_layer_norm_residual = fuse_post_layer_norm_residual
        self.edge_input_projection_mode = edge_input_projection_mode
        self.mesh2grid_full_grid_fast_path = mesh2grid_full_grid_fast_path
        self.index_add_execution = index_add_execution
        self._graph_initialized = False
        latent = self.config.latent_size
        node_input = self.config.node_input_dim
        edge_input = self.config.edge_input_dim

        self.grid2mesh_gnn = self.graphnet_type(
            edge_types=("grid2mesh",),
            edge_node_sets={"grid2mesh": ("grid_nodes", "mesh_nodes")},
            node_types=("grid_nodes", "mesh_nodes"),
            node_input_dims={
                "grid_nodes": node_input,
                "mesh_nodes": node_input,
            },
            edge_input_dims={"grid2mesh": edge_input},
            node_processor_input_dims={
                "grid_nodes": latent,
                "mesh_nodes": latent * 2,
            },
            edge_processor_input_dims={"grid2mesh": latent * 3},
            node_output_dims=None,
            extra_final_node_output_dims=None,
            edge_update_chunk_sizes=(
                {"grid2mesh": self.config.grid2mesh_edge_chunk_size}
                if self.config.grid2mesh_edge_chunk_size
                else None
            ),
            node_update_chunk_sizes=(
                {"grid_nodes": self.config.grid2mesh_node_chunk_size}
                if self.config.grid2mesh_node_chunk_size
                else None
            ),
            decoder_node_chunk_sizes=None,
            message_passing_steps=1,
            config=self.config,
            activation_checkpointing=activation_checkpointing,
            fuse_post_layer_norm_residual=fuse_post_layer_norm_residual,
            edge_input_projection_mode=edge_input_projection_mode,
            index_add_execution=index_add_execution,
        )

        self.mesh_gnn = self.graphnet_type(
            edge_types=("mesh",),
            edge_node_sets={"mesh": ("mesh_nodes", "mesh_nodes")},
            node_types=("mesh_nodes",),
            node_input_dims=None,
            edge_input_dims={"mesh": edge_input},
            node_processor_input_dims={"mesh_nodes": latent * 2},
            edge_processor_input_dims={"mesh": latent * 3},
            node_output_dims=None,
            extra_final_node_output_dims=None,
            edge_update_chunk_sizes=None,
            node_update_chunk_sizes=None,
            decoder_node_chunk_sizes=None,
            message_passing_steps=self.config.gnn_msg_steps,
            config=self.config,
            activation_checkpointing=activation_checkpointing,
            fuse_post_layer_norm_residual=fuse_post_layer_norm_residual,
            edge_input_projection_mode=edge_input_projection_mode,
            index_add_execution=index_add_execution,
        )

        self.mesh2grid_gnn = self.graphnet_type(
            edge_types=("mesh2grid",),
            edge_node_sets={"mesh2grid": ("mesh_nodes", "grid_nodes")},
            node_types=("grid_nodes", "mesh_nodes"),
            node_input_dims=None,
            edge_input_dims={"mesh2grid": edge_input},
            node_processor_input_dims={
                "grid_nodes": latent * 2,
                "mesh_nodes": latent,
            },
            edge_processor_input_dims={"mesh2grid": latent * 3},
            node_output_dims={"grid_nodes": self.config.output_dim},
            extra_final_node_output_dims=(self.config.extra_mesh2grid_final_output_heads),
            edge_update_chunk_sizes=(
                {"mesh2grid": self.config.mesh2grid_edge_chunk_size}
                if self.config.mesh2grid_edge_chunk_size
                else None
            ),
            node_update_chunk_sizes=(
                {"grid_nodes": self.config.mesh2grid_node_chunk_size}
                if self.config.mesh2grid_node_chunk_size
                else None
            ),
            decoder_node_chunk_sizes=(
                {"grid_nodes": self.config.mesh2grid_decoder_chunk_size}
                if self.config.mesh2grid_decoder_chunk_size
                else None
            ),
            message_passing_steps=1,
            config=self.config,
            activation_checkpointing=activation_checkpointing,
            fuse_post_layer_norm_residual=fuse_post_layer_norm_residual,
            edge_input_projection_mode=edge_input_projection_mode,
            index_add_execution=index_add_execution,
        )

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = enabled
        self.grid2mesh_gnn.activation_checkpointing = enabled
        self.mesh_gnn.activation_checkpointing = enabled
        self.mesh2grid_gnn.activation_checkpointing = enabled

    def set_model_parallel_devices(
        self,
        *,
        grid2mesh_device: str | torch.device,
        mesh_step_devices: Iterable[str | torch.device],
        mesh2grid_device: str | torch.device,
    ) -> None:
        mesh_step_devices = tuple(torch.device(device) for device in mesh_step_devices)
        if len(mesh_step_devices) != self.config.gnn_msg_steps:
            raise ValueError("mesh_step_devices must match config.gnn_msg_steps")
        grid2mesh_device = torch.device(grid2mesh_device)
        mesh2grid_device = torch.device(mesh2grid_device)
        self.grid2mesh_gnn.set_stage_devices(
            encoder_device=grid2mesh_device,
            processor_step_devices=[grid2mesh_device],
            decoder_device=grid2mesh_device,
        )
        self.mesh_gnn.set_stage_devices(
            encoder_device=mesh_step_devices[0],
            processor_step_devices=mesh_step_devices,
            decoder_device=mesh_step_devices[-1],
        )
        self.mesh2grid_gnn.set_stage_devices(
            encoder_device=mesh2grid_device,
            processor_step_devices=[mesh2grid_device],
            decoder_device=mesh2grid_device,
        )

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
            canonical=False,
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

    def prevalidate_static_index_add(self) -> StaticIndexAddContract:
        if self.index_add_execution != "prevalidated-static":
            raise RuntimeError("static index_add prevalidation was not requested")
        if self.training:
            raise RuntimeError("static index_add prevalidation is inference-only")
        if not self._graph_initialized:
            raise RuntimeError("the GraphCast graph must be initialized before prevalidation")
        if not self.mesh2grid_full_grid_fast_path:
            raise ValueError("prevalidated static index_add requires the full-grid fast path")

        specifications = (
            ("grid2mesh_receivers", self.grid2mesh_receivers, int(self._num_mesh_nodes)),
            ("mesh_receivers", self.mesh_receivers, int(self._num_mesh_nodes)),
            ("mesh2grid_receivers", self.mesh2grid_receivers, int(self._num_grid_nodes)),
        )
        contracts: list[StaticIndexAddBufferContract] = []
        for name, tensor, upper_bound in specifications:
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional index tensor")
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must use int32 or int64 indices")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if tensor.numel() <= 0:
                raise ValueError(f"{name} must not be empty")
            lower, upper = torch.aminmax(tensor)
            lower_value = int(lower.item())
            upper_value = int(upper.item())
            if lower_value < 0 or upper_value >= upper_bound:
                raise ValueError(
                    f"{name} is out of bounds: observed [{lower_value}, {upper_value}], "
                    f"required [0, {upper_bound})"
                )
            contracts.append(
                StaticIndexAddBufferContract(
                    name=name,
                    shape=tuple(int(value) for value in tensor.shape),
                    stride=tuple(int(value) for value in tensor.stride()),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                    data_ptr=int(tensor.data_ptr()),
                    version=int(tensor._version),
                    numel=int(tensor.numel()),
                    lower_bound=lower_value,
                    upper_value=upper_value,
                    upper_bound=upper_bound,
                )
            )
        return StaticIndexAddContract(buffers=tuple(contracts))

    def verify_static_index_add(self, contract: StaticIndexAddContract) -> None:
        for expected in contract.buffers:
            tensor = getattr(self, expected.name)
            observed = (
                tuple(int(value) for value in tensor.shape),
                tuple(int(value) for value in tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
                int(tensor.data_ptr()),
                int(tensor._version),
                int(tensor.numel()),
            )
            frozen = (
                expected.shape,
                expected.stride,
                expected.dtype,
                expected.device,
                expected.data_ptr,
                expected.version,
                expected.numel,
            )
            if observed != frozen:
                raise RuntimeError(f"{expected.name} changed after static index_add prevalidation")

    def forward(self, grid_node_features: torch.Tensor) -> torch.Tensor:
        """Run a minimal stacked-tensor forward pass.

        Args:
            grid_node_features: Tensor with shape ``[batch, grid_nodes, 183]``.

        Returns:
            Tensor with shape ``[batch, grid_nodes, 83]``.
        """
        grid2mesh_nodes, mesh_nodes_out = self.forward_pre_mesh2grid(grid_node_features)
        return self.forward_mesh2grid_partition(
            mesh_nodes=mesh_nodes_out["mesh_nodes"],
            grid_nodes=grid2mesh_nodes["grid_nodes"],
            grid_start=0,
            grid_end=int(grid2mesh_nodes["grid_nodes"].shape[1]),
        )

    def forward_from_state_forcing(
        self,
        state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 3 or forcing.ndim != 3:
            raise ValueError("state and forcing must both have [batch, grid_nodes, channels]")
        if state.shape[:-1] != forcing.shape[:-1]:
            raise ValueError("state and forcing must have matching batch and grid dimensions")
        if state.device != forcing.device or state.dtype != forcing.dtype:
            raise ValueError("state and forcing must have matching device and dtype")
        return self(torch.cat((state, forcing), dim=-1))

    def forward_pre_mesh2grid(
        self,
        grid_node_features: torch.Tensor,
        grid_output_slice: tuple[int, int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Run grid2mesh and mesh processor stages before mesh2grid."""
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
        dtype = grid_node_features.dtype

        grid_geo = self.grid_node_geo_features.to(device=device, dtype=dtype)
        mesh_geo = self.mesh_node_geo_features.to(device=device, dtype=dtype)

        grid_nodes = torch.cat(
            [grid_node_features, grid_geo.unsqueeze(0).expand(batch_size, -1, -1)],
            dim=-1,
        )
        dummy_mesh_inputs = grid_node_features.new_zeros(
            batch_size, mesh_geo.shape[0], self.config.node_input_dim - 3
        )
        mesh_nodes = torch.cat(
            [dummy_mesh_inputs, mesh_geo.unsqueeze(0).expand(batch_size, -1, -1)],
            dim=-1,
        )

        grid2mesh_edges = self._edge_features_for_batch(
            self.grid2mesh_edge_features, batch_size, device, dtype
        )
        grid2mesh_nodes, _ = self.grid2mesh_gnn(
            {"grid_nodes": grid_nodes, "mesh_nodes": mesh_nodes},
            {"grid2mesh": grid2mesh_edges},
            {
                "grid2mesh": (
                    self.grid2mesh_senders.to(device=device),
                    self.grid2mesh_receivers.to(device=device),
                )
            },
            node_output_slices=({"grid_nodes": grid_output_slice} if grid_output_slice else None),
        )

        mesh_edges = self._edge_features_for_batch(
            self.mesh_edge_features, batch_size, device, dtype
        )
        mesh_nodes_out, _ = self.mesh_gnn(
            {"mesh_nodes": grid2mesh_nodes["mesh_nodes"]},
            {"mesh": mesh_edges},
            {
                "mesh": (
                    self.mesh_senders.to(device=device),
                    self.mesh_receivers.to(device=device),
                )
            },
        )
        return grid2mesh_nodes, mesh_nodes_out

    def forward_mesh2grid_partition(
        self,
        *,
        mesh_nodes: torch.Tensor,
        grid_nodes: torch.Tensor,
        grid_start: int,
        grid_end: int,
    ) -> torch.Tensor:
        """Run mesh2grid for one contiguous grid-node partition."""
        if not self._graph_initialized:
            self.init_graph()
        if grid_start < 0 or grid_end <= grid_start:
            raise ValueError("grid partition must satisfy 0 <= start < end")
        batch_size = grid_nodes.shape[0]
        device = grid_nodes.device
        dtype = grid_nodes.dtype
        edge_features, senders, local_receivers = self._select_mesh2grid_partition_inputs(
            grid_start=grid_start,
            grid_end=grid_end,
        )
        mesh2grid_edges = self._edge_features_for_batch(edge_features, batch_size, device, dtype)
        mesh2grid_nodes, _ = self.mesh2grid_gnn(
            {
                "mesh_nodes": mesh_nodes,
                "grid_nodes": grid_nodes,
            },
            {"mesh2grid": mesh2grid_edges},
            {
                "mesh2grid": (
                    senders.to(device=device),
                    local_receivers.to(device=device),
                )
            },
        )
        return mesh2grid_nodes["grid_nodes"]

    def _select_mesh2grid_partition_inputs(
        self,
        *,
        grid_start: int,
        grid_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.mesh2grid_full_grid_fast_path
            and grid_start == 0
            and grid_end == int(self._num_grid_nodes)
        ):
            return (
                self.mesh2grid_edge_features,
                self.mesh2grid_senders,
                self.mesh2grid_receivers,
            )

        receivers = self.mesh2grid_receivers
        if grid_end > int(receivers.max().item()) + 1:
            raise ValueError("grid partition end exceeds initialized graph grid range")
        edge_mask = (receivers >= grid_start) & (receivers < grid_end)
        if not bool(edge_mask.any()):
            raise ValueError(f"mesh2grid partition [{grid_start}, {grid_end}) has no edges")
        return (
            self.mesh2grid_edge_features[edge_mask],
            self.mesh2grid_senders[edge_mask],
            receivers[edge_mask] - grid_start,
        )

    def predict_xarray(
        self,
        inputs: Any,
        targets_template: Any,
        forcings: Any,
    ) -> Any:
        """Run single-step prediction using Google's xarray argument order."""

        from .data import grid_node_outputs_to_prediction, inputs_to_grid_node_features

        latitudes = inputs.coords["lat"].values
        longitudes = inputs.coords["lon"].values
        if not self._graph_initialized or int(self.grid_node_geo_features.shape[0]) != len(
            latitudes
        ) * len(longitudes):
            self.init_graph(latitudes=latitudes, longitudes=longitudes)

        device = next(self.parameters()).device
        grid_node_features = inputs_to_grid_node_features(
            inputs,
            forcings,
            input_variables=self.config.input_variables or None,
            forcing_variables=self.config.forcing_variables or None,
        ).to(device=device)
        with torch.inference_mode():
            outputs = self(grid_node_features)
        return grid_node_outputs_to_prediction(outputs, targets_template)

    def _set_nonpersistent_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor, persistent=False)

    @staticmethod
    def _edge_features_for_batch(
        features: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return features.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


def build_google_small_compatible_model(
    config_path: str | Path = DEFAULT_CONFIG,
) -> GoogleSmallCompatibleModel:
    """Build the compatible model from the local YAML contract."""

    return GoogleSmallCompatibleModel(GoogleSmallConfig.from_yaml(config_path))
