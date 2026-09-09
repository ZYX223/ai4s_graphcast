"""Google-compatible graph construction for GraphCast_small."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.spatial
import torch
import trimesh
from scipy.spatial import transform

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_MESH_PATH = REPO_ROOT / "src/graphcast_compat/build_mesh.py"
CANONICAL_MESH6_VERTICES_PATH = (
    Path(__file__).resolve().parent / "assets" / "google_mesh6_vertices.npy"
)
CANONICAL_MESH6_VERTEX_COUNT = 40962
CANONICAL_GRAPH_TOPOLOGY_VERSION = "google_mesh6_h100_v1"
CANONICAL_OPERATIONAL_GRAPH_CONTRACT = {
    "resolution_degrees": 0.25,
    "mesh_size": 6,
    "mesh_message_passing_steps": 16,
    "grid_nodes": 1038240,
    "mesh_nodes": 40962,
    "grid2mesh_edges": 1618745,
    "mesh_edges": 327660,
    "mesh2grid_edges": 3114720,
}


def _load_build_mesh() -> Any:
    spec = importlib.util.spec_from_file_location("_graphcast_compat_build_mesh", BUILD_MESH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load build_mesh from {BUILD_MESH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_mesh = _load_build_mesh()


def canonical_mesh_hierarchy(mesh_size: int) -> list[Any]:
    """Return the pinned mesh hierarchy without host-BLAS vertex drift.

    DeepMind's mesh refinement normalizes each new float32 midpoint with
    ``np.linalg.norm``.  Different BLAS reduction orders can change a vertex by
    one ULP and can consequently change the radius-query edge set.  The mesh-6
    hierarchy is part of the model contract, so use the H100-baseline vertices
    and retain the pinned implementation's face/index construction.
    """

    if not 0 <= mesh_size <= 6:
        raise ValueError("canonical GraphCast mesh_size must be in [0, 6]")
    canonical_vertices = np.load(
        CANONICAL_MESH6_VERTICES_PATH,
        allow_pickle=False,
    )
    if canonical_vertices.shape != (CANONICAL_MESH6_VERTEX_COUNT, 3):
        raise RuntimeError(f"canonical mesh asset has unexpected shape {canonical_vertices.shape}")
    if canonical_vertices.dtype != np.float32:
        raise RuntimeError(
            f"canonical mesh asset must have float32 dtype, got {canonical_vertices.dtype}"
        )

    generated = build_mesh.get_hierarchy_of_triangular_meshes_for_sphere(mesh_size)
    return [
        build_mesh.TriangularMesh(
            vertices=canonical_vertices[: mesh.vertices.shape[0]],
            faces=mesh.faces,
        )
        for mesh in generated
    ]


def validate_canonical_operational_graph_contract(
    contract: Mapping[str, Any],
) -> None:
    """Fail closed when a formal operational graph changes structure."""

    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in CANONICAL_OPERATIONAL_GRAPH_CONTRACT.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"operational GraphCast topology disagrees with the H100 baseline: {mismatches}"
        )


@dataclass(frozen=True)
class GoogleSmallGraph:
    """Static graph tensors used by the PyTorch compatible model."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    mesh_node_latitudes: torch.Tensor
    mesh_node_longitudes: torch.Tensor
    grid_node_latitudes: torch.Tensor
    grid_node_longitudes: torch.Tensor
    grid_node_geo_features: torch.Tensor
    mesh_node_geo_features: torch.Tensor
    grid2mesh_edge_features: torch.Tensor
    grid2mesh_senders: torch.Tensor
    grid2mesh_receivers: torch.Tensor
    mesh_edge_features: torch.Tensor
    mesh_senders: torch.Tensor
    mesh_receivers: torch.Tensor
    mesh2grid_edge_features: torch.Tensor
    mesh2grid_senders: torch.Tensor
    mesh2grid_receivers: torch.Tensor

    @property
    def num_grid_nodes(self) -> int:
        return int(self.grid_node_geo_features.shape[0])

    @property
    def num_mesh_nodes(self) -> int:
        return int(self.mesh_node_geo_features.shape[0])


def default_latitudes(resolution: float) -> np.ndarray:
    """Return ERA5-style descending latitude coordinates for a given resolution."""

    return np.arange(90.0, -90.0 - resolution * 0.5, -resolution, dtype=np.float32)


def default_longitudes(resolution: float) -> np.ndarray:
    return np.arange(0.0, 360.0, resolution, dtype=np.float32)


def max_edge_distance(mesh: Any) -> float:
    senders, receivers = build_mesh.faces_to_edges(mesh.faces)
    distances = np.linalg.norm(mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    return float(distances.max())


def lat_lon_deg_to_spherical(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90.0 - node_lat)
    return phi, theta


def spherical_to_lat_lon(
    phi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lon = np.mod(np.rad2deg(phi), 360.0)
    lat = 90.0 - np.rad2deg(theta)
    return lat, lon


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):
        theta = np.arccos(z)
    return phi, theta


def spherical_to_cartesian(
    phi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta),
    )


def grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
) -> np.ndarray:
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude),
        np.deg2rad(90.0 - grid_latitude),
    )
    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: Any,
    radius: float,
    sort_neighbors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    grid_positions = grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
    kd_tree = scipy.spatial.cKDTree(mesh.vertices)
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        # cKDTree does not promise the same neighbor ordering across SciPy
        # versions.  Aggregation order is part of the floating-point execution
        # contract, so canonicalize it even when the membership is identical.
        if sort_neighbors:
            mesh_neighbors = np.sort(np.asarray(mesh_neighbors, dtype=np.int64))
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    return (
        np.concatenate(grid_edge_indices, axis=0).astype(np.int64),
        np.concatenate(mesh_edge_indices, axis=0).astype(np.int64),
    )


def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: Any,
) -> tuple[np.ndarray, np.ndarray]:
    grid_positions = grid_lat_lon_to_coordinates(grid_latitude, grid_longitude).reshape([-1, 3])
    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    _, _, query_face_indices = trimesh.proximity.closest_point(mesh_trimesh, grid_positions)

    mesh_edge_indices = mesh.faces[query_face_indices]
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])
    return (
        grid_edge_indices.reshape([-1]).astype(np.int64),
        mesh_edge_indices.reshape([-1]).astype(np.int64),
    )


def rotation_matrices_to_local_coordinates(
    reference_phi: np.ndarray,
    reference_theta: np.ndarray,
    *,
    rotate_latitude: bool,
    rotate_longitude: bool,
) -> np.ndarray:
    azimuthal_rotation = -reference_phi
    polar_rotation = -reference_theta + np.pi / 2.0

    if rotate_longitude and rotate_latitude:
        return transform.Rotation.from_euler(
            "zy", np.stack([azimuthal_rotation, polar_rotation], axis=1)
        ).as_matrix()
    if rotate_longitude:
        return transform.Rotation.from_euler(
            "z", np.expand_dims(azimuthal_rotation, axis=1)
        ).as_matrix()
    if rotate_latitude:
        return transform.Rotation.from_euler(
            "zyz",
            np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation],
                axis=1,
            ),
        ).as_matrix()
    raise ValueError("at least one of longitude and latitude should be rotated")


def rotate_with_matrices(rotation_matrices: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.einsum("...ji,...i->...j", rotation_matrices, positions)


def node_geo_features(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
) -> np.ndarray:
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lat, node_lon)
    return np.stack(
        [
            np.cos(node_theta),
            np.cos(node_phi),
            np.sin(node_phi),
        ],
        axis=-1,
    ).astype(np.float32)


def relative_position_in_receiver_local_coordinates(
    *,
    senders_node_phi: np.ndarray,
    senders_node_theta: np.ndarray,
    receivers_node_phi: np.ndarray,
    receivers_node_theta: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    sender_pos = np.stack(spherical_to_cartesian(senders_node_phi, senders_node_theta), axis=-1)
    receiver_pos = np.stack(
        spherical_to_cartesian(receivers_node_phi, receivers_node_theta), axis=-1
    )

    receiver_rotation_matrices = rotation_matrices_to_local_coordinates(
        receivers_node_phi,
        receivers_node_theta,
        rotate_latitude=True,
        rotate_longitude=True,
    )
    edge_rotation_matrices = receiver_rotation_matrices[receivers]
    receiver_pos_local = rotate_with_matrices(edge_rotation_matrices, receiver_pos[receivers])
    sender_pos_local = rotate_with_matrices(edge_rotation_matrices, sender_pos[senders])
    return sender_pos_local - receiver_pos_local


def edge_features_for_bipartite_graph(
    *,
    senders_node_lat: np.ndarray,
    senders_node_lon: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers_node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    edge_normalization_factor: float | None,
) -> np.ndarray:
    senders_phi, senders_theta = lat_lon_deg_to_spherical(senders_node_lat, senders_node_lon)
    receivers_phi, receivers_theta = lat_lon_deg_to_spherical(
        receivers_node_lat, receivers_node_lon
    )
    relative_position = relative_position_in_receiver_local_coordinates(
        senders_node_phi=senders_phi,
        senders_node_theta=senders_theta,
        receivers_node_phi=receivers_phi,
        receivers_node_theta=receivers_theta,
        senders=senders,
        receivers=receivers,
    )
    relative_distances = np.linalg.norm(relative_position, axis=-1, keepdims=True)
    if edge_normalization_factor is None:
        edge_normalization_factor = float(relative_distances.max())
    return np.concatenate(
        [
            relative_distances / edge_normalization_factor,
            relative_position / edge_normalization_factor,
        ],
        axis=-1,
    ).astype(np.float32)


def build_google_small_graph(
    *,
    resolution: float,
    mesh_size: int,
    radius_query_fraction_edge_length: float,
    mesh2grid_edge_normalization_factor: float,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
    canonical: bool = False,
) -> GoogleSmallGraph:
    latitudes = (
        default_latitudes(resolution)
        if latitudes is None
        else np.asarray(latitudes, dtype=np.float32)
    )
    longitudes = (
        default_longitudes(resolution)
        if longitudes is None
        else np.asarray(longitudes, dtype=np.float32)
    )

    meshes = (
        canonical_mesh_hierarchy(mesh_size)
        if canonical
        else build_mesh.get_hierarchy_of_triangular_meshes_for_sphere(mesh_size)
    )
    finest_mesh = meshes[-1]
    mesh_phi, mesh_theta = cartesian_to_spherical(
        finest_mesh.vertices[:, 0],
        finest_mesh.vertices[:, 1],
        finest_mesh.vertices[:, 2],
    )
    mesh_node_latitudes, mesh_node_longitudes = spherical_to_lat_lon(mesh_phi, mesh_theta)
    mesh_node_latitudes = mesh_node_latitudes.astype(np.float32)
    mesh_node_longitudes = mesh_node_longitudes.astype(np.float32)

    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    grid_node_latitudes = grid_lat.reshape([-1]).astype(np.float32)
    grid_node_longitudes = grid_lon.reshape([-1]).astype(np.float32)

    query_radius = max_edge_distance(finest_mesh) * radius_query_fraction_edge_length
    grid2mesh_grid, grid2mesh_mesh = radius_query_indices(
        grid_latitude=latitudes,
        grid_longitude=longitudes,
        mesh=finest_mesh,
        radius=query_radius,
        sort_neighbors=canonical,
    )
    grid2mesh_edge_features = edge_features_for_bipartite_graph(
        senders_node_lat=grid_node_latitudes,
        senders_node_lon=grid_node_longitudes,
        receivers_node_lat=mesh_node_latitudes,
        receivers_node_lon=mesh_node_longitudes,
        senders=grid2mesh_grid,
        receivers=grid2mesh_mesh,
        edge_normalization_factor=None,
    )

    merged_mesh = build_mesh.merge_meshes(meshes)
    mesh_senders, mesh_receivers = build_mesh.faces_to_edges(merged_mesh.faces)
    mesh_edge_features = edge_features_for_bipartite_graph(
        senders_node_lat=mesh_node_latitudes,
        senders_node_lon=mesh_node_longitudes,
        receivers_node_lat=mesh_node_latitudes,
        receivers_node_lon=mesh_node_longitudes,
        senders=mesh_senders,
        receivers=mesh_receivers,
        edge_normalization_factor=None,
    )

    mesh2grid_grid, mesh2grid_mesh = in_mesh_triangle_indices(
        grid_latitude=latitudes,
        grid_longitude=longitudes,
        mesh=finest_mesh,
    )
    mesh2grid_edge_features = edge_features_for_bipartite_graph(
        senders_node_lat=mesh_node_latitudes,
        senders_node_lon=mesh_node_longitudes,
        receivers_node_lat=grid_node_latitudes,
        receivers_node_lon=grid_node_longitudes,
        senders=mesh2grid_mesh,
        receivers=mesh2grid_grid,
        edge_normalization_factor=mesh2grid_edge_normalization_factor,
    )

    return GoogleSmallGraph(
        latitudes=latitudes,
        longitudes=longitudes,
        mesh_node_latitudes=torch.from_numpy(mesh_node_latitudes),
        mesh_node_longitudes=torch.from_numpy(mesh_node_longitudes),
        grid_node_latitudes=torch.from_numpy(grid_node_latitudes),
        grid_node_longitudes=torch.from_numpy(grid_node_longitudes),
        grid_node_geo_features=torch.from_numpy(
            node_geo_features(grid_node_latitudes, grid_node_longitudes)
        ),
        mesh_node_geo_features=torch.from_numpy(
            node_geo_features(mesh_node_latitudes, mesh_node_longitudes)
        ),
        grid2mesh_edge_features=torch.from_numpy(grid2mesh_edge_features),
        grid2mesh_senders=torch.from_numpy(grid2mesh_grid),
        grid2mesh_receivers=torch.from_numpy(grid2mesh_mesh),
        mesh_edge_features=torch.from_numpy(mesh_edge_features),
        mesh_senders=torch.from_numpy(mesh_senders.astype(np.int64)),
        mesh_receivers=torch.from_numpy(mesh_receivers.astype(np.int64)),
        mesh2grid_edge_features=torch.from_numpy(mesh2grid_edge_features),
        mesh2grid_senders=torch.from_numpy(mesh2grid_mesh),
        mesh2grid_receivers=torch.from_numpy(mesh2grid_grid),
    )
