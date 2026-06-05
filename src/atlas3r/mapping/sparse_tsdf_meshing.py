"""Dependency-safe triangle meshing for sparse TSDF block chunks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import FLOAT32
from atlas3r.mapping.mesh_chunks import (
    ObservedMeshChunk,
    chunk_id_from_block_coord,
    numeric_summary,
)
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper, SparseTSDFBlockSnapshot

SPARSE_TSDF_FALLBACK_MESHER_BACKEND = "surface_voxel_face_mesher"
MESH_INSTALL_HINT = "Optional quality meshing: python -m pip install -e .[mesh]"


@dataclass(frozen=True)
class SparseTSDFMesherConfig:
    """Configuration for sparse observed chunk meshing."""

    min_weight: float = 0.0
    surface_band: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        if self.min_weight < 0.0:
            raise ValueError("min_weight: must be non-negative")
        if self.surface_band <= 0.0:
            raise ValueError("surface_band: must be positive")


def skimage_meshing_available() -> bool:
    """Return whether optional scikit-image meshing can be imported later."""

    return importlib.util.find_spec("skimage") is not None


def mesher_backend_status() -> dict[str, object]:
    """Expose optional quality backend status without importing it."""

    return {
        "fallback_backend": SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
        "install_hint": MESH_INSTALL_HINT,
        "skimage_available": skimage_meshing_available(),
        "used_backend": SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
    }


def mesh_sparse_tsdf_block(
    mapper: SparseBlockTSDFMapper,
    block_coord_xyz: tuple[int, int, int],
    *,
    version: int,
    mesher_config: SparseTSDFMesherConfig,
) -> ObservedMeshChunk | None:
    """Mesh one observed sparse TSDF block; return None for empty chunks."""

    snapshot = mapper.block_voxel_snapshot(block_coord_xyz, include_one_voxel_halo=True)
    return mesh_sparse_tsdf_snapshot(
        mapper,
        snapshot,
        version=version,
        mesher_config=mesher_config,
    )


def mesh_sparse_tsdf_snapshot(
    mapper: SparseBlockTSDFMapper,
    snapshot: SparseTSDFBlockSnapshot,
    *,
    version: int,
    mesher_config: SparseTSDFMesherConfig,
) -> ObservedMeshChunk | None:
    """Mesh one precomputed observed sparse TSDF block snapshot."""

    if snapshot.voxel_coords_xyz.size == 0:
        return None
    config = mapper.config
    active_target_mask = snapshot.target_mask & (snapshot.weight > mesher_config.min_weight)
    surface_mask = (
        snapshot.target_mask
        & (snapshot.weight > mesher_config.min_weight)
        & (np.abs(snapshot.tsdf.astype(np.float64, copy=False)) <= mesher_config.surface_band)
    )
    if not np.any(surface_mask):
        return None
    surface_coords = snapshot.voxel_coords_xyz[surface_mask]
    vertices, normals, triangles = _mesh_surface_voxels(
        surface_coords,
        snapshot=snapshot,
        surface_band=mesher_config.surface_band,
        min_weight=mesher_config.min_weight,
        voxel_size_m=config.voxel_size_m,
    )
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        return None
    surface_weight = snapshot.weight[surface_mask].astype(np.float64, copy=False)
    surface_tsdf = snapshot.tsdf[surface_mask].astype(np.float64, copy=False)
    source_frame_ids = list(mapper.source_frame_ids)
    confidence = np.clip(surface_weight / max(float(len(source_frame_ids)), 1.0), 0.0, 1.0)
    uncertainty = (
        config.voxel_size_m / np.sqrt(np.maximum(surface_weight, 1.0))
        + np.abs(surface_tsdf) * config.truncation_distance_m
    )
    return ObservedMeshChunk(
        chunk_id=chunk_id_from_block_coord(snapshot.block_coord_xyz),
        chunk_coord_xyz=snapshot.block_coord_xyz,
        version=version,
        update_type="upsert",
        coordinate_frame=config.coordinate_frame,
        T_world_chunk=np.eye(4, dtype=FLOAT32),
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=config.truncation_distance_m,
        block_size_voxels=config.block_size_voxels,
        vertices_world_m=vertices,
        normals_world=normals,
        triangles=triangles,
        source_frame_ids=source_frame_ids,
        active_voxel_count=int(np.count_nonzero(active_target_mask)),
        surface_voxel_count=int(np.count_nonzero(surface_mask)),
        uncertainty_summary_m=numeric_summary(uncertainty),
        confidence_summary=numeric_summary(confidence),
        observed_coverage_estimate=float(
            np.count_nonzero(active_target_mask) / max(config.block_size_voxels**3, 1)
        ),
        metric_scale_source=config.metric_scale_source,
        mesher_backend=SPARSE_TSDF_FALLBACK_MESHER_BACKEND,
    )


def _mesh_surface_voxels(
    surface_coords: npt.NDArray[np.int64],
    *,
    snapshot: SparseTSDFBlockSnapshot,
    surface_band: float,
    min_weight: float,
    voxel_size_m: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    halo_surface = (snapshot.weight > min_weight) & (
        np.abs(snapshot.tsdf.astype(np.float64, copy=False)) <= surface_band
    )
    surface_set = {
        (int(coord[0]), int(coord[1]), int(coord[2]))
        for coord in snapshot.voxel_coords_xyz[halo_surface]
    }
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []
    for coord_array in surface_coords:
        coord = (int(coord_array[0]), int(coord_array[1]), int(coord_array[2]))
        for direction, normal in _FACE_NORMALS:
            neighbor = (
                coord[0] + direction[0],
                coord[1] + direction[1],
                coord[2] + direction[2],
            )
            if neighbor in surface_set:
                continue
            base_index = len(vertices)
            corners = _face_corners(coord, direction, voxel_size_m)
            vertices.extend(corners)
            normals.extend([list(normal)] * 4)
            triangles.append([base_index, base_index + 1, base_index + 2])
            triangles.append([base_index, base_index + 2, base_index + 3])
    if not vertices:
        return (
            np.empty((0, 3), dtype=FLOAT32),
            np.empty((0, 3), dtype=FLOAT32),
            np.empty((0, 3), dtype=np.uint32),
        )
    return (
        np.asarray(vertices, dtype=FLOAT32),
        np.asarray(normals, dtype=FLOAT32),
        np.asarray(triangles, dtype=np.uint32),
    )


_FACE_NORMALS = (
    ((1, 0, 0), (1.0, 0.0, 0.0)),
    ((-1, 0, 0), (-1.0, 0.0, 0.0)),
    ((0, 1, 0), (0.0, 1.0, 0.0)),
    ((0, -1, 0), (0.0, -1.0, 0.0)),
    ((0, 0, 1), (0.0, 0.0, 1.0)),
    ((0, 0, -1), (0.0, 0.0, -1.0)),
)


def _face_corners(
    coord: tuple[int, int, int],
    direction: tuple[int, int, int],
    voxel_size_m: float,
) -> list[list[float]]:
    min_corner = np.asarray(coord, dtype=np.float64) * voxel_size_m
    max_corner = min_corner + voxel_size_m
    x0, y0, z0 = min_corner.tolist()
    x1, y1, z1 = max_corner.tolist()
    if direction == (1, 0, 0):
        return [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]
    if direction == (-1, 0, 0):
        return [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]
    if direction == (0, 1, 0):
        return [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]]
    if direction == (0, -1, 0):
        return [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]
    if direction == (0, 0, 1):
        return [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    return [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]]


__all__ = [
    "MESH_INSTALL_HINT",
    "SPARSE_TSDF_FALLBACK_MESHER_BACKEND",
    "SparseTSDFMesherConfig",
    "mesh_sparse_tsdf_block",
    "mesh_sparse_tsdf_snapshot",
    "mesher_backend_status",
    "skimage_meshing_available",
]
