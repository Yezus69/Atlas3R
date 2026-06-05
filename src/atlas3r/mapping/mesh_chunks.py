"""Observed-only sparse mesh chunk contracts and payload helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import FLOAT32

MESH_CHUNK_FORMAT_NAME = "atlas3r_observed_mesh_chunk"
MESH_CHUNK_MANIFEST_FORMAT_NAME = "atlas3r_observed_mesh_chunk_manifest"
MESH_CHUNK_FORMAT_VERSION = 1
MESH_CHUNK_UPDATE_TYPES = {"upsert", "delete"}
MESH_TRUTH_FLAGS = {
    "accuracy_report": False,
    "hidden_geometry_measured": False,
    "observed_only": True,
    "predicted_completion": False,
    "realtime_claim": False,
    "rgb_only_mapping_ready": False,
    "student_rgb_only_used": False,
}


@dataclass
class ObservedMeshChunk:
    """Loadable observed-only triangle mesh chunk payload."""

    chunk_id: str
    chunk_coord_xyz: tuple[int, int, int]
    version: int
    update_type: str
    coordinate_frame: str
    T_world_chunk: npt.NDArray[np.float32]
    voxel_size_m: float
    truncation_distance_m: float
    block_size_voxels: int
    vertices_world_m: npt.NDArray[np.float32]
    normals_world: npt.NDArray[np.float32]
    triangles: npt.NDArray[np.uint32]
    source_frame_ids: list[int]
    active_voxel_count: int
    surface_voxel_count: int
    uncertainty_summary_m: dict[str, float | None]
    confidence_summary: dict[str, float | None]
    observed_coverage_estimate: float
    metric_scale_source: str
    mesher_backend: str
    truth_flags: dict[str, object] = field(default_factory=lambda: dict(MESH_TRUTH_FLAGS))

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id: must be non-empty")
        self.chunk_coord_xyz = _normalize_block_coord(self.chunk_coord_xyz)
        if self.version <= 0:
            raise ValueError("version: must be positive")
        if self.update_type not in MESH_CHUNK_UPDATE_TYPES:
            raise ValueError("update_type: must be upsert or delete")
        if not self.coordinate_frame:
            raise ValueError("coordinate_frame: must be non-empty")
        self.T_world_chunk = _validate_transform(self.T_world_chunk)
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_distance_m <= 0.0:
            raise ValueError("truncation_distance_m: must be positive")
        if self.block_size_voxels <= 0:
            raise ValueError("block_size_voxels: must be positive")
        self.vertices_world_m = _validate_vertices(self.vertices_world_m)
        self.normals_world = _validate_normals(self.normals_world, self.vertices_world_m.shape[0])
        self.triangles = _validate_triangles(self.triangles, self.vertices_world_m.shape[0])
        if self.source_frame_ids:
            self.source_frame_ids = [int(frame_id) for frame_id in self.source_frame_ids]
        if self.active_voxel_count < 0:
            raise ValueError("active_voxel_count: must be non-negative")
        if self.surface_voxel_count < 0:
            raise ValueError("surface_voxel_count: must be non-negative")
        if not 0.0 <= self.observed_coverage_estimate <= 1.0:
            raise ValueError("observed_coverage_estimate: must be in [0, 1]")
        if not self.metric_scale_source:
            raise ValueError("metric_scale_source: must be non-empty")
        if not self.mesher_backend:
            raise ValueError("mesher_backend: must be non-empty")
        self.truth_flags = dict(self.truth_flags)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_world_m.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def bbox_world_min_m(self) -> list[float]:
        if self.vertex_count == 0:
            return [0.0, 0.0, 0.0]
        return _float_list(np.min(self.vertices_world_m, axis=0))

    @property
    def bbox_world_max_m(self) -> list[float]:
        if self.vertex_count == 0:
            return [0.0, 0.0, 0.0]
        return _float_list(np.max(self.vertices_world_m, axis=0))

    def metadata_record(self) -> dict[str, object]:
        first_frame = min(self.source_frame_ids) if self.source_frame_ids else None
        last_frame = max(self.source_frame_ids) if self.source_frame_ids else None
        return {
            **self.truth_flags,
            "T_world_chunk": _json_array(self.T_world_chunk),
            "active_voxel_count": int(self.active_voxel_count),
            "bbox_world_max_m": self.bbox_world_max_m,
            "bbox_world_min_m": self.bbox_world_min_m,
            "block_size_voxels": int(self.block_size_voxels),
            "chunk_coord_xyz": list(self.chunk_coord_xyz),
            "chunk_id": self.chunk_id,
            "confidence_summary": self.confidence_summary,
            "coordinate_frame": self.coordinate_frame,
            "first_source_frame_id": first_frame,
            "format_name": MESH_CHUNK_FORMAT_NAME,
            "format_version": MESH_CHUNK_FORMAT_VERSION,
            "last_source_frame_id": last_frame,
            "mesher_backend": self.mesher_backend,
            "metric_scale_source": self.metric_scale_source,
            "observed_coverage_estimate": float(self.observed_coverage_estimate),
            "source_frame_ids": list(self.source_frame_ids),
            "surface_voxel_count": int(self.surface_voxel_count),
            "triangle_count": self.triangle_count,
            "truncation_distance_m": float(self.truncation_distance_m),
            "uncertainty_summary_m": self.uncertainty_summary_m,
            "update_type": self.update_type,
            "version": int(self.version),
            "vertex_count": self.vertex_count,
            "voxel_size_m": float(self.voxel_size_m),
        }


def chunk_id_from_block_coord(block_coord_xyz: tuple[int, int, int]) -> str:
    x, y, z = _normalize_block_coord(block_coord_xyz)
    return f"block_{x}_{y}_{z}"


def save_mesh_chunk_npz(path: str | Path, chunk: ObservedMeshChunk) -> Path:
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(chunk.metadata_record(), sort_keys=True)
    np.savez(
        payload_path,
        metadata_json=np.asarray(metadata_json),
        vertices_world_m=chunk.vertices_world_m.astype(np.float32, copy=False),
        normals_world=chunk.normals_world.astype(np.float32, copy=False),
        triangles=chunk.triangles.astype(np.uint32, copy=False),
        chunk_coord_xyz=np.asarray(chunk.chunk_coord_xyz, dtype=np.int32),
        source_frame_ids=np.asarray(chunk.source_frame_ids, dtype=np.int64),
    )
    return payload_path


def load_mesh_chunk_npz(path: str | Path) -> ObservedMeshChunk:
    payload_path = Path(path)
    try:
        with np.load(payload_path) as arrays:
            metadata_raw = arrays["metadata_json"]
            metadata = json.loads(str(metadata_raw.item()))
            vertices = np.asarray(arrays["vertices_world_m"], dtype=np.float32)
            normals = np.asarray(arrays["normals_world"], dtype=np.float32)
            triangles = np.asarray(arrays["triangles"], dtype=np.uint32)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{payload_path}: invalid mesh chunk NPZ: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{payload_path}: metadata_json must decode to an object")
    chunk_coord = cast(list[int], metadata["chunk_coord_xyz"])
    source_frame_ids = cast(list[int], metadata["source_frame_ids"])
    return ObservedMeshChunk(
        chunk_id=str(metadata["chunk_id"]),
        chunk_coord_xyz=(int(chunk_coord[0]), int(chunk_coord[1]), int(chunk_coord[2])),
        version=int(metadata["version"]),
        update_type=str(metadata["update_type"]),
        coordinate_frame=str(metadata["coordinate_frame"]),
        T_world_chunk=np.asarray(metadata["T_world_chunk"], dtype=np.float32),
        voxel_size_m=float(metadata["voxel_size_m"]),
        truncation_distance_m=float(metadata["truncation_distance_m"]),
        block_size_voxels=int(metadata["block_size_voxels"]),
        vertices_world_m=vertices,
        normals_world=normals,
        triangles=triangles,
        source_frame_ids=[int(item) for item in source_frame_ids],
        active_voxel_count=int(cast(Any, metadata["active_voxel_count"])),
        surface_voxel_count=int(cast(Any, metadata["surface_voxel_count"])),
        uncertainty_summary_m=cast(dict[str, float | None], metadata["uncertainty_summary_m"]),
        confidence_summary=cast(dict[str, float | None], metadata["confidence_summary"]),
        observed_coverage_estimate=float(metadata["observed_coverage_estimate"]),
        metric_scale_source=str(metadata["metric_scale_source"]),
        mesher_backend=str(metadata["mesher_backend"]),
    )


def write_mesh_chunk_ply(path: str | Path, chunk: ObservedMeshChunk) -> Path:
    ply_path = Path(path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment Atlas3R observed-only mesh chunk\n")
        handle.write(f"comment chunk_id {chunk.chunk_id}\n")
        handle.write("comment observed_only true\n")
        handle.write(f"element vertex {chunk.vertex_count}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float nx\n")
        handle.write("property float ny\n")
        handle.write("property float nz\n")
        handle.write(f"element face {chunk.triangle_count}\n")
        handle.write("property list uchar uint vertex_indices\n")
        handle.write("end_header\n")
        for vertex, normal in zip(chunk.vertices_world_m, chunk.normals_world, strict=True):
            handle.write(
                f"{float(vertex[0]):.7g} {float(vertex[1]):.7g} {float(vertex[2]):.7g} "
                f"{float(normal[0]):.7g} {float(normal[1]):.7g} {float(normal[2]):.7g}\n"
            )
        for tri in chunk.triangles:
            handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
    return ply_path


def numeric_summary(values: npt.NDArray[np.floating[Any]]) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    finite = values.astype(np.float64, copy=False)
    return {
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _validate_transform(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    transform = np.asarray(matrix, dtype=np.float32)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_world_chunk: must be finite 4x4")
    np.testing.assert_allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    return transform.astype(FLOAT32, copy=False)


def _validate_vertices(vertices: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    array = np.asarray(vertices, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("vertices_world_m: must be finite Nx3")
    return array.astype(FLOAT32, copy=False)


def _validate_normals(
    normals: npt.NDArray[np.float32],
    vertex_count: int,
) -> npt.NDArray[np.float32]:
    array = np.asarray(normals, dtype=np.float32)
    if array.shape != (vertex_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError("normals_world: must be finite Nx3 matching vertices")
    if vertex_count > 0:
        norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("normals_world: normals must be non-zero")
    return array.astype(FLOAT32, copy=False)


def _validate_triangles(
    triangles: npt.NDArray[np.uint32],
    vertex_count: int,
) -> npt.NDArray[np.uint32]:
    array = np.asarray(triangles, dtype=np.uint32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("triangles: must be Mx3")
    if array.size > 0 and int(np.max(array)) >= vertex_count:
        raise ValueError("triangles: indices must be in range")
    return array.astype(np.uint32, copy=False)


def _normalize_block_coord(block_coord_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(block_coord_xyz) != 3:
        raise ValueError("chunk_coord_xyz: must contain exactly three values")
    return (int(block_coord_xyz[0]), int(block_coord_xyz[1]), int(block_coord_xyz[2]))


def _json_array(array: npt.NDArray[Any]) -> list[Any]:
    return cast(list[Any], array.tolist())


def _float_list(array: npt.NDArray[np.float32]) -> list[float]:
    return [float(item) for item in array.tolist()]


__all__ = [
    "MESH_CHUNK_FORMAT_NAME",
    "MESH_CHUNK_FORMAT_VERSION",
    "MESH_CHUNK_MANIFEST_FORMAT_NAME",
    "MESH_TRUTH_FLAGS",
    "ObservedMeshChunk",
    "chunk_id_from_block_coord",
    "load_mesh_chunk_npz",
    "numeric_summary",
    "save_mesh_chunk_npz",
    "write_mesh_chunk_ply",
]
