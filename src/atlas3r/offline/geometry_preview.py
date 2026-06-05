"""Geometry preview and pixel/depth lifting for offline traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts import COORDINATE_FRAME_NAME, MapArtifact, TruthBoundary
from atlas3r.contracts.coordinates import transform_points, unproject_depth
from atlas3r.mapping import read_npz_artifact_metadata, write_mesh_ply
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint


@dataclass(frozen=True)
class GeometryPreviewResult:
    status: str
    geometry_npz_path: str
    geometry_ply_path: str | None
    point_count: int
    observed_only: bool
    predicted_completion: bool
    measured_geometry: bool
    metric_scale_source: str


def lift_depth_to_world_points(
    K: NDArray[np.float32],
    depth_m: NDArray[np.float32],
    T_world_camera: NDArray[np.float32],
) -> NDArray[np.float32]:
    points_camera = unproject_depth(K, depth_m).reshape((-1, 3))
    return transform_points(T_world_camera, points_camera)


def write_geometry_preview(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    proposal_cache: ProposalCacheResult,
    write_ply: bool,
    failure_points: list[FailurePoint],
) -> GeometryPreviewResult:
    root = Path(run_dir)
    npz_path = root / "geometry" / "geometry_preview.npz"
    if not proposal_cache.debug_depth_records:
        metadata = {
            "status": "unavailable",
            "why": "no usable depth, intrinsics, and T_world_camera proposal exists",
            "observed_only": True,
            "predicted_completion": False,
            "measured_geometry": False,
            "metric_scale_source": "unknown",
        }
        _write_geometry_npz(
            npz_path, _empty_points(), _empty_colors(), _empty_float(), (), metadata
        )
        failure_points.append(
            FailurePoint(
                module="geometry_lifter",
                code="geometry_inputs_missing",
                severity="warning",
                status="unavailable",
                why=str(metadata["why"]),
                input_missing="depth + K + T_world_camera",
                future_module="real teacher geometry proposal path",
                artifact_path="geometry/geometry_preview.npz",
            )
        )
        return GeometryPreviewResult(
            status="unavailable",
            geometry_npz_path="geometry/geometry_preview.npz",
            geometry_ply_path=None,
            point_count=0,
            observed_only=True,
            predicted_completion=False,
            measured_geometry=False,
            metric_scale_source="unknown",
        )
    frames_by_id = {frame.frame_id: frame for frame in frame_cache.frames}
    point_chunks: list[NDArray[np.float32]] = []
    color_chunks: list[NDArray[np.uint8]] = []
    uncertainty_chunks: list[NDArray[np.float32]] = []
    frame_id_chunks: list[NDArray[np.int32]] = []
    measured_geometry = False
    metric_scale_source = "debug_flat_depth"
    for proposal in proposal_cache.debug_depth_records:
        frame_id = int(str(proposal["frame_id"]))
        frame = frames_by_id[frame_id]
        depth = np.full(
            (frame.height, frame.width), float(str(proposal["constant_depth_m"])), np.float32
        )
        T_world_camera = np.asarray(proposal["T_world_camera"], dtype=np.float32)
        points_world = lift_depth_to_world_points(frame.K_model, depth, T_world_camera)
        mask = _sample_mask(frame.height, frame.width).reshape((-1,))
        point_chunks.append(points_world[mask])
        color_chunks.append(frame.rgb_u8.reshape((-1, 3))[mask])
        uncertainty_chunks.append(
            np.full(int(mask.sum()), float(str(proposal["depth_sigma_m"])), np.float32)
        )
        frame_id_chunks.append(np.full(int(mask.sum()), frame_id, np.int32))
        measured_geometry = measured_geometry or bool(proposal["measured_geometry"])
        metric_scale_source = str(proposal["metric_scale_source"])
    points = np.concatenate(point_chunks, axis=0) if point_chunks else _empty_points()
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else _empty_colors()
    uncertainty = (
        np.concatenate(uncertainty_chunks, axis=0) if uncertainty_chunks else _empty_float()
    )
    frame_ids = (
        np.concatenate(frame_id_chunks, axis=0) if frame_id_chunks else np.zeros((0,), np.int32)
    )
    metadata = {
        "status": "partial",
        "why": "debug-only flat-depth geometry preview",
        "observed_only": True,
        "predicted_completion": False,
        "measured_geometry": measured_geometry,
        "metric_scale_source": metric_scale_source,
        "coordinate_frame": COORDINATE_FRAME_NAME,
        "point_count": int(points.shape[0]),
    }
    _write_geometry_npz(npz_path, points, colors, uncertainty, tuple(frame_ids.tolist()), metadata)
    ply_rel: str | None = None
    if write_ply and points.shape[0] > 0:
        ply_path = root / "geometry" / "geometry_preview.ply"
        truth = TruthBoundary(
            label_type="synthetic_gt" if measured_geometry else "debug_synthetic",
            metric_scale_source=metric_scale_source,
            measured_geometry=measured_geometry,
            observed_only=True,
            notes="Debug flat-depth preview; not teacher-measured geometry.",
        )
        artifact = MapArtifact(
            artifact_type="mesh",
            path="geometry/geometry_preview.ply",
            coordinate_frame=COORDINATE_FRAME_NAME,
            source_frame_ids=tuple(sorted(set(int(item) for item in frame_ids.tolist()))),
            voxel_size_m=None,
            observed_coverage_estimate=0.0,
            mean_uncertainty_m=float(uncertainty.mean()) if uncertainty.size else 0.0,
            p95_uncertainty_m=float(np.percentile(uncertainty, 95)) if uncertainty.size else 0.0,
            truth_boundary=truth,
            metadata={"geometry_kind": "point_preview", "faces": 0},
        )
        write_mesh_ply(
            ply_path,
            vertices_world_m=points.astype(np.float32),
            triangles=np.zeros((0, 3), dtype=np.uint32),
            artifact=artifact,
        )
        ply_rel = "geometry/geometry_preview.ply"
    return GeometryPreviewResult(
        status="partial",
        geometry_npz_path="geometry/geometry_preview.npz",
        geometry_ply_path=ply_rel,
        point_count=int(points.shape[0]),
        observed_only=True,
        predicted_completion=False,
        measured_geometry=measured_geometry,
        metric_scale_source=metric_scale_source,
    )


def read_geometry_metadata(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("geometry metadata must be a JSON object")
    return metadata


def read_geometry_artifact_metadata(path: str | Path) -> dict[str, object]:
    metadata = read_npz_artifact_metadata(path)
    return dict(metadata)


def _write_geometry_npz(
    path: Path,
    points_world_m: NDArray[np.float32],
    colors_u8: NDArray[np.uint8],
    uncertainty_m: NDArray[np.float32],
    frame_ids: tuple[int, ...],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_world_m=points_world_m.astype(np.float32),
        colors_u8=colors_u8.astype(np.uint8),
        uncertainty_m=uncertainty_m.astype(np.float32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _sample_mask(height: int, width: int) -> NDArray[np.bool_]:
    stride = max(1, min(height, width) // 24)
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[::stride, ::stride] = True
    return mask


def _empty_points() -> NDArray[np.float32]:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_colors() -> NDArray[np.uint8]:
    return np.zeros((0, 3), dtype=np.uint8)


def _empty_float() -> NDArray[np.float32]:
    return np.zeros((0,), dtype=np.float32)
