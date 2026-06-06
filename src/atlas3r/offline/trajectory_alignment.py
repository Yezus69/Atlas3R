"""Sim3 alignment between classical SfM and Atlas3R/VGGT trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import COORDINATE_FRAME_NAME
from atlas3r.offline.colmap_import import ColmapSparseModel, write_sparse_points_ply
from atlas3r.offline.run_manifest import write_json


@dataclass(frozen=True)
class TrajectoryAlignmentResult:
    status: str
    reason: str
    common_frame_count: int = 0
    sim3_scale: float | None = None
    rotation_deg: float | None = None
    translation_norm: float | None = None
    camera_center_rmse_m: float | None = None
    camera_center_p50_m: float | None = None
    camera_center_p95_m: float | None = None
    trajectory_length_vggt_m: float | None = None
    trajectory_length_colmap_aligned_m: float | None = None
    registered_frame_ratio: float | None = None
    colmap_sparse_point_count: int = 0
    aligned_sparse_point_count: int = 0
    alignment_json_path: str = "classical/trajectory_alignment.json"
    aligned_trajectory_path: str | None = None
    aligned_points_npz_path: str | None = None
    aligned_points_ply_path: str | None = None
    transform: dict[str, object] | None = None

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        values = (
            self.alignment_json_path,
            self.aligned_trajectory_path,
            self.aligned_points_npz_path,
            self.aligned_points_ply_path,
        )
        return tuple(path for path in values if path is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "common_frame_count": self.common_frame_count,
            "sim3_scale": self.sim3_scale,
            "rotation_deg": self.rotation_deg,
            "translation_norm": self.translation_norm,
            "camera_center_rmse_m": self.camera_center_rmse_m,
            "camera_center_p50_m": self.camera_center_p50_m,
            "camera_center_p95_m": self.camera_center_p95_m,
            "trajectory_length_vggt_m": self.trajectory_length_vggt_m,
            "trajectory_length_colmap_aligned_m": self.trajectory_length_colmap_aligned_m,
            "registered_frame_ratio": self.registered_frame_ratio,
            "colmap_sparse_point_count": self.colmap_sparse_point_count,
            "aligned_sparse_point_count": self.aligned_sparse_point_count,
            "artifacts": {
                "trajectory_alignment": self.alignment_json_path,
                "aligned_colmap_camera_trajectory": self.aligned_trajectory_path,
                "aligned_colmap_sparse_points_npz": self.aligned_points_npz_path,
                "aligned_colmap_sparse_points_ply": self.aligned_points_ply_path,
            },
            "transform_colmap_to_atlas_world": self.transform,
            "coordinate_convention": COORDINATE_FRAME_NAME,
            "truth_boundary": {
                "label_type": "classical_sfm_proposal",
                "measured_geometry": False,
                "observed_only": True,
                "predicted_completion": False,
                "hidden_geometry_measured": False,
                "physical_accuracy_claim": False,
                "scale_status": "sfm_scale_unanchored",
                "metric_scale_source": "colmap_sfm_unanchored_aligned_to_vggt_world",
                "training_quality": False,
            },
        }


def write_trajectory_alignment(
    run_dir: str | Path,
    *,
    model: ColmapSparseModel | None,
    vggt_camera_records: tuple[dict[str, object], ...],
    min_common_frames: int = 3,
) -> TrajectoryAlignmentResult:
    root = Path(run_dir)
    classical_dir = root / "classical"
    classical_dir.mkdir(parents=True, exist_ok=True)
    result = align_colmap_to_vggt(
        model,
        vggt_camera_records=vggt_camera_records,
        min_common_frames=min_common_frames,
    )
    if result.status != "available" or model is None:
        write_json(classical_dir / "trajectory_alignment.json", result.to_dict())
        return result
    transform = _require_transform(result)
    aligned_trajectory = _aligned_trajectory_rows(model, transform)
    aligned_points = apply_sim3(model.points_xyz_array(), transform)
    colors = model.point_colors_array()
    np.savez_compressed(
        classical_dir / "aligned_colmap_sparse_points.npz",
        points_world_m=aligned_points.astype(np.float32),
        colors_u8=colors.astype(np.uint8),
        metadata_json=json.dumps(
            {
                "format_name": "atlas3r_aligned_colmap_sparse_points",
                "format_version": 1,
                "point_count": int(aligned_points.shape[0]),
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "metric_scale_source": "colmap_sfm_unanchored_aligned_to_vggt_world",
            },
            sort_keys=True,
        ),
    )
    write_sparse_points_ply(
        classical_dir / "aligned_colmap_sparse_points.ply",
        aligned_points.astype(np.float32),
        colors.astype(np.uint8),
        metric_scale_source="colmap_sfm_unanchored_aligned_to_vggt_world",
    )
    write_json(
        classical_dir / "aligned_colmap_camera_trajectory.json",
        {
            "format_name": "atlas3r_aligned_colmap_camera_trajectory",
            "format_version": 1,
            "trajectory_count": len(aligned_trajectory),
            "metric_scale_source": "colmap_sfm_unanchored_aligned_to_vggt_world",
            "poses": aligned_trajectory,
        },
    )
    result = TrajectoryAlignmentResult(
        **{
            **result.__dict__,
            "aligned_trajectory_path": "classical/aligned_colmap_camera_trajectory.json",
            "aligned_points_npz_path": "classical/aligned_colmap_sparse_points.npz",
            "aligned_points_ply_path": "classical/aligned_colmap_sparse_points.ply",
            "aligned_sparse_point_count": int(aligned_points.shape[0]),
        }
    )
    write_json(classical_dir / "trajectory_alignment.json", result.to_dict())
    return result


def align_colmap_to_vggt(
    model: ColmapSparseModel | None,
    *,
    vggt_camera_records: tuple[dict[str, object], ...],
    min_common_frames: int = 3,
) -> TrajectoryAlignmentResult:
    if min_common_frames < 3:
        raise ValueError("min_common_frames must be at least 3")
    if model is None:
        return TrajectoryAlignmentResult(status="unavailable", reason="no classical sparse model")
    vggt_by_frame = _vggt_centers_by_frame(vggt_camera_records)
    source_centers = []
    target_centers = []
    frame_ids = []
    for image in model.images:
        frame_id = image.frame_id
        if frame_id is None or frame_id not in vggt_by_frame:
            continue
        source_centers.append(image.camera_center_world_m.astype(np.float32))
        target_centers.append(vggt_by_frame[frame_id])
        frame_ids.append(frame_id)
    common = len(frame_ids)
    if common < min_common_frames:
        return TrajectoryAlignmentResult(
            status="unavailable",
            reason=f"need at least {min_common_frames} common frames, found {common}",
            common_frame_count=common,
            registered_frame_ratio=_registered_ratio(common, model.registered_image_count),
            colmap_sparse_point_count=model.sparse_point_count,
        )
    source = np.stack(source_centers).astype(np.float32)
    target = np.stack(target_centers).astype(np.float32)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("common camera centers must be finite")
    transform = estimate_sim3(source, target)
    aligned = apply_sim3(source, transform)
    residuals = np.linalg.norm(aligned - target, axis=1).astype(np.float32)
    scale = _transform_scale(transform)
    rotation = _transform_rotation(transform)
    translation = _transform_translation(transform)
    return TrajectoryAlignmentResult(
        status="available",
        reason="aligned classical camera centers to VGGT camera centers",
        common_frame_count=common,
        sim3_scale=scale,
        rotation_deg=_rotation_degrees(rotation),
        translation_norm=float(np.linalg.norm(translation)),
        camera_center_rmse_m=_rmse(residuals),
        camera_center_p50_m=_percentile(residuals, 50.0),
        camera_center_p95_m=_percentile(residuals, 95.0),
        trajectory_length_vggt_m=_trajectory_length(target),
        trajectory_length_colmap_aligned_m=_trajectory_length(aligned),
        registered_frame_ratio=_registered_ratio(common, model.registered_image_count),
        colmap_sparse_point_count=model.sparse_point_count,
        aligned_sparse_point_count=0,
        transform={
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "common_frame_ids": frame_ids,
        },
    )


def estimate_sim3(
    source_points: NDArray[np.float32], target_points: NDArray[np.float32]
) -> dict[str, object]:
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be shaped N,3")
    if source.shape[0] < 3:
        raise ValueError("at least three points are required for Sim3 alignment")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Sim3 input points must be finite")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_var <= 1e-12:
        raise ValueError("source points are degenerate")
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float32)
    if float(np.linalg.det(u @ vt)) < 0.0:
        correction[2, 2] = -1.0
    rotation = (u @ correction @ vt).astype(np.float32)
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_var)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("estimated Sim3 scale is invalid")
    translation = (target_mean - scale * (rotation @ source_mean)).astype(np.float32)
    return {"scale": scale, "rotation": rotation, "translation": translation}


def apply_sim3(points: NDArray[np.float32], transform: dict[str, object]) -> NDArray[np.float32]:
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("points must be shaped N,3")
    if not np.all(np.isfinite(source)):
        raise ValueError("points must be finite")
    scale = _transform_scale(transform)
    rotation = _transform_rotation(transform)
    translation = _transform_translation(transform)
    return (scale * (source @ rotation.T) + translation).astype(np.float32)


def transform_T_world_camera(
    T_world_camera: NDArray[np.float32], transform: dict[str, object]
) -> NDArray[np.float32]:
    T = np.asarray(T_world_camera, dtype=np.float32)
    rotation = _transform_rotation(transform)
    scale = _transform_scale(transform)
    translation = _transform_translation(transform)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = (rotation @ T[:3, :3]).astype(np.float32)
    result[:3, 3] = (scale * (rotation @ T[:3, 3]) + translation).astype(np.float32)
    return result


def _aligned_trajectory_rows(
    model: ColmapSparseModel, transform: dict[str, object]
) -> list[dict[str, object]]:
    rows = []
    for image in model.images:
        T_world_camera = transform_T_world_camera(image.T_world_camera, transform)
        rows.append(
            {
                "frame_id": image.frame_id,
                "image_name": image.name,
                "image_id": image.image_id,
                "T_world_camera": T_world_camera.tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].astype(np.float32).tolist(),
                "pose_source": "colmap_sfm_aligned_to_vggt",
                "metric_scale_source": "colmap_sfm_unanchored_aligned_to_vggt_world",
                "coordinate_convention": COORDINATE_FRAME_NAME,
                "pose_confidence": 0.5,
            }
        )
    return rows


def _vggt_centers_by_frame(
    vggt_camera_records: tuple[dict[str, object], ...],
) -> dict[int, NDArray[np.float32]]:
    rows = {}
    for record in vggt_camera_records:
        if "frame_id" not in record or "camera_center_world_m" not in record:
            continue
        center = np.asarray(record["camera_center_world_m"], dtype=np.float32)
        if center.shape == (3,) and np.all(np.isfinite(center)):
            rows[int(str(record["frame_id"]))] = center
    return rows


def _require_transform(result: TrajectoryAlignmentResult) -> dict[str, object]:
    if result.transform is None:
        raise ValueError("alignment result has no transform")
    return result.transform


def _rotation_degrees(rotation: NDArray[np.float32]) -> float:
    trace = float(np.trace(rotation))
    value = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(value)))


def _transform_scale(transform: dict[str, object]) -> float:
    value = transform["scale"]
    if not isinstance(value, int | float):
        raise ValueError("Sim3 transform scale must be numeric")
    return float(value)


def _transform_rotation(transform: dict[str, object]) -> NDArray[np.float32]:
    rotation = np.asarray(transform["rotation"], dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError("Sim3 transform rotation must be 3x3")
    return rotation


def _transform_translation(transform: dict[str, object]) -> NDArray[np.float32]:
    translation = np.asarray(transform["translation"], dtype=np.float32).reshape((3,))
    return translation


def _rmse(values: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _percentile(values: NDArray[np.float32], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _trajectory_length(points: NDArray[np.float32]) -> float:
    if points.shape[0] < 2:
        return 0.0
    deltas = points[1:] - points[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def _registered_ratio(common_count: int, registered_count: int) -> float:
    return float(common_count / registered_count) if registered_count else 0.0
