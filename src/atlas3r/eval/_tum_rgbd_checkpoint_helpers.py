"""Private helpers for TUM RGB-D checkpoint evaluation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import TSDFSurface, extract_tsdf_surface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.models.student.contracts import CAMERA_COORDINATE_FRAME
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.training.checkpoint_inference import TinyDepthPoseCheckpoint

POINT_THRESHOLDS_M = (
    ("1mm", 0.001),
    ("5mm", 0.005),
    ("1cm", 0.01),
    ("5cm", 0.05),
)


def predicted_observation(
    *,
    frame_id: int,
    timestamp_s: float,
    K: npt.NDArray[np.float32],
    rgb_u8: npt.NDArray[np.uint8],
    T_world_camera: npt.NDArray[np.float32],
    depth_m: npt.NDArray[np.float32],
    sigma_m: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    checkpoint: TinyDepthPoseCheckpoint,
) -> DepthObservation:
    mean_confidence = float(np.mean(confidence.astype(np.float64, copy=False)))
    return DepthObservation(
        frame_id=frame_id,
        camera=_camera(K, width=depth_m.shape[1], height=depth_m.shape[0], source="manifest"),
        pose=_pose(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            T_world_camera=T_world_camera,
            confidence=mean_confidence,
            scale_source="rgb_prior",
            diagnostics={
                "source": "tum_rgbd_checkpoint_eval_prediction",
                "coordinate_frame": CAMERA_COORDINATE_FRAME,
                "truth_boundary": dict(checkpoint.truth_boundary),
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_step": checkpoint.step,
                "pose_rotation_source": "TUM_RGBD_groundtruth_rotation_for_diagnostic",
                "pose_translation_source": "checkpoint_camera_center_world_m",
            },
        ),
        depth_m=depth_m.astype(np.float32, copy=True),
        depth_sigma_m=sigma_m.astype(np.float32, copy=True),
        confidence=confidence.astype(np.float32, copy=True),
        static_mask=np.ones(depth_m.shape, dtype=np.bool_),
        rgb_u8=rgb_u8.copy(),
        source="tum_rgbd_checkpoint_eval_prediction",
    )


def target_observation(
    *,
    frame_id: int,
    timestamp_s: float,
    K: npt.NDArray[np.float32],
    rgb_u8: npt.NDArray[np.uint8],
    T_world_camera: npt.NDArray[np.float32],
    depth_m: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
) -> DepthObservation:
    confidence = valid_mask.astype(np.float32)
    sigma = np.where(valid_mask, np.float32(0.01), np.float32(0.0)).astype(np.float32)
    return DepthObservation(
        frame_id=frame_id,
        camera=_camera(K, width=depth_m.shape[1], height=depth_m.shape[0], source="manifest"),
        pose=_pose(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            T_world_camera=T_world_camera,
            confidence=1.0,
            scale_source="external_pose",
            diagnostics={
                "source": "tum_rgbd_groundtruth_depth_pose",
                "coordinate_frame": CAMERA_COORDINATE_FRAME,
                "metric_scale_source": "TUM RGB-D depth and groundtruth.txt",
                "accuracy_report": False,
            },
        ),
        depth_m=depth_m.astype(np.float32, copy=True),
        depth_sigma_m=sigma,
        confidence=confidence,
        static_mask=valid_mask.copy(),
        rgb_u8=rgb_u8.copy(),
        source="tum_rgbd_groundtruth_depth_pose",
    )


def write_mapping_diagnostics(
    *,
    output: Path,
    checkpoint: TinyDepthPoseCheckpoint,
    split: str,
    voxel_size_m: float,
    truncation_voxels: float,
    map_max_points: int,
    predicted_observations: tuple[DepthObservation, ...],
    target_observations: tuple[DepthObservation, ...],
) -> dict[str, Any]:
    truncation_distance_m = voxel_size_m * truncation_voxels
    grid_min, grid_max = grid_bounds_from_observations(
        (*predicted_observations, *target_observations),
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_volume = integrate_observations_to_volume(
        predicted_observations,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    target_volume = integrate_observations_to_volume(
        target_observations,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_surface = _eval_surface(
        extract_tsdf_surface(predicted_volume),
        checkpoint=checkpoint,
        observations=predicted_observations,
        artifact_type="phase_4d_tum_rgbd_checkpoint_predicted_cpu_tsdf_surface_points",
    )
    target_surface = _eval_surface(
        extract_tsdf_surface(target_volume),
        checkpoint=checkpoint,
        observations=target_observations,
        artifact_type="phase_4d_tum_rgbd_target_cpu_tsdf_surface_points",
    )
    comparison = _point_set_comparison(
        predicted_surface.points_world_m,
        target_surface.points_world_m,
        max_points=map_max_points,
    )
    record: dict[str, Any] = {
        "format_name": "atlas3r_tum_rgbd_checkpoint_map_comparison",
        "format_version": 1,
        "metric_family": "real_rgbd_debug_eval",
        "diagnostic_only": True,
        "accuracy_report": False,
        "performance_report": False,
        "checkpoint_path": str(checkpoint.path),
        "split": split,
        "voxel_size_m": voxel_size_m,
        "truncation_distance_m": truncation_distance_m,
        "pose_rotation_source": "TUM RGB-D ground-truth rotation for diagnostic only",
        "pose_translation_source": "checkpoint-predicted camera center",
        "predicted_point_count": int(predicted_surface.points_world_m.shape[0]),
        "target_point_count": int(target_surface.points_world_m.shape[0]),
        "point_set_metrics": comparison,
        "known_limitations": [
            "This is a CPU TSDF and point-set diagnostic, not a final mesh accuracy report.",
            "GT rotation is used because the tiny checkpoint predicts camera center only.",
            "Threshold scores are sensitive to voxel size and sparse surface extraction.",
        ],
    }
    write_tsdf_outputs(
        output / "predicted_tsdf",
        predicted_volume,
        predicted_surface,
        {"metric_family": "real_rgbd_debug_eval", "diagnostic_only": True},
    )
    write_tsdf_outputs(
        output / "target_tsdf",
        target_volume,
        target_surface,
        {"metric_family": "real_rgbd_debug_eval", "diagnostic_only": True},
    )
    _write_json(output / "map_comparison.json", record)
    return record


def write_tum_trajectory(
    path: Path,
    rows: list[tuple[float, npt.NDArray[np.float32]]],
) -> None:
    lines = []
    for timestamp_s, transform in rows:
        qx, qy, qz, qw = quaternion_xyzw_from_rotation_matrix(transform[:3, :3])
        tx, ty, tz = transform[:3, 3]
        lines.append(
            f"{timestamp_s:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _camera(
    K: npt.NDArray[np.float32],
    *,
    width: int,
    height: int,
    source: str,
) -> CameraModel:
    return CameraModel(
        width=width,
        height=height,
        K=K.astype(np.float32, copy=True),
        distortion_model="none",
        distortion_params=None,
        rolling_shutter_row_time_s=None,
        confidence=1.0,
        source=source,
    )


def _pose(
    *,
    frame_id: int,
    timestamp_s: float,
    T_world_camera: npt.NDArray[np.float32],
    confidence: float,
    scale_source: str,
    diagnostics: dict[str, Any],
) -> PoseEstimate:
    transform = T_world_camera.astype(np.float32, copy=True)
    return PoseEstimate(
        frame_id=frame_id,
        timestamp_ns=int(round(timestamp_s * 1_000_000_000.0)),
        T_world_camera=transform,
        q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(transform[:3, :3]).astype(
            np.float32
        ),
        camera_center_world_m=transform[:3, 3].astype(np.float32, copy=True),
        covariance_6x6=np.diag(np.full(6, 1e-4, dtype=np.float32)),
        confidence=confidence,
        tracking_state="OK" if confidence >= 0.25 else "LOW_CONFIDENCE",
        scale_source=scale_source,
        diagnostics=diagnostics,
    )


def _eval_surface(
    surface: TSDFSurface,
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    observations: tuple[DepthObservation, ...],
    artifact_type: str,
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            "artifact_type": artifact_type,
            "metric_family": "real_rgbd_debug_eval",
            "diagnostic_only": True,
            "accuracy_report": False,
            "performance_report": False,
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_step": checkpoint.step,
            "truth_boundary": dict(checkpoint.truth_boundary),
            "observation_sources": _stable_strings(
                [observation.source for observation in observations]
            ),
        }
    )
    metadata["flags"] = _stable_strings(
        [*cast(list[str], metadata.get("flags", [])), "phase_4d_diagnostic", "not_accuracy_report"]
    )
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def _point_set_comparison(
    predicted_points: npt.NDArray[np.float32],
    target_points: npt.NDArray[np.float32],
    *,
    max_points: int,
) -> dict[str, Any]:
    predicted = _capped_points(predicted_points, max_points)
    target = _capped_points(target_points, max_points)
    pred_to_target = _nearest_distances(predicted, target)
    target_to_pred = _nearest_distances(target, predicted)
    thresholds: dict[str, dict[str, float]] = {}
    for label, threshold_m in POINT_THRESHOLDS_M:
        precision = float(np.mean(pred_to_target <= threshold_m))
        recall = float(np.mean(target_to_pred <= threshold_m))
        f_score = (
            0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        )
        thresholds[label] = {
            "precision_like": precision,
            "recall_like": recall,
            "f_score_like": float(f_score),
        }
    return {
        "sampled_predicted_point_count": int(predicted.shape[0]),
        "sampled_target_point_count": int(target.shape[0]),
        "predicted_to_target_mean_m": float(np.mean(pred_to_target)),
        "predicted_to_target_median_m": float(np.median(pred_to_target)),
        "predicted_to_target_p95_m": float(np.percentile(pred_to_target, 95.0)),
        "target_to_predicted_mean_m": float(np.mean(target_to_pred)),
        "target_to_predicted_median_m": float(np.median(target_to_pred)),
        "target_to_predicted_p95_m": float(np.percentile(target_to_pred, 95.0)),
        "chamfer_like_mean_m": float((np.mean(pred_to_target) + np.mean(target_to_pred)) * 0.5),
        "thresholds": thresholds,
    }


def _nearest_distances(
    source: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("point-set comparison requires non-empty point arrays")
    distances = np.empty(source.shape[0], dtype=np.float64)
    target64 = target.astype(np.float64, copy=False)
    for start in range(0, source.shape[0], 256):
        stop = min(start + 256, source.shape[0])
        chunk = source[start:stop].astype(np.float64, copy=False)
        squared = np.sum((chunk[:, np.newaxis, :] - target64[np.newaxis, :, :]) ** 2, axis=2)
        distances[start:stop] = np.sqrt(np.min(squared, axis=1))
    return distances


def _capped_points(
    points: npt.NDArray[np.float32],
    max_points: int,
) -> npt.NDArray[np.float32]:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _stable_strings(values: list[str]) -> list[str]:
    stable: list[str] = []
    for value in values:
        if value not in stable:
            stable.append(value)
    return stable


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "predicted_observation",
    "target_observation",
    "write_mapping_diagnostics",
    "write_tum_trajectory",
]
