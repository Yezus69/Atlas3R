"""Private helpers for checkpoint-to-TSDF smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.mapping.cpu_tsdf import (
    FLOAT32,
    FLOAT64,
    TSDFSurface,
    TSDFVolume,
    integrate_depth_observation,
)
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.tsdf_grid import compute_tsdf_grid_shape, voxel_centers_world
from atlas3r.models.student.contracts import CAMERA_COORDINATE_FRAME
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix, transform_points
from atlas3r.training.checkpoint_inference import (
    TinyDepthPoseCheckpoint,
    TinyDepthPosePrediction,
)
from atlas3r.training.synthetic_depth_dataset import SyntheticDepthSample


def target_observation_from_sample(sample: SyntheticDepthSample) -> DepthObservation:
    height, width = sample.depth_m.shape
    T_world_camera = sample.T_world_camera.astype(np.float32, copy=True)
    return DepthObservation(
        frame_id=sample.frame_id,
        camera=CameraModel(
            width=width,
            height=height,
            K=sample.K.astype(np.float32, copy=True),
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="procedural_synthetic_depth",
        ),
        pose=PoseEstimate(
            frame_id=sample.frame_id,
            timestamp_ns=0,
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=sample.camera_center_world_m.astype(np.float32, copy=True),
            covariance_6x6=_target_pose_covariance(sample),
            confidence=1.0,
            tracking_state="OK",
            scale_source="known_anchor",
            diagnostics={
                "source": "procedural_synthetic_depth",
                "coordinate_frame": CAMERA_COORDINATE_FRAME,
                "metric_scale_source": "known_anchor",
                "truth_boundary": {
                    "synthetic_only": True,
                    "learned_inference": False,
                    "accuracy_report": False,
                    "performance_report": False,
                },
            },
        ),
        depth_m=sample.depth_m.astype(np.float32, copy=True),
        depth_sigma_m=sample.depth_sigma_m.astype(np.float32, copy=True),
        confidence=sample.confidence.astype(np.float32, copy=True),
        static_mask=np.ones(sample.depth_m.shape, dtype=np.bool_),
        object_id=np.where(sample.object_mask, 1, -1).astype(np.int32),
        rgb_u8=sample.rgb_u8.copy(),
        source="procedural_synthetic_depth_target",
    )


def integrate_observations_to_volume(
    observations: tuple[DepthObservation, ...],
    *,
    grid_min_world_m: npt.NDArray[np.float64],
    grid_max_world_m: npt.NDArray[np.float64],
    voxel_size_m: float,
    truncation_distance_m: float,
) -> TSDFVolume:
    shape_xyz = compute_tsdf_grid_shape(grid_min_world_m, grid_max_world_m, voxel_size_m)
    centers_world = voxel_centers_world(grid_min_world_m, shape_xyz, voxel_size_m)
    tsdf_flat = np.ones(centers_world.shape[0], dtype=FLOAT64)
    weight_flat = np.zeros(centers_world.shape[0], dtype=FLOAT64)
    for observation in observations:
        integrate_depth_observation(
            observation=observation,
            centers_world_m=centers_world,
            voxel_size_m=voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            tsdf_flat=tsdf_flat,
            weight_flat=weight_flat,
        )
    scale_sources = stable_strings([observation.pose.scale_source for observation in observations])
    coordinate_frames = stable_strings(
        [
            str(observation.pose.diagnostics.get("coordinate_frame", CAMERA_COORDINATE_FRAME))
            for observation in observations
        ]
    )
    return TSDFVolume(
        grid_min_corner_world_m=grid_min_world_m.astype(FLOAT32),
        voxel_size_m=float(voxel_size_m),
        truncation_distance_m=float(truncation_distance_m),
        tsdf=tsdf_flat.reshape(shape_xyz).astype(FLOAT32),
        weight=weight_flat.reshape(shape_xyz).astype(FLOAT32),
        source_frame_ids=tuple(observation.frame_id for observation in observations),
        coordinate_frame=coordinate_frames[0] if len(coordinate_frames) == 1 else "mixed",
        metric_scale_source=scale_sources[0] if len(scale_sources) == 1 else "mixed",
    )


def grid_bounds_from_observations(
    observations: tuple[DepthObservation, ...],
    *,
    voxel_size_m: float,
    truncation_distance_m: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    point_sets = []
    for observation in observations:
        points = point_world_from_observation(observation)
        valid = (
            (observation.depth_m > 0.0)
            & (observation.confidence > 0.0)
            & np.all(np.isfinite(points), axis=2)
        )
        if np.any(valid):
            point_sets.append(points[valid].reshape(-1, 3))
    if not point_sets:
        raise ValueError("checkpoint TSDF smoke could not derive grid bounds from observations")
    all_points = np.concatenate(point_sets, axis=0).astype(FLOAT64, copy=False)
    padding = truncation_distance_m + voxel_size_m
    grid_min = np.floor((all_points.min(axis=0) - padding) / voxel_size_m) * voxel_size_m
    grid_max = np.ceil((all_points.max(axis=0) + padding) / voxel_size_m) * voxel_size_m
    extent = grid_max - grid_min
    too_small = extent < (2.0 * voxel_size_m)
    grid_min[too_small] -= voxel_size_m
    grid_max[too_small] += voxel_size_m
    return cast(npt.NDArray[np.float64], grid_min), cast(npt.NDArray[np.float64], grid_max)


def point_world_from_observation(observation: DepthObservation) -> npt.NDArray[np.float32]:
    depth = observation.depth_m.astype(FLOAT64, copy=False)
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=FLOAT64), np.arange(height, dtype=FLOAT64))
    fx = float(observation.camera.K[0, 0])
    fy = float(observation.camera.K[1, 1])
    cx = float(observation.camera.K[0, 2])
    cy = float(observation.camera.K[1, 2])
    points_camera = np.stack(
        [(u - cx) / fx * depth, (v - cy) / fy * depth, depth],
        axis=-1,
    ).reshape(-1, 3)
    points_world = transform_points(observation.pose.T_world_camera, points_camera)
    return points_world.reshape(height, width, 3).astype(FLOAT32)


def with_smoke_metadata(
    surface: TSDFSurface,
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    observations: tuple[DepthObservation, ...],
    artifact_type: str,
    input_kind: str,
    extra_metadata: dict[str, Any],
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            "artifact_type": artifact_type,
            "input_kind": input_kind,
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_step": checkpoint.step,
            "checkpoint_format": checkpoint.model_config.get("model_name", "TinyDepthPoseNet"),
            "truth_boundary": dict(checkpoint.truth_boundary),
            "accuracy_report_path": None,
            "accuracy_note": (
                "Phase 4B checkpoint TSDF smoke is a bridge diagnostic, not an accuracy report."
            ),
            "input_confidence_summary": array_summary(
                np.concatenate([obs.confidence.reshape(-1) for obs in observations])
            ),
            "input_uncertainty_summary_m": array_summary(
                np.concatenate([obs.depth_sigma_m.reshape(-1) for obs in observations])
            ),
            "observation_sources": stable_strings([obs.source for obs in observations]),
            "metric_scale_sources": stable_strings([obs.pose.scale_source for obs in observations]),
            **extra_metadata,
        }
    )
    metadata["flags"] = stable_strings(
        [
            *[str(flag) for flag in metadata.get("flags", [])],
            "phase_4b_checkpoint_inference_bridge",
            "phase_4b_reference_only",
            "observed_surface_points",
            "not_completed_surface",
            "not_accuracy_report",
        ]
    )
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def synthetic_comparison_metrics(
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    prediction: TinyDepthPosePrediction,
    sample: SyntheticDepthSample,
    predicted_volume: TSDFVolume,
    target_volume: TSDFVolume,
    predicted_surface: TSDFSurface,
    target_surface: TSDFSurface,
) -> dict[str, Any]:
    predicted_depth = prediction.depth_m[0, 0].astype(FLOAT64, copy=False)
    target_depth = sample.depth_m.astype(FLOAT64, copy=False)
    abs_error = np.abs(predicted_depth - target_depth)
    center_error = float(
        np.linalg.norm(prediction.camera_center_world_m[0, 0] - sample.camera_center_world_m)
    )
    common = (predicted_volume.weight > 0.0) & (target_volume.weight > 0.0)
    common_count = int(np.count_nonzero(common))
    common_delta = (
        float(np.mean(np.abs(predicted_volume.tsdf[common] - target_volume.tsdf[common])))
        if common_count
        else None
    )
    return {
        "metric_family": "phase_4b_checkpoint_synthetic_depth_tsdf_bridge",
        "accuracy_report": False,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_step": checkpoint.step,
        "input_kind": "procedural_synthetic_depth",
        "source_frame_ids": [sample.frame_id],
        "depth_metrics_m": {
            "mae": float(np.mean(abs_error)),
            "p95_abs": float(np.percentile(abs_error, 95.0)),
            "max_abs": float(np.max(abs_error)),
        },
        "camera_center_error_m": center_error,
        "predicted_surface_point_count": int(predicted_surface.points_world_m.shape[0]),
        "target_surface_point_count": int(target_surface.points_world_m.shape[0]),
        "observed_tsdf_voxel_count_predicted": int(np.count_nonzero(predicted_volume.weight > 0.0)),
        "observed_tsdf_voxel_count_target": int(np.count_nonzero(target_volume.weight > 0.0)),
        "common_observed_tsdf_voxel_count": common_count,
        "common_observed_tsdf_mean_abs_delta": common_delta,
        "predicted_uncertainty_summary_m": predicted_surface.metadata["uncertainty_summary_m"],
        "target_uncertainty_summary_m": target_surface.metadata["uncertainty_summary_m"],
        "truth_boundary": dict(checkpoint.truth_boundary),
        "known_limitations": [
            "This smoke output is not an accuracy report.",
            "The tiny Phase 4A checkpoint is synthetic-only and not a real-capture model.",
            "TSDF comparison is a diagnostic bridge check, not a benchmark.",
        ],
    }


def target_metrics(sample: SyntheticDepthSample) -> dict[str, Any]:
    return {
        "metric_family": "phase_4b_synthetic_target_tsdf_reference",
        "accuracy_report": False,
        "input_kind": "procedural_synthetic_depth",
        "source_frame_ids": [sample.frame_id],
        "known_limitations": [
            "Synthetic target TSDF is a diagnostic reference, not an accuracy report."
        ],
    }


def write_prediction_artifacts(
    output_path: Path,
    *,
    prediction: TinyDepthPosePrediction,
    sample: SyntheticDepthSample | None = None,
    predicted_observation: DepthObservation | None = None,
) -> tuple[Path, ...]:
    sample_path = output_path / "prediction_sample.npz"
    if sample is not None and predicted_observation is not None:
        np.savez(
            sample_path,
            rgb_u8=sample.rgb_u8,
            target_depth_m=sample.depth_m,
            predicted_depth_m=predicted_observation.depth_m.astype(np.float32),
            abs_depth_error_m=np.abs(predicted_observation.depth_m - sample.depth_m).astype(
                np.float32
            ),
            target_depth_sigma_m=sample.depth_sigma_m,
            predicted_depth_sigma_m=predicted_observation.depth_sigma_m.astype(np.float32),
            target_confidence=sample.confidence,
            predicted_confidence=predicted_observation.confidence.astype(np.float32),
            K=sample.K,
            target_T_world_camera=sample.T_world_camera,
            predicted_T_world_camera=predicted_observation.pose.T_world_camera,
            target_camera_center_world_m=sample.camera_center_world_m,
            predicted_camera_center_world_m=prediction.camera_center_world_m[0, 0],
        )
        return (sample_path,)
    np.savez(
        sample_path,
        frame_ids=np.asarray(prediction.frame_ids, dtype=np.int32),
        predicted_depth_m=prediction.depth_m[0],
        predicted_depth_sigma_m=prediction.depth_sigma_m[0],
        predicted_confidence=prediction.confidence[0],
        predicted_T_world_camera=prediction.T_world_camera[0],
        K=prediction.intrinsics[0],
    )
    return (sample_path,)


def write_tsdf_outputs(
    output_path: Path,
    volume: TSDFVolume,
    surface: TSDFSurface,
    metrics: dict[str, Any],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / "tsdf_grid.npz",
        grid_min_corner_world_m=volume.grid_min_corner_world_m,
        voxel_size_m=np.array(volume.voxel_size_m, dtype=FLOAT32),
        truncation_distance_m=np.array(volume.truncation_distance_m, dtype=FLOAT32),
        tsdf=volume.tsdf,
        weight=volume.weight,
    )
    np.savez(
        output_path / "surface_points.npz",
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
    )
    write_json(output_path / "metadata.json", surface.metadata)
    write_json(output_path / "metrics.json", metrics)


def validate_smoke_args(
    width: int,
    height: int,
    voxel_size_m: float,
    truncation_voxels: float,
) -> None:
    for field_name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")


def array_summary(values: npt.NDArray[Any]) -> dict[str, float]:
    array = np.asarray(values, dtype=FLOAT64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def stable_strings(values: list[str]) -> list[str]:
    stable: list[str] = []
    for value in values:
        if value not in stable:
            stable.append(value)
    return stable


def write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _target_pose_covariance(sample: SyntheticDepthSample) -> npt.NDArray[np.float32]:
    variance = float(np.mean(np.square(sample.depth_sigma_m.astype(FLOAT64, copy=False))))
    diagonal = np.full(6, max(variance, 1e-8), dtype=np.float32)
    return cast(npt.NDArray[np.float32], np.diag(diagonal).astype(np.float32))


__all__ = [
    "grid_bounds_from_observations",
    "integrate_observations_to_volume",
    "synthetic_comparison_metrics",
    "target_metrics",
    "target_observation_from_sample",
    "validate_smoke_args",
    "with_smoke_metadata",
    "write_prediction_artifacts",
    "write_tsdf_outputs",
]
