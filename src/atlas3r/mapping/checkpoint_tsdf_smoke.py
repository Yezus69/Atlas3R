"""CPU TSDF smoke bridge for Phase 4A tiny checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.data.frame_source import load_npz_clip_frames
from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    synthetic_comparison_metrics,
    target_metrics,
    target_observation_from_sample,
    validate_smoke_args,
    with_smoke_metadata,
    write_prediction_artifacts,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import TSDFSurface, TSDFVolume, extract_tsdf_surface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.training.checkpoint_inference import (
    TinyDepthPoseCheckpoint,
    TinyDepthPosePrediction,
    depth_observations_from_tiny_prediction,
    load_tiny_depth_pose_checkpoint,
    predict_tiny_depth_pose_frame_packets,
    predict_tiny_depth_pose_student_clip,
)
from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.synthetic_depth_dataset import (
    SyntheticDepthSample,
    generate_synthetic_depth_samples,
    sample_to_student_clip,
)


@dataclass(frozen=True)
class CheckpointTSDFSmokeResult:
    """Written Phase 4B checkpoint-to-TSDF smoke artifacts."""

    output_path: Path
    checkpoint_path: Path
    input_kind: str
    predicted_volume: TSDFVolume
    predicted_surface: TSDFSurface
    target_volume: TSDFVolume | None
    target_surface: TSDFSurface | None
    metrics: dict[str, Any]
    written_paths: tuple[Path, ...]


def run_checkpoint_tsdf_smoke(
    checkpoint_path: str | Path,
    output_folder: str | Path,
    *,
    input_npz: str | Path | None = None,
    width: int = 32,
    height: int = 24,
    seed: int = 0,
    device: str = "cpu",
    voxel_size_m: float = 0.1,
    truncation_voxels: float = 3.0,
) -> CheckpointTSDFSmokeResult:
    """Run tiny checkpoint inference and feed predictions into CPU TSDF smoke."""

    validate_smoke_args(width, height, voxel_size_m, truncation_voxels)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = load_tiny_depth_pose_checkpoint(checkpoint_path, device=device)
    truncation_distance_m = float(voxel_size_m * truncation_voxels)
    if input_npz is None:
        return _run_synthetic_checkpoint_tsdf_smoke(
            checkpoint=checkpoint,
            output_path=output_path,
            width=width,
            height=height,
            seed=seed,
            voxel_size_m=voxel_size_m,
            truncation_distance_m=truncation_distance_m,
        )
    return _run_npz_checkpoint_tsdf_smoke(
        checkpoint=checkpoint,
        output_path=output_path,
        input_npz=Path(input_npz),
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )


def write_checkpoint_tsdf_smoke(
    checkpoint_path: str | Path,
    output_folder: str | Path,
    *,
    input_npz: str | Path | None = None,
    width: int = 32,
    height: int = 24,
    seed: int = 0,
    device: str = "cpu",
    voxel_size_m: float = 0.1,
    truncation_voxels: float = 3.0,
) -> tuple[Path, ...]:
    """Write Phase 4B checkpoint TSDF smoke artifacts and return their paths."""

    result = run_checkpoint_tsdf_smoke(
        checkpoint_path,
        output_folder,
        input_npz=input_npz,
        width=width,
        height=height,
        seed=seed,
        device=device,
        voxel_size_m=voxel_size_m,
        truncation_voxels=truncation_voxels,
    )
    return result.written_paths


def _run_synthetic_checkpoint_tsdf_smoke(
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    output_path: Path,
    width: int,
    height: int,
    seed: int,
    voxel_size_m: float,
    truncation_distance_m: float,
) -> CheckpointTSDFSmokeResult:
    sample = generate_synthetic_depth_samples(count=1, width=width, height=height, seed=seed)[0]
    prediction = predict_tiny_depth_pose_student_clip(checkpoint, sample_to_student_clip(sample))
    predicted_observation = depth_observations_from_tiny_prediction(
        prediction,
        rgb_u8_by_frame_id={sample.frame_id: sample.rgb_u8},
    )[0]
    target_observation = target_observation_from_sample(sample)
    grid_min, grid_max = grid_bounds_from_observations(
        (predicted_observation, target_observation),
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_volume = integrate_observations_to_volume(
        (predicted_observation,),
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    target_volume = integrate_observations_to_volume(
        (target_observation,),
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_surface = _checkpoint_surface(
        checkpoint,
        predicted_volume,
        observations=(predicted_observation,),
        input_kind="procedural_synthetic_depth",
        target_available=True,
    )
    target_surface = _target_surface(checkpoint, target_volume, target_observation)
    metrics = synthetic_comparison_metrics(
        checkpoint=checkpoint,
        prediction=prediction,
        sample=sample,
        predicted_volume=predicted_volume,
        target_volume=target_volume,
        predicted_surface=predicted_surface,
        target_surface=target_surface,
    )
    written_paths = _write_synthetic_outputs(
        output_path,
        checkpoint=checkpoint,
        prediction=prediction,
        sample=sample,
        predicted_volume=predicted_volume,
        predicted_surface=predicted_surface,
        target_volume=target_volume,
        target_surface=target_surface,
        metrics=metrics,
    )
    return _result(
        output_path,
        checkpoint,
        "procedural_synthetic_depth",
        predicted_volume,
        predicted_surface,
        target_volume,
        target_surface,
        metrics,
        written_paths,
    )


def _run_npz_checkpoint_tsdf_smoke(
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    output_path: Path,
    input_npz: Path,
    voxel_size_m: float,
    truncation_distance_m: float,
) -> CheckpointTSDFSmokeResult:
    frames = load_npz_clip_frames(input_npz)
    prediction = predict_tiny_depth_pose_frame_packets(checkpoint, frames)
    observations = depth_observations_from_tiny_prediction(
        prediction,
        rgb_u8_by_frame_id={frame.frame_id: frame.rgb_u8 for frame in frames},
        timestamp_ns_by_frame_id={frame.frame_id: frame.timestamp_ns for frame in frames},
    )
    grid_min, grid_max = grid_bounds_from_observations(
        observations,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_volume = integrate_observations_to_volume(
        observations,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    predicted_surface = _checkpoint_surface(
        checkpoint,
        predicted_volume,
        observations=observations,
        input_kind="npz_frame_packets",
        target_available=False,
        extra_metadata={"input_npz": str(input_npz)},
    )
    metrics = _npz_metrics(checkpoint, input_npz, predicted_volume, predicted_surface)
    written = write_prediction_artifacts(output_path, prediction=prediction)
    write_tsdf_outputs(output_path, predicted_volume, predicted_surface, metrics)
    written_paths = (
        output_path / "tsdf_grid.npz",
        output_path / "surface_points.npz",
        output_path / "metadata.json",
        output_path / "metrics.json",
        *written,
    )
    return _result(
        output_path,
        checkpoint,
        "npz_frame_packets",
        predicted_volume,
        predicted_surface,
        None,
        None,
        metrics,
        written_paths,
    )


def _checkpoint_surface(
    checkpoint: TinyDepthPoseCheckpoint,
    volume: TSDFVolume,
    *,
    observations: tuple[DepthObservation, ...],
    input_kind: str,
    target_available: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> TSDFSurface:
    metadata = {"target_available": target_available}
    if extra_metadata:
        metadata.update(extra_metadata)
    return with_smoke_metadata(
        extract_tsdf_surface(volume),
        checkpoint=checkpoint,
        observations=observations,
        artifact_type="phase_4b_checkpoint_predicted_cpu_tsdf_surface_points",
        input_kind=input_kind,
        extra_metadata=metadata,
    )


def _target_surface(
    checkpoint: TinyDepthPoseCheckpoint,
    volume: TSDFVolume,
    observation: DepthObservation,
) -> TSDFSurface:
    return with_smoke_metadata(
        extract_tsdf_surface(volume),
        checkpoint=checkpoint,
        observations=(observation,),
        artifact_type="phase_4b_checkpoint_target_cpu_tsdf_surface_points",
        input_kind="procedural_synthetic_depth",
        extra_metadata={
            "target_available": True,
            "target_depth_source": "analytic_ray_box_intersection",
        },
    )


def _write_synthetic_outputs(
    output_path: Path,
    *,
    checkpoint: TinyDepthPoseCheckpoint,
    prediction: TinyDepthPosePrediction,
    sample: SyntheticDepthSample,
    predicted_volume: TSDFVolume,
    predicted_surface: TSDFSurface,
    target_volume: TSDFVolume,
    target_surface: TSDFSurface,
    metrics: dict[str, Any],
) -> tuple[Path, ...]:
    predicted_observation = depth_observations_from_tiny_prediction(
        prediction,
        rgb_u8_by_frame_id={sample.frame_id: sample.rgb_u8},
    )[0]
    written = write_prediction_artifacts(
        output_path,
        prediction=prediction,
        sample=sample,
        predicted_observation=predicted_observation,
    )
    write_tsdf_outputs(output_path, predicted_volume, predicted_surface, metrics)
    target_path = output_path / "target_tsdf"
    write_tsdf_outputs(target_path, target_volume, target_surface, target_metrics(sample))
    html_path, svg_path = write_prediction_preview(
        output_dir=output_path,
        rgb_u8=sample.rgb_u8,
        target_depth_m=sample.depth_m,
        predicted_depth_m=predicted_observation.depth_m.astype(np.float32),
        abs_error_m=np.abs(predicted_observation.depth_m - sample.depth_m).astype(np.float32),
        metrics=cast(dict[str, float], metrics["depth_metrics_m"]),
        truth_boundary=checkpoint.truth_boundary,
    )
    return (
        output_path / "tsdf_grid.npz",
        output_path / "surface_points.npz",
        output_path / "metadata.json",
        output_path / "metrics.json",
        *written,
        html_path,
        svg_path,
        target_path / "tsdf_grid.npz",
        target_path / "surface_points.npz",
        target_path / "metadata.json",
        target_path / "metrics.json",
    )


def _npz_metrics(
    checkpoint: TinyDepthPoseCheckpoint,
    input_npz: Path,
    volume: TSDFVolume,
    surface: TSDFSurface,
) -> dict[str, Any]:
    return {
        "metric_family": "not_evaluated",
        "accuracy_report": False,
        "input_kind": "npz_frame_packets",
        "input_npz": str(input_npz),
        "source_frame_ids": list(volume.source_frame_ids),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "truth_boundary": dict(checkpoint.truth_boundary),
        "reason": "NPZ RGB clips do not carry target metric depth for comparison.",
        "known_limitations": [
            "This smoke output is not an accuracy report.",
            _checkpoint_limitation(checkpoint),
        ],
    }


def _checkpoint_limitation(checkpoint: TinyDepthPoseCheckpoint) -> str:
    if bool(checkpoint.truth_boundary.get("synthetic_only", False)):
        return "The tiny checkpoint is synthetic-only and not a real-capture model."
    return "The tiny real-RGBD debug checkpoint is not usable for mapping or realtime mapping."


def _result(
    output_path: Path,
    checkpoint: TinyDepthPoseCheckpoint,
    input_kind: str,
    predicted_volume: TSDFVolume,
    predicted_surface: TSDFSurface,
    target_volume: TSDFVolume | None,
    target_surface: TSDFSurface | None,
    metrics: dict[str, Any],
    written_paths: tuple[Path, ...],
) -> CheckpointTSDFSmokeResult:
    return CheckpointTSDFSmokeResult(
        output_path=output_path,
        checkpoint_path=checkpoint.path,
        input_kind=input_kind,
        predicted_volume=predicted_volume,
        predicted_surface=predicted_surface,
        target_volume=target_volume,
        target_surface=target_surface,
        metrics=metrics,
        written_paths=written_paths,
    )


__all__ = [
    "CheckpointTSDFSmokeResult",
    "run_checkpoint_tsdf_smoke",
    "write_checkpoint_tsdf_smoke",
]
