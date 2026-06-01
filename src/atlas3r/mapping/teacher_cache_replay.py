"""Replay validated teacher-cache arrays into the CPU TSDF reference path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.data.synthetic_cube_room import create_synthetic_cube_room_scene
from atlas3r.io._teacher_cache_validation import load_validated_array_payload
from atlas3r.io.teacher_cache import LoadedTeacherPredictionCache, load_teacher_prediction_cache
from atlas3r.io.teacher_cache_schema import FRAME_SUMMARIES_PATH
from atlas3r.mapping.cpu_tsdf import (
    FLOAT32,
    FLOAT64,
    TSDFSurface,
    TSDFVolume,
    _grid_shape_xyz,
    _integrate_frame,
    _voxel_centers,
    evaluate_surface_against_synthetic_cube_room,
    extract_tsdf_surface,
)


@dataclass(frozen=True)
class TeacherCacheTSDFReplayResult:
    """In-memory result for a teacher-cache TSDF replay smoke run."""

    cache_path: Path
    volume: TSDFVolume
    surface: TSDFSurface
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TeacherCacheTSDFReplayFrame:
    """Minimal per-frame observation reconstructed for CPU TSDF replay."""

    frame_id: int
    camera: CameraModel
    pose: PoseEstimate
    depth_m: npt.NDArray[np.float32]
    depth_sigma_m: npt.NDArray[np.float32]
    confidence: npt.NDArray[np.float32]
    point_world: npt.NDArray[np.float32]


def load_teacher_cache_tsdf_replay_frames(
    cache_path: str | Path,
) -> tuple[LoadedTeacherPredictionCache, tuple[TeacherCacheTSDFReplayFrame, ...]]:
    """Load replayable per-frame TSDF inputs from a validated teacher cache."""
    cache = load_teacher_prediction_cache(cache_path)
    _require_array_payload_cache(cache)
    frames = tuple(
        _replay_frame_from_summary(cache, summary, index)
        for index, summary in enumerate(cache.frame_summaries, start=1)
    )
    return cache, frames


def run_teacher_cache_tsdf_replay(
    cache_path: str | Path,
    output_folder: str | Path,
    *,
    voxel_size_m: float = 0.1,
    truncation_voxels: float = 3.0,
) -> TeacherCacheTSDFReplayResult:
    """Replay cached teacher depth/pose/confidence/uncertainty into CPU TSDF."""
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")

    cache, frames = load_teacher_cache_tsdf_replay_frames(cache_path)
    truncation_distance_m = float(voxel_size_m * truncation_voxels)
    grid_min, grid_max = _replay_grid_bounds(cache, frames, voxel_size_m, truncation_distance_m)
    volume = _integrate_replay_frames(
        cache=cache,
        frames=frames,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    surface = extract_tsdf_surface(volume)
    surface = _with_replay_surface_metadata(cache, surface)
    metrics = _replay_metrics(cache, surface)
    _write_replay_outputs(Path(output_folder), volume, surface, metrics)
    return TeacherCacheTSDFReplayResult(
        cache_path=cache.root,
        volume=volume,
        surface=surface,
        metrics=metrics,
    )


def write_teacher_cache_tsdf_replay(
    cache_path: str | Path,
    output_folder: str | Path,
    *,
    write_mesh_sidecar: bool = False,
    write_world_map_sidecar: bool = False,
) -> tuple[Path, ...]:
    """Write deterministic TSDF replay artifacts from a teacher cache."""
    output_path = Path(output_folder)
    run_teacher_cache_tsdf_replay(cache_path, output_path)
    written_paths: tuple[Path, ...] = (
        output_path / "tsdf_grid.npz",
        output_path / "surface_points.npz",
        output_path / "metadata.json",
        output_path / "metrics.json",
    )
    if write_mesh_sidecar or write_world_map_sidecar:
        from atlas3r.mapping.mesh_sidecar import write_tsdf_surface_mesh_sidecar_from_artifacts

        sidecar_path = write_tsdf_surface_mesh_sidecar_from_artifacts(
            output_path,
            chunk_id="phase_1d_teacher_cache_tsdf_surface_reference",
        )
        written_paths = (*written_paths, sidecar_path)
    if write_world_map_sidecar:
        from atlas3r.mapping.world_map_sidecar import write_tsdf_world_map_sidecar_from_artifacts

        map_sidecar_path = write_tsdf_world_map_sidecar_from_artifacts(output_path)
        written_paths = (*written_paths, map_sidecar_path)
    return written_paths


def _require_array_payload_cache(cache: LoadedTeacherPredictionCache) -> None:
    arrays = cache.metadata["arrays"]
    if not bool(arrays["stored"]):
        raise ValueError(
            f"{cache.root / 'metadata.json'}: teacher cache TSDF replay requires "
            "arrays.stored=true; summaries-only caches cannot be replayed"
        )


def _replay_frame_from_summary(
    cache: LoadedTeacherPredictionCache,
    summary: dict[str, Any],
    line_number: int,
) -> TeacherCacheTSDFReplayFrame:
    location = Path(f"{cache.root / FRAME_SUMMARIES_PATH}:{line_number}")
    camera = _camera_from_summary(summary, location)
    pose = _pose_from_summary(summary, location)
    payload = load_validated_array_payload(cache.root, summary)
    return TeacherCacheTSDFReplayFrame(
        frame_id=int(summary["frame_id"]),
        camera=camera,
        pose=pose,
        depth_m=np.asarray(payload["depth_m"], dtype=FLOAT32),
        depth_sigma_m=np.asarray(payload["depth_sigma_m"], dtype=FLOAT32),
        confidence=np.asarray(payload["confidence"], dtype=FLOAT32),
        point_world=np.asarray(payload["point_world"], dtype=FLOAT32),
    )


def _camera_from_summary(summary: dict[str, Any], location: Path) -> CameraModel:
    camera = _dict_field(summary, "camera", location)
    return CameraModel(
        width=int(_required(camera, "width", location)),
        height=int(_required(camera, "height", location)),
        K=np.asarray(_required(camera, "K", location), dtype=FLOAT32),
        distortion_model=str(_required(camera, "distortion_model", location)),
        distortion_params=_optional_array(camera, "distortion_params", location),
        rolling_shutter_row_time_s=_optional_float(camera, "rolling_shutter_row_time_s", location),
        confidence=float(_required(camera, "confidence", location)),
        source=str(_required(camera, "source", location)),
    )


def _pose_from_summary(summary: dict[str, Any], location: Path) -> PoseEstimate:
    pose = _dict_field(summary, "pose", location)
    return PoseEstimate(
        frame_id=int(_required(summary, "frame_id", location)),
        timestamp_ns=int(_required(summary, "timestamp_ns", location)),
        T_world_camera=np.asarray(_required(pose, "T_world_camera", location), dtype=FLOAT32),
        q_world_camera_xyzw=np.asarray(
            _required(pose, "q_world_camera_xyzw", location), dtype=FLOAT32
        ),
        camera_center_world_m=np.asarray(
            _required(pose, "camera_center_world_m", location), dtype=FLOAT32
        ),
        covariance_6x6=_optional_array(pose, "covariance_6x6", location),
        confidence=float(_required(pose, "confidence", location)),
        tracking_state=str(_required(pose, "tracking_state", location)),
        scale_source=str(_required(pose, "scale_source", location)),
        diagnostics=_dict_field(pose, "diagnostics", location),
    )


def _replay_grid_bounds(
    cache: LoadedTeacherPredictionCache,
    frames: tuple[TeacherCacheTSDFReplayFrame, ...],
    voxel_size_m: float,
    truncation_distance_m: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if _is_synthetic_cube_room_fixture(cache):
        scene = create_synthetic_cube_room_scene()
        return scene.room_bounds_m.min_corner_m, scene.room_bounds_m.max_corner_m
    points = np.concatenate(
        [
            frame.point_world[
                (frame.depth_m > 0.0)
                & (frame.confidence > 0.0)
                & np.all(np.isfinite(frame.point_world), axis=2)
            ].reshape(-1, 3)
            for frame in frames
        ],
        axis=0,
    )
    if points.size == 0:
        raise ValueError(
            f"{cache.root}: teacher cache replay could not derive grid bounds "
            "from finite positive-confidence point_world payloads"
        )
    padding = truncation_distance_m + voxel_size_m
    grid_min = np.floor((points.min(axis=0).astype(FLOAT64) - padding) / voxel_size_m)
    grid_min *= voxel_size_m
    grid_max = np.ceil((points.max(axis=0).astype(FLOAT64) + padding) / voxel_size_m)
    grid_max *= voxel_size_m
    extent = grid_max - grid_min
    too_small = extent < (2.0 * voxel_size_m)
    grid_min[too_small] -= voxel_size_m
    grid_max[too_small] += voxel_size_m
    return grid_min, grid_max


def _integrate_replay_frames(
    *,
    cache: LoadedTeacherPredictionCache,
    frames: tuple[TeacherCacheTSDFReplayFrame, ...],
    grid_min_world_m: npt.NDArray[np.float64],
    grid_max_world_m: npt.NDArray[np.float64],
    voxel_size_m: float,
    truncation_distance_m: float,
) -> TSDFVolume:
    shape_xyz = _grid_shape_xyz(grid_min_world_m, grid_max_world_m, voxel_size_m)
    centers_world = _voxel_centers(grid_min_world_m, shape_xyz, voxel_size_m)
    tsdf_flat = np.ones(centers_world.shape[0], dtype=FLOAT64)
    weight_flat = np.zeros(centers_world.shape[0], dtype=FLOAT64)
    for frame in frames:
        _integrate_frame(
            frame=frame,  # type: ignore[arg-type]
            centers_world_m=centers_world,
            voxel_size_m=voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            tsdf_flat=tsdf_flat,
            weight_flat=weight_flat,
        )
    scale_sources = [str(item) for item in cache.metadata["scale_sources"]]
    return TSDFVolume(
        grid_min_corner_world_m=grid_min_world_m.astype(FLOAT32),
        voxel_size_m=float(voxel_size_m),
        truncation_distance_m=float(truncation_distance_m),
        tsdf=tsdf_flat.reshape(shape_xyz).astype(FLOAT32),
        weight=weight_flat.reshape(shape_xyz).astype(FLOAT32),
        source_frame_ids=tuple(frame.frame_id for frame in frames),
        coordinate_frame=str(cache.metadata["coordinate_frame"]),
        metric_scale_source=scale_sources[0] if len(scale_sources) == 1 else "mixed",
    )


def _with_replay_surface_metadata(
    cache: LoadedTeacherPredictionCache,
    surface: TSDFSurface,
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            "artifact_type": "phase_1d_teacher_cache_cpu_tsdf_surface_points",
            "source_teacher_cache": str(cache.root),
            "teacher_adapter": cache.adapter_status.name,
            "cache_arrays_stored": True,
            "accuracy_report_path": None,
            "accuracy_note": (
                "Teacher-cache TSDF replay is a deterministic smoke output, not an accuracy report."
            ),
            "metric_scale_sources": list(cache.metadata["scale_sources"]),
            "input_confidence_summaries": [
                {
                    "frame_id": summary["frame_id"],
                    "summary": summary["confidence_summary"],
                }
                for summary in cache.frame_summaries
            ],
            "input_uncertainty_summaries_m": [
                {
                    "frame_id": summary["frame_id"],
                    "summary": summary["uncertainty_summary"],
                }
                for summary in cache.frame_summaries
            ],
        }
    )
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def _replay_metrics(
    cache: LoadedTeacherPredictionCache,
    surface: TSDFSurface,
) -> dict[str, Any]:
    if _is_synthetic_cube_room_fixture(cache):
        metrics = evaluate_surface_against_synthetic_cube_room(
            create_synthetic_cube_room_scene(),
            surface,
        )
        metrics["metric_family"] = "phase_1d_teacher_cache_replay_synthetic_cube_room_reference"
        metrics["source_teacher_cache"] = str(cache.root)
        metrics["accuracy_report"] = False
        return metrics
    return {
        "metric_family": "not_evaluated",
        "source_teacher_cache": str(cache.root),
        "accuracy_report": False,
        "reason": (
            "No fixture metrics were computed because cache metadata does not prove "
            "the synthetic cube-room fixture source."
        ),
        "known_limitations": [
            "This smoke output is not an accuracy report.",
            "No benchmark or calibration ground truth was provided for this teacher cache.",
        ],
    }


def _is_synthetic_cube_room_fixture(cache: LoadedTeacherPredictionCache) -> bool:
    metadata = cache.metadata.get("prediction_metadata", {})
    return (
        cache.adapter_status.name == "fixture-cube-room"
        and isinstance(metadata, dict)
        and metadata.get("fixture") is True
        and metadata.get("fixture_session_type") == "synthetic_cube_room"
        and cache.metadata.get("coordinate_frame") == "synthetic_world"
    )


def _write_replay_outputs(
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
    _write_json(output_path / "metadata.json", surface.metadata)
    _write_json(output_path / "metrics.json", metrics)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _required(record: dict[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in record:
        raise ValueError(f"{path}: missing replay field {field_name}")
    return record[field_name]


def _dict_field(record: dict[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    value = _required(record, field_name, path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: replay field {field_name} must be an object")
    return cast(dict[str, Any], value)


def _optional_array(
    record: dict[str, Any],
    field_name: str,
    path: Path,
) -> npt.NDArray[np.float32] | None:
    value = _required(record, field_name, path)
    if value is None:
        return None
    return np.asarray(value, dtype=FLOAT32)


def _optional_float(record: dict[str, Any], field_name: str, path: Path) -> float | None:
    value = _required(record, field_name, path)
    if value is None:
        return None
    return float(value)


__all__ = [
    "TeacherCacheTSDFReplayFrame",
    "TeacherCacheTSDFReplayResult",
    "load_teacher_cache_tsdf_replay_frames",
    "run_teacher_cache_tsdf_replay",
    "write_teacher_cache_tsdf_replay",
]
