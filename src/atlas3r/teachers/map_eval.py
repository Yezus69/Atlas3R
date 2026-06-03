"""Teacher-signal inspection and CPU TSDF mapping diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import extract_tsdf_surface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.teachers._diagnostic_helpers import (
    aggregate_within_percent,
    max_or_none,
    mean_metric,
    mean_or_none,
    rmse,
    unique_ints,
    within_percent,
    write_json,
    write_jsonl,
    write_preview_html,
    write_preview_svg,
)
from atlas3r.teachers._map_bridge_helpers import (
    map_metrics,
    map_summary,
    with_teacher_surface_metadata,
)
from atlas3r.teachers.signals import (
    load_teacher_signal_manifest,
    read_teacher_signal_payload_from_entry,
    teacher_signal_manifest_path_from_input,
    validate_payload_matches_signal_entry,
    validate_teacher_signal_payload,
)


@dataclass(frozen=True)
class TeacherSignalInspectConfig:
    clip_cache: Path
    teacher_cache: Path
    output: Path
    max_clips: int = 64


@dataclass(frozen=True)
class TeacherSignalMapConfig:
    teacher_cache: Path
    output: Path
    max_clips: int = 16
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0


def inspect_teacher_signals(config: TeacherSignalInspectConfig) -> dict[str, object]:
    """Compare teacher signal depth/pose against the source clip cache."""

    _validate_inspect_config(config)
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    teacher_manifest_path = teacher_signal_manifest_path_from_input(config.teacher_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    teacher_manifest = load_teacher_signal_manifest(
        teacher_manifest_path,
        clip_cache=clip_manifest_path,
        validate_payloads=False,
    )
    clip_entries = _entries(clip_manifest, "clips")
    signal_entries = _entries(teacher_manifest, "signals")
    selected = signal_entries[: min(config.max_clips, len(signal_entries))]
    if not selected:
        raise ValueError(f"{teacher_manifest_path}: no teacher signals selected for inspection")

    config.output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    accumulator = _new_depth_accumulator()
    for index, signal_entry in enumerate(selected):
        source_clip_id = _int_field(signal_entry, "source_clip_id")
        clip_entry = clip_entries[source_clip_id]
        clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
        _validate_clip_payload_for_manifest(clip_payload, clip_manifest)
        teacher_payload = read_teacher_signal_payload_from_entry(
            teacher_manifest_path.parent,
            signal_entry,
        )
        _validate_signal_payload_for_manifest(teacher_payload, teacher_manifest)
        validate_payload_matches_signal_entry(teacher_payload, signal_entry, index=index)
        record = _per_clip_metrics(
            signal_index=index,
            signal_entry=signal_entry,
            clip_payload=clip_payload,
            teacher_payload=teacher_payload,
            accumulator=accumulator,
        )
        metrics.append(record)

    summary = _inspection_summary(
        teacher_manifest=teacher_manifest,
        selected_count=len(selected),
        accumulator=accumulator,
        per_clip_metrics=metrics,
    )
    write_json(config.output / "summary.json", summary)
    write_jsonl(config.output / "per_clip_metrics.jsonl", metrics)
    write_preview_html(config.output / "preview.html", summary, metrics)
    write_preview_svg(config.output / "preview.svg", metrics)
    return {
        "format_name": "atlas3r_teacher_signal_inspection_result",
        "summary_path": str(config.output / "summary.json"),
        "per_clip_metrics_path": str(config.output / "per_clip_metrics.jsonl"),
        "preview_html_path": str(config.output / "preview.html"),
        "preview_svg_path": str(config.output / "preview.svg"),
        "summary": summary,
    }


def map_teacher_signals(config: TeacherSignalMapConfig) -> dict[str, object]:
    """Fuse teacher signal depth/pose into existing CPU TSDF diagnostics."""

    _validate_map_config(config)
    teacher_manifest_path = teacher_signal_manifest_path_from_input(config.teacher_cache)
    teacher_manifest = load_teacher_signal_manifest(teacher_manifest_path, validate_payloads=False)
    signal_entries = _entries(teacher_manifest, "signals")
    selected = signal_entries[: min(config.max_clips, len(signal_entries))]
    if not selected:
        raise ValueError(f"{teacher_manifest_path}: no teacher signals selected for mapping")
    observations = _observations_from_signals(
        teacher_manifest_path=teacher_manifest_path,
        teacher_manifest=teacher_manifest,
        signal_entries=selected,
    )
    truncation_distance_m = config.voxel_size_m * config.truncation_voxels
    grid_min, grid_max = grid_bounds_from_observations(
        observations,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    volume = integrate_observations_to_volume(
        observations,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    surface = extract_tsdf_surface(volume)
    surface = with_teacher_surface_metadata(surface, teacher_manifest, observations)
    metrics = map_metrics(teacher_manifest, surface)

    tsdf_dir = config.output / "teacher_tsdf"
    config.output.mkdir(parents=True, exist_ok=True)
    write_tsdf_outputs(tsdf_dir, volume, surface, metrics)
    summary = map_summary(
        teacher_manifest=teacher_manifest,
        surface=surface,
        observations=observations,
        tsdf_dir=tsdf_dir,
        unique_source_frame_ids=unique_ints([obs.frame_id for obs in observations]),
    )
    write_json(config.output / "map_summary.json", summary)
    return {
        "format_name": "atlas3r_teacher_signal_map_result",
        "teacher_tsdf_path": str(tsdf_dir),
        "map_summary_path": str(config.output / "map_summary.json"),
        "map_summary": summary,
    }


def _observations_from_signals(
    *,
    teacher_manifest_path: Path,
    teacher_manifest: dict[str, object],
    signal_entries: list[dict[str, object]],
) -> tuple[DepthObservation, ...]:
    observations: list[DepthObservation] = []
    for signal_index, signal_entry in enumerate(signal_entries):
        payload = read_teacher_signal_payload_from_entry(teacher_manifest_path.parent, signal_entry)
        _validate_signal_payload_for_manifest(payload, teacher_manifest)
        validate_payload_matches_signal_entry(payload, signal_entry, index=signal_index)
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32)
        timestamps = np.asarray(payload["timestamps_s"], dtype=np.float64)
        for frame_offset, frame_id in enumerate(frame_ids.tolist()):
            observations.append(
                _observation_from_payload_frame(
                    teacher_manifest=teacher_manifest,
                    payload=payload,
                    frame_offset=frame_offset,
                    frame_id=int(frame_id),
                    timestamp_s=float(timestamps[frame_offset]),
                )
            )
    return tuple(observations)


def _observation_from_payload_frame(
    *,
    teacher_manifest: dict[str, object],
    payload: dict[str, Any],
    frame_offset: int,
    frame_id: int,
    timestamp_s: float,
) -> DepthObservation:
    depth = np.asarray(payload["depth_m"][frame_offset], dtype=np.float32)
    height, width = depth.shape
    T_world_camera = np.asarray(payload["T_world_camera"][frame_offset], dtype=np.float32)
    confidence = np.asarray(payload["confidence"][frame_offset], dtype=np.float32)
    sigma = np.asarray(payload["depth_sigma_m"][frame_offset], dtype=np.float32)
    mean_sigma = float(np.mean(sigma[confidence > 0.0])) if np.any(confidence > 0.0) else 0.0
    covariance = np.diag(np.full(6, max(mean_sigma * mean_sigma, 1e-8), dtype=np.float32))
    truth_boundary = cast(dict[str, object], teacher_manifest["truth_boundary"])
    return DepthObservation(
        frame_id=frame_id,
        camera=CameraModel(
            width=width,
            height=height,
            K=np.asarray(payload["K"][frame_offset], dtype=np.float32),
            distortion_model="none",
            distortion_params=None,
            rolling_shutter_row_time_s=None,
            confidence=1.0,
            source="teacher_signal_cache",
        ),
        pose=PoseEstimate(
            frame_id=frame_id,
            timestamp_ns=int(round(timestamp_s * 1_000_000_000.0)),
            T_world_camera=T_world_camera,
            q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
                np.float32
            ),
            camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
            covariance_6x6=covariance.astype(np.float32),
            confidence=1.0,
            tracking_state="OK",
            scale_source="external_pose",
            diagnostics={
                "source": str(teacher_manifest["teacher_name"]),
                "coordinate_frame": "x_right_y_down_z_forward",
                "truth_boundary": truth_boundary,
            },
        ),
        depth_m=depth,
        depth_sigma_m=sigma,
        confidence=confidence,
        static_mask=np.asarray(payload["valid_mask"][frame_offset], dtype=np.bool_),
        object_id=(
            np.asarray(payload["object_mask_ids"][frame_offset], dtype=np.int32)
            if "object_mask_ids" in payload
            else None
        ),
        source="teacher_signal_cache",
    )


def _per_clip_metrics(
    *,
    signal_index: int,
    signal_entry: dict[str, object],
    clip_payload: dict[str, Any],
    teacher_payload: dict[str, Any],
    accumulator: dict[str, Any],
) -> dict[str, object]:
    clip_depth = np.asarray(clip_payload["depth_m"], dtype=np.float64)
    teacher_depth = np.asarray(teacher_payload["depth_m"], dtype=np.float64)
    overlap = np.asarray(clip_payload["valid_depth_mask"], dtype=np.bool_) & np.asarray(
        teacher_payload["valid_mask"], dtype=np.bool_
    )
    abs_error = np.abs(teacher_depth[overlap] - clip_depth[overlap])
    abs_rel = abs_error / np.maximum(clip_depth[overlap], 1e-12)
    _accumulate_depth(accumulator, abs_error, abs_rel)
    confidence = np.asarray(teacher_payload["confidence"], dtype=np.float64)
    center_delta = _pose_center_delta(clip_payload, teacher_payload)
    return {
        "signal_index": signal_index,
        "source_clip_id": _int_field(signal_entry, "source_clip_id"),
        "frame_ids": list(cast(list[int], signal_entry["frame_ids"])),
        "overlap_valid_pixel_count": int(abs_error.size),
        "valid_pixel_overlap_percent": float(np.count_nonzero(overlap) / overlap.size * 100.0),
        "depth_rmse_m": rmse(abs_error),
        "depth_mae_m": mean_or_none(abs_error),
        "depth_absrel": mean_or_none(abs_rel),
        "within_1mm_percent": within_percent(abs_error, 0.001),
        "within_5mm_percent": within_percent(abs_error, 0.005),
        "within_1cm_percent": within_percent(abs_error, 0.01),
        "within_5cm_percent": within_percent(abs_error, 0.05),
        "within_10cm_percent": within_percent(abs_error, 0.10),
        "pose_center_delta_m_mean": mean_or_none(center_delta),
        "pose_center_delta_m_max": max_or_none(center_delta),
        "confidence_mean": float(np.mean(confidence)) if confidence.size else 0.0,
    }


def _inspection_summary(
    *,
    teacher_manifest: dict[str, object],
    selected_count: int,
    accumulator: dict[str, Any],
    per_clip_metrics: list[dict[str, object]],
) -> dict[str, object]:
    count = int(accumulator["count"])
    within = cast(dict[float, int], accumulator["within"])
    return {
        "format_name": "atlas3r_teacher_signal_inspection_summary",
        "format_version": 1,
        "teacher_name": str(teacher_manifest["teacher_name"]),
        "teacher_source_type": str(teacher_manifest["teacher_source_type"]),
        "selected_signal_count": selected_count,
        "overlap_valid_pixel_count": count,
        "aggregate": {
            "depth_rmse_m": (
                float(np.sqrt(float(accumulator["squared_error_sum"]) / count)) if count else None
            ),
            "depth_mae_m": float(accumulator["abs_error_sum"] / count) if count else None,
            "depth_absrel": float(accumulator["abs_rel_sum"] / count) if count else None,
            "within_1mm_percent": aggregate_within_percent(within, 0.001, count),
            "within_5mm_percent": aggregate_within_percent(within, 0.005, count),
            "within_1cm_percent": aggregate_within_percent(within, 0.01, count),
            "within_5cm_percent": aggregate_within_percent(within, 0.05, count),
            "within_10cm_percent": aggregate_within_percent(within, 0.10, count),
            "valid_pixel_overlap_percent_mean": mean_metric(
                per_clip_metrics, "valid_pixel_overlap_percent"
            ),
            "confidence_mean": mean_metric(per_clip_metrics, "confidence_mean"),
            "pose_center_delta_m_mean": mean_metric(per_clip_metrics, "pose_center_delta_m_mean"),
        },
        "truth_boundary": dict(cast(dict[str, object], teacher_manifest["truth_boundary"])),
        "known_limitations": [
            "Inspection compares teacher signals against the source clip cache only.",
            "This diagnostic is not an accuracy report or performance report.",
        ],
    }


def _validate_clip_payload_for_manifest(
    payload: dict[str, Any],
    manifest: dict[str, object],
) -> None:
    validate_clip_payload(
        payload,
        clip_length=_int_field(manifest, "clip_length"),
        height=_int_field(manifest, "image_height"),
        width=_int_field(manifest, "image_width"),
    )


def _validate_signal_payload_for_manifest(
    payload: dict[str, Any],
    manifest: dict[str, object],
) -> None:
    validate_teacher_signal_payload(
        payload,
        clip_length=_int_field(manifest, "clip_length"),
        height=_int_field(manifest, "image_height"),
        width=_int_field(manifest, "image_width"),
    )


def _new_depth_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "abs_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "abs_rel_sum": 0.0,
        "within": {0.001: 0, 0.005: 0, 0.01: 0, 0.05: 0, 0.10: 0},
    }


def _accumulate_depth(
    accumulator: dict[str, Any],
    abs_error: npt.NDArray[np.float64],
    abs_rel: npt.NDArray[np.float64],
) -> None:
    accumulator["count"] = int(accumulator["count"]) + int(abs_error.size)
    accumulator["abs_error_sum"] = float(accumulator["abs_error_sum"]) + float(np.sum(abs_error))
    accumulator["squared_error_sum"] = float(accumulator["squared_error_sum"]) + float(
        np.sum(np.square(abs_error))
    )
    accumulator["abs_rel_sum"] = float(accumulator["abs_rel_sum"]) + float(np.sum(abs_rel))
    within = cast(dict[float, int], accumulator["within"])
    for threshold in tuple(within):
        within[threshold] += int(np.count_nonzero(abs_error <= threshold))


def _pose_center_delta(
    clip_payload: dict[str, Any],
    teacher_payload: dict[str, Any],
) -> npt.NDArray[np.float64]:
    clip_t = np.asarray(clip_payload["T_world_camera"], dtype=np.float64)[:, :3, 3]
    teacher_t = np.asarray(teacher_payload["T_world_camera"], dtype=np.float64)[:, :3, 3]
    return cast(npt.NDArray[np.float64], np.linalg.norm(teacher_t - clip_t, axis=1))


def _entries(manifest: dict[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]


def _validate_inspect_config(config: TeacherSignalInspectConfig) -> None:
    if config.max_clips <= 0:
        raise ValueError("max_clips: must be positive")


def _validate_map_config(config: TeacherSignalMapConfig) -> None:
    if config.max_clips <= 0:
        raise ValueError("max_clips: must be positive")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if config.truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "TeacherSignalInspectConfig",
    "TeacherSignalMapConfig",
    "inspect_teacher_signals",
    "map_teacher_signals",
]
