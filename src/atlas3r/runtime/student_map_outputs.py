"""TSDF metadata and summary helpers for Phase 5E student runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import stable_strings
from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.student_map_reports import DIAGNOSTIC_TRUTH_FLAGS, point_set_comparison
from atlas3r.runtime.student_map_runtime_types import StudentMapRuntimeConfig
from atlas3r.runtime.student_stream import StreamFrame
from atlas3r.teachers.map_eval import TeacherSignalMapConfig, map_teacher_signals
from atlas3r.teachers.signals import load_teacher_signal_manifest


def with_runtime_surface_metadata(
    surface: TSDFSurface,
    *,
    pose_mode: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    observations: tuple[DepthObservation, ...],
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            "artifact_type": "phase_5e_streaming_student_cpu_tsdf_surface_points",
            "pose_mode": pose_mode,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "truth_boundary": dict(cast(dict[str, object], checkpoint["truth_boundary"])),
            "accuracy_report_path": None,
            "accuracy_note": "Phase 5E runtime output is diagnostic, not an accuracy report.",
            "observation_sources": stable_strings([obs.source for obs in observations]),
            **DIAGNOSTIC_TRUTH_FLAGS,
        }
    )
    metadata["flags"] = stable_strings(
        [
            *[str(flag) for flag in metadata.get("flags", [])],
            "phase_5e_streaming_student_runtime",
            "observed_surface_points",
            "not_completed_surface",
            "not_accuracy_report",
            "not_realtime_report",
            "not_mapping_ready",
        ]
    )
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def teacher_map_comparison(
    config: StudentMapRuntimeConfig,
    output: Path,
    surface: TSDFSurface,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    try:
        teacher_manifest = load_teacher_signal_manifest(
            config.teacher_cache, validate_payloads=False
        )
        signal_count = _int_field(teacher_manifest, "signal_count")
        result = map_teacher_signals(
            TeacherSignalMapConfig(
                teacher_cache=config.teacher_cache,
                output=output / "teacher_reference_map",
                max_clips=max(1, signal_count),
                voxel_size_m=config.voxel_size_m,
                truncation_voxels=config.truncation_voxels,
            )
        )
        teacher_surface_path = (
            output / "teacher_reference_map" / "teacher_tsdf" / "surface_points.npz"
        )
        with np.load(teacher_surface_path, allow_pickle=False) as payload:
            teacher_points = np.asarray(payload["points_world_m"], dtype=np.float32)
        comparison = point_set_comparison(surface.points_world_m, teacher_points)
        return cast(dict[str, object], result["map_summary"]), comparison
    except ValueError as exc:
        return {"teacher_map_error": str(exc), "diagnostic_only": True}, None


def tsdf_metrics(surface: TSDFSurface, pose_mode: str) -> dict[str, object]:
    return {
        "metric_family": "phase_5e_streaming_student_tsdf_diagnostic",
        **DIAGNOSTIC_TRUTH_FLAGS,
        "pose_mode": pose_mode,
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "known_limitations": [
            "CPU TSDF output contains observed fused surface points only.",
            "The exported PLY is a point cloud, not a triangle mesh.",
        ],
    }


def tsdf_summary(
    surface: TSDFSurface,
    tsdf_dir: Path,
    teacher_map_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    metadata = surface.metadata
    return {
        "tsdf_path": str(tsdf_dir),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "source_frame_ids": list(cast(list[int], metadata["source_frame_ids"])),
        "voxel_size_m": float(metadata["voxel_size_m"]),
        "observed_coverage_estimate": float(metadata["observed_coverage_estimate"]),
        "uncertainty_summary_m": dict(cast(dict[str, object], metadata["uncertainty_summary_m"])),
        "teacher_map_summary": None if teacher_map_summary is None else dict(teacher_map_summary),
        **DIAGNOSTIC_TRUTH_FLAGS,
    }


def summary_record(
    *,
    config: StudentMapRuntimeConfig,
    pose_mode: str,
    output: Path,
    checkpoint: Mapping[str, Any],
    resolved_device: str,
    stream: tuple[StreamFrame, ...],
    observations: tuple[DepthObservation, ...],
    surface: TSDFSurface,
    tsdf_summary_record: Mapping[str, object],
    quality_report: Mapping[str, object],
    latency_report: Mapping[str, object],
    ply_path: Path,
) -> dict[str, object]:
    return {
        "format_name": "atlas3r_phase5e_stream_student_map_summary",
        "format_version": 1,
        **DIAGNOSTIC_TRUTH_FLAGS,
        "pose_mode": pose_mode,
        "checkpoint": str(config.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "device": resolved_device,
        "clip_cache": str(config.clip_cache),
        "teacher_cache": str(config.teacher_cache),
        "stream_frame_count": len(stream),
        "observation_count": len(observations),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "artifacts": {
            "runtime_events": _rel(output, output / "runtime_events.jsonl"),
            "summary": _rel(output, output / "summary.json"),
            "observations_summary": _rel(output, output / "observations_summary.jsonl"),
            "latency_report": _rel(output, output / "latency_report.json"),
            "quality_report": _rel(output, output / "quality_report.json"),
            "per_frame_quality": _rel(output, output / "per_frame_quality.jsonl"),
            "tsdf": _rel(output, output / "tsdf"),
            "point_cloud": _rel(output, ply_path),
            "map_preview": _rel(output, output / "map_preview.html"),
        },
        "cpu_tsdf": dict(tsdf_summary_record),
        "quality_aggregate": dict(cast(dict[str, object], quality_report["aggregate"])),
        "latency": dict(cast(dict[str, object], latency_report["fps_equivalent"])),
        "student_relative_pose_note": (
            "student-relative pose is diagnostic only and keeps source rotation"
            if pose_mode == "student-relative"
            else None
        ),
        "known_blockers_for_realtime_mapping": [
            "No realtime scheduler or bounded GPU mapper is implemented in this phase.",
            "Student-relative pose is not validated as a mapping-ready tracker.",
            "Quality is compared only against the provided teacher cache, not a benchmark.",
        ],
    }


def mode_comparison(mode_results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "diagnostic_only": True,
        "accuracy_report": False,
        "modes": [
            {
                "pose_mode": result["pose_mode"],
                "surface_point_count": result["surface_point_count"],
                "depth_rmse_m": cast(dict[str, object], result["quality_aggregate"]).get(
                    "depth_rmse_m"
                ),
            }
            for result in mode_results
        ],
    }


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "mode_comparison",
    "summary_record",
    "teacher_map_comparison",
    "tsdf_metrics",
    "tsdf_summary",
    "with_runtime_surface_metadata",
]
