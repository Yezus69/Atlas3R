"""Private helpers for teacher-signal TSDF bridge summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import array_summary, stable_strings
from atlas3r.mapping.observations import DepthObservation

TEACHER_FORMAT_NOTE = "atlas3r_teacher_signal_cache_v1"


def with_teacher_surface_metadata(
    surface: Any,
    teacher_manifest: dict[str, object],
    observations: tuple[DepthObservation, ...],
) -> Any:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            "artifact_type": "phase_5b_teacher_signal_cpu_tsdf_surface_points",
            "source_teacher_cache_format": TEACHER_FORMAT_NOTE,
            "teacher_name": str(teacher_manifest["teacher_name"]),
            "teacher_source_type": str(teacher_manifest["teacher_source_type"]),
            "truth_boundary": dict(cast(dict[str, object], teacher_manifest["truth_boundary"])),
            "accuracy_report_path": None,
            "accuracy_note": "Teacher-signal map output is diagnostic, not an accuracy report.",
            "observation_sources": stable_strings([obs.source for obs in observations]),
            "input_uncertainty_summary_m": array_summary(
                np.concatenate([obs.depth_sigma_m.reshape(-1) for obs in observations])
            ),
        }
    )
    metadata["flags"] = stable_strings(
        [
            *[str(flag) for flag in metadata.get("flags", [])],
            "phase_5b_teacher_signal_mapping_bridge",
            "observed_surface_points",
            "not_completed_surface",
            "not_accuracy_report",
            "not_realtime_report",
        ]
    )
    return type(surface)(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def map_metrics(teacher_manifest: dict[str, object], surface: Any) -> dict[str, object]:
    return {
        "metric_family": "phase_5b_teacher_signal_tsdf_diagnostic",
        "accuracy_report": False,
        "performance_report": False,
        "teacher_name": str(teacher_manifest["teacher_name"]),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "known_limitations": [
            "No benchmark ground truth is used by this map diagnostic.",
            "TSDF output contains observed fused surfaces only and no hidden-geometry completion.",
        ],
    }


def map_summary(
    *,
    teacher_manifest: dict[str, object],
    surface: Any,
    observations: tuple[DepthObservation, ...],
    tsdf_dir: Path,
    unique_source_frame_ids: list[int],
) -> dict[str, object]:
    metadata = cast(dict[str, object], surface.metadata)
    return {
        "format_name": "atlas3r_teacher_signal_map_summary",
        "format_version": 1,
        "teacher_tsdf_path": str(tsdf_dir),
        "teacher_name": str(teacher_manifest["teacher_name"]),
        "teacher_source_type": str(teacher_manifest["teacher_source_type"]),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "source_frame_ids": [obs.frame_id for obs in observations],
        "unique_source_frame_ids": unique_source_frame_ids,
        "voxel_size_m": _float_metadata(metadata, "voxel_size_m"),
        "observed_coverage_estimate": _float_metadata(metadata, "observed_coverage_estimate"),
        "uncertainty_summary_m": dict(cast(dict[str, object], metadata["uncertainty_summary_m"])),
        "truth_boundary": dict(cast(dict[str, object], teacher_manifest["truth_boundary"])),
        "accuracy_report": False,
        "performance_report": False,
    }


def _float_metadata(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"surface.metadata.{key}: must be numeric")
    return float(value)


__all__ = [
    "map_metrics",
    "map_summary",
    "with_teacher_surface_metadata",
]
