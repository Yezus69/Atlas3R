"""Dependency-free runtime fixture output folder inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas3r.io.session import validate_session
from atlas3r.io.teacher_cache import LoadedTeacherPredictionCache, load_teacher_prediction_cache
from atlas3r.mapping.tsdf_output_inspection import tsdf_output_folder_inspection_record
from atlas3r.runtime._fixture_inspection_helpers import (
    bool_field,
    dict_field,
    read_json_object,
    read_jsonl,
    require_equal,
    require_false,
    required,
    str_field,
    string_list_field,
    validate_events,
    validate_relative_path,
    validate_summary,
    validate_summary_header,
)
from atlas3r.runtime.events import (
    RUNTIME_EVENT_LOG_FORMAT_NAME,
    RUNTIME_EVENT_LOG_FORMAT_VERSION,
)
from atlas3r.runtime.scheduler import (
    RUNTIME_EVENTS_FILENAME,
    RUNTIME_SUMMARY_FILENAME,
)

RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME = "atlas3r_runtime_fixture_output_inspection"
RUNTIME_FIXTURE_INSPECTION_FORMAT_VERSION = 1
RUNTIME_FIXTURE_DIAGNOSTIC_NOTE = (
    "Runtime fixture inspection is a diagnostic only; it is not a performance report "
    "and not an accuracy report."
)


def runtime_fixture_inspection_record(input_folder: str | Path) -> dict[str, Any]:
    """Validate a Phase 2D runtime fixture output folder."""
    root = Path(input_folder)
    if not root.is_dir():
        raise ValueError(f"{root}: expected a runtime fixture output directory")

    summary_path = root / RUNTIME_SUMMARY_FILENAME
    event_log_path = root / RUNTIME_EVENTS_FILENAME
    events = read_jsonl(event_log_path)
    summary = read_json_object(summary_path, missing_kind="runtime summary")
    frame_ids = validate_summary_header(summary, summary_path)
    event_validation = validate_events(root, event_log_path, events, frame_ids)
    summary_record = validate_summary(summary, summary_path, event_validation, frame_ids)

    session_record = _validate_session_artifact(
        root / "synthetic_cube_room.atlas3r",
        frame_ids,
    )
    teacher_cache = load_teacher_prediction_cache(root / "teacher_cache")
    teacher_cache_record = _validate_teacher_cache_artifact(teacher_cache, frame_ids, root)

    tsdf_output_path = root / "teacher_cache_tsdf"
    tsdf_inspection = tsdf_output_folder_inspection_record(tsdf_output_path, mode="complete")
    require_equal(
        tsdf_inspection["surface"]["source_frame_ids"],
        frame_ids,
        tsdf_output_path / "metadata.json",
        "source_frame_ids",
        "runtime_summary.json frame_ids",
    )
    require_false(
        bool(tsdf_inspection["truth_boundary"]["accuracy_report"]),
        tsdf_output_path / "metadata.json",
        "teacher_cache_tsdf truth_boundary.accuracy_report",
    )

    artifact_record = _validate_artifacts(
        root,
        summary,
        frame_ids,
        teacher_cache_record["array_payloads"],
    )
    return {
        "format_name": RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME,
        "format_version": RUNTIME_FIXTURE_INSPECTION_FORMAT_VERSION,
        "artifacts": artifact_record,
        "event_log": _event_log_record(events, event_validation),
        "summary": summary_record,
        "session": session_record,
        "teacher_cache": teacher_cache_record,
        "teacher_cache_tsdf": _teacher_cache_tsdf_record(tsdf_inspection),
        "cross_checks": {
            "event_count_matches_summary": True,
            "frame_ids_match_session": True,
            "frame_ids_match_teacher_cache": True,
            "frame_ids_match_teacher_cache_tsdf": True,
            "bounded_memory_matches_summary": True,
            "all_recorded_paths_relative": True,
            "required_artifacts_present": True,
            "teacher_cache_tsdf_complete_inspection_passed": True,
            "passed": True,
        },
        "diagnostic_boundary": {
            "diagnostic_only": True,
            "performance_report": False,
            "accuracy_report": False,
            "note": RUNTIME_FIXTURE_DIAGNOSTIC_NOTE,
        },
    }


def format_runtime_fixture_inspection(input_folder: str | Path) -> str:
    """Format deterministic runtime fixture inspection JSON."""
    return (
        json.dumps(
            runtime_fixture_inspection_record(input_folder),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _event_log_record(events: tuple[dict[str, Any], ...], event_validation: Any) -> dict[str, Any]:
    return {
        "format_name": RUNTIME_EVENT_LOG_FORMAT_NAME,
        "format_version": RUNTIME_EVENT_LOG_FORMAT_VERSION,
        "event_count": len(events),
        "stage_sequence": event_validation.stage_sequence,
        "stage_counts": event_validation.stage_counts,
        "frame_ids_by_stage": event_validation.frame_ids_by_stage,
        "timestamp_placeholder_ns_step": 1_000_000,
        "latency_ns_values": [0],
        "dropped_frame_count": event_validation.dropped_frame_count,
        "bounded_memory": {
            "configured_frame_array_bound": event_validation.configured_frame_array_bound,
            "peak_frame_arrays_in_memory": event_validation.peak_frame_arrays_in_memory,
            "processed_frame_count": event_validation.processed_frame_count,
            "dropped_frame_count": event_validation.dropped_frame_count,
            "queued_frame_count": event_validation.queued_frame_count,
        },
        "relative_paths_checked": event_validation.relative_paths,
    }


def _teacher_cache_tsdf_record(tsdf_inspection: dict[str, Any]) -> dict[str, Any]:
    surface = tsdf_inspection["surface"]
    return {
        "format_name": tsdf_inspection["format_name"],
        "format_version": tsdf_inspection["format_version"],
        "inspect_mode": tsdf_inspection["inspect_mode"],
        "surface": {
            "artifact_type": surface["artifact_type"],
            "coordinate_frame": surface["coordinate_frame"],
            "source_frame_ids": surface["source_frame_ids"],
            "metric_scale_source": surface["metric_scale_source"],
            "voxel_size_m": surface["voxel_size_m"],
            "observed_coverage_estimate": surface["observed_coverage_estimate"],
            "point_count": surface["point_count"],
            "uncertainty_summary_m": surface["uncertainty_summary_m"],
        },
        "mesh_chunk": tsdf_inspection["mesh_chunk"],
        "world_map": tsdf_inspection["world_map"],
        "cross_checks": tsdf_inspection["cross_checks"],
        "truth_boundary": tsdf_inspection["truth_boundary"],
    }


def _validate_session_artifact(session_path: Path, frame_ids: list[int]) -> dict[str, Any]:
    session = validate_session(session_path)
    session_frame_ids = [int(pose.frame_id) for pose in session.poses]
    require_equal(
        session_frame_ids,
        frame_ids,
        session_path / "poses.jsonl",
        "frame_ids",
        "runtime_summary.json",
    )
    require_equal(
        str(session.metadata["session_type"]),
        "synthetic_cube_room",
        session_path / "metadata.json",
        "session_type",
        "synthetic_cube_room",
    )
    return {
        "path": "synthetic_cube_room.atlas3r",
        "session_type": session.metadata["session_type"],
        "frame_count": session.frame_count,
        "frame_ids": session_frame_ids,
        "depth_payload_count": len(session.depth_files),
    }


def _validate_teacher_cache_artifact(
    cache: LoadedTeacherPredictionCache,
    frame_ids: list[int],
    root: Path,
) -> dict[str, Any]:
    metadata = cache.metadata
    require_equal(
        cache.adapter_status.name,
        "fixture-cube-room",
        cache.root / "metadata.json",
        "adapter.name",
        "fixture-cube-room",
    )
    require_equal(
        list(metadata["frame_ids"]),
        frame_ids,
        cache.root / "metadata.json",
        "frame_ids",
        "runtime_summary.json",
    )
    arrays = dict_field(metadata, "arrays", cache.root / "metadata.json")
    require_equal(
        bool_field(arrays, "stored", cache.root / "metadata.json"),
        True,
        cache.root / "metadata.json",
        "arrays.stored",
        "true",
    )
    payloads = []
    for summary in cache.frame_summaries:
        arrays_path = str_field(summary, "arrays_path", cache.root / "frame_summaries.jsonl")
        payload_path = cache.root / arrays_path
        if not payload_path.is_file():
            raise ValueError(f"{payload_path}: missing teacher cache array payload")
        payloads.append({"frame_id": int(summary["frame_id"]), "path": _rel(root, payload_path)})
    return {
        "path": "teacher_cache",
        "adapter": cache.adapter_status.name,
        "frame_count": cache.frame_count,
        "frame_ids": list(metadata["frame_ids"]),
        "coordinate_frame": metadata["coordinate_frame"],
        "scale_sources": metadata["scale_sources"],
        "arrays_stored": arrays["stored"],
        "array_payload_count": len(payloads),
        "array_payloads": payloads,
    }


def _validate_artifacts(
    root: Path,
    summary: dict[str, Any],
    frame_ids: list[int],
    array_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    artifacts = dict_field(summary, "artifacts", root / RUNTIME_SUMMARY_FILENAME)
    artifact_paths = string_list_field(artifacts, "paths", root / RUNTIME_SUMMARY_FILENAME)
    for relative in artifact_paths:
        validate_relative_path(root, relative, root / RUNTIME_SUMMARY_FILENAME, "artifacts.paths")
    required_paths = _required_artifact_paths(frame_ids)
    missing_from_summary = sorted(set(required_paths) - set(artifact_paths))
    if missing_from_summary:
        raise ValueError(
            f"{root / RUNTIME_SUMMARY_FILENAME}: artifacts.paths missing {missing_from_summary}"
        )
    for relative in required_paths:
        validate_relative_path(
            root,
            relative,
            root / RUNTIME_SUMMARY_FILENAME,
            "required_artifacts",
        )
    require_equal(
        int(required(artifacts, "artifact_count", root / RUNTIME_SUMMARY_FILENAME)),
        len(artifact_paths),
        root / RUNTIME_SUMMARY_FILENAME,
        "artifacts.artifact_count",
        "artifacts.paths",
    )
    return {
        "runtime_events_jsonl": {"path": RUNTIME_EVENTS_FILENAME, "present": True},
        "runtime_summary_json": {"path": RUNTIME_SUMMARY_FILENAME, "present": True},
        "session": {
            "path": "synthetic_cube_room.atlas3r",
            "metadata_json": "synthetic_cube_room.atlas3r/metadata.json",
            "present": True,
        },
        "teacher_cache": {
            "path": "teacher_cache",
            "metadata_json": "teacher_cache/metadata.json",
            "frame_summaries_jsonl": "teacher_cache/frame_summaries.jsonl",
            "array_payloads": array_payloads,
            "present": True,
        },
        "teacher_cache_tsdf": {
            "path": "teacher_cache_tsdf",
            "required_files": [
                "teacher_cache_tsdf/tsdf_grid.npz",
                "teacher_cache_tsdf/surface_points.npz",
                "teacher_cache_tsdf/metadata.json",
                "teacher_cache_tsdf/metrics.json",
                "teacher_cache_tsdf/mesh_chunk_sidecar.json",
                "teacher_cache_tsdf/world_map_sidecar.json",
            ],
            "present": True,
        },
        "summary_artifact_count": len(artifact_paths),
    }


def _required_artifact_paths(frame_ids: list[int]) -> list[str]:
    return [
        RUNTIME_EVENTS_FILENAME,
        RUNTIME_SUMMARY_FILENAME,
        "synthetic_cube_room.atlas3r/metadata.json",
        "teacher_cache/metadata.json",
        "teacher_cache/frame_summaries.jsonl",
        *[f"teacher_cache/arrays/frame_{frame_id:06d}.npz" for frame_id in frame_ids],
        "teacher_cache_tsdf/tsdf_grid.npz",
        "teacher_cache_tsdf/surface_points.npz",
        "teacher_cache_tsdf/metadata.json",
        "teacher_cache_tsdf/metrics.json",
        "teacher_cache_tsdf/mesh_chunk_sidecar.json",
        "teacher_cache_tsdf/world_map_sidecar.json",
    ]


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "RUNTIME_FIXTURE_DIAGNOSTIC_NOTE",
    "RUNTIME_FIXTURE_INSPECTION_FORMAT_NAME",
    "RUNTIME_FIXTURE_INSPECTION_FORMAT_VERSION",
    "format_runtime_fixture_inspection",
    "runtime_fixture_inspection_record",
]
