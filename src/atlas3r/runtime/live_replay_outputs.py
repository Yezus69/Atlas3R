"""Artifact writers for Phase 6E live replay diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import (
    SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    SparseBlockTSDFMapper,
    SparseTSDFConfig,
)
from atlas3r.mapping.sparse_tsdf_artifacts import sparse_state_array_bytes
from atlas3r.recording.schema import Atlas3RRecording
from atlas3r.runtime.live_replay_types import (
    LIVE_REPLAY_FORMAT_VERSION,
    LIVE_REPLAY_SUMMARY_FORMAT_NAME,
    LiveReplayConfig,
    LiveReplayState,
)
from atlas3r.runtime.recording_fusion_incremental_helpers import CPU_SPARSE_BACKEND
from atlas3r.runtime.student_map_report_common import DIAGNOSTIC_TRUTH_FLAGS


def extract_or_empty_surface(
    mapper: SparseBlockTSDFMapper,
    sparse_config: SparseTSDFConfig,
    recording: Atlas3RRecording,
    config: LiveReplayConfig,
) -> TSDFSurface:
    try:
        surface = mapper.extract_surface()
        points = surface.points_world_m
        confidence = surface.confidence
        uncertainty = surface.uncertainty_m
        voxel_indices = surface.voxel_indices_xyz
        metadata = dict(surface.metadata)
    except ValueError as exc:
        points = np.zeros((0, 3), dtype=np.float32)
        confidence = np.zeros((0,), dtype=np.float32)
        uncertainty = np.zeros((0,), dtype=np.float32)
        voxel_indices = np.zeros((0, 3), dtype=np.int32)
        metadata = {
            "active_block_count": mapper.active_block_count,
            "active_voxel_count": mapper.active_voxel_count,
            "allocated_voxel_count": mapper.allocated_voxel_count,
            "approximate_state_bytes": mapper.approximate_state_bytes,
            "block_size_voxels": sparse_config.block_size_voxels,
            "coordinate_frame": sparse_config.coordinate_frame,
            "empty_surface_reason": str(exc),
            "metric_scale_source": sparse_config.metric_scale_source,
            "source_frame_ids": list(mapper.source_frame_ids),
            "surface_count": 0,
            "truncation_distance_m": sparse_config.truncation_distance_m,
            "voxel_size_m": sparse_config.voxel_size_m,
        }
    metadata.update(
        {
            **DIAGNOSTIC_TRUTH_FLAGS,
            "accuracy_note": "Live replay is diagnostic scheduler evidence, not a benchmark.",
            "artifact_type": "phase6e_live_replay_sparse_surface_points",
            "hidden_geometry_measured": False,
            "mapper_backend": config.mapper_backend,
            "recording": str(recording.root),
            "rgb_only_mapping_ready": False,
            "student_rgb_only_used": False,
            "truth_boundary": truth_boundary(
                measured_depth_used=bool(mapper.source_frame_ids),
                measured_pose_used=bool(mapper.source_frame_ids),
            ),
        }
    )
    return TSDFSurface(
        points_world_m=points,
        confidence=confidence,
        uncertainty_m=uncertainty,
        voxel_indices_xyz=voxel_indices,
        metadata=metadata,
    )


def live_replay_summary(
    *,
    config: LiveReplayConfig,
    recording: Atlas3RRecording,
    state: LiveReplayState,
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    source_observations: tuple[DepthObservation, ...],
    latency_report: dict[str, object],
    point_cloud_path: Path | None,
    mesh_status: dict[str, object],
    mesh_manifest: dict[str, object] | None,
    dirty_block_counts: tuple[int, ...],
) -> dict[str, object]:
    mesh_artifacts: dict[str, object] = {}
    if mesh_manifest is not None:
        mesh_artifacts = {
            "mesh_chunk_manifest": "mesh_chunks/mesh_chunk_manifest.json",
            "mesh_chunk_updates": "mesh_chunks/mesh_chunk_updates.jsonl",
            "mesh_chunks": "mesh_chunks",
        }
    mesh_payload_bytes = _status_int(mesh_status, "mesh_payload_bytes")
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": mapper.active_block_count,
        "active_mesh_chunk_count": _status_int(mesh_status, "active_mesh_chunk_count"),
        "active_voxel_count": mapper.active_voxel_count,
        "artifacts": {
            "events": "live_replay_events.jsonl",
            "latency_report": "live_replay_latency_report.json",
            "mesh_status": "mesh_status.json",
            "point_cloud": None if point_cloud_path is None else point_cloud_path.name,
            "report": "live_replay_report.md",
            "sparse_tsdf": "sparse_tsdf",
            "summary": "live_replay_summary.json",
            **mesh_artifacts,
        },
        "deleted_mesh_chunk_count": _status_int(mesh_status, "deleted_mesh_chunk_count"),
        "drop_policy": config.drop_policy,
        "dropped_frame_count": state.dropped_frame_count,
        "dropped_keyframe_count": state.dropped_keyframe_count,
        "dirty_block_count": int(sum(dirty_block_counts)),
        "dirty_blocks_per_update": _int_summary(dirty_block_counts),
        "format_name": LIVE_REPLAY_SUMMARY_FORMAT_NAME,
        "format_version": LIVE_REPLAY_FORMAT_VERSION,
        "frame_count_emitted": state.frame_count_emitted,
        "frame_count_seen": state.frame_count_seen,
        "keyframe_selected_count": state.keyframe_selected_count,
        "known_limitations": known_limitations(),
        "latency": cast(dict[str, object], latency_report["segments"]),
        "map_keyframe_stride": config.map_keyframe_stride,
        "map_update_count": state.map_update_count,
        "mapper_backend": config.mapper_backend,
        "max_capture_queue": config.max_capture_queue,
        "max_capture_queue_depth_observed": state.max_capture_queue_depth_observed,
        "max_map_queue": config.max_map_queue,
        "max_map_queue_depth_observed": state.max_map_queue_depth_observed,
        "memory_counters": {
            "active_block_count": mapper.active_block_count,
            "active_voxel_count": mapper.active_voxel_count,
            "allocated_voxel_count": mapper.allocated_voxel_count,
            "approximate_sparse_state_bytes": mapper.approximate_state_bytes,
            "mesh_payload_bytes": mesh_payload_bytes,
            "sparse_output_array_bytes": sparse_state_array_bytes(mapper, surface),
        },
        "mesh": mesh_status,
        "mesh_backend": mesh_status.get("mesh_backend"),
        "mesh_chunk_count": _status_int(mesh_status, "mesh_chunk_count"),
        "mesh_chunk_update_count": _status_int(mesh_status, "mesh_chunk_update_count"),
        "mesh_format": mesh_status.get("mesh_format"),
        "output": str(config.output),
        "pixel_stride": config.pixel_stride,
        "pose_update_count": state.pose_update_count,
        "recording": str(recording.root),
        "recording_source": {
            "source_dataset": str(recording.manifest["source_dataset"]),
            "source_sequence": str(recording.manifest["source_sequence"]),
        },
        "simulated_pacing": not config.wall_clock_pacing,
        "source_frame_ids_mapped": [observation.frame_id for observation in source_observations],
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "target_fps": config.target_fps,
        "total_triangle_count": _status_int(mesh_status, "total_triangle_count"),
        "total_vertex_count": _status_int(mesh_status, "total_vertex_count"),
        "truth_boundary": truth_boundary(
            measured_depth_used=state.measured_depth_used,
            measured_pose_used=state.measured_pose_used,
        ),
        "voxel_size_m": config.voxel_size_m,
    }


def sparse_metrics(
    config: LiveReplayConfig,
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    state: LiveReplayState,
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "dropped_keyframe_count": state.dropped_keyframe_count,
        "known_limitations": known_limitations(),
        "mapper_backend": config.mapper_backend,
        "metric_family": "phase6e_live_replay_sparse_tsdf_diagnostic",
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    }


def write_sparse_mesh_status(output: Path, surface: TSDFSurface) -> dict[str, object]:
    status = {
        "backend": CPU_SPARSE_BACKEND,
        "format_name": "atlas3r_phase6e_live_replay_mesh_status",
        "format_version": 1,
        "mesh_exported": False,
        "reason": "live replay currently exports observed sparse surface points, not triangles",
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }
    with (output / "mesh_status.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return status


def write_live_replay_report(path: Path, summary: dict[str, object]) -> None:
    truth = cast(dict[str, object], summary["truth_boundary"])
    latency = cast(dict[str, object], summary["latency"])
    map_latency = latency.get("map_update", {})
    mesh_latency = latency.get("mesh_update", {})
    combined_latency = latency.get("map_mesh_update", {})
    mesh = cast(dict[str, object], summary["mesh"])
    memory = cast(dict[str, object], summary["memory_counters"])
    ply_requested = summary["mesh_format"] in {"ply", "both"}
    target_line = (
        f"- Target FPS: `{summary['target_fps']}`; "
        f"simulated pacing: `{summary['simulated_pacing']}`"
    )
    frame_line = (
        f"- Frames seen/emitted: `{summary['frame_count_seen']}` / "
        f"`{summary['frame_count_emitted']}`"
    )
    drop_line = (
        f"- Dropped frames/keyframes: `{summary['dropped_frame_count']}` / "
        f"`{summary['dropped_keyframe_count']}`"
    )
    queue_line = (
        f"- Max capture/map queue depth: `{summary['max_capture_queue_depth_observed']}` / "
        f"`{summary['max_map_queue_depth_observed']}`"
    )
    lines = [
        "# Phase 6E Live Replay Report",
        "",
        f"- Replayed: `{summary['recording']}`",
        target_line,
        f"- Measured pose used: `{truth['measured_pose_used']}`",
        f"- Measured depth used: `{truth['measured_depth_used']}`",
        frame_line,
        f"- Pose updates: `{summary['pose_update_count']}`",
        f"- Keyframes selected: `{summary['keyframe_selected_count']}`",
        f"- Map updates: `{summary['map_update_count']}`",
        drop_line,
        queue_line,
        f"- Mapper: `{summary['mapper_backend']}` at voxel size `{summary['voxel_size_m']}` m",
        f"- Pixel stride: `{summary['pixel_stride']}`",
        f"- Surface points: `{summary['surface_point_count']}`",
        f"- Mesh chunks: `{summary['active_mesh_chunk_count']}` active, "
        f"`{summary['mesh_chunk_update_count']}` updates",
        f"- Mesh vertices/triangles: `{summary['total_vertex_count']}` / "
        f"`{summary['total_triangle_count']}`",
        f"- Mesh backend/format: `{summary['mesh_backend']}` / `{summary['mesh_format']}`",
        f"- Mesh payload bytes: `{memory['mesh_payload_bytes']}`",
        f"- Map update latency: `{map_latency}`",
        f"- Mesh update latency: `{mesh_latency}`",
        f"- Combined map+mesh latency: `{combined_latency}`",
        f"- NPZ mesh chunks loadable by contract: `{bool(mesh.get('mesh_exported'))}`",
        f"- PLY requested/written when format asks for it: `{ply_requested}`",
        "",
        "This is observed-only measured-depth/pose replay evidence and remains not "
        "live-mapping-ready evidence. It replays measured depth and measured pose through "
        "bounded queues, a CPU sparse TSDF mapper, and optional observed mesh chunk updates; "
        "no RGB-only student mapping, object-aware fusion, loop closure, hidden geometry "
        "completion, realtime claim, benchmark accuracy claim, or millimeter claim is made.",
        "",
        "Remaining bottlenecks: CPU sparse map-update latency, blocky fallback meshing quality, "
        "no real camera hardware exercise in CI, no object/map update removals, and no "
        "accelerated mapper.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def truth_boundary(*, measured_depth_used: bool, measured_pose_used: bool) -> dict[str, object]:
    return {
        "accuracy_report": False,
        "diagnostic_only": True,
        "hidden_geometry_measured": False,
        "measured_depth_used": measured_depth_used,
        "measured_pose_used": measured_pose_used,
        "observed_only": True,
        "performance_report": False,
        "predicted_completion": False,
        "realtime_claim": False,
        "rgb_only_mapping_ready": False,
        "student_rgb_only_used": False,
    }


def known_limitations() -> list[str]:
    return [
        "Replay uses measured recording pose when present; missing pose skips mapping.",
        "Replay uses measured recording depth when present; missing depth skips mapping.",
        "CPU sparse TSDF is a diagnostic mapper, not an accelerated realtime mapper.",
        "Hidden or unobserved geometry is not emitted as measured geometry.",
        "No object-aware fusion, loop closure, benchmark accuracy, "
        "or millimeter-level claim is made.",
    ]


def _int_summary(values: tuple[int, ...]) -> dict[str, float | int | None]:
    if not values:
        return {"max": None, "mean": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "max": int(np.max(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _status_int(status: dict[str, object], field_name: str) -> int:
    return int(cast(Any, status.get(field_name, 0)))


__all__ = [
    "extract_or_empty_surface",
    "known_limitations",
    "live_replay_summary",
    "sparse_metrics",
    "truth_boundary",
    "write_live_replay_report",
    "write_sparse_mesh_status",
]
