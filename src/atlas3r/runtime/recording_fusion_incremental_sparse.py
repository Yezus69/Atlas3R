"""Sparse block measured-recording incremental TSDF backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import (
    SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    SparseBlockTSDFMapper,
    SparseTSDFConfig,
)
from atlas3r.mapping.sparse_tsdf_artifacts import (
    sparse_state_array_bytes,
    write_sparse_tsdf_outputs,
)
from atlas3r.mapping.sparse_tsdf_compare import compare_sparse_to_dense_persistent
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import Atlas3RRecording, RecordingFrame, load_recording
from atlas3r.runtime.recording_fusion import (
    FuseRecordingConfig,
    _quality_report,
    _RuntimeEventRecorder,
    _summary,
    _write_map_preview,
)
from atlas3r.runtime.recording_fusion_incremental_helpers import (
    CPU_SPARSE_BACKEND,
    _incremental_limitations,
    _selected_and_dropped_frames,
    _single_observation_array_bytes,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


def run_cpu_sparse_incremental(config: FuseRecordingConfig, backend: str) -> dict[str, object]:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    recording = load_recording(config.recording)
    recorder = LatencyRecorder()
    runtime_events = _RuntimeEventRecorder(
        configured_frame_array_bound=max(1, config.keyframe_stride)
    )
    run_start_ns = time.perf_counter_ns()
    truncation_distance_m = config.voxel_size_m * config.truncation_voxels
    mapper = SparseBlockTSDFMapper(
        SparseTSDFConfig(
            voxel_size_m=config.voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            coordinate_frame=str(recording.manifest["coordinate_frame"]),
            metric_scale_source="external_pose",
        )
    )
    runtime_events.emit(
        "runtime_start",
        metadata={
            **DIAGNOSTIC_TRUTH_FLAGS,
            "backend": backend,
            "depth_source": config.depth_source,
            "fixed_bounds_precomputed_offline": False,
            "mode": config.mode,
            "pose_source": config.pose_source,
            "recording": str(config.recording),
            "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        },
    )

    observations: list[DepthObservation] = []
    per_frame_events: list[dict[str, object]] = []
    sparse_update_latencies_ns: list[int] = []
    for frame, selected in _selected_and_dropped_frames(recording, config):
        if not selected:
            per_frame_events.append(_sparse_per_frame_event(mapper, backend, frame, selected=False))
            runtime_events.emit(
                "keyframe_dropped",
                frame_id=frame.frame_id,
                metadata={
                    "backend": backend,
                    "mode": config.mode,
                    "reason": "keyframe_stride",
                    "selected_keyframe": False,
                },
            )
            continue

        load_start_ns = time.perf_counter_ns()
        observation = observation_from_recording_frame(
            recording,
            frame,
            depth_sigma_floor_m=max(0.01, config.voxel_size_m * 0.25),
        )
        load_latency_ns = time.perf_counter_ns() - load_start_ns
        recorder.add("per_frame_observation_load", load_latency_ns)
        observations.append(observation)
        runtime_events.emit(
            "depth_observation",
            frame_id=observation.frame_id,
            latency_ns=load_latency_ns,
            frame_arrays_in_memory=1,
            count_processed_frame=True,
            metadata={"backend": backend, "mode": config.mode, "source": observation.source},
        )

        update_start_ns = time.perf_counter_ns()
        update_stats = mapper.integrate(observation)
        update_latency_ns = time.perf_counter_ns() - update_start_ns
        sparse_update_latencies_ns.append(update_latency_ns)
        recorder.add("per_frame_map_update", update_latency_ns)
        runtime_events.emit(
            "incremental_map_update",
            frame_id=observation.frame_id,
            latency_ns=update_latency_ns,
            metadata={
                "active_block_count": update_stats.active_block_count,
                "active_voxel_count": update_stats.active_voxel_count,
                "approximate_state_bytes": update_stats.approximate_state_bytes,
                "backend": backend,
                "current_observation_count": update_stats.integrated_observation_count,
                "mode": config.mode,
                "new_voxel_count": update_stats.new_voxel_count,
                "skipped_duplicate_frame": update_stats.skipped_duplicate_frame,
                "surface_point_count": None,
                "update_implementation": update_stats.update_implementation,
                "updated_voxel_count": update_stats.updated_voxel_count,
            },
        )
        per_frame_events.append(
            _sparse_per_frame_event(
                mapper,
                backend,
                frame,
                selected=True,
                observation_load_latency_ns=load_latency_ns,
                map_update_latency_ns=update_latency_ns,
                observation_array_bytes=_single_observation_array_bytes(observation),
                valid_depth_sample_count=update_stats.valid_depth_sample_count,
                candidate_voxel_count=update_stats.candidate_voxel_count,
                new_voxel_count=update_stats.new_voxel_count,
                updated_voxel_count=update_stats.updated_voxel_count,
                skipped_duplicate_frame=update_stats.skipped_duplicate_frame,
                current_observation_count=update_stats.integrated_observation_count,
            )
        )
    if not observations:
        raise ValueError("recording: no keyframes selected for incremental fusion")
    observation_tuple = tuple(observations)

    surface_start_ns = time.perf_counter_ns()
    final_surface = _with_sparse_metadata(mapper.extract_surface(), recording=recording)
    recorder.add("final_surface_extraction", time.perf_counter_ns() - surface_start_ns)
    sparse_state_bytes = mapper.approximate_state_bytes

    comparison_start_ns = time.perf_counter_ns()
    comparison = compare_sparse_to_dense_persistent(
        observations=observation_tuple,
        sparse_surface=final_surface,
        sparse_state_bytes=sparse_state_bytes,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
        sparse_update_latencies_ns=tuple(sparse_update_latencies_ns),
        coordinate_frame=str(final_surface.metadata["coordinate_frame"]),
        metric_scale_source=str(final_surface.metadata["metric_scale_source"]),
    ).comparison
    recorder.add("backend_comparison", time.perf_counter_ns() - comparison_start_ns)

    export_start_ns = time.perf_counter_ns()
    sparse_dir = output / "sparse_tsdf"
    write_sparse_tsdf_outputs(
        sparse_dir,
        mapper=mapper,
        surface=final_surface,
        metrics=_sparse_metrics(final_surface, backend),
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(output / "surface_points.ply", final_surface)
    mesh_status = _write_sparse_mesh_status(output, config.export_mesh, final_surface)
    recorder.add("geometry_export", time.perf_counter_ns() - export_start_ns)
    recorder.add("total_pipeline", time.perf_counter_ns() - run_start_ns)

    latency_report = recorder.report()
    latency_report.update(
        {
            "backend": backend,
            "fixed_bounds_precomputed_offline": False,
            "format_name": "atlas3r_phase6d_sparse_incremental_latency_report",
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        }
    )
    memory_report = _sparse_memory_report(
        observations=observation_tuple,
        mapper=mapper,
        surface=final_surface,
        sparse_output_array_bytes=sparse_state_array_bytes(mapper, final_surface),
        comparison=comparison,
    )
    quality_report = _quality_report(observation_tuple, final_surface)
    quality_report.update(
        {"backend": backend, "known_limitations": _incremental_limitations(backend)}
    )
    summary = _summary(
        config=config,
        recording=recording,
        observations=observation_tuple,
        surface=final_surface,
        tsdf_dir=sparse_dir,
        point_cloud_path=point_cloud_path,
        mesh_status=mesh_status,
        latency_report=latency_report,
        memory_report=memory_report,
        quality_report=quality_report,
    )
    cast(dict[str, object], summary["artifacts"]).update(
        {
            "backend_comparison": "backend_comparison.json",
            "per_frame_events": "per_frame_events.jsonl",
            "sparse_tsdf": "sparse_tsdf",
        }
    )
    summary.update(
        {
            "backend": backend,
            "cpu_tsdf_note": "Sparse block TSDF is a diagnostic CPU mapper, not a realtime claim.",
            "fixed_bounds_precomputed_offline": False,
            "format_name": "atlas3r_phase6d_sparse_fuse_recording_summary",
            "known_limitations": _incremental_limitations(backend),
            "sparse_tsdf": {
                "active_block_count": mapper.active_block_count,
                "active_voxel_count": mapper.active_voxel_count,
                "approximate_state_bytes": mapper.approximate_state_bytes,
                "block_size_voxels": final_surface.metadata["block_size_voxels"],
                "pixel_stride": final_surface.metadata["pixel_stride"],
            },
            "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        }
    )
    write_json(output / "backend_comparison.json", comparison)
    write_json(output / "latency_report.json", latency_report)
    write_json(output / "memory_report.json", memory_report)
    write_json(output / "quality_report.json", quality_report)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "per_frame_events.jsonl", per_frame_events)
    _write_map_preview(output / "map_preview.html", summary=summary, latency_report=latency_report)
    runtime_events.emit(
        "runtime_complete",
        paths={
            "backend_comparison": "backend_comparison.json",
            "per_frame_events": "per_frame_events.jsonl",
            "point_cloud": "" if point_cloud_path is None else point_cloud_path.name,
            "summary": "summary.json",
        },
        metadata={
            "active_block_count": mapper.active_block_count,
            "active_voxel_count": mapper.active_voxel_count,
            "backend": backend,
            "mesh_exported": bool(mesh_status.get("mesh_exported", False)),
            "mode": config.mode,
            "surface_point_count": int(final_surface.points_world_m.shape[0]),
        },
    )
    write_jsonl(
        output / "runtime_events.jsonl", [event.to_json_record() for event in runtime_events.events]
    )
    return summary


def _sparse_per_frame_event(
    mapper: SparseBlockTSDFMapper,
    backend: str,
    frame: RecordingFrame,
    *,
    selected: bool,
    observation_load_latency_ns: int = 0,
    map_update_latency_ns: int = 0,
    observation_array_bytes: int = 0,
    valid_depth_sample_count: int = 0,
    candidate_voxel_count: int = 0,
    new_voxel_count: int = 0,
    updated_voxel_count: int = 0,
    skipped_duplicate_frame: bool = False,
    current_observation_count: int = 0,
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "approximate_state_bytes": mapper.approximate_state_bytes,
        "backend": backend,
        "candidate_voxel_count": candidate_voxel_count,
        "current_observation_count": current_observation_count,
        "dropped_frame": not selected,
        "dropped_keyframe": not selected,
        "format_name": "atlas3r_phase6d_sparse_incremental_frame_event",
        "format_version": 1,
        "frame_id": frame.frame_id,
        "map_update_latency_ns": map_update_latency_ns,
        "mode": "incremental",
        "new_voxel_count": new_voxel_count,
        "observation_array_bytes": observation_array_bytes,
        "observation_load_latency_ns": observation_load_latency_ns,
        "selected_keyframe": selected,
        "skipped_duplicate_frame": skipped_duplicate_frame,
        "surface_point_count": None,
        "timestamp_s": frame.timestamp_s,
        "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        "updated_voxel_count": updated_voxel_count,
        "valid_depth_sample_count": valid_depth_sample_count,
    }


def _with_sparse_metadata(
    surface: TSDFSurface,
    *,
    recording: Atlas3RRecording,
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            **DIAGNOSTIC_TRUTH_FLAGS,
            "accuracy_note": "Sparse measured recording fusion is diagnostic, not a benchmark.",
            "backend": CPU_SPARSE_BACKEND,
            "recording": str(recording.root),
            "truth_boundary": dict(cast(dict[str, object], recording.manifest["truth_boundary"])),
            "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
        }
    )
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def _sparse_metrics(surface: TSDFSurface, backend: str) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": surface.metadata["active_block_count"],
        "active_voxel_count": surface.metadata["active_voxel_count"],
        "backend": backend,
        "known_limitations": _incremental_limitations(backend),
        "metric_family": "phase6d_sparse_block_tsdf_recording_diagnostic",
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    }


def _sparse_memory_report(
    *,
    observations: tuple[DepthObservation, ...],
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    sparse_output_array_bytes: int,
    comparison: dict[str, object],
) -> dict[str, object]:
    observation_bytes = sum(
        int(
            observation.depth_m.nbytes
            + observation.depth_sigma_m.nbytes
            + observation.confidence.nbytes
        )
        for observation in observations
    )
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "active_block_count": mapper.active_block_count,
        "active_voxel_count": mapper.active_voxel_count,
        "allocated_voxel_count": mapper.allocated_voxel_count,
        "approximate_sparse_state_bytes": mapper.approximate_state_bytes,
        "backend": CPU_SPARSE_BACKEND,
        "dense_persistent_state_bytes": comparison["dense_persistent_state_bytes"],
        "format_name": "atlas3r_phase6d_sparse_memory_report",
        "format_version": 1,
        "known_limitations": _incremental_limitations(CPU_SPARSE_BACKEND),
        "mode": "incremental",
        "observation_array_bytes": int(observation_bytes),
        "sparse_output_array_bytes": sparse_output_array_bytes,
        "sparse_to_dense_state_memory_ratio": comparison["sparse_to_dense_state_memory_ratio"],
        "surface_array_bytes": int(
            surface.points_world_m.nbytes + surface.confidence.nbytes + surface.uncertainty_m.nbytes
        ),
        "update_implementation": SPARSE_TSDF_UPDATE_IMPLEMENTATION,
    }


def _write_sparse_mesh_status(
    output: Path,
    export_mode: str,
    surface: TSDFSurface,
) -> dict[str, object]:
    if export_mode == "required":
        raise ValueError("cpu-sparse mesh export: triangle mesh export is not available yet")
    status = {
        "backend": CPU_SPARSE_BACKEND,
        "export_mode": export_mode,
        "format_name": "atlas3r_phase6d_sparse_mesh_status",
        "format_version": 1,
        "mesh_exported": False,
        "reason": "sparse backend currently exports observed surface points, not triangles",
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }
    with (output / "mesh_status.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return status


__all__ = ["run_cpu_sparse_incremental"]
