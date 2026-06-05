"""Phase 6B-compatible rebuild-per-keyframe incremental backend."""

from __future__ import annotations

import time
from typing import cast

from atlas3r.mapping.mesh_extraction import export_tsdf_triangle_mesh
from atlas3r.mapping.observations import DepthObservation
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import load_recording
from atlas3r.runtime.recording_fusion import (
    FuseRecordingConfig,
    _memory_report,
    _quality_report,
    _RuntimeEventRecorder,
    _summary,
    _write_map_preview,
)
from atlas3r.runtime.recording_fusion_incremental_helpers import (
    CPU_REBUILD_UPDATE_IMPLEMENTATION,
    _event_int,
    _incremental_limitations,
    _observation_array_bytes,
    _observed_voxel_count,
    _per_frame_event,
    _rebuild_cpu_tsdf,
    _selected_and_dropped_frames,
    _surface_array_bytes,
    _volume_array_bytes,
    _write_final_tsdf_artifacts,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


def run_cpu_rebuild_incremental(config: FuseRecordingConfig, backend: str) -> dict[str, object]:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    recording = load_recording(config.recording)
    recorder = LatencyRecorder()
    runtime_events = _RuntimeEventRecorder(
        configured_frame_array_bound=max(1, config.keyframe_stride)
    )
    run_start_ns = time.perf_counter_ns()
    runtime_events.emit(
        "runtime_start",
        metadata={
            **DIAGNOSTIC_TRUTH_FLAGS,
            "backend": backend,
            "depth_source": config.depth_source,
            "mode": config.mode,
            "pose_source": config.pose_source,
            "recording": str(config.recording),
            "update_implementation": CPU_REBUILD_UPDATE_IMPLEMENTATION,
        },
    )
    observations: list[DepthObservation] = []
    per_frame_events: list[dict[str, object]] = []
    final_update = None
    for frame, selected in _selected_and_dropped_frames(recording, config):
        if not selected:
            per_frame_events.append(
                _per_frame_event(
                    backend=backend,
                    frame=frame,
                    selected_keyframe=False,
                    dropped_keyframe=True,
                    current_observation_count=len(observations),
                    observed_voxel_count=0
                    if final_update is None
                    else _observed_voxel_count(final_update.volume),
                    update_implementation=CPU_REBUILD_UPDATE_IMPLEMENTATION,
                )
            )
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
        final_update = _rebuild_cpu_tsdf(config, recording, tuple(observations), backend)
        recorder.add("per_frame_map_update", final_update.latency_ns)
        observed_voxel_count = _observed_voxel_count(final_update.volume)
        runtime_events.emit(
            "incremental_map_update",
            frame_id=observation.frame_id,
            latency_ns=final_update.latency_ns,
            metadata={
                "backend": backend,
                "current_observation_count": len(observations),
                "mode": config.mode,
                "observed_voxel_count": observed_voxel_count,
                "surface_point_count": int(final_update.surface.points_world_m.shape[0]),
                "update_implementation": CPU_REBUILD_UPDATE_IMPLEMENTATION,
            },
        )
        per_frame_events.append(
            _per_frame_event(
                backend=backend,
                frame=frame,
                selected_keyframe=True,
                dropped_keyframe=False,
                current_observation_count=len(observations),
                observed_voxel_count=observed_voxel_count,
                observation_load_latency_ns=load_latency_ns,
                map_update_latency_ns=final_update.latency_ns,
                observation_array_bytes=_observation_array_bytes(tuple(observations)),
                tsdf_array_bytes=_volume_array_bytes(final_update.volume),
                surface_array_bytes=_surface_array_bytes(final_update.surface),
                surface_point_count=int(final_update.surface.points_world_m.shape[0]),
                update_implementation=CPU_REBUILD_UPDATE_IMPLEMENTATION,
            )
        )
    if final_update is None or not observations:
        raise ValueError("recording: no keyframes selected for incremental fusion")

    export_start_ns = time.perf_counter_ns()
    tsdf_dir = _write_final_tsdf_artifacts(
        config,
        output,
        final_update,
        backend=backend,
        update_implementation=CPU_REBUILD_UPDATE_IMPLEMENTATION,
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(
            output / "surface_points.ply", final_update.surface
        )
    mesh_status = export_tsdf_triangle_mesh(
        output,
        volume=final_update.volume,
        surface=final_update.surface,
        export_mode=config.export_mesh,
    )
    recorder.add("geometry_export", time.perf_counter_ns() - export_start_ns)
    recorder.add("total_pipeline", time.perf_counter_ns() - run_start_ns)
    observation_tuple = tuple(observations)
    latency_report = recorder.report()
    latency_report.update(
        {
            "backend": backend,
            "format_name": "atlas3r_phase6b_incremental_latency_report",
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "update_implementation": CPU_REBUILD_UPDATE_IMPLEMENTATION,
        }
    )
    memory_report = _memory_report(observation_tuple, final_update.surface, tsdf_dir)
    memory_report.update(
        {
            "backend": backend,
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "peak_incremental_tsdf_array_bytes": max(
                _event_int(event, "tsdf_array_bytes") for event in per_frame_events
            ),
            "update_implementation": CPU_REBUILD_UPDATE_IMPLEMENTATION,
        }
    )
    quality_report = _quality_report(observation_tuple, final_update.surface)
    quality_report.update(
        {"backend": backend, "known_limitations": _incremental_limitations(backend)}
    )
    summary = _summary(
        config=config,
        recording=recording,
        observations=observation_tuple,
        surface=final_update.surface,
        tsdf_dir=tsdf_dir,
        point_cloud_path=point_cloud_path,
        mesh_status=mesh_status,
        latency_report=latency_report,
        memory_report=memory_report,
        quality_report=quality_report,
    )
    cast(dict[str, object], summary["artifacts"])["per_frame_events"] = "per_frame_events.jsonl"
    summary["backend"] = backend
    summary["known_limitations"] = _incremental_limitations(backend)
    summary["update_implementation"] = CPU_REBUILD_UPDATE_IMPLEMENTATION
    write_json(output / "latency_report.json", latency_report)
    write_json(output / "memory_report.json", memory_report)
    write_json(output / "quality_report.json", quality_report)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "per_frame_events.jsonl", per_frame_events)
    _write_map_preview(output / "map_preview.html", summary=summary, latency_report=latency_report)
    runtime_events.emit(
        "runtime_complete",
        paths={
            "per_frame_events": "per_frame_events.jsonl",
            "point_cloud": "" if point_cloud_path is None else point_cloud_path.name,
            "summary": "summary.json",
        },
        metadata={
            "backend": backend,
            "mesh_exported": bool(mesh_status.get("mesh_exported", False)),
            "mode": config.mode,
            "surface_point_count": int(final_update.surface.points_world_m.shape[0]),
        },
    )
    write_jsonl(
        output / "runtime_events.jsonl", [event.to_json_record() for event in runtime_events.events]
    )
    return summary


__all__ = ["run_cpu_rebuild_incremental"]
