"""Persistent measured-recording incremental TSDF backend."""

from __future__ import annotations

import time
from typing import cast

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import grid_bounds_from_observations
from atlas3r.mapping.cpu_tsdf import extract_tsdf_surface
from atlas3r.mapping.incremental_tsdf import (
    PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
    IncrementalTSDFConfig,
    PersistentIncrementalTSDFMapper,
)
from atlas3r.mapping.mesh_extraction import export_tsdf_triangle_mesh
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
    _compare_persistent_to_batch_cpu_tsdf,
    _incremental_limitations,
    _MapUpdate,
    _per_frame_event,
    _preload_selected_observations,
    _selected_and_dropped_frames,
    _single_observation_array_bytes,
    _unique_or_mixed,
    _with_incremental_metadata,
    _write_final_tsdf_artifacts,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


def run_cpu_persistent_incremental(
    config: FuseRecordingConfig,
    backend: str,
) -> dict[str, object]:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    recording = load_recording(config.recording)
    selected_records = _selected_and_dropped_frames(recording, config)
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
            "fixed_bounds_precomputed_offline": True,
            "mode": config.mode,
            "pose_source": config.pose_source,
            "recording": str(config.recording),
            "update_implementation": PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
        },
    )

    preloaded = _preload_selected_observations(
        recording=recording,
        selected_records=selected_records,
        config=config,
        recorder=recorder,
    )
    if not preloaded:
        raise ValueError("recording: no keyframes selected for incremental fusion")
    observations = tuple(record.observation for record in preloaded)

    bounds_start_ns = time.perf_counter_ns()
    truncation_distance_m = config.voxel_size_m * config.truncation_voxels
    grid_min, grid_max = grid_bounds_from_observations(
        observations,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    bounds_latency_ns = time.perf_counter_ns() - bounds_start_ns
    recorder.add("fixed_grid_bounds_precompute", bounds_latency_ns)
    runtime_events.emit(
        "fixed_grid_bounds_precompute",
        latency_ns=bounds_latency_ns,
        metadata={
            "backend": backend,
            "selected_observation_count": len(observations),
            "truncation_distance_m": truncation_distance_m,
            "voxel_size_m": config.voxel_size_m,
        },
    )

    mapper_start_ns = time.perf_counter_ns()
    mapper = PersistentIncrementalTSDFMapper(
        IncrementalTSDFConfig(
            grid_min_world_m=grid_min,
            grid_max_world_m=grid_max,
            voxel_size_m=config.voxel_size_m,
            truncation_distance_m=truncation_distance_m,
            coordinate_frame=_unique_or_mixed(
                [
                    str(
                        observation.pose.diagnostics.get(
                            "coordinate_frame",
                            recording.manifest["coordinate_frame"],
                        )
                    )
                    for observation in observations
                ]
            ),
            metric_scale_source=_unique_or_mixed(
                [observation.pose.scale_source for observation in observations]
            ),
        )
    )
    mapper_init_latency_ns = time.perf_counter_ns() - mapper_start_ns
    recorder.add("persistent_tsdf_initialization", mapper_init_latency_ns)
    runtime_events.emit(
        "persistent_tsdf_initialization",
        latency_ns=mapper_init_latency_ns,
        metadata={
            "backend": backend,
            "centers_array_bytes": mapper.centers_array_bytes,
            "shape_xyz": list(mapper.shape_xyz),
            "tsdf_array_bytes": mapper.tsdf_array_bytes,
            "voxel_count": mapper.voxel_count,
        },
    )

    per_frame_events: list[dict[str, object]] = []
    preloaded_index = 0
    current_observation_count = 0
    current_observed_voxel_count = 0
    for frame, selected in selected_records:
        if not selected:
            per_frame_events.append(
                _per_frame_event(
                    backend=backend,
                    frame=frame,
                    selected_keyframe=False,
                    dropped_keyframe=True,
                    current_observation_count=current_observation_count,
                    observed_voxel_count=current_observed_voxel_count,
                    update_implementation=PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
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

        preloaded_record = preloaded[preloaded_index]
        preloaded_index += 1
        observation = preloaded_record.observation
        runtime_events.emit(
            "depth_observation",
            frame_id=observation.frame_id,
            latency_ns=preloaded_record.load_latency_ns,
            frame_arrays_in_memory=1,
            count_processed_frame=True,
            metadata={
                "backend": backend,
                "loaded_during_fixed_bounds_precompute": True,
                "mode": config.mode,
                "source": observation.source,
            },
        )
        update_start_ns = time.perf_counter_ns()
        update_stats = mapper.integrate(observation)
        update_latency_ns = time.perf_counter_ns() - update_start_ns
        recorder.add("per_frame_map_update", update_latency_ns)
        current_observation_count = update_stats.integrated_observation_count
        current_observed_voxel_count = update_stats.observed_voxel_count
        runtime_events.emit(
            "incremental_map_update",
            frame_id=observation.frame_id,
            latency_ns=update_latency_ns,
            metadata={
                "backend": backend,
                "current_observation_count": current_observation_count,
                "mode": config.mode,
                "observed_voxel_count": current_observed_voxel_count,
                "surface_point_count": None,
                "update_implementation": update_stats.update_implementation,
            },
        )
        per_frame_events.append(
            _per_frame_event(
                backend=backend,
                frame=frame,
                selected_keyframe=True,
                dropped_keyframe=False,
                current_observation_count=current_observation_count,
                observed_voxel_count=current_observed_voxel_count,
                observation_load_latency_ns=preloaded_record.load_latency_ns,
                map_update_latency_ns=update_latency_ns,
                observation_array_bytes=_single_observation_array_bytes(observation),
                tsdf_array_bytes=update_stats.tsdf_array_bytes,
                centers_array_bytes=update_stats.centers_array_bytes,
                surface_array_bytes=0,
                surface_point_count=None,
                update_implementation=update_stats.update_implementation,
            )
        )

    surface_start_ns = time.perf_counter_ns()
    final_volume = mapper.volume()
    final_surface = _with_incremental_metadata(
        extract_tsdf_surface(final_volume),
        recording=recording,
        observations=observations,
        backend=backend,
        update_implementation=PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
    )
    recorder.add("final_surface_extraction", time.perf_counter_ns() - surface_start_ns)

    comparison_start_ns = time.perf_counter_ns()
    backend_comparison = _compare_persistent_to_batch_cpu_tsdf(
        backend=backend,
        observations=observations,
        persistent_volume=final_volume,
        persistent_surface=final_surface,
        grid_min_world_m=grid_min,
        grid_max_world_m=grid_max,
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=truncation_distance_m,
    )
    recorder.add("backend_comparison", time.perf_counter_ns() - comparison_start_ns)

    final_update = _MapUpdate(volume=final_volume, surface=final_surface, latency_ns=0)
    export_start_ns = time.perf_counter_ns()
    tsdf_dir = _write_final_tsdf_artifacts(
        config,
        output,
        final_update,
        backend=backend,
        update_implementation=PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(output / "surface_points.ply", final_surface)
    mesh_status = export_tsdf_triangle_mesh(
        output,
        volume=final_volume,
        surface=final_surface,
        export_mode=config.export_mesh,
    )
    recorder.add("geometry_export", time.perf_counter_ns() - export_start_ns)
    recorder.add("total_pipeline", time.perf_counter_ns() - run_start_ns)

    latency_report = recorder.report()
    latency_report.update(
        {
            "backend": backend,
            "fixed_bounds_precomputed_offline": True,
            "format_name": "atlas3r_phase6c_incremental_latency_report",
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "update_implementation": PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
        }
    )
    memory_report = _memory_report(observations, final_surface, tsdf_dir)
    memory_report.update(
        {
            "backend": backend,
            "centers_array_bytes": mapper.centers_array_bytes,
            "known_limitations": _incremental_limitations(backend),
            "mode": config.mode,
            "persistent_tsdf_array_bytes": mapper.tsdf_array_bytes,
            "peak_incremental_tsdf_array_bytes": mapper.tsdf_array_bytes,
            "update_implementation": PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
        }
    )
    quality_report = _quality_report(observations, final_surface)
    quality_report.update(
        {"backend": backend, "known_limitations": _incremental_limitations(backend)}
    )
    summary = _summary(
        config=config,
        recording=recording,
        observations=observations,
        surface=final_surface,
        tsdf_dir=tsdf_dir,
        point_cloud_path=point_cloud_path,
        mesh_status=mesh_status,
        latency_report=latency_report,
        memory_report=memory_report,
        quality_report=quality_report,
    )
    artifacts = cast(dict[str, object], summary["artifacts"])
    artifacts["backend_comparison"] = "backend_comparison.json"
    artifacts["per_frame_events"] = "per_frame_events.jsonl"
    summary.update(
        {
            "backend": backend,
            "fixed_bounds_precomputed_offline": True,
            "format_name": "atlas3r_phase6c_fuse_recording_summary",
            "known_limitations": _incremental_limitations(backend),
            "update_implementation": PERSISTENT_CPU_UPDATE_IMPLEMENTATION,
        }
    )
    write_json(output / "backend_comparison.json", backend_comparison)
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


__all__ = ["run_cpu_persistent_incremental"]
