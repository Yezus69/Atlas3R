"""Incremental timing wrapper for measured recording CPU TSDF fusion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import TSDFSurface, TSDFVolume, extract_tsdf_surface
from atlas3r.mapping.mesh_extraction import export_tsdf_triangle_mesh
from atlas3r.mapping.mesh_sidecar import write_tsdf_surface_mesh_sidecar_from_artifacts
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.world_map_sidecar import write_tsdf_world_map_sidecar_from_artifacts
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import Atlas3RRecording, RecordingFrame, load_recording
from atlas3r.runtime.recording_fusion import (
    FuseRecordingConfig,
    _memory_report,
    _quality_report,
    _RuntimeEventRecorder,
    _summary,
    _tsdf_metrics,
    _with_recording_metadata,
    _write_map_preview,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


@dataclass(frozen=True)
class _MapUpdate:
    volume: TSDFVolume
    surface: TSDFSurface
    latency_ns: int


class IncrementalRecordingFusion:
    """Expose per-keyframe measured-input load and CPU TSDF update timings."""

    def __init__(self, config: FuseRecordingConfig) -> None:
        self._config = config

    def run(self) -> dict[str, object]:
        config = self._config
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
                "depth_source": config.depth_source,
                "mode": config.mode,
                "pose_source": config.pose_source,
                "recording": str(config.recording),
                "update_implementation": "cpu_tsdf_full_rebuild_per_selected_keyframe",
            },
        )
        observations: list[DepthObservation] = []
        per_frame_events: list[dict[str, object]] = []
        final_update: _MapUpdate | None = None
        for frame, selected in _selected_and_dropped_frames(recording, config):
            if not selected:
                per_frame_events.append(
                    _per_frame_event(
                        frame=frame,
                        selected_keyframe=False,
                        dropped_keyframe=True,
                        current_observation_count=len(observations),
                    )
                )
                runtime_events.emit(
                    "keyframe_dropped",
                    frame_id=frame.frame_id,
                    metadata={
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
                metadata={"mode": config.mode, "source": observation.source},
            )
            final_update = _rebuild_cpu_tsdf(config, recording, tuple(observations))
            recorder.add("per_frame_map_update", final_update.latency_ns)
            runtime_events.emit(
                "incremental_map_update",
                frame_id=observation.frame_id,
                latency_ns=final_update.latency_ns,
                metadata={
                    "current_observation_count": len(observations),
                    "mode": config.mode,
                    "surface_point_count": int(final_update.surface.points_world_m.shape[0]),
                    "update_implementation": "cpu_tsdf_full_rebuild_per_selected_keyframe",
                },
            )
            per_frame_events.append(
                _per_frame_event(
                    frame=frame,
                    selected_keyframe=True,
                    dropped_keyframe=False,
                    current_observation_count=len(observations),
                    observation_load_latency_ns=load_latency_ns,
                    map_update_latency_ns=final_update.latency_ns,
                    observation_array_bytes=_observation_array_bytes(tuple(observations)),
                    tsdf_array_bytes=_volume_array_bytes(final_update.volume),
                    surface_array_bytes=_surface_array_bytes(final_update.surface),
                    surface_point_count=int(final_update.surface.points_world_m.shape[0]),
                )
            )
        if final_update is None or not observations:
            raise ValueError("recording: no keyframes selected for incremental fusion")
        export_start_ns = time.perf_counter_ns()
        tsdf_dir = _write_final_tsdf_artifacts(config, output, final_update)
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
                "format_name": "atlas3r_phase6b_incremental_latency_report",
                "known_limitations": _incremental_limitations(),
                "mode": config.mode,
            }
        )
        memory_report = _memory_report(observation_tuple, final_update.surface, tsdf_dir)
        memory_report.update(
            {
                "known_limitations": [
                    *_string_list(memory_report.get("known_limitations", [])),
                    "Incremental mode currently rebuilds CPU TSDF state per selected keyframe.",
                ],
                "mode": config.mode,
                "peak_incremental_tsdf_array_bytes": max(
                    _event_int(event, "tsdf_array_bytes") for event in per_frame_events
                ),
            }
        )
        quality_report = _quality_report(observation_tuple, final_update.surface)
        quality_report.update(
            {"known_limitations": _incremental_limitations(), "mode": config.mode}
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
        summary["known_limitations"] = _incremental_limitations()
        summary["update_implementation"] = "cpu_tsdf_full_rebuild_per_selected_keyframe"
        write_json(output / "latency_report.json", latency_report)
        write_json(output / "memory_report.json", memory_report)
        write_json(output / "quality_report.json", quality_report)
        write_json(output / "summary.json", summary)
        write_jsonl(output / "per_frame_events.jsonl", per_frame_events)
        _write_map_preview(
            output / "map_preview.html", summary=summary, latency_report=latency_report
        )
        runtime_events.emit(
            "runtime_complete",
            paths={
                "per_frame_events": "per_frame_events.jsonl",
                "point_cloud": "" if point_cloud_path is None else point_cloud_path.name,
                "summary": "summary.json",
            },
            metadata={
                "mesh_exported": bool(mesh_status.get("mesh_exported", False)),
                "mode": config.mode,
                "surface_point_count": int(final_update.surface.points_world_m.shape[0]),
            },
        )
        write_jsonl(
            output / "runtime_events.jsonl",
            [event.to_json_record() for event in runtime_events.events],
        )
        return summary


def _selected_and_dropped_frames(
    recording: Atlas3RRecording,
    config: FuseRecordingConfig,
) -> tuple[tuple[RecordingFrame, bool], ...]:
    selected_count = 0
    records: list[tuple[RecordingFrame, bool]] = []
    for index, frame in enumerate(recording.frames):
        if config.max_frames is not None and selected_count >= config.max_frames:
            break
        selected = index % config.keyframe_stride == 0
        records.append((frame, selected))
        if selected:
            selected_count += 1
    return tuple(records)


def _rebuild_cpu_tsdf(
    config: FuseRecordingConfig,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
) -> _MapUpdate:
    start_ns = time.perf_counter_ns()
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
    surface = _with_recording_metadata(extract_tsdf_surface(volume), recording, observations)
    return _MapUpdate(volume=volume, surface=surface, latency_ns=time.perf_counter_ns() - start_ns)


def _write_final_tsdf_artifacts(
    config: FuseRecordingConfig,
    output: Path,
    update: _MapUpdate,
) -> Path:
    tsdf_dir = output / "tsdf"
    metrics = _tsdf_metrics(update.surface)
    metrics.update(
        {
            "known_limitations": _incremental_limitations(),
            "mode": config.mode,
            "update_implementation": "cpu_tsdf_full_rebuild_per_selected_keyframe",
        }
    )
    write_tsdf_outputs(tsdf_dir, update.volume, update.surface, metrics)
    write_tsdf_surface_mesh_sidecar_from_artifacts(
        tsdf_dir,
        chunk_id="phase6b_incremental_recording_cpu_tsdf_surface",
    )
    write_tsdf_world_map_sidecar_from_artifacts(tsdf_dir)
    return tsdf_dir


def _per_frame_event(
    *,
    frame: RecordingFrame,
    selected_keyframe: bool,
    dropped_keyframe: bool,
    current_observation_count: int,
    observation_load_latency_ns: int = 0,
    map_update_latency_ns: int = 0,
    observation_array_bytes: int = 0,
    tsdf_array_bytes: int = 0,
    surface_array_bytes: int = 0,
    surface_point_count: int = 0,
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "current_observation_count": current_observation_count,
        "dropped_keyframe": dropped_keyframe,
        "format_name": "atlas3r_phase6b_incremental_frame_event",
        "format_version": 1,
        "frame_id": frame.frame_id,
        "map_update_latency_ns": map_update_latency_ns,
        "mode": "incremental",
        "observation_array_bytes": observation_array_bytes,
        "observation_load_latency_ns": observation_load_latency_ns,
        "selected_keyframe": selected_keyframe,
        "surface_array_bytes": surface_array_bytes,
        "surface_point_count": surface_point_count,
        "timestamp_s": frame.timestamp_s,
        "tsdf_array_bytes": tsdf_array_bytes,
        "update_implementation": "cpu_tsdf_full_rebuild_per_selected_keyframe",
    }


def _observation_array_bytes(observations: tuple[DepthObservation, ...]) -> int:
    return int(
        sum(
            observation.depth_m.nbytes
            + observation.depth_sigma_m.nbytes
            + observation.confidence.nbytes
            for observation in observations
        )
    )


def _volume_array_bytes(volume: TSDFVolume) -> int:
    return int(volume.tsdf.nbytes + volume.weight.nbytes)


def _surface_array_bytes(surface: TSDFSurface) -> int:
    return int(
        surface.points_world_m.nbytes + surface.confidence.nbytes + surface.uncertainty_m.nbytes
    )


def _incremental_limitations() -> list[str]:
    return [
        "CPU TSDF is a diagnostic reference path and is not realtime.",
        "Incremental mode currently rebuilds the CPU TSDF from selected observations each step.",
        "Only measured depth and pose inputs are fused; RGB-only geometry is not inferred here.",
        "Hidden or unobserved geometry is not emitted as measured geometry.",
        "No object-aware fusion is performed.",
    ]


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _event_int(event: dict[str, object], key: str) -> int:
    value = event.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = ["IncrementalRecordingFusion"]
