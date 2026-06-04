"""Measured recording-to-CPU-TSDF product-slice runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

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
from atlas3r.recording.observations import observations_from_recording
from atlas3r.recording.schema import Atlas3RRecording, load_recording
from atlas3r.runtime.events import BoundedMemoryCounters, RuntimeEvent
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


@dataclass(frozen=True)
class FuseRecordingConfig:
    recording: Path
    output: Path
    pose_source: str = "recording"
    depth_source: str = "recording"
    max_frames: int | None = None
    keyframe_stride: int = 1
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0
    export_point_cloud: bool = False
    export_mesh: str = "auto"


def run_fuse_recording(config: FuseRecordingConfig) -> dict[str, object]:
    """Fuse measured recording depth+pose through the existing CPU TSDF path."""

    _validate_config(config)
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    recording = load_recording(config.recording)
    recorder = LatencyRecorder()
    events = _RuntimeEventRecorder(configured_frame_array_bound=max(1, config.keyframe_stride))
    run_start_ns = time.perf_counter_ns()
    events.emit(
        "runtime_start",
        metadata={
            **DIAGNOSTIC_TRUTH_FLAGS,
            "depth_source": config.depth_source,
            "pose_source": config.pose_source,
            "recording": str(config.recording),
        },
    )

    observation_start_ns = time.perf_counter_ns()
    observations = observations_from_recording(
        recording,
        max_frames=config.max_frames,
        keyframe_stride=config.keyframe_stride,
        depth_sigma_floor_m=max(0.01, config.voxel_size_m * 0.25),
    )
    recorder.add("recording_observation_load", time.perf_counter_ns() - observation_start_ns)
    for observation in observations:
        events.emit(
            "depth_observation",
            frame_id=observation.frame_id,
            frame_arrays_in_memory=1,
            count_processed_frame=True,
            metadata={"source": observation.source},
        )

    surface, tsdf_dir, tsdf_latency_ns = _write_tsdf_outputs(
        config, output, recording, observations
    )
    recorder.add("cpu_tsdf_mapping", tsdf_latency_ns)
    events.emit(
        "cpu_tsdf_mapping",
        latency_ns=tsdf_latency_ns,
        paths={"tsdf": "tsdf"},
        metadata={"surface_point_count": int(surface.points_world_m.shape[0])},
    )

    export_start_ns = time.perf_counter_ns()
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(output / "surface_points.ply", surface)
    mesh_status = export_tsdf_triangle_mesh(
        output,
        volume=_load_volume_from_tsdf_dir(tsdf_dir, observations),
        surface=surface,
        export_mode=config.export_mesh,
    )
    recorder.add("geometry_export", time.perf_counter_ns() - export_start_ns)
    recorder.add("total_pipeline", time.perf_counter_ns() - run_start_ns)

    latency_report = recorder.report()
    latency_report["format_name"] = "atlas3r_phase6a_latency_report"
    memory_report = _memory_report(observations, surface, tsdf_dir)
    quality_report = _quality_report(observations, surface)
    summary = _summary(
        config=config,
        recording=recording,
        observations=observations,
        surface=surface,
        tsdf_dir=tsdf_dir,
        point_cloud_path=point_cloud_path,
        mesh_status=mesh_status,
        latency_report=latency_report,
        memory_report=memory_report,
        quality_report=quality_report,
    )
    write_json(output / "latency_report.json", latency_report)
    write_json(output / "memory_report.json", memory_report)
    write_json(output / "quality_report.json", quality_report)
    write_json(output / "summary.json", summary)
    _write_map_preview(output / "map_preview.html", summary=summary, latency_report=latency_report)
    events.emit(
        "runtime_complete",
        paths={
            "point_cloud": "" if point_cloud_path is None else point_cloud_path.name,
            "summary": "summary.json",
        },
        metadata={
            "mesh_exported": bool(mesh_status.get("mesh_exported", False)),
            "surface_point_count": int(surface.points_world_m.shape[0]),
        },
    )
    write_jsonl(
        output / "runtime_events.jsonl", [event.to_json_record() for event in events.events]
    )
    return summary


def _write_tsdf_outputs(
    config: FuseRecordingConfig,
    output: Path,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
) -> tuple[TSDFSurface, Path, int]:
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
    tsdf_dir = output / "tsdf"
    write_tsdf_outputs(tsdf_dir, volume, surface, _tsdf_metrics(surface))
    write_tsdf_surface_mesh_sidecar_from_artifacts(
        tsdf_dir,
        chunk_id="phase6a_recording_cpu_tsdf_surface",
    )
    write_tsdf_world_map_sidecar_from_artifacts(tsdf_dir)
    return surface, tsdf_dir, time.perf_counter_ns() - start_ns


def _load_volume_from_tsdf_dir(
    tsdf_dir: Path,
    observations: tuple[DepthObservation, ...],
) -> TSDFVolume:
    with np.load(tsdf_dir / "tsdf_grid.npz", allow_pickle=False) as payload:
        return TSDFVolume(
            grid_min_corner_world_m=np.asarray(
                payload["grid_min_corner_world_m"], dtype=np.float32
            ),
            voxel_size_m=float(np.asarray(payload["voxel_size_m"]).item()),
            truncation_distance_m=float(np.asarray(payload["truncation_distance_m"]).item()),
            tsdf=np.asarray(payload["tsdf"], dtype=np.float32),
            weight=np.asarray(payload["weight"], dtype=np.float32),
            source_frame_ids=tuple(observation.frame_id for observation in observations),
            coordinate_frame="x_right_y_down_z_forward",
            metric_scale_source="external_pose",
        )


def _with_recording_metadata(
    surface: TSDFSurface,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
) -> TSDFSurface:
    metadata = dict(surface.metadata)
    metadata.update(
        {
            **DIAGNOSTIC_TRUTH_FLAGS,
            "accuracy_note": "Phase 6A measured recording fusion is diagnostic, not a benchmark.",
            "artifact_type": "phase6a_recording_cpu_tsdf_surface_points",
            "recording": str(recording.root),
            "truth_boundary": dict(cast(dict[str, object], recording.manifest["truth_boundary"])),
        }
    )
    metadata["flags"] = _stable_strings(
        [
            *[str(flag) for flag in metadata.get("flags", [])],
            "phase6a_product_slice",
            "cpu_tsdf_not_realtime",
            "observed_surface_points",
            "not_completed_surface",
            "not_accuracy_report",
            "not_performance_report",
        ]
    )
    metadata["source_frame_ids"] = [observation.frame_id for observation in observations]
    return TSDFSurface(
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
        metadata=metadata,
    )


def _tsdf_metrics(surface: TSDFSurface) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "known_limitations": [
            "CPU TSDF is a diagnostic reference path and is not realtime.",
            "Only observed depth samples are fused; hidden geometry is not completed.",
        ],
        "metric_family": "phase6a_recording_cpu_tsdf_diagnostic",
        "surface_point_count": int(surface.points_world_m.shape[0]),
    }


def _quality_report(
    observations: tuple[DepthObservation, ...],
    surface: TSDFSurface,
) -> dict[str, object]:
    valid_counts = np.asarray(
        [int(np.count_nonzero(observation.confidence > 0.0)) for observation in observations],
        dtype=np.float64,
    )
    total_pixels = float(sum(observation.depth_m.size for observation in observations))
    depths = np.concatenate(
        [
            observation.depth_m[observation.confidence > 0.0].reshape(-1).astype(np.float64)
            for observation in observations
        ]
    )
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "format_name": "atlas3r_phase6a_quality_report",
        "format_version": 1,
        "known_limitations": [
            "Quality report summarizes measured input coverage; it is not a prediction benchmark.",
            "No millimeter, realtime, or mapping-readiness claim is made.",
        ],
        "measured_depth_available": True,
        "observation_count": len(observations),
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "valid_depth_pixel_count": int(np.sum(valid_counts)),
        "valid_depth_pixel_ratio": float(np.sum(valid_counts) / max(total_pixels, 1.0)),
        "valid_depth_range_m": {
            "max": None if depths.size == 0 else float(np.max(depths)),
            "mean": None if depths.size == 0 else float(np.mean(depths)),
            "min": None if depths.size == 0 else float(np.min(depths)),
        },
    }


def _memory_report(
    observations: tuple[DepthObservation, ...],
    surface: TSDFSurface,
    tsdf_dir: Path,
) -> dict[str, object]:
    with np.load(tsdf_dir / "tsdf_grid.npz", allow_pickle=False) as payload:
        tsdf_bytes = int(np.asarray(payload["tsdf"]).nbytes + np.asarray(payload["weight"]).nbytes)
    observation_bytes = sum(
        observation.depth_m.nbytes
        + observation.depth_sigma_m.nbytes
        + observation.confidence.nbytes
        for observation in observations
    )
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "format_name": "atlas3r_phase6a_memory_report",
        "format_version": 1,
        "known_limitations": [
            "Counters are deterministic NumPy array byte counts, not process RSS telemetry.",
            "CPU TSDF is not a bounded GPU mapper.",
        ],
        "observation_array_bytes": int(observation_bytes),
        "surface_array_bytes": int(
            surface.points_world_m.nbytes + surface.confidence.nbytes + surface.uncertainty_m.nbytes
        ),
        "tsdf_array_bytes": tsdf_bytes,
    }


def _summary(
    *,
    config: FuseRecordingConfig,
    recording: Atlas3RRecording,
    observations: tuple[DepthObservation, ...],
    surface: TSDFSurface,
    tsdf_dir: Path,
    point_cloud_path: Path | None,
    mesh_status: dict[str, object],
    latency_report: dict[str, object],
    memory_report: dict[str, object],
    quality_report: dict[str, object],
) -> dict[str, object]:
    return {
        **DIAGNOSTIC_TRUTH_FLAGS,
        "artifacts": {
            "latency_report": "latency_report.json",
            "map_preview": "map_preview.html",
            "memory_report": "memory_report.json",
            "mesh_status": "mesh_status.json",
            "point_cloud": None if point_cloud_path is None else point_cloud_path.name,
            "quality_report": "quality_report.json",
            "runtime_events": "runtime_events.jsonl",
            "summary": "summary.json",
            "tsdf": tsdf_dir.name,
        },
        "cpu_tsdf_note": "CPU TSDF is diagnostic and not the final realtime GPU mapper.",
        "format_name": "atlas3r_phase6a_fuse_recording_summary",
        "format_version": 1,
        "frame_count": len(observations),
        "keyframe_stride": config.keyframe_stride,
        "known_limitations": [
            "Measured depth+pose are fused; no learned geometry is generated in this path.",
            "Hidden or unobserved geometry is not emitted as measured geometry.",
            "CPU TSDF is not realtime and not the final GPU mapper.",
        ],
        "latency": cast(dict[str, object], latency_report["segments"]),
        "memory": memory_report,
        "mesh": mesh_status,
        "output": str(config.output),
        "quality": quality_report,
        "recording": str(recording.root),
        "source_frame_ids": [observation.frame_id for observation in observations],
        "surface_point_count": int(surface.points_world_m.shape[0]),
        "voxel_size_m": config.voxel_size_m,
    }


def _write_map_preview(
    path: Path,
    *,
    summary: dict[str, object],
    latency_report: dict[str, object],
) -> None:
    total_latency = cast(dict[str, object], latency_report["segments"]).get("total_pipeline")
    mesh_exported = cast(dict[str, object], summary["mesh"]).get("mesh_exported")
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html><head><meta charset="utf-8"><title>Atlas3R Phase 6A</title>',
                "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}"
                "code{background:#eee;padding:.1rem .25rem}</style></head><body>",
                "<h1>Atlas3R Phase 6A Recording Fusion</h1>",
                "<p>Diagnostic measured recording to CPU TSDF output.</p>",
                "<ul>",
                f"<li>Frames fused: {summary['frame_count']}</li>",
                f"<li>Surface points: {summary['surface_point_count']}</li>",
                f"<li>Mesh exported: {mesh_exported}</li>",
                f"<li>Total pipeline latency: {total_latency}</li>",
                "</ul>",
                "<p>No realtime, benchmark accuracy, mapping-readiness, or millimeter "
                "claim is made.</p>",
                "</body></html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


class _RuntimeEventRecorder:
    def __init__(self, *, configured_frame_array_bound: int) -> None:
        self._configured_frame_array_bound = configured_frame_array_bound
        self._events: list[RuntimeEvent] = []
        self._start_ns = time.perf_counter_ns()
        self._processed_frame_count = 0

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    def emit(
        self,
        stage_name: str,
        *,
        latency_ns: int = 0,
        frame_id: int | None = None,
        frame_arrays_in_memory: int = 0,
        count_processed_frame: bool = False,
        paths: dict[str, str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if count_processed_frame:
            self._processed_frame_count += 1
        counters = BoundedMemoryCounters(
            configured_frame_array_bound=max(1, self._configured_frame_array_bound),
            frame_arrays_in_memory=frame_arrays_in_memory,
            peak_frame_arrays_in_memory=1 if self._processed_frame_count else 0,
            processed_frame_count=self._processed_frame_count,
            dropped_frame_count=0,
            queued_frame_count=0,
        )
        self._events.append(
            RuntimeEvent(
                event_index=len(self._events),
                stage_name=stage_name,
                timestamp_ns=time.perf_counter_ns() - self._start_ns,
                latency_ns=latency_ns,
                dropped_frame=False,
                memory_counters=counters,
                frame_id=frame_id,
                paths=paths or {},
                metadata=metadata or {},
            )
        )


def _validate_config(config: FuseRecordingConfig) -> None:
    if config.pose_source != "recording":
        raise ValueError("pose_source: only 'recording' is supported in Phase 6A")
    if config.depth_source != "recording":
        raise ValueError("depth_source: only 'recording' is supported in Phase 6A")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    if config.keyframe_stride <= 0:
        raise ValueError("keyframe_stride: must be positive")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if config.truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")
    if config.export_mesh not in {"off", "auto", "required"}:
        raise ValueError("export_mesh: must be off, auto, or required")


def _stable_strings(values: list[str]) -> list[str]:
    stable: list[str] = []
    for value in values:
        if value not in stable:
            stable.append(value)
    return stable


__all__ = [
    "FuseRecordingConfig",
    "run_fuse_recording",
]
