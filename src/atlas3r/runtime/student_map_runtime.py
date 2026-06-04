"""Phase 5G streaming temporal-student to CPU TSDF runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from atlas3r.mapping._checkpoint_tsdf_smoke_helpers import (
    grid_bounds_from_observations,
    integrate_observations_to_volume,
    write_tsdf_outputs,
)
from atlas3r.mapping.cpu_tsdf import TSDFSurface, extract_tsdf_surface
from atlas3r.mapping.mesh_sidecar import write_tsdf_surface_mesh_sidecar_from_artifacts
from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.world_map_sidecar import write_tsdf_world_map_sidecar_from_artifacts
from atlas3r.runtime.events import BoundedMemoryCounters, RuntimeEvent
from atlas3r.runtime.student_map_observations import (
    StudentOdometryState,
    load_teacher_reference_frames,
    observation_from_prediction,
    window_arrays,
)
from atlas3r.runtime.student_map_outputs import (
    mode_comparison,
    summary_record,
    teacher_map_comparison,
    tsdf_metrics,
    tsdf_summary,
    with_runtime_surface_metadata,
)
from atlas3r.runtime.student_map_reports import (
    DIAGNOSTIC_TRUTH_FLAGS,
    LatencyRecorder,
    TeacherReferenceFrame,
    write_json,
    write_jsonl,
    write_latency_report,
    write_map_preview,
    write_observation_summaries,
    write_point_cloud_ply,
    write_quality_reports,
)
from atlas3r.runtime.student_map_runtime_types import POSE_MODES, StudentMapRuntimeConfig
from atlas3r.runtime.student_stream import (
    StreamFrame,
    build_stream_window,
    load_unique_frame_stream,
)
from atlas3r.training.teacher_signal_temporal_artifacts import (
    load_teacher_signal_temporal_checkpoint,
)
from atlas3r.training.torch_runtime import require_torch


def run_stream_student_map(config: StudentMapRuntimeConfig) -> dict[str, object]:
    """Run the Phase 5G streaming student map diagnostic."""

    _validate_config(config)
    loaded = load_teacher_signal_temporal_checkpoint(config.checkpoint, device=config.device)
    stream = load_unique_frame_stream(config.clip_cache, max_frames=config.max_frames)
    references = load_teacher_reference_frames(
        config.teacher_cache,
        frame_ids=tuple(frame.frame_id for frame in stream),
    )
    modes = ("oracle", "student-relative") if config.pose_mode == "both" else (config.pose_mode,)
    mode_results = [
        _run_single_pose_mode(
            config=config,
            pose_mode=mode,
            loaded_checkpoint=loaded,
            stream=stream,
            references_by_frame_id=references,
        )
        for mode in modes
    ]
    root_summary = {
        "format_name": "atlas3r_phase5g_stream_student_map_run",
        "format_version": 1,
        **DIAGNOSTIC_TRUTH_FLAGS,
        "checkpoint": str(config.checkpoint),
        "clip_cache": str(config.clip_cache),
        "teacher_cache": str(config.teacher_cache),
        "output": str(config.output),
        "requested_pose_mode": config.pose_mode,
        "modes": mode_results,
        "mode_comparison": mode_comparison(mode_results),
    }
    config.output.mkdir(parents=True, exist_ok=True)
    write_json(config.output / "summary.json", root_summary)
    return root_summary


def _run_single_pose_mode(
    *,
    config: StudentMapRuntimeConfig,
    pose_mode: str,
    loaded_checkpoint: Mapping[str, Any],
    stream: tuple[StreamFrame, ...],
    references_by_frame_id: Mapping[int, TeacherReferenceFrame],
) -> dict[str, object]:
    torch = require_torch()
    model = loaded_checkpoint["model"]
    checkpoint = cast(dict[str, Any], loaded_checkpoint["checkpoint"])
    if pose_mode == "student-odometry" and not bool(
        loaded_checkpoint.get("has_trained_rotation_head", False)
    ):
        raise ValueError(
            "student-odometry: checkpoint does not contain trained SE(3) rotation-head weights"
        )
    resolved_device = str(loaded_checkpoint["device"])
    output = config.output / pose_mode
    output.mkdir(parents=True, exist_ok=True)
    recorder = LatencyRecorder()
    events = _RuntimeEventRecorder(configured_frame_array_bound=config.window_size)
    run_start_ns = time.perf_counter_ns()
    events.emit(
        "runtime_start",
        metadata={
            "pose_mode": pose_mode,
            "checkpoint": str(config.checkpoint),
            "device": resolved_device,
            "stream_frame_count": len(stream),
            **DIAGNOSTIC_TRUTH_FLAGS,
        },
    )

    observations: list[DepthObservation] = []
    observation_records: list[dict[str, object]] = []
    odometry_state = StudentOdometryState() if pose_mode == "student-odometry" else None
    model.eval()
    with torch.no_grad():
        for stream_index in range(len(stream)):
            observation, record, frame_latency_ns = _run_frame_window(
                model=model,
                resolved_device=resolved_device,
                config=config,
                stream=stream,
                stream_index=stream_index,
                pose_mode=pose_mode,
                checkpoint=checkpoint,
                recorder=recorder,
                odometry_state=odometry_state,
            )
            observations.append(observation)
            observation_records.append(record)
            events.emit(
                "depth_observation",
                frame_id=observation.frame_id,
                latency_ns=frame_latency_ns,
                frame_arrays_in_memory=config.window_size,
                count_processed_frame=True,
                metadata={
                    "pose_mode": pose_mode,
                    "window_frame_ids": cast(dict[str, object], record["window"])[
                        "window_frame_ids"
                    ],
                    "padded_positions": cast(dict[str, object], record["window"])[
                        "padded_positions"
                    ],
                },
            )
    observations_tuple = tuple(observations)
    surface, tsdf_dir, tsdf_latency_ns = _write_tsdf_outputs(
        config,
        output,
        pose_mode,
        checkpoint,
        observations_tuple,
    )
    recorder.add("cpu_tsdf_mapping", tsdf_latency_ns)
    events.emit(
        "cpu_tsdf_mapping",
        latency_ns=tsdf_latency_ns,
        paths={"tsdf": _rel(output, tsdf_dir)},
        metadata={"surface_point_count": int(surface.points_world_m.shape[0])},
    )

    teacher_map_summary, point_comparison = teacher_map_comparison(config, output, surface)
    tsdf_summary_record = tsdf_summary(surface, tsdf_dir, teacher_map_summary)
    quality_report = write_quality_reports(
        output,
        observations=observations_tuple,
        references_by_frame_id=references_by_frame_id,
        pose_mode=pose_mode,
        tsdf_summary=tsdf_summary_record,
        map_point_set_comparison=point_comparison,
    )

    geometry_start_ns = time.perf_counter_ns()
    ply_path = write_point_cloud_ply(output / "point_cloud.ply", surface)
    write_map_preview(
        output / "map_preview.html",
        pose_mode=pose_mode,
        summary=tsdf_summary_record,
        quality_report=quality_report,
        latency_report=recorder.report(),
    )
    recorder.add("geometry_export", time.perf_counter_ns() - geometry_start_ns)
    write_observation_summaries(output, observation_records)
    recorder.add("total_pipeline", time.perf_counter_ns() - run_start_ns)
    latency_report = write_latency_report(output, recorder)

    summary = summary_record(
        config=config,
        pose_mode=pose_mode,
        output=output,
        checkpoint=checkpoint,
        resolved_device=resolved_device,
        stream=stream,
        observations=observations_tuple,
        surface=surface,
        tsdf_summary_record=tsdf_summary_record,
        quality_report=quality_report,
        latency_report=latency_report,
        ply_path=ply_path,
    )
    write_json(output / "summary.json", summary)
    events.emit(
        "runtime_complete",
        paths={"summary": "summary.json", "point_cloud": "point_cloud.ply"},
        metadata={
            "surface_point_count": int(surface.points_world_m.shape[0]),
            "mapping_ready": False,
        },
    )
    write_jsonl(
        output / "runtime_events.jsonl", [event.to_json_record() for event in events.events]
    )
    return summary


def _run_frame_window(
    *,
    model: Any,
    resolved_device: str,
    config: StudentMapRuntimeConfig,
    stream: tuple[StreamFrame, ...],
    stream_index: int,
    pose_mode: str,
    checkpoint: Mapping[str, Any],
    recorder: LatencyRecorder,
    odometry_state: StudentOdometryState | None,
) -> tuple[DepthObservation, dict[str, object], int]:
    torch = require_torch()
    frame_start_ns = time.perf_counter_ns()
    window_start_ns = time.perf_counter_ns()
    window = build_stream_window(
        stream,
        stream_index=stream_index,
        window_size=config.window_size,
    )
    images_np, intrinsics_np = window_arrays(window)
    recorder.add("window_assembly", time.perf_counter_ns() - window_start_ns)
    transfer_start_ns = time.perf_counter_ns()
    images = torch.from_numpy(images_np).to(resolved_device)
    intrinsics = torch.from_numpy(intrinsics_np).to(resolved_device)
    recorder.add("tensor_transfer", time.perf_counter_ns() - transfer_start_ns)
    inference_start_ns = time.perf_counter_ns()
    prediction = model(images, intrinsics)
    recorder.add("model_inference", time.perf_counter_ns() - inference_start_ns)
    observation_start_ns = time.perf_counter_ns()
    observation, record = observation_from_prediction(
        prediction=prediction,
        window=window,
        pose_mode=pose_mode,
        checkpoint_path=config.checkpoint,
        checkpoint=checkpoint,
        odometry_state=odometry_state,
    )
    recorder.add("observation_construction", time.perf_counter_ns() - observation_start_ns)
    return observation, record, time.perf_counter_ns() - frame_start_ns


def _write_tsdf_outputs(
    config: StudentMapRuntimeConfig,
    output: Path,
    pose_mode: str,
    checkpoint: Mapping[str, Any],
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
    surface = with_runtime_surface_metadata(
        extract_tsdf_surface(volume),
        pose_mode=pose_mode,
        checkpoint_path=config.checkpoint,
        checkpoint=checkpoint,
        observations=observations,
    )
    tsdf_dir = output / "tsdf"
    write_tsdf_outputs(tsdf_dir, volume, surface, tsdf_metrics(surface, pose_mode))
    write_tsdf_surface_mesh_sidecar_from_artifacts(
        tsdf_dir,
        chunk_id=f"phase_5e_{pose_mode.replace('-', '_')}_student_surface",
    )
    write_tsdf_world_map_sidecar_from_artifacts(tsdf_dir)
    return surface, tsdf_dir, time.perf_counter_ns() - start_ns


def _validate_config(config: StudentMapRuntimeConfig) -> None:
    if config.pose_mode not in POSE_MODES:
        raise ValueError(f"pose_mode: expected one of {', '.join(POSE_MODES)}")
    if config.max_frames <= 0:
        raise ValueError("max_frames: must be positive")
    if config.window_size <= 0:
        raise ValueError("window_size: must be positive")
    if config.voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    if config.truncation_voxels <= 0.0:
        raise ValueError("truncation_voxels: must be positive")


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
        peak_frame_arrays = self._configured_frame_array_bound if self._processed_frame_count else 0
        counters = BoundedMemoryCounters(
            configured_frame_array_bound=self._configured_frame_array_bound,
            frame_arrays_in_memory=frame_arrays_in_memory,
            peak_frame_arrays_in_memory=peak_frame_arrays,
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


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "POSE_MODES",
    "StudentMapRuntimeConfig",
    "load_teacher_reference_frames",
    "run_stream_student_map",
]
