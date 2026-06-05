"""Live-style replay scheduler over measured Atlas3R recordings."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.mapping.sparse_tsdf import (
    SparseBlockTSDFMapper,
    SparseTSDFConfig,
)
from atlas3r.mapping.sparse_tsdf_artifacts import write_sparse_tsdf_outputs
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import Atlas3RRecording, RecordingFrame, load_recording
from atlas3r.runtime.live_replay_outputs import (
    extract_or_empty_surface,
    known_limitations,
    live_replay_summary,
    sparse_metrics,
    write_live_replay_report,
    write_sparse_mesh_status,
)
from atlas3r.runtime.live_replay_types import (
    LiveReplayConfig,
    LiveReplayEventRecorder,
    LiveReplayState,
)
from atlas3r.runtime.student_map_report_common import (
    DIAGNOSTIC_TRUTH_FLAGS,
    write_json,
    write_jsonl,
)
from atlas3r.runtime.student_map_reports import LatencyRecorder, write_point_cloud_ply


@dataclass(frozen=True)
class _MapQueueItem:
    frame: RecordingFrame
    observation: DepthObservation
    keyframe_reason: str
    observation_load_latency_ns: int


def run_live_replay_recording(config: LiveReplayConfig) -> dict[str, object]:
    """Replay a measured recording through bounded live-style scheduler queues."""

    recording = load_recording(config.recording)
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    recorder = LatencyRecorder()
    events = LiveReplayEventRecorder()
    state = LiveReplayState()
    capture_queue: deque[RecordingFrame] = deque()
    map_queue: deque[_MapQueueItem] = deque()
    seen_frame_ids: set[int] = set()
    source_observations: list[DepthObservation] = []
    map_update_latencies_ns: list[int] = []
    previous_keyframe: RecordingFrame | None = None
    replay_start_ns = time.perf_counter_ns()
    sparse_config = SparseTSDFConfig(
        voxel_size_m=config.voxel_size_m,
        truncation_distance_m=config.voxel_size_m * config.truncation_voxels,
        coordinate_frame=str(recording.manifest["coordinate_frame"]),
        metric_scale_source="external_pose",
    )
    mapper = SparseBlockTSDFMapper(sparse_config)
    events.emit(
        "runtime_start",
        metadata={
            **DIAGNOSTIC_TRUTH_FLAGS,
            "mapper_backend": config.mapper_backend,
            "recording": str(config.recording),
            "simulated_pacing": not config.wall_clock_pacing,
            "target_fps": config.target_fps,
        },
    )

    frames = (
        recording.frames if config.max_frames is None else recording.frames[: config.max_frames]
    )
    for source_index, frame in enumerate(frames, start=1):
        state.frame_count_seen += 1
        _maybe_sleep_for_wall_clock_pacing(config, source_index, replay_start_ns)
        _enqueue_capture_frame(config, events, state, capture_queue, frame)
        if source_index % config.capture_service_interval_frames == 0:
            previous_keyframe = _process_capture_queue(
                config=config,
                recording=recording,
                events=events,
                recorder=recorder,
                state=state,
                capture_queue=capture_queue,
                map_queue=map_queue,
                seen_frame_ids=seen_frame_ids,
                previous_keyframe=previous_keyframe,
            )
        if source_index % config.map_service_interval_frames == 0:
            _process_map_queue(
                events=events,
                recorder=recorder,
                state=state,
                map_queue=map_queue,
                mapper=mapper,
                source_observations=source_observations,
                map_update_latencies_ns=map_update_latencies_ns,
            )

    while capture_queue:
        previous_keyframe = _process_capture_queue(
            config=config,
            recording=recording,
            events=events,
            recorder=recorder,
            state=state,
            capture_queue=capture_queue,
            map_queue=map_queue,
            seen_frame_ids=seen_frame_ids,
            previous_keyframe=previous_keyframe,
        )
    while map_queue:
        _process_map_queue(
            events=events,
            recorder=recorder,
            state=state,
            map_queue=map_queue,
            mapper=mapper,
            source_observations=source_observations,
            map_update_latencies_ns=map_update_latencies_ns,
        )

    surface_start_ns = time.perf_counter_ns()
    surface = extract_or_empty_surface(mapper, sparse_config, recording, config)
    recorder.add("final_surface_extraction", time.perf_counter_ns() - surface_start_ns)
    export_start_ns = time.perf_counter_ns()
    sparse_dir = output / "sparse_tsdf"
    write_sparse_tsdf_outputs(
        sparse_dir,
        mapper=mapper,
        surface=surface,
        metrics=sparse_metrics(config, mapper, surface, state),
    )
    point_cloud_path = None
    if config.export_point_cloud:
        point_cloud_path = write_point_cloud_ply(output / "surface_points.ply", surface)
    mesh_status = write_sparse_mesh_status(output, surface)
    recorder.add("geometry_export", time.perf_counter_ns() - export_start_ns)
    recorder.add("total_pipeline", time.perf_counter_ns() - replay_start_ns)
    latency_report = recorder.report()
    latency_report.update(
        {
            "format_name": "atlas3r_phase6e_live_replay_latency_report",
            "known_limitations": known_limitations(),
            "mapper_backend": config.mapper_backend,
            "simulated_pacing": not config.wall_clock_pacing,
        }
    )
    summary = live_replay_summary(
        config=config,
        recording=recording,
        state=state,
        mapper=mapper,
        surface=surface,
        source_observations=tuple(source_observations),
        latency_report=latency_report,
        point_cloud_path=point_cloud_path,
        mesh_status=mesh_status,
    )
    events.emit(
        "runtime_complete",
        paths={
            "events": "live_replay_events.jsonl",
            "report": "live_replay_report.md",
            "summary": "live_replay_summary.json",
        },
        metadata={
            "map_update_count": state.map_update_count,
            "surface_point_count": int(surface.points_world_m.shape[0]),
        },
    )
    write_jsonl(output / "live_replay_events.jsonl", events.events)
    write_json(output / "live_replay_latency_report.json", latency_report)
    write_json(output / "live_replay_summary.json", summary)
    write_live_replay_report(output / "live_replay_report.md", summary)
    return summary


def _enqueue_capture_frame(
    config: LiveReplayConfig,
    events: LiveReplayEventRecorder,
    state: LiveReplayState,
    capture_queue: deque[RecordingFrame],
    frame: RecordingFrame,
) -> None:
    timestamp_ns = int(round(frame.timestamp_s * 1_000_000_000.0))
    if len(capture_queue) >= config.max_capture_queue:
        if config.drop_policy == "newest":
            state.dropped_frame_count += 1
            events.emit(
                "capture_drop",
                frame_id=frame.frame_id,
                timestamp_ns=timestamp_ns,
                dropped_frame=True,
                drop_reason="capture_queue_full",
                capture_queue_depth=len(capture_queue),
                metadata={"drop_policy": config.drop_policy},
            )
            return
        dropped = capture_queue.popleft()
        state.dropped_frame_count += 1
        events.emit(
            "capture_drop",
            frame_id=dropped.frame_id,
            timestamp_ns=int(round(dropped.timestamp_s * 1_000_000_000.0)),
            dropped_frame=True,
            drop_reason="capture_queue_full",
            capture_queue_depth=len(capture_queue),
            metadata={"drop_policy": config.drop_policy},
        )
    capture_queue.append(frame)
    state.max_capture_queue_depth_observed = max(
        state.max_capture_queue_depth_observed,
        len(capture_queue),
    )
    events.emit(
        "capture_enqueue",
        frame_id=frame.frame_id,
        timestamp_ns=timestamp_ns,
        capture_queue_depth=len(capture_queue),
    )


def _process_capture_queue(
    *,
    config: LiveReplayConfig,
    recording: Atlas3RRecording,
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
    state: LiveReplayState,
    capture_queue: deque[RecordingFrame],
    map_queue: deque[_MapQueueItem],
    seen_frame_ids: set[int],
    previous_keyframe: RecordingFrame | None,
) -> RecordingFrame | None:
    if not capture_queue:
        return previous_keyframe
    frame = capture_queue.popleft()
    timestamp_ns = int(round(frame.timestamp_s * 1_000_000_000.0))
    if frame.frame_id in seen_frame_ids:
        state.dropped_frame_count += 1
        events.emit(
            "capture_drop",
            frame_id=frame.frame_id,
            timestamp_ns=timestamp_ns,
            dropped_frame=True,
            drop_reason="duplicate_frame",
            capture_queue_depth=len(capture_queue),
        )
        return previous_keyframe
    seen_frame_ids.add(frame.frame_id)
    state.frame_count_emitted += 1
    recorder.add("pose_output", 0)
    if frame.T_world_camera is None:
        state.dropped_keyframe_count += 1
        events.emit(
            "keyframe_drop",
            frame_id=frame.frame_id,
            timestamp_ns=timestamp_ns,
            drop_reason="missing_pose",
            keyframe_selected=False,
            keyframe_reason="not_selected",
            capture_queue_depth=len(capture_queue),
            map_queue_depth=len(map_queue),
        )
        return previous_keyframe
    state.pose_update_count += 1
    state.measured_pose_used = True
    events.emit(
        "pose_update",
        frame_id=frame.frame_id,
        timestamp_ns=timestamp_ns,
        capture_queue_depth=len(capture_queue),
        map_queue_depth=len(map_queue),
        metadata={"pose_source": "recording", "T_world_camera": True},
    )
    selected, reason = _keyframe_decision(config, state, frame, previous_keyframe)
    events.emit(
        "keyframe_decision",
        frame_id=frame.frame_id,
        timestamp_ns=timestamp_ns,
        keyframe_selected=selected,
        keyframe_reason=reason,
        capture_queue_depth=len(capture_queue),
        map_queue_depth=len(map_queue),
    )
    if not selected:
        state.dropped_keyframe_count += 1
        events.emit(
            "keyframe_drop",
            frame_id=frame.frame_id,
            timestamp_ns=timestamp_ns,
            drop_reason="not_keyframe",
            keyframe_selected=False,
            keyframe_reason="not_selected",
            capture_queue_depth=len(capture_queue),
            map_queue_depth=len(map_queue),
        )
        return previous_keyframe
    state.keyframe_selected_count += 1
    if frame.depth_path is None:
        state.dropped_keyframe_count += 1
        events.emit(
            "mapping_skip",
            frame_id=frame.frame_id,
            timestamp_ns=timestamp_ns,
            drop_reason="missing_depth",
            keyframe_selected=True,
            keyframe_reason=reason,
            metadata={"measured_depth_available": False},
        )
        return frame
    load_start_ns = time.perf_counter_ns()
    observation = observation_from_recording_frame(
        recording,
        frame,
        depth_sigma_floor_m=max(0.01, config.voxel_size_m * 0.25),
    )
    load_latency_ns = time.perf_counter_ns() - load_start_ns
    recorder.add("observation_load", load_latency_ns)
    state.measured_depth_used = True
    _enqueue_map_item(
        config=config,
        events=events,
        state=state,
        map_queue=map_queue,
        item=_MapQueueItem(
            frame=frame,
            observation=observation,
            keyframe_reason=reason,
            observation_load_latency_ns=load_latency_ns,
        ),
    )
    return frame


def _enqueue_map_item(
    *,
    config: LiveReplayConfig,
    events: LiveReplayEventRecorder,
    state: LiveReplayState,
    map_queue: deque[_MapQueueItem],
    item: _MapQueueItem,
) -> None:
    if len(map_queue) >= config.max_map_queue:
        if config.drop_policy == "newest":
            state.dropped_keyframe_count += 1
            events.emit(
                "keyframe_drop",
                frame_id=item.frame.frame_id,
                timestamp_ns=item.observation.pose.timestamp_ns,
                drop_reason="map_queue_full",
                keyframe_selected=True,
                keyframe_reason=item.keyframe_reason,
                map_queue_depth=len(map_queue),
                metadata={"drop_policy": config.drop_policy},
            )
            return
        dropped = map_queue.popleft()
        state.dropped_keyframe_count += 1
        events.emit(
            "keyframe_drop",
            frame_id=dropped.frame.frame_id,
            timestamp_ns=dropped.observation.pose.timestamp_ns,
            drop_reason="map_queue_full",
            keyframe_selected=True,
            keyframe_reason=dropped.keyframe_reason,
            map_queue_depth=len(map_queue),
            metadata={"drop_policy": config.drop_policy},
        )
    map_queue.append(item)
    state.max_map_queue_depth_observed = max(state.max_map_queue_depth_observed, len(map_queue))
    events.emit(
        "map_queue_enqueue",
        frame_id=item.frame.frame_id,
        timestamp_ns=item.observation.pose.timestamp_ns,
        latency_ns=item.observation_load_latency_ns,
        keyframe_selected=True,
        keyframe_reason=item.keyframe_reason,
        map_queue_depth=len(map_queue),
    )


def _process_map_queue(
    *,
    events: LiveReplayEventRecorder,
    recorder: LatencyRecorder,
    state: LiveReplayState,
    map_queue: deque[_MapQueueItem],
    mapper: SparseBlockTSDFMapper,
    source_observations: list[DepthObservation],
    map_update_latencies_ns: list[int],
) -> None:
    if not map_queue:
        return
    item = map_queue.popleft()
    start_ns = time.perf_counter_ns()
    update_stats = mapper.integrate(item.observation)
    latency_ns = time.perf_counter_ns() - start_ns
    recorder.add("map_update", latency_ns)
    map_update_latencies_ns.append(latency_ns)
    source_observations.append(item.observation)
    state.map_update_count += 1
    events.emit(
        "map_update",
        frame_id=item.frame.frame_id,
        timestamp_ns=item.observation.pose.timestamp_ns,
        latency_ns=latency_ns,
        keyframe_selected=True,
        keyframe_reason=item.keyframe_reason,
        map_queue_depth=len(map_queue),
        metadata={
            "active_block_count": update_stats.active_block_count,
            "active_voxel_count": update_stats.active_voxel_count,
            "approximate_state_bytes": update_stats.approximate_state_bytes,
            "candidate_voxel_count": update_stats.candidate_voxel_count,
            "new_voxel_count": update_stats.new_voxel_count,
            "update_implementation": update_stats.update_implementation,
            "updated_voxel_count": update_stats.updated_voxel_count,
        },
    )


def _keyframe_decision(
    config: LiveReplayConfig,
    state: LiveReplayState,
    frame: RecordingFrame,
    previous_keyframe: RecordingFrame | None,
) -> tuple[bool, str]:
    emitted_index = state.frame_count_emitted - 1
    if state.keyframe_selected_count == 0:
        return True, "first_frame"
    if emitted_index % config.map_keyframe_stride == 0:
        return True, "stride"
    if (
        config.translation_threshold_m is not None
        and previous_keyframe is not None
        and previous_keyframe.T_world_camera is not None
        and frame.T_world_camera is not None
    ):
        delta = frame.T_world_camera[:3, 3] - previous_keyframe.T_world_camera[:3, 3]
        if float(np.linalg.norm(delta.astype(np.float64))) >= config.translation_threshold_m:
            return True, "translation_threshold"
    if (
        config.rotation_threshold_deg is not None
        and previous_keyframe is not None
        and previous_keyframe.T_world_camera is not None
        and frame.T_world_camera is not None
    ):
        if _rotation_delta_deg(previous_keyframe.T_world_camera, frame.T_world_camera) >= (
            config.rotation_threshold_deg
        ):
            return True, "rotation_threshold"
    return False, "not_selected"


def _rotation_delta_deg(
    previous_T: npt.NDArray[np.float32],
    current_T: npt.NDArray[np.float32],
) -> float:
    rel = previous_T[:3, :3].astype(np.float64).T @ current_T[:3, :3].astype(np.float64)
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def _maybe_sleep_for_wall_clock_pacing(
    config: LiveReplayConfig,
    source_index: int,
    replay_start_ns: int,
) -> None:
    if not config.wall_clock_pacing:
        return
    target_elapsed_s = (source_index - 1) / config.target_fps
    elapsed_s = (time.perf_counter_ns() - replay_start_ns) / 1_000_000_000.0
    if target_elapsed_s > elapsed_s:
        time.sleep(target_elapsed_s - elapsed_s)


__all__ = ["LiveReplayConfig", "run_live_replay_recording"]
