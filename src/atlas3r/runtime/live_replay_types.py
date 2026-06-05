"""Public types for Phase 6E live replay diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atlas3r.runtime.recording_fusion_incremental_helpers import CPU_SPARSE_BACKEND

LIVE_REPLAY_EVENT_FORMAT_NAME = "atlas3r_live_replay_event"
LIVE_REPLAY_SUMMARY_FORMAT_NAME = "atlas3r_live_replay_summary"
LIVE_REPLAY_FORMAT_VERSION = 1

DROP_REASONS = {
    "capture_queue_full",
    "map_queue_full",
    "missing_depth",
    "missing_pose",
    "duplicate_frame",
    "not_keyframe",
    "scheduler_shutdown",
    "other",
}
KEYFRAME_REASONS = {
    "first_frame",
    "stride",
    "translation_threshold",
    "rotation_threshold",
    "uncertainty_threshold",
    "forced",
    "not_selected",
}


@dataclass(frozen=True)
class LiveReplayConfig:
    recording: Path
    output: Path
    target_fps: float = 30.0
    max_frames: int | None = None
    mapper_backend: str = CPU_SPARSE_BACKEND
    map_keyframe_stride: int = 1
    max_capture_queue: int = 4
    max_map_queue: int = 2
    drop_policy: str = "oldest"
    voxel_size_m: float = 0.05
    truncation_voxels: float = 3.0
    pixel_stride: int = 8
    export_point_cloud: bool = False
    export_mesh_chunks: bool = False
    mesh_format: str = "npz"
    mesh_update_interval_frames: int = 1
    mesh_max_dirty_chunks_per_frame: int | None = None
    mesh_min_weight: float = 0.0
    wall_clock_pacing: bool = False
    capture_service_interval_frames: int = 1
    map_service_interval_frames: int = 1
    translation_threshold_m: float | None = None
    rotation_threshold_deg: float | None = None

    def __post_init__(self) -> None:
        if self.target_fps <= 0.0:
            raise ValueError("target_fps: must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames: must be positive when provided")
        if self.mapper_backend != CPU_SPARSE_BACKEND:
            raise ValueError("mapper_backend: only cpu-sparse is supported in Phase 6E")
        if self.map_keyframe_stride <= 0:
            raise ValueError("map_keyframe_stride: must be positive")
        if self.max_capture_queue <= 0:
            raise ValueError("max_capture_queue: must be positive")
        if self.max_map_queue <= 0:
            raise ValueError("max_map_queue: must be positive")
        if self.drop_policy not in {"oldest", "newest"}:
            raise ValueError("drop_policy: must be oldest or newest")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m: must be positive")
        if self.truncation_voxels <= 0.0:
            raise ValueError("truncation_voxels: must be positive")
        if self.pixel_stride <= 0:
            raise ValueError("pixel_stride: must be positive")
        if self.mesh_format not in {"npz", "ply", "both"}:
            raise ValueError("mesh_format: must be npz, ply, or both")
        if self.mesh_update_interval_frames <= 0:
            raise ValueError("mesh_update_interval_frames: must be positive")
        if (
            self.mesh_max_dirty_chunks_per_frame is not None
            and self.mesh_max_dirty_chunks_per_frame <= 0
        ):
            raise ValueError("mesh_max_dirty_chunks_per_frame: must be positive when provided")
        if self.mesh_min_weight < 0.0:
            raise ValueError("mesh_min_weight: must be non-negative")
        if self.capture_service_interval_frames <= 0:
            raise ValueError("capture_service_interval_frames: must be positive")
        if self.map_service_interval_frames <= 0:
            raise ValueError("map_service_interval_frames: must be positive")
        if self.translation_threshold_m is not None and self.translation_threshold_m <= 0.0:
            raise ValueError("translation_threshold_m: must be positive when provided")
        if self.rotation_threshold_deg is not None and self.rotation_threshold_deg <= 0.0:
            raise ValueError("rotation_threshold_deg: must be positive when provided")


@dataclass
class LiveReplayState:
    frame_count_seen: int = 0
    frame_count_emitted: int = 0
    pose_update_count: int = 0
    keyframe_selected_count: int = 0
    map_update_count: int = 0
    dropped_frame_count: int = 0
    dropped_keyframe_count: int = 0
    max_capture_queue_depth_observed: int = 0
    max_map_queue_depth_observed: int = 0
    measured_depth_used: bool = False
    measured_pose_used: bool = False


class LiveReplayEventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(
        self,
        stage_name: str,
        *,
        frame_id: int | None = None,
        timestamp_ns: int = 0,
        latency_ns: int = 0,
        dropped_frame: bool = False,
        drop_reason: str | None = None,
        keyframe_selected: bool | None = None,
        keyframe_reason: str | None = None,
        capture_queue_depth: int = 0,
        map_queue_depth: int = 0,
        metadata: dict[str, object] | None = None,
        paths: dict[str, str] | None = None,
    ) -> None:
        if drop_reason is not None and drop_reason not in DROP_REASONS:
            raise ValueError(f"drop_reason: unsupported reason {drop_reason!r}")
        if keyframe_reason is not None and keyframe_reason not in KEYFRAME_REASONS:
            raise ValueError(f"keyframe_reason: unsupported reason {keyframe_reason!r}")
        self.events.append(
            {
                "capture_queue_depth": capture_queue_depth,
                "drop_reason": drop_reason,
                "dropped_frame": dropped_frame,
                "event_index": len(self.events),
                "format_name": LIVE_REPLAY_EVENT_FORMAT_NAME,
                "format_version": LIVE_REPLAY_FORMAT_VERSION,
                "frame_id": frame_id,
                "keyframe_reason": keyframe_reason,
                "keyframe_selected": keyframe_selected,
                "latency_ns": latency_ns,
                "map_queue_depth": map_queue_depth,
                "metadata": metadata or {},
                "paths": paths or {},
                "stage_name": stage_name,
                "timestamp_ns": timestamp_ns,
            }
        )


__all__ = [
    "DROP_REASONS",
    "KEYFRAME_REASONS",
    "LIVE_REPLAY_EVENT_FORMAT_NAME",
    "LIVE_REPLAY_FORMAT_VERSION",
    "LIVE_REPLAY_SUMMARY_FORMAT_NAME",
    "LiveReplayConfig",
    "LiveReplayEventRecorder",
    "LiveReplayState",
]
