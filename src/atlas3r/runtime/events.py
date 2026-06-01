"""Deterministic runtime event contracts for smoke scheduler plumbing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

RUNTIME_EVENT_LOG_FORMAT_NAME = "atlas3r_runtime_fixture_event_log"
RUNTIME_EVENT_LOG_FORMAT_VERSION = 1


class RuntimeStage(str, Enum):
    """Known deterministic scheduler stages for the Phase 2D fixture smoke."""

    RUNTIME_START = "runtime_start"
    SESSION_WRITE = "session_write"
    SOURCE_FRAME = "source_frame"
    ADAPTER_CACHE_WRITE = "adapter_cache_write"
    ADAPTER_CACHE_FRAME = "adapter_cache_frame"
    TSDF_REPLAY = "tsdf_replay"
    TSDF_REPLAY_FRAME = "tsdf_replay_frame"
    TSDF_OUTPUT_WRITE = "tsdf_output_write"
    RUNTIME_COMPLETE = "runtime_complete"


@dataclass(frozen=True)
class BoundedMemoryCounters:
    """Deterministic scheduler-owned bounded-memory counters."""

    configured_frame_array_bound: int
    frame_arrays_in_memory: int
    peak_frame_arrays_in_memory: int
    processed_frame_count: int
    dropped_frame_count: int
    queued_frame_count: int

    def __post_init__(self) -> None:
        if self.configured_frame_array_bound <= 0:
            raise ValueError("configured_frame_array_bound: must be positive")
        fields = {
            "frame_arrays_in_memory": self.frame_arrays_in_memory,
            "peak_frame_arrays_in_memory": self.peak_frame_arrays_in_memory,
            "processed_frame_count": self.processed_frame_count,
            "dropped_frame_count": self.dropped_frame_count,
            "queued_frame_count": self.queued_frame_count,
        }
        for field_name, value in fields.items():
            if value < 0:
                raise ValueError(f"{field_name}: must be non-negative")
        if self.frame_arrays_in_memory > self.configured_frame_array_bound:
            raise ValueError("frame_arrays_in_memory: exceeds configured bound")
        if self.peak_frame_arrays_in_memory > self.configured_frame_array_bound:
            raise ValueError("peak_frame_arrays_in_memory: exceeds configured bound")

    def to_json_record(self) -> dict[str, int]:
        """Return a deterministic JSON-serializable counter record."""
        return {
            "configured_frame_array_bound": self.configured_frame_array_bound,
            "dropped_frame_count": self.dropped_frame_count,
            "frame_arrays_in_memory": self.frame_arrays_in_memory,
            "peak_frame_arrays_in_memory": self.peak_frame_arrays_in_memory,
            "processed_frame_count": self.processed_frame_count,
            "queued_frame_count": self.queued_frame_count,
        }


@dataclass(frozen=True)
class RuntimeEvent:
    """One deterministic per-stage runtime event record."""

    event_index: int
    stage_name: RuntimeStage | str
    timestamp_ns: int
    latency_ns: int
    dropped_frame: bool
    memory_counters: BoundedMemoryCounters
    frame_id: int | None = None
    paths: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_index < 0:
            raise ValueError("event_index: must be non-negative")
        if self.frame_id is not None and self.frame_id < 0:
            raise ValueError("frame_id: must be non-negative when present")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns: must be non-negative")
        if self.latency_ns < 0:
            raise ValueError("latency_ns: must be non-negative")
        stage_name = (
            self.stage_name.value if isinstance(self.stage_name, RuntimeStage) else self.stage_name
        )
        if not stage_name:
            raise ValueError("stage_name: must be non-empty")
        object.__setattr__(self, "stage_name", stage_name)
        object.__setattr__(self, "paths", dict(self.paths))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable event record."""
        return {
            "dropped_frame": self.dropped_frame,
            "event_index": self.event_index,
            "format_name": RUNTIME_EVENT_LOG_FORMAT_NAME,
            "format_version": RUNTIME_EVENT_LOG_FORMAT_VERSION,
            "frame_id": self.frame_id,
            "latency_ns": self.latency_ns,
            "memory_counters": self.memory_counters.to_json_record(),
            "metadata": dict(self.metadata),
            "paths": dict(self.paths),
            "stage_name": str(self.stage_name),
            "timestamp_ns": self.timestamp_ns,
        }


__all__ = [
    "BoundedMemoryCounters",
    "RUNTIME_EVENT_LOG_FORMAT_NAME",
    "RUNTIME_EVENT_LOG_FORMAT_VERSION",
    "RuntimeEvent",
    "RuntimeStage",
]
