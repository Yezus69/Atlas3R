"""Chronological clip-cache stream builder for student runtime inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)


@dataclass(frozen=True)
class StreamFrame:
    """One unique chronological frame loaded from a clip-cache payload."""

    stream_index: int
    frame_id: int
    timestamp_s: float
    K: npt.NDArray[np.float32]
    T_world_camera: npt.NDArray[np.float32]
    rgb_u8: npt.NDArray[np.uint8]
    depth_m: npt.NDArray[np.float32]
    valid_depth_mask: npt.NDArray[np.bool_]
    source_clip_id: int
    source_clip_offset: int
    source_payload_path: str
    duplicate_source_count: int

    def metadata(self) -> dict[str, object]:
        """Return compact JSON-safe source metadata."""

        return {
            "stream_index": self.stream_index,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "source_clip_id": self.source_clip_id,
            "source_clip_offset": self.source_clip_offset,
            "source_payload_path": self.source_payload_path,
            "duplicate_source_count": self.duplicate_source_count,
        }


@dataclass(frozen=True)
class StreamWindow:
    """A deterministic temporal inference window around one output frame."""

    output: StreamFrame
    frames: tuple[StreamFrame, ...]
    center_index: int
    requested_stream_indices: tuple[int, ...]
    source_stream_indices: tuple[int, ...]
    padded_positions: tuple[int, ...]
    padding_policy: str

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self.frames)

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_s for frame in self.frames)

    def metadata(self) -> dict[str, object]:
        """Return compact JSON-safe window metadata."""

        return {
            "output_frame_id": self.output.frame_id,
            "output_stream_index": self.output.stream_index,
            "window_frame_ids": list(self.frame_ids),
            "window_timestamps_s": list(self.timestamps_s),
            "center_index": self.center_index,
            "requested_stream_indices": list(self.requested_stream_indices),
            "source_stream_indices": list(self.source_stream_indices),
            "padded_positions": list(self.padded_positions),
            "padding_policy": self.padding_policy,
        }


def load_unique_frame_stream(
    clip_cache: str | Path,
    *,
    max_frames: int | None = None,
) -> tuple[StreamFrame, ...]:
    """Load a chronological stream of unique frames from overlapping clip-cache clips."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")
    manifest_path = manifest_path_from_input(clip_cache)
    manifest = load_clip_cache_manifest(manifest_path)
    clip_entries = _entries(manifest, "clips")
    clip_length = _int_field(manifest, "clip_length")
    height = _int_field(manifest, "image_height")
    width = _int_field(manifest, "image_width")
    first_by_frame_id: dict[int, StreamFrame] = {}
    duplicate_counts: dict[int, int] = {}
    for clip_entry in clip_entries:
        payload = read_clip_payload_from_entry(manifest_path.parent, clip_entry)
        validate_clip_payload(payload, clip_length=clip_length, height=height, width=width)
        clip_id = _int_field(clip_entry, "clip_id")
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
        timestamps = np.asarray(payload["timestamps_s"], dtype=np.float64)
        for offset, frame_id_value in enumerate(frame_ids.tolist()):
            frame = _frame_from_payload(
                payload=payload,
                clip_entry=clip_entry,
                clip_id=clip_id,
                offset=offset,
                frame_id=int(frame_id_value),
                timestamp_s=float(timestamps[offset]),
            )
            existing = first_by_frame_id.get(frame.frame_id)
            if existing is None:
                first_by_frame_id[frame.frame_id] = frame
                duplicate_counts[frame.frame_id] = 1
                continue
            _validate_duplicate_frame(existing, frame)
            duplicate_counts[frame.frame_id] += 1
    if not first_by_frame_id:
        raise ValueError(f"{manifest_path}: no frames found in clip cache")
    ordered = sorted(first_by_frame_id.values(), key=lambda item: (item.timestamp_s, item.frame_id))
    if max_frames is not None:
        ordered = ordered[:max_frames]
    return tuple(
        replace(
            frame,
            stream_index=index,
            duplicate_source_count=duplicate_counts[frame.frame_id],
        )
        for index, frame in enumerate(ordered)
    )


def build_stream_window(
    stream: tuple[StreamFrame, ...],
    *,
    stream_index: int,
    window_size: int,
) -> StreamWindow:
    """Build a fixed-length window with deterministic edge padding by frame reuse."""

    if not stream:
        raise ValueError("stream: must contain at least one frame")
    if stream_index < 0 or stream_index >= len(stream):
        raise ValueError("stream_index: out of range")
    if window_size <= 0:
        raise ValueError("window_size: must be positive")
    center_index = window_size // 2
    requested = tuple(stream_index + position - center_index for position in range(window_size))
    source_indices = tuple(min(max(index, 0), len(stream) - 1) for index in requested)
    padded_positions = tuple(
        position
        for position, (raw, clamped) in enumerate(zip(requested, source_indices, strict=True))
        if raw != clamped
    )
    return StreamWindow(
        output=stream[stream_index],
        frames=tuple(stream[index] for index in source_indices),
        center_index=center_index,
        requested_stream_indices=requested,
        source_stream_indices=source_indices,
        padded_positions=padded_positions,
        padding_policy="edge_reuse_clamp",
    )


def _frame_from_payload(
    *,
    payload: dict[str, Any],
    clip_entry: dict[str, object],
    clip_id: int,
    offset: int,
    frame_id: int,
    timestamp_s: float,
) -> StreamFrame:
    return StreamFrame(
        stream_index=-1,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        K=np.asarray(payload["K"][offset], dtype=np.float32).copy(),
        T_world_camera=np.asarray(payload["T_world_camera"][offset], dtype=np.float32).copy(),
        rgb_u8=np.asarray(payload["images_rgb_u8"][offset], dtype=np.uint8).copy(),
        depth_m=np.asarray(payload["depth_m"][offset], dtype=np.float32).copy(),
        valid_depth_mask=np.asarray(payload["valid_depth_mask"][offset], dtype=np.bool_).copy(),
        source_clip_id=clip_id,
        source_clip_offset=offset,
        source_payload_path=str(clip_entry["payload_path"]),
        duplicate_source_count=1,
    )


def _validate_duplicate_frame(first: StreamFrame, duplicate: StreamFrame) -> None:
    checks = (
        ("timestamp_s", np.array(first.timestamp_s), np.array(duplicate.timestamp_s)),
        ("K", first.K, duplicate.K),
        ("T_world_camera", first.T_world_camera, duplicate.T_world_camera),
        ("depth_m", first.depth_m, duplicate.depth_m),
    )
    for field_name, left, right in checks:
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=1e-5):
            raise ValueError(f"duplicate frame_id {first.frame_id}: {field_name} differs")
    if not np.array_equal(first.rgb_u8, duplicate.rgb_u8):
        raise ValueError(f"duplicate frame_id {first.frame_id}: rgb_u8 differs")
    if not np.array_equal(first.valid_depth_mask, duplicate.valid_depth_mask):
        raise ValueError(f"duplicate frame_id {first.frame_id}: valid_depth_mask differs")


def _entries(manifest: dict[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    entries: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.{key}[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], item))
    return entries


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "StreamFrame",
    "StreamWindow",
    "build_stream_window",
    "load_unique_frame_stream",
]
