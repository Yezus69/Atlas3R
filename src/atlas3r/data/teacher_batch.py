"""Bridge RGB FramePacket records into the teacher adapter FrameBatch boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from atlas3r.api import FramePacket
from atlas3r.data.frame_source import RGBFrameSource
from atlas3r.models.adapters import FrameBatch

_CAMERA_COORDINATE_FRAME = "x_right_y_down_z_forward"
_RESERVED_METADATA_KEYS = frozenset(
    {
        "source",
        "frame_count",
        "frame_ids",
        "input_order_preserved",
        "coordinate_frame",
        "truth_boundary_note",
        "source_formats",
    }
)


def teacher_frame_batch_from_frame_packets(
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "frame-packet-teacher-batch",
    metadata: Mapping[str, Any] | None = None,
) -> FrameBatch:
    """Convert an ordered FramePacket sequence into one teacher FrameBatch."""

    return _frame_batch_from_packets(
        frames,
        batch_id=batch_id,
        metadata=metadata,
        source="frame_packets",
    )


def teacher_frame_batch_from_rgb_source(
    source: RGBFrameSource,
    *,
    batch_id: str = "rgb-frame-source-teacher-batch",
    metadata: Mapping[str, Any] | None = None,
) -> FrameBatch:
    """Read an RGBFrameSource once and convert its packets into a teacher FrameBatch."""

    return _frame_batch_from_packets(
        tuple(source.frames()),
        batch_id=batch_id,
        metadata=metadata,
        source="rgb_frame_source",
    )


def _frame_batch_from_packets(
    frames: Sequence[FramePacket],
    *,
    batch_id: str,
    metadata: Mapping[str, Any] | None,
    source: str,
) -> FrameBatch:
    packets = _validate_frame_sequence(frames)
    frame_ids = _validate_unique_frame_ids(packets)
    batch_metadata = _base_metadata(packets, frame_ids=frame_ids, source=source)
    _merge_extra_metadata(batch_metadata, metadata)
    return FrameBatch(frames=packets, batch_id=batch_id, metadata=batch_metadata)


def _validate_frame_sequence(frames: Sequence[FramePacket]) -> tuple[FramePacket, ...]:
    if not isinstance(frames, Sequence):
        raise ValueError("frames: must be a sequence of FramePacket records")
    if len(frames) == 0:
        raise ValueError("frames: must contain at least one FramePacket")
    packets: list[FramePacket] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, FramePacket):
            raise ValueError(f"frames[{index}]: must be a FramePacket")
        packets.append(frame)
    return tuple(packets)


def _validate_unique_frame_ids(packets: tuple[FramePacket, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    frame_ids: list[int] = []
    for index, packet in enumerate(packets):
        frame_id = packet.frame_id
        if frame_id in seen:
            raise ValueError(f"frames[{index}].frame_id: duplicate frame_id {frame_id}")
        seen.add(frame_id)
        frame_ids.append(frame_id)
    return tuple(frame_ids)


def _base_metadata(
    packets: tuple[FramePacket, ...],
    *,
    frame_ids: tuple[int, ...],
    source: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source,
        "frame_count": len(packets),
        "frame_ids": frame_ids,
        "input_order_preserved": True,
        "coordinate_frame": _CAMERA_COORDINATE_FRAME,
        "truth_boundary_note": "Batch assembly only; no teacher inference or geometry output.",
    }
    source_formats = _source_formats(packets)
    if source_formats:
        metadata["source_formats"] = source_formats
    return metadata


def _merge_extra_metadata(
    batch_metadata: dict[str, Any],
    extra_metadata: Mapping[str, Any] | None,
) -> None:
    if extra_metadata is None:
        return
    if not isinstance(extra_metadata, Mapping):
        raise ValueError("metadata: must be a mapping")
    for key, value in extra_metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"metadata[{key!r}]: keys must be strings")
        if key in _RESERVED_METADATA_KEYS:
            raise ValueError(f"metadata.{key}: reserved bridge metadata field")
        batch_metadata[key] = value


def _source_formats(packets: tuple[FramePacket, ...]) -> tuple[str, ...]:
    source_formats: list[str] = []
    seen: set[str] = set()
    for packet in packets:
        value = packet.camera_metadata.get("source_format")
        if value is None:
            continue
        source_format = str(value)
        if source_format not in seen:
            seen.add(source_format)
            source_formats.append(source_format)
    return tuple(source_formats)


__all__ = [
    "teacher_frame_batch_from_frame_packets",
    "teacher_frame_batch_from_rgb_source",
]
