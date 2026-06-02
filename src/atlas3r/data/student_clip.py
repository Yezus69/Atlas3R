"""Bridge FramePacket clips into the NumPy-only student model boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api import FramePacket
from atlas3r.models.student.contracts import CAMERA_COORDINATE_FRAME, StudentClipInput

Array = npt.NDArray[Any]


def student_clip_from_frame_packets(
    frames: Sequence[FramePacket],
    *,
    batch_id: str = "frame-packet-clip",
) -> StudentClipInput:
    """Convert an ordered FramePacket sequence into one StudentClipInput batch."""

    packets = _validate_frame_sequence(frames)
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id: must be a non-empty string")

    frame_ids = _validate_unique_frame_ids(packets)
    image_shape = _validate_shared_rgb_model_shape(packets)
    _validate_k_model_shapes(packets)

    images_rgb = np.stack([np.asarray(packet.rgb_model) for packet in packets], axis=0)
    if images_rgb.shape != (len(packets), *image_shape):
        raise ValueError("images_rgb: internal stack shape mismatch")
    images_rgb = images_rgb[np.newaxis, ...].copy()
    intrinsics = np.stack([np.asarray(packet.K_model) for packet in packets], axis=0)
    intrinsics = intrinsics[np.newaxis, ...].astype(np.float32, copy=True)

    metadata: dict[str, object] = {
        "batch_id": batch_id,
        "source": "frame_packets",
        "frame_count": len(packets),
        "image_layout": "B,T,3,H,W",
        "input_order_preserved": True,
        "coordinate_frame": CAMERA_COORDINATE_FRAME,
        "truth_boundary_note": "Clip assembly only; no learned inference or mapper output.",
    }
    source_formats = _source_formats(packets)
    if source_formats:
        metadata["source_formats"] = source_formats

    return StudentClipInput(
        frame_ids=frame_ids,
        images_rgb=images_rgb,
        intrinsics=intrinsics,
        metadata=metadata,
    )


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


def _validate_shared_rgb_model_shape(packets: tuple[FramePacket, ...]) -> tuple[int, int, int]:
    first_shape = _validate_rgb_model_shape("frames[0].rgb_model", packets[0].rgb_model)
    for index, packet in enumerate(packets[1:], start=1):
        shape = _validate_rgb_model_shape(f"frames[{index}].rgb_model", packet.rgb_model)
        if shape != first_shape:
            raise ValueError(
                f"frames[{index}].rgb_model: shape must match frames[0].rgb_model "
                f"{first_shape}, got {shape}"
            )
    return first_shape


def _validate_rgb_model_shape(field_name: str, value: Array) -> tuple[int, int, int]:
    rgb_model = np.asarray(value)
    if rgb_model.ndim != 3 or rgb_model.shape[0] != 3:
        raise ValueError(f"{field_name}: must have shape 3xHxW")
    height = int(rgb_model.shape[1])
    width = int(rgb_model.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError(f"{field_name}: H and W must be positive")
    return (3, height, width)


def _validate_k_model_shapes(packets: tuple[FramePacket, ...]) -> None:
    for index, packet in enumerate(packets):
        if np.asarray(packet.K_model).shape != (3, 3):
            raise ValueError(f"frames[{index}].K_model: must have shape 3x3")


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
    "student_clip_from_frame_packets",
]
