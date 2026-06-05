"""Frame and camera contracts for dependency-safe ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts._arrays import (
    array_to_list,
    as_float,
    as_float32_array,
    as_int,
    as_uint8_array,
    require_probability,
)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    K: NDArray[np.float32]
    distortion_model: str = "none"
    distortion_params: NDArray[np.float32] | None = None
    rolling_shutter_row_time_s: float | None = None
    confidence: float = 0.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        as_float32_array(self.K, (3, 3), "K")
        if float(self.K[0, 0]) <= 0 or float(self.K[1, 1]) <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.distortion_model not in {"none", "brown_conrady", "fisheye"}:
            raise ValueError(f"unknown distortion_model: {self.distortion_model}")
        if self.distortion_params is not None:
            as_float32_array(self.distortion_params, (None,), "distortion_params")
        if self.rolling_shutter_row_time_s is not None and self.rolling_shutter_row_time_s < 0:
            raise ValueError("rolling_shutter_row_time_s must be non-negative")
        require_probability(float(self.confidence), "confidence")
        if not self.source:
            raise ValueError("camera source must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "K": array_to_list(self.K),
            "distortion_model": self.distortion_model,
            "distortion_params": array_to_list(self.distortion_params),
            "rolling_shutter_row_time_s": self.rolling_shutter_row_time_s,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CameraModel:
        return cls(
            width=as_int(data["width"], "width"),
            height=as_int(data["height"], "height"),
            K=as_float32_array(data["K"], (3, 3), "K"),
            distortion_model=str(data.get("distortion_model", "none")),
            distortion_params=(
                None
                if data.get("distortion_params") is None
                else as_float32_array(data["distortion_params"], (None,), "distortion_params")
            ),
            rolling_shutter_row_time_s=(
                None
                if data.get("rolling_shutter_row_time_s") is None
                else as_float(data["rolling_shutter_row_time_s"], "rolling_shutter_row_time_s")
            ),
            confidence=as_float(data.get("confidence", 0.0), "confidence"),
            source=str(data.get("source", "unknown")),
        )


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ns: int
    rgb_u8: NDArray[np.uint8]
    K_original: NDArray[np.float32] | None
    K_model: NDArray[np.float32]
    resize_transform: NDArray[np.float32]
    camera_metadata: Mapping[str, object]
    source_uri: str = ""

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        rgb = as_uint8_array(self.rgb_u8, (None, None, 3), "rgb_u8")
        if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
            raise ValueError("rgb_u8 must have positive height and width")
        if self.K_original is not None:
            as_float32_array(self.K_original, (3, 3), "K_original")
        as_float32_array(self.K_model, (3, 3), "K_model")
        as_float32_array(self.resize_transform, (3, 3), "resize_transform")

    @property
    def height(self) -> int:
        return int(self.rgb_u8.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb_u8.shape[1])

    def rgb_model_f32(self) -> NDArray[np.float32]:
        return np.transpose(self.rgb_u8.astype(np.float32) / 255.0, (2, 0, 1))

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "rgb_u8": array_to_list(self.rgb_u8),
            "K_original": array_to_list(self.K_original),
            "K_model": array_to_list(self.K_model),
            "resize_transform": array_to_list(self.resize_transform),
            "camera_metadata": dict(self.camera_metadata),
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FramePacket:
        return cls(
            frame_id=as_int(data["frame_id"], "frame_id"),
            timestamp_ns=as_int(data["timestamp_ns"], "timestamp_ns"),
            rgb_u8=as_uint8_array(data["rgb_u8"], (None, None, 3), "rgb_u8"),
            K_original=(
                None
                if data.get("K_original") is None
                else as_float32_array(data["K_original"], (3, 3), "K_original")
            ),
            K_model=as_float32_array(data["K_model"], (3, 3), "K_model"),
            resize_transform=as_float32_array(data["resize_transform"], (3, 3), "resize_transform"),
            camera_metadata=_mapping(data.get("camera_metadata", {}), "camera_metadata"),
            source_uri=str(data.get("source_uri", "")),
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
