"""Array normalization helpers for Depth Pro teacher-signal outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np
import numpy.typing as npt


@dataclass
class DepthProRunStats:
    clip_count: int = 0
    clip_frame_slot_count: int = 0
    unique_frame_prediction_count: int = 0
    duplicate_prediction_reuse_count: int = 0
    resized_prediction_array_count: int = 0
    _resize_events: dict[tuple[str, tuple[int, int], tuple[int, int], str], int] = field(
        default_factory=dict
    )

    def record_clip(self, frame_slot_count: int) -> None:
        self.clip_count += 1
        self.clip_frame_slot_count += frame_slot_count

    def record_unique_prediction(self) -> None:
        self.unique_frame_prediction_count += 1

    def record_duplicate_reuse(self) -> None:
        self.duplicate_prediction_reuse_count += 1

    def record_resize(
        self,
        *,
        key: str,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
        method: str,
    ) -> None:
        self.resized_prediction_array_count += 1
        event_key = (key, source_shape, target_shape, method)
        self._resize_events[event_key] = self._resize_events.get(event_key, 0) + 1

    def result_fields(self) -> dict[str, object]:
        return {
            "clip_count": self.clip_count,
            "clip_frame_slot_count": self.clip_frame_slot_count,
            "unique_frame_prediction_count": self.unique_frame_prediction_count,
            "duplicate_prediction_reuse_count": self.duplicate_prediction_reuse_count,
            "resized_prediction_array_count": self.resized_prediction_array_count,
            "prediction_resize_events": self.resize_events(),
        }

    def source_metadata(self) -> dict[str, object]:
        return {
            "clip_count": self.clip_count,
            "clip_frame_slot_count": self.clip_frame_slot_count,
            "unique_frame_prediction_count": self.unique_frame_prediction_count,
            "duplicate_prediction_reuse_count": self.duplicate_prediction_reuse_count,
            "prediction_resize_policy": (
                "Resize Depth Pro depth/confidence/sigma arrays to the source clip-cache "
                "height and width with Pillow bilinear when available, otherwise NumPy "
                "nearest-neighbor."
            ),
            "resized_prediction_array_count": self.resized_prediction_array_count,
            "prediction_resize_events": self.resize_events(),
        }

    def resize_events(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for (key, source_shape, target_shape, method), count in sorted(self._resize_events.items()):
            events.append(
                {
                    "array": key,
                    "source_shape_hw": list(source_shape),
                    "target_shape_hw": list(target_shape),
                    "method": method,
                    "count": count,
                }
            )
        return events


def normalise_frame_prediction_arrays(
    *,
    depth_m: npt.NDArray[Any],
    confidence: npt.NDArray[Any] | None,
    depth_sigma_m: npt.NDArray[Any] | None,
    height: int,
    width: int,
    stats: DepthProRunStats,
) -> dict[str, npt.NDArray[Any]]:
    depth = _image_float32(depth_m, "depth_m", height, width, stats=stats)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth = np.where(valid, depth, np.float32(0.0)).astype(np.float32, copy=False)
    if confidence is None:
        confidence_array = np.where(valid, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    else:
        confidence_array = _image_float32(confidence, "confidence", height, width, stats=stats)
        _validate_probability("confidence", confidence_array)
        confidence_array = np.where(valid, confidence_array, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
    valid = valid & (confidence_array > 0.0)
    if depth_sigma_m is None:
        sigma = _heuristic_sigma(
            depth, confidence_array, valid, confidence_provided=confidence is not None
        )
    else:
        sigma = _image_float32(depth_sigma_m, "depth_sigma_m", height, width, stats=stats)
        if np.any(sigma < 0.0):
            raise ValueError("depth_sigma_m: must be non-negative")
        if np.any(sigma[valid] <= 0.0):
            raise ValueError("depth_sigma_m: must be positive on valid pixels")
        sigma = np.where(valid, sigma, np.float32(0.0)).astype(np.float32, copy=False)
    return {
        "depth_m": np.where(valid, depth, np.float32(0.0)).astype(np.float32, copy=False),
        "depth_sigma_m": sigma,
        "confidence": np.where(valid, confidence_array, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        ),
        "valid_mask": valid.astype(np.bool_, copy=False),
    }


def _heuristic_sigma(
    depth: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    *,
    confidence_provided: bool,
) -> npt.NDArray[np.float32]:
    sigma = np.zeros_like(depth, dtype=np.float32)
    base = np.maximum(np.float32(0.05), depth * np.float32(0.05))
    if confidence_provided:
        base = base / np.maximum(confidence, np.float32(0.25))
    sigma[valid] = base[valid]
    return sigma


def _image_float32(
    value: npt.NDArray[Any],
    key: str,
    height: int,
    width: int,
    *,
    stats: DepthProRunStats,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{key}: expected a 2D image array, got shape {array.shape}")
    if array.shape != (height, width):
        source_shape = (int(array.shape[0]), int(array.shape[1]))
        array, method = _resize_float32_image(array, height=height, width=width)
        stats.record_resize(
            key=key,
            source_shape=source_shape,
            target_shape=(height, width),
            method=method,
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: must contain finite values")
    return array.astype(np.float32, copy=False)


def _resize_float32_image(
    array: npt.NDArray[np.float32],
    *,
    height: int,
    width: int,
) -> tuple[npt.NDArray[np.float32], str]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return _resize_nearest(array, height=height, width=width), "numpy_nearest"
    image = image_module.fromarray(array.astype(np.float32, copy=False), mode="F")
    if hasattr(image_module, "Resampling"):
        resampling = image_module.Resampling.BILINEAR
    else:
        resampling = image_module.BILINEAR
    resized = image.resize((width, height), resample=resampling)
    return np.asarray(resized, dtype=np.float32), "pillow_bilinear"


def _resize_nearest(
    array: npt.NDArray[np.float32],
    *,
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    source_height, source_width = array.shape
    y_indices = np.floor((np.arange(height, dtype=np.float64) + 0.5) * source_height / height)
    x_indices = np.floor((np.arange(width, dtype=np.float64) + 0.5) * source_width / width)
    y = np.clip(y_indices.astype(np.int64), 0, source_height - 1)
    x = np.clip(x_indices.astype(np.int64), 0, source_width - 1)
    resized = array[y[:, np.newaxis], x[np.newaxis, :]].astype(np.float32, copy=False)
    return cast(npt.NDArray[np.float32], resized)


def _validate_probability(key: str, array: npt.NDArray[np.float32]) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{key}: must contain values in [0, 1]")


__all__ = [
    "DepthProRunStats",
    "normalise_frame_prediction_arrays",
]
