"""Array normalization and diagnostic alignment for VGGT teacher outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.teachers.external.vggt_alignment import (
    VGGTPoseAlignment,
    alignment_for_policy,
    apply_alignment_to_points,
    apply_alignment_to_transforms,
)

ALIGNMENT_POLICIES = {"diagnostic_sim3", "diagnostic_se3", "none"}
_DEPTH_SIGMA_HEURISTIC = (
    "If VGGT output does not provide depth_sigma_m, Atlas3R sets sigma to "
    "max(0.05 m, 0.05 * depth_m) / max(confidence, 0.25); invalid pixels use "
    "depth/sigma/confidence 0."
)


@dataclass
class VGGTRunStats:
    clip_count: int = 0
    resized_prediction_array_count: int = 0
    output_fields_found: set[str] = field(default_factory=set)
    confidence_normalization: str = "not_used"
    _resize_events: dict[tuple[str, tuple[int, int], tuple[int, int], str], int] = field(
        default_factory=dict
    )
    _alignments: list[dict[str, object]] = field(default_factory=list)

    def record_clip(self) -> None:
        self.clip_count += 1

    def record_field(self, key: str) -> None:
        self.output_fields_found.add(key)

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

    def record_alignment(self, metadata: Mapping[str, object]) -> None:
        self._alignments.append(dict(metadata))

    def result_fields(self) -> dict[str, object]:
        return {
            "clip_count": self.clip_count,
            "resized_prediction_array_count": self.resized_prediction_array_count,
            "output_fields_found": sorted(self.output_fields_found),
            "prediction_resize_events": self.resize_events(),
            "alignment_records": self.alignment_records(),
        }

    def source_metadata(self) -> dict[str, object]:
        return {
            "clip_count": self.clip_count,
            "depth_sigma_m_heuristic": _DEPTH_SIGMA_HEURISTIC,
            "confidence_normalization": self.confidence_normalization,
            "prediction_resize_policy": (
                "Resize VGGT sequence arrays to source clip-cache height and width with "
                "Pillow bilinear when available, otherwise NumPy nearest-neighbor."
            ),
            "resized_prediction_array_count": self.resized_prediction_array_count,
            "prediction_resize_events": self.resize_events(),
            "output_fields_found": sorted(self.output_fields_found),
            "alignment_records": self.alignment_records(),
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

    def alignment_records(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._alignments]


def normalise_vggt_clip_prediction(
    raw_prediction: object,
    *,
    clip_payload: Mapping[str, Any],
    align_to_source_pose: str,
    stats: VGGTRunStats,
) -> dict[str, npt.NDArray[Any] | np.generic]:
    """Convert one VGGT prediction object/dict into a teacher-signal payload."""

    if align_to_source_pose not in ALIGNMENT_POLICIES:
        raise ValueError("align_to_source_pose: expected diagnostic_sim3, diagnostic_se3, or none")
    stats.record_clip()
    raw = _prediction_mapping(raw_prediction)
    clip_depth = np.asarray(clip_payload["depth_m"], dtype=np.float32)
    clip_length, height, width = clip_depth.shape

    depth_value, depth_key = _required_prediction(raw, ("depth_m", "depth", "depth_map", "depths"))
    stats.record_field(depth_key)
    depth, model_hw = _sequence_image_float32(
        depth_value,
        depth_key,
        clip_length,
        height,
        width,
        stats=stats,
    )
    confidence_value, confidence_key = _optional_prediction(
        raw,
        ("confidence", "depth_conf", "depth_confidence", "conf"),
    )
    if confidence_value is None:
        confidence = np.where(depth > 0.0, np.float32(0.5), np.float32(0.0))
        stats.confidence_normalization = "missing_confidence_default_0.5"
    else:
        stats.record_field(confidence_key)
        confidence, _confidence_hw = _sequence_image_float32(
            confidence_value,
            confidence_key,
            clip_length,
            height,
            width,
            stats=stats,
        )
        confidence = _normalise_confidence(confidence, stats)
    valid = np.isfinite(depth) & (depth > 0.0) & (confidence > 0.0)
    depth = np.where(valid, depth, np.float32(0.0)).astype(np.float32, copy=False)
    confidence = np.where(valid, confidence, np.float32(0.0)).astype(np.float32, copy=False)

    sigma_value, sigma_key = _optional_prediction(
        raw,
        ("depth_sigma_m", "depth_uncertainty_m", "uncertainty"),
    )
    if sigma_value is None:
        sigma = _heuristic_sigma(depth, confidence, valid)
    else:
        stats.record_field(sigma_key)
        sigma, _sigma_hw = _sequence_image_float32(
            sigma_value,
            sigma_key,
            clip_length,
            height,
            width,
            stats=stats,
        )
        if np.any(sigma < 0.0):
            raise ValueError("depth_sigma_m: must be non-negative")
        if np.any(sigma[valid] <= 0.0):
            raise ValueError("depth_sigma_m: must be positive on valid pixels")
        sigma = np.where(valid, sigma, np.float32(0.0)).astype(np.float32, copy=False)

    T_world_camera, pose_source = _transforms_from_prediction(
        raw,
        clip_payload=clip_payload,
        clip_length=clip_length,
        stats=stats,
    )
    alignment = alignment_for_policy(
        source_T_world_camera=np.asarray(clip_payload["T_world_camera"], dtype=np.float32),
        teacher_T_world_camera=T_world_camera,
        policy=align_to_source_pose,
    )
    if alignment is not None:
        T_world_camera = apply_alignment_to_transforms(T_world_camera, alignment)
        stats.record_alignment(alignment.metadata(pose_source=pose_source))

    payload: dict[str, npt.NDArray[Any] | np.generic] = {
        "depth_m": depth,
        "depth_sigma_m": sigma,
        "confidence": confidence,
        "valid_mask": valid.astype(np.bool_, copy=False),
        "K": _intrinsics_from_prediction(
            raw,
            clip_payload=clip_payload,
            clip_length=clip_length,
            model_hw=model_hw,
            target_hw=(height, width),
            stats=stats,
        ),
        "T_world_camera": T_world_camera.astype(np.float32, copy=False),
        "frame_ids": np.asarray(clip_payload["frame_ids"], dtype=np.int32),
        "timestamps_s": np.asarray(clip_payload["timestamps_s"], dtype=np.float64),
    }
    _add_optional_pointmaps(
        payload,
        raw,
        clip_length=clip_length,
        height=height,
        width=width,
        stats=stats,
        alignment=alignment,
    )
    _add_optional_normal(
        payload,
        raw,
        clip_length=clip_length,
        height=height,
        width=width,
        stats=stats,
    )
    return payload


def _prediction_mapping(raw_prediction: object) -> dict[str, object]:
    if isinstance(raw_prediction, Mapping):
        return dict(raw_prediction)
    keys = (
        "depth",
        "depth_m",
        "depth_conf",
        "confidence",
        "depth_sigma_m",
        "extrinsic",
        "extrinsics",
        "intrinsic",
        "intrinsics",
        "K",
        "T_world_camera",
        "world_points",
        "pointmap_world_m",
        "pointmap_camera_m",
        "normal_camera",
    )
    return {key: getattr(raw_prediction, key) for key in keys if hasattr(raw_prediction, key)}


def _required_prediction(raw: Mapping[str, object], keys: tuple[str, ...]) -> tuple[object, str]:
    value, key = _optional_prediction(raw, keys)
    if value is None:
        raise ValueError(f"VGGT prediction: missing one of {keys}")
    return value, key


def _optional_prediction(
    raw: Mapping[str, object],
    keys: tuple[str, ...],
) -> tuple[object | None, str]:
    for key in keys:
        if key in raw:
            return raw[key], key
    return None, keys[0]


def _sequence_image_float32(
    value: object,
    key: str,
    clip_length: int,
    height: int,
    width: int,
    *,
    stats: VGGTRunStats,
) -> tuple[npt.NDArray[np.float32], tuple[int, int]]:
    array = _to_numpy(value).astype(np.float32, copy=False)
    array = _strip_batch_and_channel_dims(array, clip_length=clip_length, key=key)
    if array.ndim != 3 or array.shape[0] != clip_length:
        raise ValueError(f"{key}: expected shape T,H,W, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: must contain finite values")
    source_hw = (int(array.shape[1]), int(array.shape[2]))
    if source_hw != (height, width):
        resized = np.zeros((clip_length, height, width), dtype=np.float32)
        method = "none"
        for index in range(clip_length):
            resized[index], method = _resize_float32_image(array[index], height=height, width=width)
        stats.record_resize(
            key=key,
            source_shape=source_hw,
            target_shape=(height, width),
            method=method,
        )
        array = resized
    return array.astype(np.float32, copy=False), source_hw


def _sequence_vector_float32(
    value: object,
    key: str,
    clip_length: int,
    height: int,
    width: int,
    *,
    stats: VGGTRunStats,
) -> npt.NDArray[np.float32]:
    array = _to_numpy(value).astype(np.float32, copy=False)
    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[0] == clip_length and array.shape[-1] == 3:
        sequence = array
    elif array.ndim == 4 and array.shape[0] == clip_length and array.shape[1] == 3:
        sequence = np.transpose(array, (0, 2, 3, 1))
    else:
        raise ValueError(f"{key}: expected shape T,H,W,3 or T,3,H,W, got {array.shape}")
    if not np.all(np.isfinite(sequence)):
        raise ValueError(f"{key}: must contain finite values")
    source_hw = (int(sequence.shape[1]), int(sequence.shape[2]))
    if source_hw != (height, width):
        resized = np.zeros((clip_length, height, width, 3), dtype=np.float32)
        method = "none"
        for frame_index in range(clip_length):
            for channel in range(3):
                resized[frame_index, :, :, channel], method = _resize_float32_image(
                    sequence[frame_index, :, :, channel],
                    height=height,
                    width=width,
                )
        stats.record_resize(
            key=key,
            source_shape=source_hw,
            target_shape=(height, width),
            method=method,
        )
        sequence = resized
    return sequence.astype(np.float32, copy=False)


def _strip_batch_and_channel_dims(
    array: npt.NDArray[np.float32],
    *,
    clip_length: int,
    key: str,
) -> npt.NDArray[np.float32]:
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    if array.ndim == 2 and clip_length == 1:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"{key}: expected sequence image array, got {array.shape}")
    return array


def _normalise_confidence(
    confidence: npt.NDArray[np.float32],
    stats: VGGTRunStats,
) -> npt.NDArray[np.float32]:
    if np.any(confidence < 0.0):
        raise ValueError("confidence: must be non-negative")
    max_value = float(np.max(confidence)) if confidence.size else 0.0
    if max_value <= 1.0:
        stats.confidence_normalization = "already_unit_interval"
        return confidence.astype(np.float32, copy=False)
    stats.confidence_normalization = "positive_confidence_x_over_1_plus_x"
    return (confidence / (confidence + np.float32(1.0))).astype(np.float32, copy=False)


def _heuristic_sigma(
    depth: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    sigma = np.zeros_like(depth, dtype=np.float32)
    base = np.maximum(np.float32(0.05), depth * np.float32(0.05))
    base = base / np.maximum(confidence, np.float32(0.25))
    sigma[valid] = base[valid]
    return sigma


def _intrinsics_from_prediction(
    raw: Mapping[str, object],
    *,
    clip_payload: Mapping[str, Any],
    clip_length: int,
    model_hw: tuple[int, int],
    target_hw: tuple[int, int],
    stats: VGGTRunStats,
) -> npt.NDArray[np.float32]:
    value, key = _optional_prediction(raw, ("K", "intrinsic", "intrinsics"))
    if value is None:
        return np.asarray(clip_payload["K"], dtype=np.float32)
    stats.record_field(key)
    K = _sequence_intrinsics(value, key=key, clip_length=clip_length)
    source_h, source_w = model_hw
    target_h, target_w = target_hw
    if source_w > 0 and source_h > 0 and (source_w, source_h) != (target_w, target_h):
        K = K.copy()
        K[:, 0, :] *= np.float32(target_w / source_w)
        K[:, 1, :] *= np.float32(target_h / source_h)
        K[:, 2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return K.astype(np.float32, copy=False)


def _sequence_intrinsics(value: object, *, key: str, clip_length: int) -> npt.NDArray[np.float32]:
    array = _to_numpy(value).astype(np.float32, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (3, 3):
        array = np.repeat(array[np.newaxis, :, :], clip_length, axis=0)
    if array.shape != (clip_length, 3, 3):
        raise ValueError(f"{key}: expected shape T,3,3, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _transforms_from_prediction(
    raw: Mapping[str, object],
    *,
    clip_payload: Mapping[str, Any],
    clip_length: int,
    stats: VGGTRunStats,
) -> tuple[npt.NDArray[np.float32], str]:
    value, key = _optional_prediction(raw, ("T_world_camera",))
    if value is not None:
        stats.record_field(key)
        return _sequence_transforms(value, key=key, clip_length=clip_length), "vggt_T_world_camera"
    value, key = _optional_prediction(raw, ("extrinsic", "extrinsics", "T_camera_world"))
    if value is not None:
        stats.record_field(key)
        return (
            _invert_camera_from_world_sequence(value, key=key, clip_length=clip_length),
            "vggt_extrinsic_camera_from_world",
        )
    return (
        np.asarray(clip_payload["T_world_camera"], dtype=np.float32),
        "source_clip_cache_pose_fallback",
    )


def _sequence_transforms(value: object, *, key: str, clip_length: int) -> npt.NDArray[np.float32]:
    array = _to_numpy(value).astype(np.float32, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (clip_length, 4, 4):
        raise ValueError(f"{key}: expected shape T,4,4, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _invert_camera_from_world_sequence(
    value: object,
    *,
    key: str,
    clip_length: int,
) -> npt.NDArray[np.float32]:
    array = _to_numpy(value).astype(np.float64, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (clip_length, 3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[np.newaxis, :, :], clip_length, axis=0)
        padded[:, :3, :] = array
        array = padded
    if array.shape != (clip_length, 4, 4):
        raise ValueError(f"{key}: expected shape T,3,4 or T,4,4, got {array.shape}")
    inverted = np.linalg.inv(array)
    return inverted.astype(np.float32, copy=False)


def _add_optional_pointmaps(
    payload: dict[str, npt.NDArray[Any] | np.generic],
    raw: Mapping[str, object],
    *,
    clip_length: int,
    height: int,
    width: int,
    stats: VGGTRunStats,
    alignment: VGGTPoseAlignment | None,
) -> None:
    value, key = _optional_prediction(raw, ("pointmap_camera_m", "points_camera", "pts3d_cam"))
    if value is not None:
        stats.record_field(key)
        payload["pointmap_camera_m"] = _sequence_vector_float32(
            value, key, clip_length, height, width, stats=stats
        )
    value, key = _optional_prediction(
        raw,
        ("pointmap_world_m", "world_points", "points_world", "pts3d", "points3d"),
    )
    if value is not None:
        stats.record_field(key)
        pointmap = _sequence_vector_float32(value, key, clip_length, height, width, stats=stats)
        if alignment is not None:
            pointmap = apply_alignment_to_points(pointmap, alignment)
        payload["pointmap_world_m"] = pointmap


def _add_optional_normal(
    payload: dict[str, npt.NDArray[Any] | np.generic],
    raw: Mapping[str, object],
    *,
    clip_length: int,
    height: int,
    width: int,
    stats: VGGTRunStats,
) -> None:
    value, key = _optional_prediction(raw, ("normal_camera", "normals_camera"))
    if value is None:
        return
    stats.record_field(key)
    payload["normal_camera"] = _sequence_vector_float32(
        value, key, clip_length, height, width, stats=stats
    )


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
    return np.asarray(image.resize((width, height), resample=resampling), dtype=np.float32), (
        "pillow_bilinear"
    )


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
    return cast(npt.NDArray[np.float32], array[y[:, np.newaxis], x[np.newaxis, :]])


def _to_numpy(value: object) -> npt.NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "ALIGNMENT_POLICIES",
    "VGGTRunStats",
    "normalise_vggt_clip_prediction",
]
