"""Student model boundary contracts for Atlas3R."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import (
    validate_confidence_array,
    validate_finite_numeric_array,
    validate_int_sequence,
    validate_intrinsics,
    validate_mapping,
    validate_nonempty_str,
    validate_same_shape,
    validate_transform,
)

Array = npt.NDArray[Any]

CAMERA_COORDINATE_FRAME = "x_right_y_down_z_forward"
STUDENT_TRUTH_BOUNDARY_FLAGS = (
    "shape_only",
    "learned_inference",
    "usable_for_mapping",
    "accuracy_report",
    "performance_report",
)


def _set(instance: object, field_name: str, value: object) -> None:
    object.__setattr__(instance, field_name, value)


def _validate_coordinate_frame(value: str) -> str:
    coordinate_frame = validate_nonempty_str("coordinate_frame", value)
    if coordinate_frame != CAMERA_COORDINATE_FRAME:
        raise ValueError(
            "coordinate_frame: must be x_right_y_down_z_forward for the Phase 3A "
            "student camera boundary"
        )
    return coordinate_frame


def _validate_metadata(field_name: str, value: Mapping[str, object]) -> dict[str, object]:
    metadata = validate_mapping(field_name, value)
    for key in metadata:
        if not isinstance(key, str):
            raise ValueError(f"{field_name}: keys must be strings")
    return dict(metadata)


def _validate_frame_ids(frame_ids: tuple[int, ...], frame_count: int) -> tuple[int, ...]:
    normalized = tuple(frame_ids)
    validate_int_sequence("frame_ids", normalized, non_empty=True)
    if len(normalized) != frame_count:
        raise ValueError("frame_ids: length must match clip time dimension T")
    return normalized


def _validate_images_rgb(value: Array) -> Array:
    images = np.asarray(value)
    if images.ndim != 5:
        raise ValueError("images_rgb: must have shape BxTx3xHxW")
    batch_size, frame_count, channel_count, height, width = images.shape
    if batch_size <= 0 or frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError("images_rgb: B, T, H, and W must be positive")
    if channel_count != 3:
        raise ValueError("images_rgb: channel count must be exactly 3")
    allowed_dtypes = (np.dtype(np.uint8), np.dtype(np.float32), np.dtype(np.float64))
    if images.dtype not in allowed_dtypes:
        raise ValueError("images_rgb: dtype must be uint8, float32, or float64")
    if np.issubdtype(images.dtype, np.floating) and not np.all(np.isfinite(images)):
        raise ValueError("images_rgb: floating-point images must contain only finite values")
    return images


def _validate_clip_intrinsics(
    field_name: str,
    value: Array,
    batch_size: int,
    frame_count: int,
) -> Array:
    intrinsics = validate_finite_numeric_array(field_name, value)
    if intrinsics.shape == (3, 3):
        validate_intrinsics(field_name, intrinsics)
        return intrinsics
    if intrinsics.shape != (batch_size, frame_count, 3, 3):
        raise ValueError(
            f"{field_name}: must have shape (3, 3) or ({batch_size}, {frame_count}, 3, 3)"
        )
    for batch_index in range(batch_size):
        for frame_index in range(frame_count):
            validate_intrinsics(
                f"{field_name}[{batch_index},{frame_index}]",
                intrinsics[batch_index, frame_index],
            )
    return intrinsics


def _validate_transform_clip(
    field_name: str,
    value: Array,
    batch_size: int,
    frame_count: int,
) -> Array:
    transforms = validate_finite_numeric_array(field_name, value)
    if transforms.shape != (batch_size, frame_count, 4, 4):
        raise ValueError(f"{field_name}: must have shape ({batch_size}, {frame_count}, 4, 4)")
    for batch_index in range(batch_size):
        for frame_index in range(frame_count):
            validate_transform(
                f"{field_name}[{batch_index},{frame_index}]",
                transforms[batch_index, frame_index],
            )
    return transforms


def _validate_float_array(field_name: str, value: Array, shape: tuple[int, ...]) -> Array:
    array = validate_same_shape(field_name, value, shape)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{field_name}: must be a floating-point array")
    return array


def _validate_bthw_array(field_name: str, value: Array) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.ndim != 4:
        raise ValueError(f"{field_name}: must have shape BxTxHxW")
    batch_size, frame_count, height, width = array.shape
    if batch_size <= 0 or frame_count <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{field_name}: B, T, H, and W must be positive")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{field_name}: must be a floating-point array")
    return array


def _validate_truth_boundary(value: Mapping[str, object]) -> dict[str, object]:
    truth_boundary = _validate_metadata("truth_boundary", value)
    for flag_name in STUDENT_TRUTH_BOUNDARY_FLAGS:
        if not isinstance(truth_boundary.get(flag_name), bool):
            raise ValueError(f"truth_boundary.{flag_name}: must be a bool")
    if bool(truth_boundary["shape_only"]):
        for flag_name in (
            "learned_inference",
            "usable_for_mapping",
            "accuracy_report",
            "performance_report",
        ):
            if bool(truth_boundary[flag_name]):
                raise ValueError(
                    f"truth_boundary.{flag_name}: must be false for shape-only outputs"
                )
    return truth_boundary


def broadcast_student_intrinsics(
    intrinsics: Array,
    batch_size: int,
    frame_count: int,
) -> npt.NDArray[np.float32]:
    """Return clip-shaped intrinsics without mutating the source array."""

    validated = _validate_clip_intrinsics("intrinsics", intrinsics, batch_size, frame_count)
    intrinsics_f32 = np.asarray(validated, dtype=np.float32)
    if intrinsics_f32.shape == (3, 3):
        return np.broadcast_to(intrinsics_f32, (batch_size, frame_count, 3, 3)).copy()
    return intrinsics_f32.copy()


@dataclass(frozen=True)
class StudentClipInput:
    frame_ids: tuple[int, ...]
    images_rgb: Array
    intrinsics: Array
    T_world_camera_prior: Array | None = None
    coordinate_frame: str = CAMERA_COORDINATE_FRAME
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        images_rgb = _validate_images_rgb(self.images_rgb)
        batch_size = int(images_rgb.shape[0])
        frame_count = int(images_rgb.shape[1])
        frame_ids = _validate_frame_ids(self.frame_ids, frame_count)
        intrinsics = _validate_clip_intrinsics(
            "intrinsics",
            self.intrinsics,
            batch_size,
            frame_count,
        )
        if self.T_world_camera_prior is not None:
            _set(
                self,
                "T_world_camera_prior",
                _validate_transform_clip(
                    "T_world_camera_prior",
                    self.T_world_camera_prior,
                    batch_size,
                    frame_count,
                ),
            )
        _set(self, "frame_ids", frame_ids)
        _set(self, "images_rgb", images_rgb)
        _set(self, "intrinsics", intrinsics)
        _set(self, "coordinate_frame", _validate_coordinate_frame(self.coordinate_frame))
        _set(self, "metadata", _validate_metadata("metadata", self.metadata))

    @property
    def batch_size(self) -> int:
        return int(self.images_rgb.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.images_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.images_rgb.shape[3])

    @property
    def width(self) -> int:
        return int(self.images_rgb.shape[4])

    def batched_intrinsics(self) -> npt.NDArray[np.float32]:
        return broadcast_student_intrinsics(self.intrinsics, self.batch_size, self.frame_count)


@dataclass(frozen=True)
class StudentForwardOutput:
    frame_ids: tuple[int, ...]
    depth_m: Array
    depth_sigma_m: Array
    confidence: Array
    dynamic_probability: Array
    normals_camera: Array
    pointmap_camera_m: Array
    T_world_camera: Array
    intrinsics: Array
    truth_boundary: dict[str, object]
    coordinate_frame: str = CAMERA_COORDINATE_FRAME

    def __post_init__(self) -> None:
        depth_m = _validate_bthw_array("depth_m", self.depth_m)
        if np.any(depth_m < 0.0):
            raise ValueError("depth_m: must be non-negative")
        batch_size, frame_count, height, width = (
            int(depth_m.shape[0]),
            int(depth_m.shape[1]),
            int(depth_m.shape[2]),
            int(depth_m.shape[3]),
        )
        expected_bthw = (batch_size, frame_count, height, width)
        depth_sigma_m = _validate_float_array("depth_sigma_m", self.depth_sigma_m, expected_bthw)
        if np.any(depth_sigma_m < 0.0):
            raise ValueError("depth_sigma_m: must be non-negative")
        confidence = _validate_float_array("confidence", self.confidence, expected_bthw)
        validate_confidence_array("confidence", confidence)
        dynamic_probability = _validate_float_array(
            "dynamic_probability",
            self.dynamic_probability,
            expected_bthw,
        )
        validate_confidence_array("dynamic_probability", dynamic_probability)
        normals_camera = _validate_float_array(
            "normals_camera",
            self.normals_camera,
            (batch_size, frame_count, 3, height, width),
        )
        pointmap_camera_m = _validate_float_array(
            "pointmap_camera_m",
            self.pointmap_camera_m,
            (batch_size, frame_count, 3, height, width),
        )
        T_world_camera = _validate_transform_clip(
            "T_world_camera",
            self.T_world_camera,
            batch_size,
            frame_count,
        )
        intrinsics = _validate_clip_intrinsics(
            "intrinsics",
            self.intrinsics,
            batch_size,
            frame_count,
        )
        if intrinsics.shape != (batch_size, frame_count, 3, 3):
            raise ValueError("intrinsics: output must have shape BxTx3x3")
        _set(self, "frame_ids", _validate_frame_ids(self.frame_ids, frame_count))
        _set(self, "depth_m", depth_m)
        _set(self, "depth_sigma_m", depth_sigma_m)
        _set(self, "confidence", confidence)
        _set(self, "dynamic_probability", dynamic_probability)
        _set(self, "normals_camera", normals_camera)
        _set(self, "pointmap_camera_m", pointmap_camera_m)
        _set(self, "T_world_camera", T_world_camera)
        _set(self, "intrinsics", intrinsics)
        _set(self, "truth_boundary", _validate_truth_boundary(self.truth_boundary))
        _set(self, "coordinate_frame", _validate_coordinate_frame(self.coordinate_frame))


__all__ = [
    "CAMERA_COORDINATE_FRAME",
    "STUDENT_TRUTH_BOUNDARY_FLAGS",
    "StudentClipInput",
    "StudentForwardOutput",
    "broadcast_student_intrinsics",
]
