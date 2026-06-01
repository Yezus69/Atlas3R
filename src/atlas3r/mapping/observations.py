"""Public mapper observation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api import CameraModel, PoseEstimate
from atlas3r.api.validation import (
    validate_confidence_array,
    validate_finite_numeric_array,
    validate_non_negative_int,
    validate_nonempty_str,
    validate_same_shape,
)
from atlas3r.data.synthetic_cube_room import SyntheticCubeRoomFrame


def _set(instance: object, field_name: str, value: object) -> None:
    object.__setattr__(instance, field_name, value)


def _require_type(field_name: str, value: object, expected_type: type[object]) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name}: must be a {expected_type.__name__}")


def _validate_float_hw(
    field_name: str,
    value: npt.NDArray[Any],
    expected_shape: tuple[int, int],
    *,
    non_negative: bool,
) -> npt.NDArray[np.floating[Any]]:
    array = validate_same_shape(field_name, value, expected_shape)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{field_name}: must be a floating-point HxW array")
    if non_negative and np.any(array < 0.0):
        raise ValueError(f"{field_name}: must be non-negative")
    return array


@dataclass(frozen=True)
class DepthObservation:
    """One validated depth observation accepted by mapper fusion code."""

    frame_id: int
    camera: CameraModel
    pose: PoseEstimate
    depth_m: npt.NDArray[np.floating[Any]]
    depth_sigma_m: npt.NDArray[np.floating[Any]]
    confidence: npt.NDArray[np.floating[Any]]
    static_mask: npt.NDArray[Any] | None = None
    object_id: npt.NDArray[np.integer[Any]] | None = None
    rgb_u8: npt.NDArray[np.uint8] | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        validate_non_negative_int("frame_id", self.frame_id)
        _require_type("camera", self.camera, CameraModel)
        _require_type("pose", self.pose, PoseEstimate)
        expected_shape = (self.camera.height, self.camera.width)
        depth_m = _validate_float_hw(
            "depth_m",
            self.depth_m,
            expected_shape,
            non_negative=True,
        )
        depth_sigma_m = _validate_float_hw(
            "depth_sigma_m",
            self.depth_sigma_m,
            expected_shape,
            non_negative=True,
        )
        confidence = _validate_float_hw(
            "confidence",
            self.confidence,
            expected_shape,
            non_negative=False,
        )
        validate_confidence_array("confidence", confidence)

        if self.static_mask is not None:
            static_mask = validate_same_shape("static_mask", self.static_mask, expected_shape)
            if static_mask.dtype != np.bool_:
                validate_confidence_array("static_mask", static_mask)
            _set(self, "static_mask", static_mask)
        if self.object_id is not None:
            object_id = validate_same_shape("object_id", self.object_id, expected_shape)
            if not np.issubdtype(object_id.dtype, np.integer):
                raise ValueError("object_id: must be an integer HxW array")
            _set(self, "object_id", object_id)
        if self.rgb_u8 is not None:
            rgb_u8 = validate_finite_numeric_array("rgb_u8", self.rgb_u8)
            if rgb_u8.dtype != np.uint8 or rgb_u8.shape != (*expected_shape, 3):
                raise ValueError("rgb_u8: must be a uint8 HxWx3 array")
            _set(self, "rgb_u8", rgb_u8)

        _set(self, "source", validate_nonempty_str("source", self.source))
        _set(self, "depth_m", depth_m)
        _set(self, "depth_sigma_m", depth_sigma_m)
        _set(self, "confidence", confidence)


def depth_observation_from_synthetic_frame(frame: SyntheticCubeRoomFrame) -> DepthObservation:
    """Convert a deterministic synthetic frame into the public mapper contract."""
    return DepthObservation(
        frame_id=frame.frame_id,
        camera=frame.camera,
        pose=frame.pose,
        depth_m=frame.depth_m,
        depth_sigma_m=frame.depth_sigma_m,
        confidence=frame.confidence,
        static_mask=np.ones(frame.depth_m.shape, dtype=np.bool_),
        object_id=frame.object_id,
        source="synthetic_cube_room",
    )


__all__ = [
    "DepthObservation",
    "depth_observation_from_synthetic_frame",
]
