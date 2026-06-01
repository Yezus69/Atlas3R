"""Pinhole camera projection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import (
    validate_finite_numeric_array,
    validate_image_hw,
    validate_intrinsics,
    validate_matrix_nx2,
    validate_matrix_nx3,
    validate_positive_scalar,
)

Array = npt.NDArray[Any]


def project_points_camera(
    points_camera_m: Array, K: Array
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project camera-frame points into pixel coordinates.

    Points with non-positive camera z are rejected for this foundation API.
    """
    points = validate_matrix_nx3("points_camera_m", points_camera_m).astype(np.float64, copy=False)
    intrinsics = validate_intrinsics("K", K).astype(np.float64, copy=False)
    depth_m = points[:, 2].copy()
    if np.any(depth_m <= 0.0):
        raise ValueError("points_camera_m: z values must be positive")
    projected_h = (intrinsics @ points.T).T
    pixels_uv = projected_h[:, :2] / projected_h[:, 2:3]
    return pixels_uv, depth_m


def unproject_pixels(pixels_uv: Array, depth_m: Array, K: Array) -> npt.NDArray[np.float64]:
    """Unproject Nx2 pixels and N depths into camera-frame points."""
    pixels = validate_matrix_nx2("pixels_uv", pixels_uv).astype(np.float64, copy=False)
    depth = validate_finite_numeric_array("depth_m", depth_m).astype(np.float64, copy=False)
    if depth.shape != (pixels.shape[0],):
        raise ValueError("depth_m: must have length N")
    if np.any(depth <= 0.0):
        raise ValueError("depth_m: values must be positive")
    intrinsics = validate_intrinsics("K", K).astype(np.float64, copy=False)
    homogeneous_pixels = np.column_stack(
        [pixels[:, 0], pixels[:, 1], np.ones(pixels.shape[0], dtype=np.float64)]
    )
    rays = (np.linalg.inv(intrinsics) @ homogeneous_pixels.T).T
    return rays * depth[:, None]


def unproject_depth_map(depth_m: Array, K: Array) -> npt.NDArray[np.float64]:
    """Unproject an HxW depth map into HxWx3 camera-frame points."""
    depth = validate_image_hw("depth_m", depth_m).astype(np.float64, copy=False)
    if np.any(depth <= 0.0):
        raise ValueError("depth_m: values must be positive")
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.column_stack([u.reshape(-1), v.reshape(-1)])
    points = unproject_pixels(pixels, depth.reshape(-1), K)
    return points.reshape(height, width, 3)


def scale_intrinsics(K: Array, sx: float, sy: float) -> npt.NDArray[np.float64]:
    """Scale focal lengths and principal point for resized images."""
    validate_positive_scalar("sx", sx)
    validate_positive_scalar("sy", sy)
    scaled = validate_intrinsics("K", K).astype(np.float64, copy=True)
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


__all__ = [
    "project_points_camera",
    "scale_intrinsics",
    "unproject_depth_map",
    "unproject_pixels",
]
