"""Coordinate-frame utilities for Atlas3R contracts."""

from __future__ import annotations

from typing import cast

import numpy as np

from atlas3r.contracts._arrays import FloatArray, as_float32_array

COORDINATE_FRAME_NAME = "x_right_y_down_z_forward"
CAMERA_FRAME = "camera_x_right_y_down_z_forward"
WORLD_FRAME_NAME = "world_from_anchor"


def validate_T_A_B(value: object, name: str = "T_A_B") -> FloatArray:
    T_A_B = as_float32_array(value, (4, 4), name)
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(T_A_B[3], expected_last_row, atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous 4x4 transform")
    return T_A_B


def invert_T_A_B(T_A_B: object) -> FloatArray:
    transform = validate_T_A_B(T_A_B)
    return cast(FloatArray, np.linalg.inv(transform).astype(np.float32))


def transform_points(T_A_B: object, points_B: object) -> FloatArray:
    transform = validate_T_A_B(T_A_B)
    points = as_float32_array(points_B, (None, 3), "points_B")
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = homogeneous @ transform.T
    return cast(FloatArray, transformed[:, :3].astype(np.float32))


def project_points(K: object, points_camera: object) -> FloatArray:
    intrinsics = as_float32_array(K, (3, 3), "K")
    points = as_float32_array(points_camera, (None, 3), "points_camera")
    z = points[:, 2]
    if np.any(z <= 0):
        raise ValueError("points_camera must have positive z for projection")
    pixels_h = points @ intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return cast(FloatArray, pixels.astype(np.float32))


def unproject_depth(K: object, depth_m: object) -> FloatArray:
    intrinsics = as_float32_array(K, (3, 3), "K")
    depth = as_float32_array(depth_m, (None, None), "depth_m")
    if np.any(depth < 0):
        raise ValueError("depth_m must be non-negative")
    height, width = depth.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("K focal lengths must be positive")
    x = (u_grid - cx) * depth / fx
    y = (v_grid - cy) * depth / fy
    return cast(FloatArray, np.stack([x, y, depth], axis=-1).astype(np.float32))
