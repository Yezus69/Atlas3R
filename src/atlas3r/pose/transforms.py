"""Pure coordinate-frame transform utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import (
    validate_matrix_nx3,
    validate_rotation_matrix,
    validate_transform,
    validate_vector,
)

Array = npt.NDArray[Any]


def make_transform(R_world_camera: Array, t_world_camera: Array) -> npt.NDArray[np.float64]:
    """Create a homogeneous transform mapping camera-frame points into world."""
    R = validate_rotation_matrix("R_world_camera", R_world_camera).astype(np.float64, copy=False)
    t = validate_vector("t_world_camera", t_world_camera, 3).astype(np.float64, copy=False)
    T_world_camera = np.eye(4, dtype=np.float64)
    T_world_camera[:3, :3] = R
    T_world_camera[:3, 3] = t
    return T_world_camera


def invert_transform(T_A_B: Array) -> npt.NDArray[np.float64]:
    """Invert a homogeneous transform."""
    T = validate_transform("T_A_B", T_A_B).astype(np.float64, copy=False)
    R_A_B = T[:3, :3]
    t_A_B = T[:3, 3]
    T_B_A = np.eye(4, dtype=np.float64)
    T_B_A[:3, :3] = R_A_B.T
    T_B_A[:3, 3] = -(R_A_B.T @ t_A_B)
    return T_B_A


def compose_transforms(T_A_B: Array, T_B_C: Array) -> npt.NDArray[np.float64]:
    """Compose transforms so the result maps points from C into A."""
    left = validate_transform("T_A_B", T_A_B).astype(np.float64, copy=False)
    right = validate_transform("T_B_C", T_B_C).astype(np.float64, copy=False)
    return left @ right


def transform_points(T_A_B: Array, points_B: Array) -> npt.NDArray[np.float64]:
    """Transform Nx3 points from frame B into frame A."""
    T = validate_transform("T_A_B", T_A_B).astype(np.float64, copy=False)
    points = validate_matrix_nx3("points_B", points_B).astype(np.float64, copy=False)
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def camera_center_from_T_world_camera(T_world_camera: Array) -> npt.NDArray[np.float64]:
    """Return the camera center in world coordinates."""
    T = validate_transform("T_world_camera", T_world_camera).astype(np.float64, copy=False)
    return T[:3, 3].copy()


def rotation_matrix_from_quaternion_xyzw(q_xyzw: Array) -> npt.NDArray[np.float64]:
    """Convert a quaternion in xyzw order into a 3x3 rotation matrix."""
    q = validate_vector("q_xyzw", q_xyzw, 4).astype(np.float64, copy=False)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("q_xyzw: must have non-zero norm")
    x, y, z, w = q / norm
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    validate_rotation_matrix("R", R)
    return R


def quaternion_xyzw_from_rotation_matrix(R: Array) -> npt.NDArray[np.float64]:
    """Convert a 3x3 rotation matrix into a unit quaternion in xyzw order."""
    matrix = validate_rotation_matrix("R", R).astype(np.float64, copy=False)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


__all__ = [
    "camera_center_from_T_world_camera",
    "compose_transforms",
    "invert_transform",
    "make_transform",
    "quaternion_xyzw_from_rotation_matrix",
    "rotation_matrix_from_quaternion_xyzw",
    "transform_points",
]
