"""Pure Sim3 math for RGB teacher window stitching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Sim3Transform:
    scale: float
    rotation: npt.NDArray[np.float32]
    translation: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale: must be finite and positive")
        rotation = np.asarray(self.rotation, dtype=np.float32)
        translation = np.asarray(self.translation, dtype=np.float32)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation: must be finite 3x3")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation: must be finite length-3")
        if not np.allclose(rotation.T @ rotation, np.eye(3, dtype=np.float32), atol=1e-3):
            raise ValueError("rotation: must be orthonormal")
        if float(np.linalg.det(rotation.astype(np.float64))) <= 0.0:
            raise ValueError("rotation: determinant must be positive")
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @staticmethod
    def identity() -> Sim3Transform:
        return Sim3Transform(
            scale=1.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros((3,), dtype=np.float32),
        )

    def apply_points(self, points_local: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
        points = points_array(points_local, "points_local")
        transformed = self.scale * (points @ self.rotation.T) + self.translation[None, :]
        return transformed.astype(np.float32, copy=False)

    def to_record(self) -> dict[str, object]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.astype(float).tolist(),
            "translation_m": self.translation.astype(float).tolist(),
        }


def estimate_sim3_umeyama(
    source_points: npt.NDArray[np.floating[Any]],
    target_points: npt.NDArray[np.floating[Any]],
) -> Sim3Transform:
    """Estimate `target = s * R @ source + t` with Umeyama least squares."""

    source = points_array(source_points, "source_points").astype(np.float64, copy=False)
    target = points_array(target_points, "target_points").astype(np.float64, copy=False)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have matching shape")
    if source.shape[0] < 2:
        raise ValueError("source_points: at least two correspondences are required")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean[None, :]
    target_centered = target - target_mean[None, :]
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-12:
        raise ValueError("source_points: degenerate point set")
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3Transform(
        scale=scale,
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
    )


def apply_sim3_to_camera_pose(
    T_local_world_camera: npt.NDArray[np.floating[Any]],
    sim3_local_to_global: Sim3Transform,
) -> npt.NDArray[np.float32]:
    """Transform an SE(3) camera pose from local teacher world into global world."""

    T_local = np.asarray(T_local_world_camera, dtype=np.float64)
    if T_local.shape != (4, 4) or not np.all(np.isfinite(T_local)):
        raise ValueError("T_local_world_camera: must be finite 4x4")
    if not np.allclose(T_local[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError("T_local_world_camera: bottom row must be [0,0,0,1]")
    T_global = np.eye(4, dtype=np.float64)
    R = sim3_local_to_global.rotation.astype(np.float64)
    T_global[:3, :3] = R @ T_local[:3, :3]
    T_global[:3, 3] = sim3_local_to_global.scale * (
        R @ T_local[:3, 3]
    ) + sim3_local_to_global.translation.astype(np.float64)
    return T_global.astype(np.float32, copy=False)


def apply_sim3_scale_to_depth(
    depth_m: npt.NDArray[np.floating[Any]],
    sigma_m: npt.NDArray[np.floating[Any]],
    scale: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale: must be finite and positive")
    depth = np.asarray(depth_m, dtype=np.float32)
    sigma = np.asarray(sigma_m, dtype=np.float32)
    if depth.shape != sigma.shape:
        raise ValueError("depth_m and sigma_m must have matching shape")
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(sigma)):
        raise ValueError("depth_m and sigma_m must be finite")
    if np.any(depth < 0.0) or np.any(sigma < 0.0):
        raise ValueError("depth_m and sigma_m must be non-negative")
    return (
        (depth * np.float32(scale)).astype(np.float32, copy=False),
        (sigma * np.float32(scale)).astype(np.float32, copy=False),
    )


def points_array(
    points: npt.NDArray[np.floating[Any]],
    name: str,
) -> npt.NDArray[np.float32]:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: must be finite Nx3")
    return array


__all__ = [
    "Sim3Transform",
    "apply_sim3_scale_to_depth",
    "apply_sim3_to_camera_pose",
    "estimate_sim3_umeyama",
    "points_array",
]
