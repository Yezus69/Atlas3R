"""Diagnostic pose/point alignment helpers for VGGT pseudo labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class VGGTPoseAlignment:
    policy: str
    scale: float
    R_source_teacher: npt.NDArray[np.float64]
    t_source_teacher: npt.NDArray[np.float64]
    center_rmse_before_m: float
    center_rmse_after_m: float

    def metadata(self, *, pose_source: str) -> dict[str, object]:
        return {
            "alignment_policy": self.policy,
            "pose_source_before_alignment": pose_source,
            "diagnostic_alignment_against_source_pose": True,
            "scale": self.scale,
            "center_rmse_before_m": self.center_rmse_before_m,
            "center_rmse_after_m": self.center_rmse_after_m,
            "aligned_outputs_are_measured_geometry": False,
        }


def alignment_for_policy(
    *,
    source_T_world_camera: npt.NDArray[np.float32],
    teacher_T_world_camera: npt.NDArray[np.float32],
    policy: str,
) -> VGGTPoseAlignment | None:
    if policy == "none":
        return None
    source_centers = source_T_world_camera[:, :3, 3].astype(np.float64)
    teacher_centers = teacher_T_world_camera[:, :3, 3].astype(np.float64)
    before = _center_rmse(teacher_centers, source_centers)
    scale, rotation, translation = _fit_similarity(teacher_centers, source_centers, scale=policy)
    aligned = (scale * (rotation @ teacher_centers.T)).T + translation
    after = _center_rmse(aligned, source_centers)
    return VGGTPoseAlignment(
        policy=policy,
        scale=float(scale),
        R_source_teacher=rotation,
        t_source_teacher=translation,
        center_rmse_before_m=before,
        center_rmse_after_m=after,
    )


def apply_alignment_to_transforms(
    transforms: npt.NDArray[np.float32],
    alignment: VGGTPoseAlignment,
) -> npt.NDArray[np.float32]:
    aligned = transforms.astype(np.float64, copy=True)
    for index in range(aligned.shape[0]):
        aligned[index, :3, :3] = alignment.R_source_teacher @ aligned[index, :3, :3]
        center = aligned[index, :3, 3]
        aligned[index, :3, 3] = (
            alignment.scale * (alignment.R_source_teacher @ center) + alignment.t_source_teacher
        )
    return aligned.astype(np.float32, copy=False)


def apply_alignment_to_points(
    pointmap: npt.NDArray[np.float32],
    alignment: VGGTPoseAlignment,
) -> npt.NDArray[np.float32]:
    flat = pointmap.reshape(-1, 3).astype(np.float64)
    aligned = (alignment.scale * (alignment.R_source_teacher @ flat.T)).T
    aligned += alignment.t_source_teacher
    return aligned.reshape(pointmap.shape).astype(np.float32, copy=False)


def _fit_similarity(
    source_points: npt.NDArray[np.float64],
    target_points: npt.NDArray[np.float64],
    *,
    scale: str,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    src_mean = np.mean(source_points, axis=0)
    tgt_mean = np.mean(target_points, axis=0)
    src_centered = source_points - src_mean
    tgt_centered = target_points - tgt_mean
    variance = float(np.sum(src_centered * src_centered) / max(source_points.shape[0], 1))
    if source_points.shape[0] < 2 or variance <= 1e-12:
        rotation = np.eye(3, dtype=np.float64)
        fitted_scale = 1.0
    else:
        covariance = (tgt_centered.T @ src_centered) / source_points.shape[0]
        U, singular_values, Vt = np.linalg.svd(covariance)
        sign = np.sign(np.linalg.det(U @ Vt))
        correction = np.diag([1.0, 1.0, sign])
        rotation = U @ correction @ Vt
        fitted_scale = (
            float(np.sum(singular_values * np.diag(correction)) / variance)
            if scale == "diagnostic_sim3"
            else 1.0
        )
    translation = tgt_mean - fitted_scale * (rotation @ src_mean)
    return fitted_scale, rotation, translation


def _center_rmse(
    estimated: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> float:
    if estimated.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(estimated - reference), axis=1))))


__all__ = [
    "VGGTPoseAlignment",
    "alignment_for_policy",
    "apply_alignment_to_points",
    "apply_alignment_to_transforms",
]
