"""Minimal Sim3 stitching for overlapping teacher-geometry windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import validate_T_A_B


@dataclass(frozen=True)
class Sim3Estimate:
    scale: float
    rotation: NDArray[np.float32]
    translation: NDArray[np.float32]
    rmse_m: float
    accepted: bool
    reason: str


def estimate_sim3_umeyama(
    source_points: NDArray[np.float32],
    target_points: NDArray[np.float32],
    *,
    residual_threshold_m: float = 0.25,
) -> Sim3Estimate:
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both be shaped N,3")
    if source.shape[0] < 3:
        return _rejected("need at least three overlap centers")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return _rejected("overlap centers must be finite")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_var <= 1e-12:
        return _rejected("source overlap centers are degenerate")
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    det = float(np.linalg.det(u @ vt))
    correction = np.eye(3, dtype=np.float32)
    correction[2, 2] = 1.0 if det >= 0 else -1.0
    rotation = (u @ correction @ vt).astype(np.float32)
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_var)
    if not np.isfinite(scale) or scale <= 0:
        return _rejected("estimated scale is invalid")
    translation = (target_mean - scale * (rotation @ source_mean)).astype(np.float32)
    transformed = (scale * (source @ rotation.T) + translation).astype(np.float32)
    residuals = np.linalg.norm(transformed - target, axis=1)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    accepted = bool(rmse <= residual_threshold_m)
    return Sim3Estimate(
        scale=scale,
        rotation=rotation,
        translation=translation,
        rmse_m=rmse,
        accepted=accepted,
        reason="accepted" if accepted else "overlap residual too high",
    )


def transform_pose_with_sim3(T_window_camera: object, sim3: Sim3Estimate) -> NDArray[np.float32]:
    transform = validate_T_A_B(T_window_camera, "T_window_camera")
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = (sim3.rotation @ transform[:3, :3]).astype(np.float32)
    result[:3, 3] = (sim3.scale * (sim3.rotation @ transform[:3, 3]) + sim3.translation).astype(
        np.float32
    )
    return result


def empty_stitch_stats(stitch_mode: str) -> dict[str, object]:
    return {
        "vggt_window_count": 0,
        "stitch_mode": stitch_mode,
        "stitch_edge_count": 0,
        "accepted_edge_count": 0,
        "rejected_edge_count": 0,
        "pseudo_submap_count": 0,
        "overlap_center_rmse_m": None,
        "scale_min": None,
        "scale_median": None,
        "scale_max": None,
        "rejected_windows": [],
    }


def none_stitch_stats(window_ids: set[int], *, has_records: bool) -> dict[str, object]:
    return {
        "vggt_window_count": len(window_ids),
        "stitch_mode": "none",
        "stitch_edge_count": 0,
        "accepted_edge_count": 0,
        "rejected_edge_count": 0,
        "pseudo_submap_count": len(window_ids),
        "overlap_center_rmse_m": None,
        "scale_min": 1.0 if has_records else None,
        "scale_median": 1.0 if has_records else None,
        "scale_max": 1.0 if has_records else None,
        "rejected_windows": [],
    }


def _rejected(reason: str) -> Sim3Estimate:
    return Sim3Estimate(
        scale=1.0,
        rotation=np.eye(3, dtype=np.float32),
        translation=np.zeros((3,), dtype=np.float32),
        rmse_m=float("inf"),
        accepted=False,
        reason=reason,
    )
