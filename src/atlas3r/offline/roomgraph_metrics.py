"""RoomGraph metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RoomGraphMetrics:
    reprojection_error_mean_px: float
    reprojection_error_p95_px: float
    cross_view_depth_residual_mean_m: float
    cross_view_depth_residual_p95_m: float
    mapped_teacher_disagreement_rel_mean: float
    track_inlier_ratio: float
    track_count: int
    observation_count: int
    camera_trajectory_length_m: float
    camera_bbox_size_m: tuple[float, float, float]
    camera_collapse_score_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "reprojection_error_mean_px": self.reprojection_error_mean_px,
            "reprojection_error_p95_px": self.reprojection_error_p95_px,
            "cross_view_depth_residual_mean_m": self.cross_view_depth_residual_mean_m,
            "cross_view_depth_residual_p95_m": self.cross_view_depth_residual_p95_m,
            "mapped_teacher_disagreement_rel_mean": self.mapped_teacher_disagreement_rel_mean,
            "track_inlier_ratio": self.track_inlier_ratio,
            "track_count": self.track_count,
            "observation_count": self.observation_count,
            "camera_trajectory_length_m": self.camera_trajectory_length_m,
            "camera_bbox_size_m": list(self.camera_bbox_size_m),
            "camera_collapse_score_m": self.camera_collapse_score_m,
        }


def compute_roomgraph_metrics(
    problem: Any,
    centers: NDArray[np.float32],
    depth_scale: NDArray[np.float32],
    depth_bias_m: NDArray[np.float32],
    points: NDArray[np.float32],
    focal_scale: float,
) -> RoomGraphMetrics:
    reproj: list[float] = []
    depth_residuals: list[float] = []
    teacher_rel: list[float] = []
    inliers = 0
    observations = problem.observations
    frames = problem.frames
    for obs in observations:
        frame = frames[obs.frame_index]
        point = points[obs.track_index]
        pc = frame.R_world_camera.T @ (point - centers[obs.frame_index])
        if pc[2] <= 1e-5 or not np.all(np.isfinite(pc)):
            reproj.append(1e3)
            depth_residuals.append(10.0)
            continue
        K = frame.K.copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
        pixel = (K @ pc)[:2] / max(float(pc[2]), 1e-6)
        err_px = float(np.linalg.norm(pixel - np.asarray(obs.xy_px, dtype=np.float32)))
        target_depth = depth_scale[obs.frame_index] * obs.depth_pro_m
        target_depth += depth_bias_m[obs.frame_index]
        depth_err = abs(float(pc[2] - target_depth))
        rel = abs(float(obs.vggt_depth_m - target_depth)) / max(abs(float(target_depth)), 1e-3)
        reproj.append(err_px)
        depth_residuals.append(depth_err)
        teacher_rel.append(rel)
        if err_px < 8.0 and depth_err < 0.25:
            inliers += 1
    centers64 = centers.astype(np.float64)
    steps = np.linalg.norm(np.diff(centers64, axis=0), axis=1) if centers.shape[0] > 1 else []
    bbox_size = centers64.max(axis=0) - centers64.min(axis=0) if centers64.size else np.zeros(3)
    return RoomGraphMetrics(
        reprojection_error_mean_px=_mean(reproj),
        reprojection_error_p95_px=_percentile(reproj, 95),
        cross_view_depth_residual_mean_m=_mean(depth_residuals),
        cross_view_depth_residual_p95_m=_percentile(depth_residuals, 95),
        mapped_teacher_disagreement_rel_mean=_mean(teacher_rel),
        track_inlier_ratio=float(inliers / len(observations)) if observations else 0.0,
        track_count=int(problem.point_count),
        observation_count=len(observations),
        camera_trajectory_length_m=float(np.sum(steps)) if len(centers64) > 1 else 0.0,
        camera_bbox_size_m=(float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])),
        camera_collapse_score_m=float(np.linalg.norm(bbox_size)),
    )


def improvement(before: RoomGraphMetrics, after: RoomGraphMetrics) -> dict[str, float]:
    return {
        "reprojection_error_mean_px": _lower_is_better(
            before.reprojection_error_mean_px, after.reprojection_error_mean_px
        ),
        "cross_view_depth_residual_mean_m": _lower_is_better(
            before.cross_view_depth_residual_mean_m,
            after.cross_view_depth_residual_mean_m,
        ),
        "mapped_teacher_disagreement_rel_mean": _lower_is_better(
            before.mapped_teacher_disagreement_rel_mean,
            after.mapped_teacher_disagreement_rel_mean,
        ),
        "camera_collapse_score_m": 0.0
        if before.camera_collapse_score_m <= 1e-6
        else (after.camera_collapse_score_m - before.camera_collapse_score_m)
        / before.camera_collapse_score_m,
    }


def _lower_is_better(before: float, after: float) -> float:
    return 0.0 if before <= 1e-8 else (before - after) / before


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile)) if values else 0.0
