"""Ground-plane plus camera-height scale cue for observed room maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

GROUND_PLANE_METRIC_SCALE_SOURCE = "ground_plane_camera_height_depth_pro_vggt_near_metric"
GROUND_PLANE_SCALE_STATUS = "near_metric_unanchored"
DEFAULT_CAMERA_HEIGHT_PRIOR_M = 1.45


@dataclass(frozen=True)
class GroundPlaneScaleEstimate:
    available: bool
    scale_factor: float = 1.0
    confidence: str = "unavailable"
    reason: str = "not_run"
    vertical_axis: int | None = None
    ground_side: str | None = None
    ground_level_m: float | None = None
    estimated_camera_height_m: float | None = None
    camera_height_prior_m: float = DEFAULT_CAMERA_HEIGHT_PRIOR_M
    metric_scale_source: str = GROUND_PLANE_METRIC_SCALE_SOURCE
    scale_status: str = GROUND_PLANE_SCALE_STATUS
    cue_sources: tuple[str, ...] = (
        "ground_plane_percentile",
        "phone_camera_height_prior",
        "depth_pro_vggt_soft_metric_prior",
    )
    cross_cue_disagreement_ratio: float | None = None
    bbox_size_before_m: tuple[float, float, float] | None = None
    bbox_size_after_m: tuple[float, float, float] | None = None
    anchor_world_m: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "scale_factor": self.scale_factor,
            "confidence": self.confidence,
            "reason": self.reason,
            "vertical_axis": self.vertical_axis,
            "ground_side": self.ground_side,
            "ground_level_m": self.ground_level_m,
            "estimated_camera_height_m": self.estimated_camera_height_m,
            "camera_height_prior_m": self.camera_height_prior_m,
            "metric_scale_source": self.metric_scale_source,
            "scale_status": self.scale_status,
            "cue_sources": list(self.cue_sources),
            "cross_cue_disagreement_ratio": self.cross_cue_disagreement_ratio,
            "bbox_size_before_m": self.bbox_size_before_m,
            "bbox_size_after_m": self.bbox_size_after_m,
            "anchor_world_m": self.anchor_world_m,
            "physical_accuracy_claim": False,
            "measured_geometry": False,
            "truth_status": "near_metric_prior_not_measured_truth",
        }


@dataclass(frozen=True)
class _ScaleCandidate:
    axis: int
    side: str
    ground_level_m: float
    camera_height_m: float
    scale_factor: float
    bbox_size_after_m: NDArray[np.float32]
    score: tuple[float, float, float, float]


def estimate_ground_plane_camera_height_scale(
    points_world_m: NDArray[np.float32],
    trajectory: list[dict[str, object]],
    *,
    camera_height_prior_m: float = DEFAULT_CAMERA_HEIGHT_PRIOR_M,
) -> GroundPlaneScaleEstimate:
    points = _finite_points(points_world_m)
    centers = _camera_centers(trajectory)
    if points.shape[0] < 32:
        return GroundPlaneScaleEstimate(available=False, reason="insufficient_finite_points")
    if centers.shape[0] == 0:
        return GroundPlaneScaleEstimate(available=False, reason="no_finite_camera_centers")
    if camera_height_prior_m <= 0.0 or not math.isfinite(camera_height_prior_m):
        return GroundPlaneScaleEstimate(available=False, reason="invalid_camera_height_prior")

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).astype(np.float32)
    camera_anchor = centers[0].astype(np.float32)
    camera_median = np.median(centers, axis=0).astype(np.float32)
    candidates = _scale_candidates(points, camera_median, bbox_size, camera_height_prior_m)
    if not candidates:
        return GroundPlaneScaleEstimate(
            available=False,
            reason="no_roomlike_ground_plane_camera_height_candidate",
            bbox_size_before_m=_tuple3(bbox_size),
            anchor_world_m=_tuple3(camera_anchor),
        )
    selected = max(candidates, key=lambda item: item.score)
    disagreement_ratio = max(selected.scale_factor, 1.0 / selected.scale_factor)
    confidence = _confidence_for_disagreement(disagreement_ratio)
    return GroundPlaneScaleEstimate(
        available=True,
        scale_factor=float(selected.scale_factor),
        confidence=confidence,
        reason="ground_plane_camera_height_prior_applied",
        vertical_axis=selected.axis,
        ground_side=selected.side,
        ground_level_m=float(selected.ground_level_m),
        estimated_camera_height_m=float(selected.camera_height_m),
        camera_height_prior_m=float(camera_height_prior_m),
        cross_cue_disagreement_ratio=float(disagreement_ratio),
        bbox_size_before_m=_tuple3(bbox_size),
        bbox_size_after_m=_tuple3(selected.bbox_size_after_m),
        anchor_world_m=_tuple3(camera_anchor),
    )


def scale_trajectory(
    trajectory: list[dict[str, object]],
    *,
    scale_factor: float,
    anchor_world_m: tuple[float, float, float] | None,
    metric_scale_source: str = GROUND_PLANE_METRIC_SCALE_SOURCE,
) -> list[dict[str, object]]:
    anchor = np.asarray(anchor_world_m or (0.0, 0.0, 0.0), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for row in trajectory:
        out = dict(row)
        center = _center_from_row(out)
        if center is None:
            rows.append(out)
            continue
        scaled_center = anchor + scale_factor * (center - anchor)
        transform = np.asarray(out.get("T_world_camera"), dtype=np.float32)
        if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
            transform = transform.copy()
            transform[:3, 3] = scaled_center
            out["T_world_camera"] = transform.astype(float).tolist()
        out["camera_center_world_m"] = scaled_center.astype(float).tolist()
        out["metric_scale_source"] = metric_scale_source
        rows.append(out)
    return rows


def _scale_candidates(
    points: NDArray[np.float32],
    camera_median: NDArray[np.float32],
    bbox_size: NDArray[np.float32],
    camera_height_prior_m: float,
) -> list[_ScaleCandidate]:
    candidates: list[_ScaleCandidate] = []
    for axis in range(3):
        values = points[:, axis]
        for side, percentile in (("min", 5.0), ("max", 95.0)):
            ground_level = float(np.percentile(values, percentile))
            camera_height = abs(float(camera_median[axis]) - ground_level)
            if camera_height < 0.15 or camera_height > 2.5:
                continue
            scale_factor = float(camera_height_prior_m / camera_height)
            if scale_factor < 0.5 or scale_factor > 6.0:
                continue
            scaled_bbox = (bbox_size * scale_factor).astype(np.float32)
            room_dims = float(np.count_nonzero(scaled_bbox >= 1.2))
            if room_dims < 2.0 or float(scaled_bbox.min()) < 0.45:
                continue
            other_axes = [index for index in range(3) if index != axis]
            horizontal_area = float(scaled_bbox[other_axes[0]] * scaled_bbox[other_axes[1]])
            score = (
                room_dims,
                float(scaled_bbox.min()),
                horizontal_area,
                -abs(math.log(scale_factor)),
            )
            candidates.append(
                _ScaleCandidate(
                    axis=axis,
                    side=side,
                    ground_level_m=ground_level,
                    camera_height_m=camera_height,
                    scale_factor=scale_factor,
                    bbox_size_after_m=scaled_bbox,
                    score=score,
                )
            )
    return candidates


def _finite_points(points_world_m: NDArray[np.float32]) -> NDArray[np.float32]:
    points = np.asarray(points_world_m, dtype=np.float32).reshape((-1, 3))
    return cast(NDArray[np.float32], points[np.isfinite(points).all(axis=1)])


def _camera_centers(trajectory: list[dict[str, object]]) -> NDArray[np.float32]:
    centers: list[NDArray[np.float32]] = []
    for row in trajectory:
        center = _center_from_row(row)
        if center is not None:
            centers.append(center)
    if not centers:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32).reshape((-1, 3))


def _center_from_row(row: dict[str, object]) -> NDArray[np.float32] | None:
    center_value = row.get("camera_center_world_m")
    center = np.asarray(center_value, dtype=np.float32)
    if center.shape == (3,) and np.all(np.isfinite(center)):
        return center
    transform = np.asarray(row.get("T_world_camera"), dtype=np.float32)
    if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
        return transform[:3, 3].astype(np.float32)
    return None


def _confidence_for_disagreement(disagreement_ratio: float) -> str:
    if disagreement_ratio <= 1.35:
        return "medium"
    if disagreement_ratio <= 2.5:
        return "low"
    return "very_low"


def _tuple3(values: NDArray[np.float32]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


__all__ = [
    "DEFAULT_CAMERA_HEIGHT_PRIOR_M",
    "GROUND_PLANE_METRIC_SCALE_SOURCE",
    "GROUND_PLANE_SCALE_STATUS",
    "GroundPlaneScaleEstimate",
    "estimate_ground_plane_camera_height_scale",
    "scale_trajectory",
]
