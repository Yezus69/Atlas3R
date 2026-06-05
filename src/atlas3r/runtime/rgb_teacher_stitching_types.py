"""Typed contracts for RGB teacher window stitching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.rgb_teacher_stitching_math import Sim3Transform


@dataclass(frozen=True)
class StitchThresholds:
    min_overlap_frames: int = 4
    max_center_rmse_m: float = 0.25
    max_scale_ratio: float = 2.0
    min_inliers: int = 4

    def __post_init__(self) -> None:
        if self.min_overlap_frames <= 0:
            raise ValueError("min_overlap_frames: must be positive")
        if self.max_center_rmse_m <= 0.0:
            raise ValueError("max_center_rmse_m: must be positive")
        if self.max_scale_ratio < 1.0:
            raise ValueError("max_scale_ratio: must be >= 1")
        if self.min_inliers <= 0:
            raise ValueError("min_inliers: must be positive")


@dataclass(frozen=True)
class TeacherWindowPrediction:
    window_index: int
    frame_ids: tuple[int, ...]
    observations: tuple[DepthObservation, ...]
    payload: Mapping[str, Any]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.window_index < 0:
            raise ValueError("window_index: must be non-negative")
        if not self.frame_ids:
            raise ValueError("frame_ids: at least one frame is required")
        if len(self.frame_ids) != len(self.observations):
            raise ValueError("frame_ids: must match observations length")


@dataclass(frozen=True)
class TeacherWindowNode:
    window_index: int
    frame_ids: tuple[int, ...]
    sim3_local_to_global: Sim3Transform
    pseudo_submap_id: int
    accepted: bool
    confidence_scale: float = 1.0
    rejection_reason: str | None = None


@dataclass(frozen=True)
class TeacherWindowOverlapEdge:
    source_window_index: int
    target_window_index: int
    overlap_frame_ids: tuple[int, ...]
    sim3_source_to_global: Sim3Transform
    accepted: bool
    rejection_reason: str | None
    metrics: dict[str, object]


@dataclass(frozen=True)
class StitchGraph:
    nodes: tuple[TeacherWindowNode, ...]
    edges: tuple[TeacherWindowOverlapEdge, ...]


@dataclass(frozen=True)
class StitchDiagnostics:
    stitch_mode: str
    graph: StitchGraph
    rejected_windows: tuple[dict[str, object], ...]
    boundary_center_jumps_m: tuple[float, ...]

    def summary_fields(self) -> dict[str, object]:
        edges = self.graph.edges
        accepted_edges = [edge for edge in edges if edge.accepted]
        center_rmse = _edge_metric_floats(accepted_edges, "overlap_camera_center_rmse_m")
        scales = _edge_metric_floats(accepted_edges, "sim3_scale")
        accepted_submaps = {
            node.pseudo_submap_id
            for node in self.graph.nodes
            if node.accepted and node.pseudo_submap_id >= 0
        }
        jumps = np.asarray(self.boundary_center_jumps_m, dtype=np.float64)
        return {
            "stitch_mode": self.stitch_mode,
            "stitch_window_count": len(self.graph.nodes),
            "stitch_edge_count": len(edges),
            "stitch_accepted_edge_count": len(accepted_edges),
            "stitch_rejected_edge_count": len(edges) - len(accepted_edges),
            "stitch_submap_count": len(accepted_submaps),
            "stitch_mean_overlap_center_rmse_m": _mean_or_none(center_rmse),
            "stitch_p95_overlap_center_rmse_m": _percentile_or_none(center_rmse, 95.0),
            "stitch_max_overlap_center_rmse_m": max(center_rmse) if center_rmse else None,
            "stitch_scale_min": min(scales) if scales else None,
            "stitch_scale_median": _percentile_or_none(scales, 50.0),
            "stitch_scale_max": max(scales) if scales else None,
            "stitch_boundary_jump_mean_m": float(np.mean(jumps)) if jumps.size else None,
            "stitch_boundary_jump_p95_m": float(np.percentile(jumps, 95.0)) if jumps.size else None,
            "stitch_boundary_jump_max_m": float(np.max(jumps)) if jumps.size else None,
            "rejected_windows": [dict(item) for item in self.rejected_windows],
        }


@dataclass(frozen=True)
class StitchedTeacherObservationBatch:
    observations: tuple[DepthObservation, ...]
    graph: StitchGraph
    diagnostics: StitchDiagnostics


def _edge_metric_floats(
    edges: Sequence[TeacherWindowOverlapEdge],
    field_name: str,
) -> list[float]:
    return [
        float(cast(float, edge.metrics[field_name]))
        for edge in edges
        if edge.metrics.get(field_name) is not None
    ]


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


__all__ = [
    "StitchDiagnostics",
    "StitchGraph",
    "StitchThresholds",
    "StitchedTeacherObservationBatch",
    "TeacherWindowNode",
    "TeacherWindowOverlapEdge",
    "TeacherWindowPrediction",
]
