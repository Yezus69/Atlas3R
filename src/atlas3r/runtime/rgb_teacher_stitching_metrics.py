"""Diagnostic metrics for RGB teacher window stitching."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.rgb_teacher_stitching_math import Sim3Transform
from atlas3r.runtime.rgb_teacher_stitching_types import (
    TeacherWindowNode,
    TeacherWindowOverlapEdge,
    TeacherWindowPrediction,
)


def overlap_pose_depth_metrics(
    source_window: TeacherWindowPrediction,
    target_window: TeacherWindowPrediction,
    target_node: TeacherWindowNode,
    sim3: Sim3Transform,
    overlap_frame_ids: Sequence[int],
) -> tuple[npt.NDArray[np.float64], list[float], list[float]]:
    source_by_id = observations_by_id(source_window)
    target_by_id = observations_by_id(target_window)
    center_residuals: list[float] = []
    rotation_errors: list[float] = []
    depth_ratios: list[float] = []
    for frame_id in overlap_frame_ids:
        source_obs = source_by_id[frame_id]
        target_obs = target_by_id[frame_id]
        source_center = sim3.apply_points(source_obs.pose.camera_center_world_m[None, :])[0]
        target_center = target_node.sim3_local_to_global.apply_points(
            target_obs.pose.camera_center_world_m[None, :]
        )[0]
        center_residuals.append(float(np.linalg.norm(source_center - target_center)))
        rotation_errors.append(
            _rotation_delta_deg(
                sim3.rotation @ source_obs.pose.T_world_camera[:3, :3],
                target_node.sim3_local_to_global.rotation @ target_obs.pose.T_world_camera[:3, :3],
            )
        )
        ratio = _depth_scale_ratio(source_obs, target_obs, target_node.sim3_local_to_global.scale)
        if ratio is not None:
            depth_ratios.append(ratio)
    return np.asarray(center_residuals, dtype=np.float64), rotation_errors, depth_ratios


def raw_adjacent_center_jumps(windows: Sequence[TeacherWindowPrediction]) -> list[float]:
    jumps: list[float] = []
    for left, right in zip(windows[:-1], windows[1:], strict=False):
        left_by_id = observations_by_id(left)
        right_by_id = observations_by_id(right)
        for frame_id in sorted(set(left.frame_ids).intersection(right.frame_ids)):
            jumps.append(
                float(
                    np.linalg.norm(
                        left_by_id[frame_id].pose.camera_center_world_m
                        - right_by_id[frame_id].pose.camera_center_world_m
                    )
                )
            )
    return jumps


def edge_center_jumps(edges: Sequence[TeacherWindowOverlapEdge]) -> list[float]:
    jumps: list[float] = []
    for edge in edges:
        values = edge.metrics.get("boundary_center_jumps_m", [])
        if isinstance(values, list):
            jumps.extend(float(value) for value in values)
    return jumps


def rotation_angle_deg(rotation: npt.NDArray[np.float32]) -> float:
    cos_theta = np.clip((float(np.trace(rotation.astype(np.float64))) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def observations_by_id(window: TeacherWindowPrediction) -> dict[int, DepthObservation]:
    return {observation.frame_id: observation for observation in window.observations}


def _depth_scale_ratio(
    source_obs: DepthObservation,
    target_obs: DepthObservation,
    target_scale: float,
) -> float | None:
    source_valid = np.asarray(source_obs.static_mask, dtype=np.bool_) & (source_obs.depth_m > 0.0)
    target_valid = np.asarray(target_obs.static_mask, dtype=np.bool_) & (target_obs.depth_m > 0.0)
    valid = source_valid & target_valid
    if not np.any(valid):
        return None
    source_depth = np.asarray(source_obs.depth_m[valid], dtype=np.float64)
    target_depth = np.asarray(target_obs.depth_m[valid], dtype=np.float64) * float(target_scale)
    median_source = float(np.median(source_depth))
    if median_source <= 1e-12:
        return None
    return float(np.median(target_depth) / median_source)


def _rotation_delta_deg(
    left: npt.NDArray[np.float32],
    right: npt.NDArray[np.float32],
) -> float:
    relative = left.astype(np.float64).T @ right.astype(np.float64)
    cos_theta = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))
