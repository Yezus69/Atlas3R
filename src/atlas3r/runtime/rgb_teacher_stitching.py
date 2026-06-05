"""Sim3 stitching for independent RGB teacher prediction windows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import PoseEstimate
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import quaternion_xyzw_from_rotation_matrix
from atlas3r.runtime.rgb_teacher_stitching_math import (
    Sim3Transform,
    apply_sim3_scale_to_depth,
    apply_sim3_to_camera_pose,
    estimate_sim3_umeyama,
)
from atlas3r.runtime.rgb_teacher_stitching_metrics import (
    edge_center_jumps,
    observations_by_id,
    overlap_pose_depth_metrics,
    raw_adjacent_center_jumps,
    rotation_angle_deg,
)
from atlas3r.runtime.rgb_teacher_stitching_points import pointmap_correspondences
from atlas3r.runtime.rgb_teacher_stitching_types import (
    StitchDiagnostics,
    StitchedTeacherObservationBatch,
    StitchGraph,
    StitchThresholds,
    TeacherWindowNode,
    TeacherWindowOverlapEdge,
    TeacherWindowPrediction,
)


def stitch_teacher_windows(
    windows: Sequence[TeacherWindowPrediction],
    *,
    overlap_policy: str,
    thresholds: StitchThresholds,
) -> StitchedTeacherObservationBatch:
    if overlap_policy not in {"none", "sim3-overlap"}:
        raise ValueError("overlap_policy: must be none or sim3-overlap")
    if not windows:
        raise ValueError("windows: at least one teacher window is required")
    ordered = tuple(sorted(windows, key=lambda item: item.window_index))
    if overlap_policy == "none":
        return _no_stitch_batch(ordered)
    nodes: list[TeacherWindowNode] = [
        TeacherWindowNode(
            window_index=ordered[0].window_index,
            frame_ids=ordered[0].frame_ids,
            sim3_local_to_global=Sim3Transform.identity(),
            pseudo_submap_id=0,
            accepted=True,
        )
    ]
    edges: list[TeacherWindowOverlapEdge] = []
    rejected_windows: list[dict[str, object]] = []
    accepted_by_index = {ordered[0].window_index: nodes[0]}
    for window in ordered[1:]:
        target_window, target_node = _best_overlap_target(window, ordered, accepted_by_index)
        if target_window is None or target_node is None:
            reason = "insufficient_overlap_with_accepted_windows"
            nodes.append(_rejected_node(window, reason))
            rejected_windows.append({"window_index": window.window_index, "reason": reason})
            continue
        edge = _estimate_overlap_edge(
            source_window=window,
            target_window=target_window,
            target_node=target_node,
            thresholds=thresholds,
        )
        edges.append(edge)
        if not edge.accepted:
            nodes.append(_rejected_node(window, edge.rejection_reason or "rejected_edge"))
            rejected_windows.append(
                {
                    "window_index": window.window_index,
                    "reason": edge.rejection_reason or "rejected_edge",
                }
            )
            continue
        rmse = float(cast(float, edge.metrics["overlap_camera_center_rmse_m"]))
        confidence_scale = max(0.25, 1.0 - rmse / max(thresholds.max_center_rmse_m, 1e-6))
        node = TeacherWindowNode(
            window_index=window.window_index,
            frame_ids=window.frame_ids,
            sim3_local_to_global=edge.sim3_source_to_global,
            pseudo_submap_id=0,
            accepted=True,
            confidence_scale=confidence_scale,
        )
        nodes.append(node)
        accepted_by_index[window.window_index] = node
    graph = StitchGraph(nodes=tuple(nodes), edges=tuple(edges))
    observations = _deduped_transformed_observations(ordered, graph)
    validate_stitched_observations(observations)
    diagnostics = StitchDiagnostics(
        stitch_mode=overlap_policy,
        graph=graph,
        rejected_windows=tuple(rejected_windows),
        boundary_center_jumps_m=tuple(edge_center_jumps(edges)),
    )
    return StitchedTeacherObservationBatch(
        observations=observations,
        graph=graph,
        diagnostics=diagnostics,
    )


def validate_stitched_observations(observations: Sequence[DepthObservation]) -> None:
    if not observations:
        raise ValueError("stitched observations: no observations were accepted")
    seen: set[int] = set()
    for observation in observations:
        if observation.frame_id in seen:
            raise ValueError(f"stitched observations: duplicate frame_id {observation.frame_id}")
        seen.add(observation.frame_id)
        truth = observation.pose.diagnostics.get("truth_boundary")
        if isinstance(truth, Mapping):
            if bool(truth.get("measured_depth_used")) or bool(truth.get("measured_pose_used")):
                raise ValueError("stitched observations: measured mapping flags must remain false")


def _no_stitch_batch(
    windows: Sequence[TeacherWindowPrediction],
) -> StitchedTeacherObservationBatch:
    nodes = tuple(
        TeacherWindowNode(
            window_index=window.window_index,
            frame_ids=window.frame_ids,
            sim3_local_to_global=Sim3Transform.identity(),
            pseudo_submap_id=window.window_index,
            accepted=True,
        )
        for window in windows
    )
    graph = StitchGraph(nodes=nodes, edges=())
    observations = _deduped_transformed_observations(windows, graph)
    diagnostics = StitchDiagnostics(
        stitch_mode="none",
        graph=graph,
        rejected_windows=(),
        boundary_center_jumps_m=tuple(raw_adjacent_center_jumps(windows)),
    )
    return StitchedTeacherObservationBatch(
        observations=observations,
        graph=graph,
        diagnostics=diagnostics,
    )


def _best_overlap_target(
    window: TeacherWindowPrediction,
    windows: Sequence[TeacherWindowPrediction],
    accepted_by_index: Mapping[int, TeacherWindowNode],
) -> tuple[TeacherWindowPrediction | None, TeacherWindowNode | None]:
    best: tuple[int, TeacherWindowPrediction, TeacherWindowNode] | None = None
    source_ids = set(window.frame_ids)
    for candidate in windows:
        node = accepted_by_index.get(candidate.window_index)
        if node is None:
            continue
        overlap = len(source_ids.intersection(candidate.frame_ids))
        if best is None or overlap > best[0]:
            best = (overlap, candidate, node)
    if best is None or best[0] <= 0:
        return None, None
    return best[1], best[2]


def _estimate_overlap_edge(
    *,
    source_window: TeacherWindowPrediction,
    target_window: TeacherWindowPrediction,
    target_node: TeacherWindowNode,
    thresholds: StitchThresholds,
) -> TeacherWindowOverlapEdge:
    overlap_frame_ids = tuple(
        sorted(set(source_window.frame_ids).intersection(target_window.frame_ids))
    )
    if len(overlap_frame_ids) < thresholds.min_overlap_frames:
        return _rejected_edge(
            source_window,
            target_window,
            overlap_frame_ids,
            "insufficient_overlap_frames",
        )
    source_points, target_points = _overlap_correspondences(
        source_window,
        target_window,
        target_node,
        overlap_frame_ids,
    )
    try:
        sim3, residuals, inliers = _robust_sim3(source_points, target_points, thresholds)
    except ValueError as exc:
        return _rejected_edge(source_window, target_window, overlap_frame_ids, str(exc))
    center_residuals, rotation_errors, depth_ratios = overlap_pose_depth_metrics(
        source_window,
        target_window,
        target_node,
        sim3,
        overlap_frame_ids,
    )
    center_rmse = _rmse(center_residuals)
    scale_ratio = max(sim3.scale, 1.0 / sim3.scale)
    reason = None
    if int(np.count_nonzero(inliers)) < thresholds.min_inliers:
        reason = "insufficient_inliers"
    elif center_rmse is None or center_rmse > thresholds.max_center_rmse_m:
        reason = "overlap_center_rmse_exceeded"
    elif scale_ratio > thresholds.max_scale_ratio:
        reason = "sim3_scale_ratio_exceeded"
    metrics = {
        "overlap_camera_center_rmse_m": center_rmse,
        "overlap_rotation_error_deg": _mean_or_none(rotation_errors),
        "overlap_depth_scale_ratio": _mean_or_none(depth_ratios),
        "accepted_inlier_count": int(np.count_nonzero(inliers)) if reason is None else 0,
        "rejected_outlier_count": int(inliers.size - np.count_nonzero(inliers)),
        "sim3_scale": float(sim3.scale),
        "sim3_translation_norm_m": float(np.linalg.norm(sim3.translation.astype(np.float64))),
        "sim3_rotation_deg": rotation_angle_deg(sim3.rotation),
        "correspondence_count": int(source_points.shape[0]),
        "boundary_center_jumps_m": center_residuals.astype(float).tolist(),
    }
    return TeacherWindowOverlapEdge(
        source_window_index=source_window.window_index,
        target_window_index=target_window.window_index,
        overlap_frame_ids=overlap_frame_ids,
        sim3_source_to_global=sim3,
        accepted=reason is None,
        rejection_reason=reason,
        metrics=metrics,
    )


def _robust_sim3(
    source_points: npt.NDArray[np.float32],
    target_points: npt.NDArray[np.float32],
    thresholds: StitchThresholds,
) -> tuple[Sim3Transform, npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    if source_points.shape[0] < thresholds.min_inliers:
        raise ValueError("insufficient_correspondences")
    initial = estimate_sim3_umeyama(source_points, target_points)
    residuals = _sim3_residuals(initial, source_points, target_points)
    keep_count = max(thresholds.min_inliers, int(math.ceil(0.8 * residuals.size)))
    keep = np.argsort(residuals)[:keep_count]
    trimmed = estimate_sim3_umeyama(source_points[keep], target_points[keep])
    residuals = _sim3_residuals(trimmed, source_points, target_points)
    inliers = residuals <= thresholds.max_center_rmse_m
    if int(np.count_nonzero(inliers)) >= thresholds.min_inliers:
        trimmed = estimate_sim3_umeyama(source_points[inliers], target_points[inliers])
        residuals = _sim3_residuals(trimmed, source_points, target_points)
        inliers = residuals <= thresholds.max_center_rmse_m
    return trimmed, residuals, inliers


def _overlap_correspondences(
    source_window: TeacherWindowPrediction,
    target_window: TeacherWindowPrediction,
    target_node: TeacherWindowNode,
    overlap_frame_ids: Sequence[int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    source_by_id = observations_by_id(source_window)
    target_by_id = observations_by_id(target_window)
    source_parts: list[npt.NDArray[np.float32]] = []
    target_parts: list[npt.NDArray[np.float32]] = []
    for frame_id in overlap_frame_ids:
        source_parts.append(source_by_id[frame_id].pose.camera_center_world_m[None, :])
        target_parts.append(
            target_node.sim3_local_to_global.apply_points(
                target_by_id[frame_id].pose.camera_center_world_m[None, :]
            )
        )
        source_pm, target_pm = pointmap_correspondences(
            source_window,
            target_window,
            target_node,
            frame_id,
        )
        if source_pm.size and target_pm.size:
            source_parts.append(source_pm)
            target_parts.append(target_pm)
    return (
        np.concatenate(source_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
    )


def _deduped_transformed_observations(
    windows: Sequence[TeacherWindowPrediction],
    graph: StitchGraph,
) -> tuple[DepthObservation, ...]:
    nodes_by_index = {node.window_index: node for node in graph.nodes}
    observations: list[DepthObservation] = []
    seen: set[int] = set()
    for window in windows:
        node = nodes_by_index[window.window_index]
        if not node.accepted:
            continue
        for observation in window.observations:
            if observation.frame_id in seen:
                continue
            seen.add(observation.frame_id)
            observations.append(_transform_observation(observation, node))
    return tuple(observations)


def _transform_observation(
    observation: DepthObservation,
    node: TeacherWindowNode,
) -> DepthObservation:
    depth, sigma = apply_sim3_scale_to_depth(
        observation.depth_m,
        observation.depth_sigma_m,
        node.sim3_local_to_global.scale,
    )
    T_world_camera = apply_sim3_to_camera_pose(
        observation.pose.T_world_camera,
        node.sim3_local_to_global,
    )
    confidence = np.clip(
        observation.confidence.astype(np.float32, copy=False) * np.float32(node.confidence_scale),
        0.0,
        1.0,
    )
    pose = PoseEstimate(
        frame_id=observation.pose.frame_id,
        timestamp_ns=observation.pose.timestamp_ns,
        T_world_camera=T_world_camera,
        q_world_camera_xyzw=quaternion_xyzw_from_rotation_matrix(T_world_camera[:3, :3]).astype(
            np.float32
        ),
        camera_center_world_m=T_world_camera[:3, 3].astype(np.float32),
        covariance_6x6=observation.pose.covariance_6x6,
        confidence=float(np.clip(observation.pose.confidence * node.confidence_scale, 0.0, 1.0)),
        tracking_state=observation.pose.tracking_state,
        scale_source=observation.pose.scale_source,
        diagnostics={
            **dict(observation.pose.diagnostics),
            "stitching": {
                "window_index": node.window_index,
                "pseudo_submap_id": node.pseudo_submap_id,
                "sim3_local_to_global": node.sim3_local_to_global.to_record(),
                "confidence_scale": node.confidence_scale,
            },
        },
    )
    return DepthObservation(
        frame_id=observation.frame_id,
        camera=observation.camera,
        pose=pose,
        depth_m=depth,
        depth_sigma_m=sigma,
        confidence=confidence,
        static_mask=observation.static_mask,
        object_id=observation.object_id,
        rgb_u8=observation.rgb_u8,
        source=observation.source,
    )


def _rejected_node(window: TeacherWindowPrediction, reason: str) -> TeacherWindowNode:
    return TeacherWindowNode(
        window_index=window.window_index,
        frame_ids=window.frame_ids,
        sim3_local_to_global=Sim3Transform.identity(),
        pseudo_submap_id=-1,
        accepted=False,
        rejection_reason=reason,
    )


def _rejected_edge(
    source_window: TeacherWindowPrediction,
    target_window: TeacherWindowPrediction,
    overlap_frame_ids: tuple[int, ...],
    reason: str,
) -> TeacherWindowOverlapEdge:
    return TeacherWindowOverlapEdge(
        source_window_index=source_window.window_index,
        target_window_index=target_window.window_index,
        overlap_frame_ids=overlap_frame_ids,
        sim3_source_to_global=Sim3Transform.identity(),
        accepted=False,
        rejection_reason=reason,
        metrics={
            "overlap_camera_center_rmse_m": None,
            "overlap_rotation_error_deg": None,
            "overlap_depth_scale_ratio": None,
            "accepted_inlier_count": 0,
            "rejected_outlier_count": 0,
            "sim3_scale": None,
            "sim3_translation_norm_m": None,
            "sim3_rotation_deg": None,
            "correspondence_count": 0,
            "boundary_center_jumps_m": [],
        },
    )


def _sim3_residuals(
    sim3: Sim3Transform,
    source: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    transformed = sim3.apply_points(source).astype(np.float64, copy=False)
    return cast(npt.NDArray[np.float64], np.linalg.norm(transformed - target, axis=1))


def _rmse(values: npt.NDArray[np.float64]) -> float | None:
    if values.size == 0:
        return None
    return float(math.sqrt(float(np.mean(values * values))))


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))
