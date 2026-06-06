"""RoomGraph optimizer orchestration and artifact export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.best_map_selection import BestMapSelectionResult
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.fused_world_map_artifacts import (
    FusedPointCloud,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    write_map_artifacts,
)
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.roomgraph_core import (
    RoomGraphFrame,
    RoomGraphVariantResult,
    build_roomgraph_problem,
    optimize_roomgraph_variant,
)
from atlas3r.offline.roomgraph_metrics import RoomGraphMetrics
from atlas3r.offline.roomgraph_report import render_roomgraph_report_markdown
from atlas3r.offline.roomgraph_tracks import (
    RoomGraphTrackSource,
    TrackBuildResult,
    build_roomgraph_tracks,
)
from atlas3r.offline.run_manifest import FailurePoint, write_json

ROOMGRAPH_TRUTH_BOUNDARY: dict[str, object] = {
    "label_type": "unanchored_teacher_consensus_map",
    "measured_geometry": False,
    "observed_only": True,
    "predicted_completion": False,
    "hidden_geometry_measured": False,
    "metric_scale_source": "depth_pro_vggt_roomgraph_soft_metric_prior",
    "scale_status": "soft_metric_unanchored",
    "physical_accuracy_claim": False,
    "training_quality": False,
    "realtime_claim": False,
    "optimized_world_state": "roomgraph_track_depth_pose_optimizer",
    "accuracy_report": False,
    "usable_for_training": False,
}


@dataclass(frozen=True)
class RoomGraphOptions:
    enabled: bool = False
    device: str = "cuda:0"
    cotracker_checkpoint: str | None = None
    max_keyframes: int = 24
    track_grid_size: int = 16
    max_tracks: int = 48
    max_iterations: int = 25
    export_world_map: bool = True
    track_source: RoomGraphTrackSource = "auto"


@dataclass(frozen=True)
class RoomGraphOptimizerResult:
    status: str
    selected_variant: str = "none"
    track_source: str = "none"
    track_count: int = 0
    track_inlier_ratio: float = 0.0
    point_count: int = 0
    occupied_voxel_count: int = 0
    mesh_triangle_count: int = 0
    trajectory_count: int = 0
    metrics_path: str | None = None
    report_path: str | None = None
    fused_points_npz_path: str | None = None
    fused_points_ply_path: str | None = None
    occupancy_grid_npz_path: str | None = None
    observed_voxel_mesh_ply_path: str | None = None
    camera_trajectory_path: str | None = None
    before_metrics: dict[str, object] | None = None
    after_metrics: dict[str, object] | None = None
    improvement: dict[str, float] | None = None
    reason: str | None = None

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        paths = (
            "roomgraph/track_summary.json",
            "roomgraph/tracks.npz",
            self.metrics_path,
            self.report_path,
            self.fused_points_npz_path,
            self.fused_points_ply_path,
            self.occupancy_grid_npz_path,
            self.observed_voxel_mesh_ply_path,
            self.camera_trajectory_path,
        )
        return tuple(path for path in paths if path is not None)


def write_roomgraph_optimizer(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    best_map: BestMapSelectionResult,
    voxel_size_m: float,
    options: RoomGraphOptions,
    failure_points: list[FailurePoint],
) -> RoomGraphOptimizerResult:
    if not options.enabled:
        return RoomGraphOptimizerResult(status="disabled")
    root = Path(run_dir)
    (root / "roomgraph").mkdir(parents=True, exist_ok=True)
    usable_frame_ids = _usable_frame_ids(proposal_cache)
    tracks = build_roomgraph_tracks(
        root,
        frame_cache=frame_cache,
        keyframes=keyframes,
        usable_frame_ids=usable_frame_ids,
        checkpoint_path=options.cotracker_checkpoint,
        device=options.device,
        max_keyframes=options.max_keyframes,
        grid_size=options.track_grid_size,
        max_tracks=options.max_tracks,
        requested_source=options.track_source,
    )
    if tracks.status != "available":
        _append_failure(failure_points, tracks.reason or "no tracks")
        return RoomGraphOptimizerResult(
            status="unavailable",
            track_source=tracks.source,
            metrics_path="world_map_roomgraph/roomgraph_metrics.json",
            report_path="world_map_roomgraph/roomgraph_report.md",
            reason=tracks.reason,
        )
    problem = build_roomgraph_problem(
        proposal_cache=proposal_cache,
        disagreement=disagreement,
        tracks=tracks.tracks,
    )
    variants = [
        optimize_roomgraph_variant(
            problem, variant="depth_only", max_iterations=options.max_iterations
        ),
        optimize_roomgraph_variant(
            problem, variant="pose_only", max_iterations=options.max_iterations
        ),
        optimize_roomgraph_variant(problem, variant="joint", max_iterations=options.max_iterations),
    ]
    selected = _select_variant(variants)
    cloud = _build_roomgraph_cloud(frame_cache, problem.frames, selected, max_points=2_000_000)
    occupancy = build_sparse_occupancy(
        cloud.points_world_m,
        cloud.colors_u8,
        cloud.confidence,
        voxel_size_m=voxel_size_m,
    )
    mesh = build_observed_voxel_mesh(occupancy)
    trajectory = _trajectory(problem.frames, selected.centers_world_m)
    paths = write_map_artifacts(
        root,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory=trajectory,
        write_observed_mesh=True,
        rejected_pose_count=0,
        map_dir_name="world_map_roomgraph",
        truth_boundary=ROOMGRAPH_TRUTH_BOUNDARY,
    )
    before = _merge_before_metrics(best_map, selected.before_metrics)
    after = selected.after_metrics.to_dict() | {
        "map_bbox_size_m": _bbox_size(cloud.points_world_m),
        "fused_point_count": int(cloud.points_world_m.shape[0]),
        "occupied_voxel_count": occupancy.occupied_voxel_count,
        "observed_mesh_triangle_count": mesh.triangle_count,
        "camera_trajectory_count": len(trajectory),
    }
    before_for_report = before | {
        "fused_point_count": best_map.point_count,
        "occupied_voxel_count": best_map.occupied_voxel_count,
        "observed_mesh_triangle_count": best_map.mesh_triangle_count,
        "camera_trajectory_count": best_map.trajectory_count,
    }
    payload = _metrics_payload(
        input_path=input_path,
        tracks=tracks,
        variants=variants,
        selected=selected,
        before=before_for_report,
        after=after,
        artifact_paths=paths,
    )
    map_dir = root / "world_map_roomgraph"
    write_json(map_dir / "roomgraph_metrics.json", payload)
    (map_dir / "roomgraph_report.md").write_text(
        render_roomgraph_report_markdown(payload), encoding="utf-8"
    )
    return RoomGraphOptimizerResult(
        status=selected.status,
        selected_variant=selected.variant,
        track_source=tracks.source,
        track_count=tracks.track_count,
        track_inlier_ratio=selected.after_metrics.track_inlier_ratio,
        point_count=int(cloud.points_world_m.shape[0]),
        occupied_voxel_count=occupancy.occupied_voxel_count,
        mesh_triangle_count=mesh.triangle_count,
        trajectory_count=len(trajectory),
        metrics_path="world_map_roomgraph/roomgraph_metrics.json",
        report_path="world_map_roomgraph/roomgraph_report.md",
        fused_points_npz_path="world_map_roomgraph/fused_points.npz",
        fused_points_ply_path=paths.get("fused_points_ply"),
        occupancy_grid_npz_path="world_map_roomgraph/occupancy_grid.npz",
        observed_voxel_mesh_ply_path=paths.get("observed_voxel_mesh"),
        camera_trajectory_path="world_map_roomgraph/camera_trajectory.json",
        before_metrics=before_for_report,
        after_metrics=after,
        improvement=selected.improvement,
    )


def _usable_frame_ids(proposal_cache: ProposalCacheResult) -> set[int]:
    cameras = {int(str(record["frame_id"])) for record in proposal_cache.vggt_camera_records}
    depth_pro = {int(str(record["frame_id"])) for record in proposal_cache.depth_pro_depth_records}
    vggt_depth = {int(str(record["frame_id"])) for record in proposal_cache.vggt_depth_records}
    return cameras & depth_pro & vggt_depth


def _select_variant(variants: list[RoomGraphVariantResult]) -> RoomGraphVariantResult:
    def score(item: RoomGraphVariantResult) -> float:
        if item.after_metrics.track_count == 0:
            return -1.0
        return (
            max(item.improvement.values() or [0.0]) + 0.02 * item.after_metrics.track_inlier_ratio
        )

    return max(variants, key=score)


def _build_roomgraph_cloud(
    frame_cache: FrameCacheResult,
    frames: tuple[RoomGraphFrame, ...],
    selected: RoomGraphVariantResult,
    *,
    max_points: int,
) -> FusedPointCloud:
    frame_rgb = {
        frame.frame_id: packet.rgb_u8
        for frame, packet in zip(frame_cache.records, frame_cache.frames, strict=False)
    }
    chunks: list[NDArray[np.float32]] = []
    colors: list[NDArray[np.uint8]] = []
    confs: list[NDArray[np.float32]] = []
    frame_ids: list[NDArray[np.int64]] = []
    keyframe_ids: list[NDArray[np.int64]] = []
    source_ids: list[NDArray[np.int32]] = []
    sigma: list[NDArray[np.float32]] = []
    disagreement: list[NDArray[np.float32]] = []
    per_frame_counts: dict[str, int] = {}
    stride = 8
    for index, frame in enumerate(frames):
        depth = selected.depth_scale[index] * frame.depth_pro_m + selected.depth_bias_m[index]
        valid = frame.depth_valid & np.isfinite(depth) & (depth > 0.0)
        sample = np.zeros(depth.shape, dtype=np.bool_)
        sample[::stride, ::stride] = True
        mask = valid & sample
        if not np.any(mask):
            continue
        points = _lift_depth(frame, selected.centers_world_m[index], depth, mask)
        if points.size == 0:
            continue
        rgb = frame_rgb.get(frame.frame_id)
        if rgb is None:
            color = np.full((points.shape[0], 3), 180, dtype=np.uint8)
        else:
            color = _colors_for_mask(rgb, depth.shape, mask)
        rel = np.abs(frame.vggt_depth_m[mask] - depth[mask]) / np.maximum(depth[mask], 1e-3)
        conf = np.clip(1.0 / (1.0 + 2.0 * rel), 0.05, 0.95).astype(np.float32)
        chunks.append(points)
        colors.append(color)
        confs.append(conf)
        frame_ids.append(np.full(points.shape[0], frame.frame_id, dtype=np.int64))
        keyframe_ids.append(np.full(points.shape[0], frame.keyframe_id, dtype=np.int64))
        source_ids.append(np.full(points.shape[0], 6, dtype=np.int32))
        sigma.append(np.maximum(0.05, 1.0 - conf).astype(np.float32))
        disagreement.append(rel.astype(np.float32))
        per_frame_counts[str(frame.frame_id)] = int(points.shape[0])
    if not chunks:
        return _empty_cloud()
    points_all = np.concatenate(chunks, axis=0).astype(np.float32)
    keep = (
        np.linspace(0, points_all.shape[0] - 1, min(max_points, points_all.shape[0]))
        .round()
        .astype(np.int64)
    )
    return FusedPointCloud(
        points_world_m=points_all[keep],
        colors_u8=np.concatenate(colors, axis=0).astype(np.uint8)[keep],
        confidence=np.concatenate(confs, axis=0).astype(np.float32)[keep],
        source_frame_ids=np.concatenate(frame_ids, axis=0).astype(np.int64)[keep],
        source_keyframe_ids=np.concatenate(keyframe_ids, axis=0).astype(np.int64)[keep],
        depth_source_id=np.concatenate(source_ids, axis=0).astype(np.int32)[keep],
        disagreement_rel=np.concatenate(disagreement, axis=0).astype(np.float32)[keep],
        point_sigma_m=np.concatenate(sigma, axis=0).astype(np.float32)[keep],
        depth_source="roomgraph_depthpro_vggt_track_consensus",
        metric_scale_source=str(ROOMGRAPH_TRUTH_BOUNDARY["metric_scale_source"]),
        valid_depth_ratio=1.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=float(np.mean(np.concatenate(disagreement)))
        if disagreement
        else None,
        mapped_disagreement_p50=float(np.percentile(np.concatenate(disagreement), 50))
        if disagreement
        else None,
        mapped_disagreement_p95=float(np.percentile(np.concatenate(disagreement), 95))
        if disagreement
        else None,
        per_frame_point_counts=per_frame_counts,
    )


def _lift_depth(
    frame: RoomGraphFrame,
    center_world_m: NDArray[np.float32],
    depth: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - float(frame.K[0, 2])) * z / float(frame.K[0, 0])
    y = (v.astype(np.float32) - float(frame.K[1, 2])) * z / float(frame.K[1, 1])
    points_camera = np.stack([x, y, z], axis=1).astype(np.float32)
    points = points_camera @ frame.R_world_camera.T + center_world_m[None, :]
    return cast(NDArray[np.float32], points.astype(np.float32))


def _colors_for_mask(
    rgb_u8: NDArray[np.uint8], depth_shape: tuple[int, int], mask: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    h, w = depth_shape
    src_h, src_w = rgb_u8.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, h)).astype(np.int32), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, w)).astype(np.int32), 0, src_w - 1)
    colors = rgb_u8[y_idx[:, None], x_idx[None, :]][mask].astype(np.uint8)
    return cast(NDArray[np.uint8], colors)


def _trajectory(
    frames: tuple[RoomGraphFrame, ...], centers: NDArray[np.float32]
) -> list[dict[str, object]]:
    rows = []
    for index, frame in enumerate(frames):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = frame.R_world_camera
        T[:3, 3] = centers[index]
        rows.append(
            {
                "frame_id": frame.frame_id,
                "keyframe_id": frame.keyframe_id,
                "timestamp_ns": None,
                "T_world_camera": T.astype(float).tolist(),
                "camera_center_world_m": centers[index].astype(float).tolist(),
                "pose_source": "roomgraph_optimized_vggt_rotation_track_translation",
                "pose_confidence": 0.5,
                "metric_scale_source": ROOMGRAPH_TRUTH_BOUNDARY["metric_scale_source"],
                "pseudo_submap_id": 0,
            }
        )
    return rows


def _metrics_payload(
    *,
    input_path: str,
    tracks: TrackBuildResult,
    variants: list[RoomGraphVariantResult],
    selected: RoomGraphVariantResult,
    before: dict[str, object],
    after: dict[str, object],
    artifact_paths: dict[str, str],
) -> dict[str, object]:
    return {
        "format_name": "atlas3r_roomgraph_metrics",
        "format_version": 1,
        "input_path": input_path,
        "status": selected.status,
        "track_source": tracks.source,
        "track_count": tracks.track_count,
        "track_observation_count": tracks.observation_count,
        "selected_variant": selected.variant,
        "variants": [
            {
                "variant": item.variant,
                "status": item.status,
                "before": item.before_metrics.to_dict(),
                "after": item.after_metrics.to_dict(),
                "improvement": item.improvement,
                "cost_initial": item.cost_initial,
                "cost_final": item.cost_final,
                "iterations": item.iterations,
            }
            for item in variants
        ],
        "before": before,
        "after": after,
        "improvement": selected.improvement,
        "truth_boundary": ROOMGRAPH_TRUTH_BOUNDARY,
        "artifacts": artifact_paths
        | {
            "roomgraph_metrics": "world_map_roomgraph/roomgraph_metrics.json",
            "roomgraph_report": "world_map_roomgraph/roomgraph_report.md",
        },
    }


def _merge_before_metrics(
    best_map: BestMapSelectionResult, track_before: RoomGraphMetrics
) -> dict[str, object]:
    payload = track_before.to_dict()
    if best_map.bbox_world_min_m is not None and best_map.bbox_world_max_m is not None:
        bbox_min = np.asarray(best_map.bbox_world_min_m, dtype=np.float32)
        bbox_max = np.asarray(best_map.bbox_world_max_m, dtype=np.float32)
        payload["map_bbox_size_m"] = (bbox_max - bbox_min).astype(float).tolist()
    else:
        payload["map_bbox_size_m"] = None
    return payload


def _bbox_size(points: NDArray[np.float32]) -> list[float] | None:
    if points.size == 0:
        return None
    values = (points.max(axis=0) - points.min(axis=0)).astype(float)
    return [float(value) for value in values.tolist()]


def _empty_cloud() -> FusedPointCloud:
    return FusedPointCloud(
        points_world_m=np.zeros((0, 3), dtype=np.float32),
        colors_u8=np.zeros((0, 3), dtype=np.uint8),
        confidence=np.zeros((0,), dtype=np.float32),
        source_frame_ids=np.zeros((0,), dtype=np.int64),
        source_keyframe_ids=np.zeros((0,), dtype=np.int64),
        depth_source_id=np.zeros((0,), dtype=np.int32),
        disagreement_rel=np.zeros((0,), dtype=np.float32),
        point_sigma_m=np.zeros((0,), dtype=np.float32),
        depth_source="roomgraph_empty",
        metric_scale_source=str(ROOMGRAPH_TRUTH_BOUNDARY["metric_scale_source"]),
        valid_depth_ratio=0.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=None,
        mapped_disagreement_p50=None,
        mapped_disagreement_p95=None,
        per_frame_point_counts={},
    )


def _append_failure(failure_points: list[FailurePoint], why: str) -> None:
    failure_points.append(
        FailurePoint(
            module="roomgraph_optimizer",
            code="roomgraph_tracks_unavailable",
            severity="warning",
            status="unavailable",
            why=why,
            input_missing="long-lived point tracks",
            dependency_missing=None,
            future_module="provide CoTracker checkpoint or OpenCV-readable frames",
            artifact_path="world_map_roomgraph/roomgraph_metrics.json",
        )
    )
