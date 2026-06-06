"""Best inspectable map selection for no-anchor room-walk runs."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.fused_world_map import (
    FusedWorldMapOptions,
    FusedWorldMapResult,
    write_fused_world_map,
)
from atlas3r.offline.fused_world_map_artifacts import (
    OPTIMIZED_TRUTH_BOUNDARY,
    FusedPointCloud,
    bbox,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    write_map_artifacts,
    write_quality_and_manifest,
)
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.map_consistency_optimizer import MapConsistencyOptimizerResult
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint, write_json
from atlas3r.offline.topdown_preview import (
    write_inspection_instructions,
    write_topdown_preview,
)


@dataclass(frozen=True)
class BestMapSelectionResult:
    status: str
    selected_source: str = "none"
    manifest_path: str | None = None
    fused_points_npz_path: str | None = None
    fused_points_ply_path: str | None = None
    occupancy_grid_npz_path: str | None = None
    occupancy_grid_metadata_path: str | None = None
    observed_voxel_mesh_ply_path: str | None = None
    camera_trajectory_path: str | None = None
    map_quality_json_path: str | None = None
    map_quality_markdown_path: str | None = None
    topdown_preview_path: str | None = None
    inspection_instructions_path: str | None = None
    point_count: int = 0
    occupied_voxel_count: int = 0
    mesh_triangle_count: int = 0
    trajectory_count: int = 0
    bbox_world_min_m: tuple[float, float, float] | None = None
    bbox_world_max_m: tuple[float, float, float] | None = None
    cleanup: dict[str, object] | None = None
    failure_reasons: tuple[str, ...] = ()

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        paths = (
            self.manifest_path,
            self.fused_points_npz_path,
            self.fused_points_ply_path,
            self.occupancy_grid_npz_path,
            self.occupancy_grid_metadata_path,
            self.observed_voxel_mesh_ply_path,
            self.camera_trajectory_path,
            self.map_quality_json_path,
            self.map_quality_markdown_path,
            self.topdown_preview_path,
            self.inspection_instructions_path,
        )
        return tuple(path for path in paths if path is not None)


def write_best_world_map(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    raw_world_map: FusedWorldMapResult,
    optimizer: MapConsistencyOptimizerResult,
    map_options: FusedWorldMapOptions,
    export_best_world_map: bool,
    failure_points: list[FailurePoint],
) -> BestMapSelectionResult:
    if not export_best_world_map:
        return BestMapSelectionResult(status="disabled")
    root = Path(run_dir)
    source_map = _select_source_map(raw_world_map, optimizer)
    fallback_map: FusedWorldMapResult | None = None
    if (
        source_map is None
        and proposal_cache.vggt_camera_records
        and proposal_cache.vggt_depth_records
    ):
        fallback_map = write_fused_world_map(
            root,
            input_path=input_path,
            frame_cache=frame_cache,
            keyframes=keyframes,
            proposal_cache=proposal_cache,
            disagreement=disagreement,
            options=replace(
                map_options,
                export_world_map=True,
                depth_source="vggt",
                output_dir_name="world_map_vggt_fallback",
                write_observed_mesh=True,
                write_occupancy=True,
            ),
            failure_points=failure_points,
        )
        source_map = ("vggt_fallback", fallback_map)
    if source_map is None:
        _append_best_map_failure(failure_points, "no raw, optimized, or VGGT fallback map exists")
        return BestMapSelectionResult(
            status="unavailable",
            failure_reasons=("no raw, optimized, or VGGT fallback map exists",),
        )
    selected_source, selected = source_map
    if selected.fused_points_npz_path is None or selected.camera_trajectory_path is None:
        reason = f"selected {selected_source} map has no point cloud or trajectory artifact"
        _append_best_map_failure(failure_points, reason)
        return BestMapSelectionResult(status="unavailable", failure_reasons=(reason,))
    try:
        cloud = _read_cloud(root / selected.fused_points_npz_path)
        trajectory, rejected_pose_count = _read_trajectory(root / selected.camera_trajectory_path)
    except (OSError, ValueError, KeyError) as exc:
        reason = f"selected {selected_source} map could not be loaded: {exc}"
        _append_best_map_failure(failure_points, reason)
        return BestMapSelectionResult(status="unavailable", failure_reasons=(reason,))
    cleanup_reasons: list[str] = []
    clean_cloud, cleanup = _cleanup_cloud(
        cloud,
        voxel_size_m=map_options.voxel_size_m,
        cleanup_reasons=cleanup_reasons,
    )
    if clean_cloud.points_world_m.shape[0] == 0 and cloud.points_world_m.shape[0] > 0:
        cleanup_reasons.append("cleanup would empty map; restored selected source cloud")
        clean_cloud = cloud
    occupancy = build_sparse_occupancy(
        clean_cloud.points_world_m,
        clean_cloud.colors_u8,
        clean_cloud.confidence,
        voxel_size_m=map_options.voxel_size_m,
    )
    mesh = build_observed_voxel_mesh(occupancy)
    paths = write_map_artifacts(
        root,
        cloud=clean_cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory=trajectory,
        write_observed_mesh=True,
        rejected_pose_count=rejected_pose_count,
        map_dir_name="world_map_best",
        truth_boundary=OPTIMIZED_TRUTH_BOUNDARY,
    )
    quality = write_quality_and_manifest(
        root,
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        cloud=clean_cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory_count=len(trajectory),
        write_observed_mesh=True,
        voxel_size_m=map_options.voxel_size_m,
        min_confidence=map_options.min_confidence,
        max_relative_disagreement=map_options.max_relative_disagreement,
        paths=paths,
        failure_reasons=cleanup_reasons,
        rejected_pose_count=rejected_pose_count,
        map_dir_name="world_map_best",
        truth_boundary=OPTIMIZED_TRUTH_BOUNDARY,
    )
    cleanup_payload = cleanup | {
        "selected_best_map_source": selected_source,
        "failure_reasons": cleanup_reasons,
    }
    _patch_best_json(root / "world_map_best" / "world_map_manifest.json", cleanup_payload)
    _patch_best_json(root / "world_map_best" / "map_quality.json", cleanup_payload)
    write_topdown_preview(root / "world_map_best" / "topdown_preview.svg", clean_cloud, trajectory)
    write_inspection_instructions(root / "world_map_best" / "inspection_instructions.md")
    bbox_min, bbox_max = bbox(clean_cloud.points_world_m)
    if clean_cloud.points_world_m.shape[0] > 0 and (
        occupancy.occupied_voxel_count == 0 or mesh.triangle_count == 0
    ):
        _append_best_map_failure(
            failure_points, "best map cleanup produced empty occupancy or mesh"
        )
    return BestMapSelectionResult(
        status="available" if bool(quality["inspectable_map_available"]) else "unavailable",
        selected_source=selected_source,
        manifest_path="world_map_best/world_map_manifest.json",
        fused_points_npz_path="world_map_best/fused_points.npz",
        fused_points_ply_path=paths.get("fused_points_ply"),
        occupancy_grid_npz_path="world_map_best/occupancy_grid.npz",
        occupancy_grid_metadata_path="world_map_best/occupancy_grid_metadata.json",
        observed_voxel_mesh_ply_path="world_map_best/observed_voxel_mesh.ply",
        camera_trajectory_path="world_map_best/camera_trajectory.json",
        map_quality_json_path="world_map_best/map_quality.json",
        map_quality_markdown_path="world_map_best/map_quality.md",
        topdown_preview_path="world_map_best/topdown_preview.svg",
        inspection_instructions_path="world_map_best/inspection_instructions.md",
        point_count=int(clean_cloud.points_world_m.shape[0]),
        occupied_voxel_count=occupancy.occupied_voxel_count,
        mesh_triangle_count=mesh.triangle_count,
        trajectory_count=len(trajectory),
        bbox_world_min_m=_bbox_tuple(bbox_min),
        bbox_world_max_m=_bbox_tuple(bbox_max),
        cleanup=cleanup_payload,
        failure_reasons=tuple(cleanup_reasons),
    )


def _select_source_map(
    raw_world_map: FusedWorldMapResult, optimizer: MapConsistencyOptimizerResult
) -> tuple[str, FusedWorldMapResult] | None:
    optimized = optimizer.optimized_world_map
    raw_points = max(1, raw_world_map.point_count)
    optimized_retained = optimized.point_count / raw_points
    if (
        optimizer.improvement_passed
        and optimized.point_count > 0
        and optimized.occupied_voxel_count > 0
        and optimized.mesh_triangle_count > 0
        and optimized_retained >= 0.5
    ):
        return "optimized", optimized
    if (
        raw_world_map.point_count > 0
        and raw_world_map.occupied_voxel_count > 0
        and raw_world_map.mesh_triangle_count > 0
    ):
        return "raw_consensus", raw_world_map
    return None


def _read_cloud(path: Path) -> FusedPointCloud:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        return FusedPointCloud(
            points_world_m=payload["points_world_m"].astype(np.float32),
            colors_u8=payload["colors_u8"].astype(np.uint8),
            confidence=payload["confidence"].astype(np.float32),
            source_frame_ids=payload["source_frame_ids"].astype(np.int64),
            source_keyframe_ids=payload["source_keyframe_ids"].astype(np.int64),
            depth_source_id=payload["depth_source_id"].astype(np.int32),
            disagreement_rel=payload["disagreement_rel"].astype(np.float32),
            point_sigma_m=payload["point_sigma_m"].astype(np.float32),
            depth_source=str(metadata.get("depth_source", "unknown")),
            metric_scale_source=str(
                metadata.get("metric_scale_source", "depth_pro_vggt_soft_metric_prior")
            ),
            valid_depth_ratio=0.0,
            rejected_low_confidence_ratio=0.0,
            rejected_high_disagreement_ratio=0.0,
            mapped_disagreement_mean=None,
            mapped_disagreement_p50=None,
            mapped_disagreement_p95=None,
            per_frame_point_counts={},
        )


def _read_trajectory(path: Path) -> tuple[list[dict[str, object]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = payload.get("poses", [])
    if not isinstance(poses, list):
        raise ValueError("camera trajectory poses must be a list")
    return [dict(item) for item in poses if isinstance(item, dict)], int(
        payload.get("rejected_nonfinite_pose_count", 0)
    )


def _cleanup_cloud(
    cloud: FusedPointCloud, *, voxel_size_m: float, cleanup_reasons: list[str]
) -> tuple[FusedPointCloud, dict[str, object]]:
    original = int(cloud.points_world_m.shape[0])
    finite_mask = (
        np.isfinite(cloud.points_world_m).all(axis=1)
        & np.isfinite(cloud.confidence)
        & np.isfinite(cloud.point_sigma_m)
    )
    filtered = _filter_cloud(cloud, finite_mask)
    after_finite = int(filtered.points_world_m.shape[0])
    percentile_mask = _percentile_mask(filtered.points_world_m)
    if percentile_mask.sum() >= max(1, math.ceil(0.5 * after_finite)):
        filtered = _filter_cloud(filtered, percentile_mask)
    else:
        cleanup_reasons.append("percentile outlier cleanup skipped to avoid dropping >50%")
    after_percentile = int(filtered.points_world_m.shape[0])
    filtered, component_stats = _cleanup_voxel_components(filtered, voxel_size_m, cleanup_reasons)
    final = int(filtered.points_world_m.shape[0])
    return filtered, {
        "cleanup_original_point_count": original,
        "cleanup_after_finite_point_count": after_finite,
        "cleanup_after_percentile_point_count": after_percentile,
        "cleanup_final_point_count": final,
        "cleanup_retained_point_ratio": float(final / original) if original else 0.0,
        **component_stats,
        "cleanup_aggressive": bool(original and final / original < 0.5),
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
    }


def _filter_cloud(cloud: FusedPointCloud, mask: NDArray[np.bool_]) -> FusedPointCloud:
    return FusedPointCloud(
        points_world_m=cloud.points_world_m[mask],
        colors_u8=cloud.colors_u8[mask],
        confidence=cloud.confidence[mask],
        source_frame_ids=cloud.source_frame_ids[mask],
        source_keyframe_ids=cloud.source_keyframe_ids[mask],
        depth_source_id=cloud.depth_source_id[mask],
        disagreement_rel=cloud.disagreement_rel[mask],
        point_sigma_m=cloud.point_sigma_m[mask],
        depth_source=cloud.depth_source,
        metric_scale_source=cloud.metric_scale_source,
        valid_depth_ratio=cloud.valid_depth_ratio,
        rejected_low_confidence_ratio=cloud.rejected_low_confidence_ratio,
        rejected_high_disagreement_ratio=cloud.rejected_high_disagreement_ratio,
        mapped_disagreement_mean=cloud.mapped_disagreement_mean,
        mapped_disagreement_p50=cloud.mapped_disagreement_p50,
        mapped_disagreement_p95=cloud.mapped_disagreement_p95,
        per_frame_point_counts=cloud.per_frame_point_counts,
    )


def _percentile_mask(points: NDArray[np.float32]) -> NDArray[np.bool_]:
    if points.shape[0] < 16:
        return np.ones((points.shape[0],), dtype=np.bool_)
    low = np.percentile(points, 1.0, axis=0)
    high = np.percentile(points, 99.0, axis=0)
    return cast(NDArray[np.bool_], np.all((points >= low) & (points <= high), axis=1))


def _cleanup_voxel_components(
    cloud: FusedPointCloud, voxel_size_m: float, cleanup_reasons: list[str]
) -> tuple[FusedPointCloud, dict[str, object]]:
    occupancy = build_sparse_occupancy(
        cloud.points_world_m, cloud.colors_u8, cloud.confidence, voxel_size_m=voxel_size_m
    )
    total = occupancy.occupied_voxel_count
    if total <= 1:
        return cloud, {"connected_component_count": total, "kept_component_count": total}
    components = _voxel_components(occupancy.voxel_indices_ijk)
    sorted_components = sorted(components, key=len, reverse=True)
    keep_target = max(1, math.ceil(0.5 * total))
    kept: set[tuple[int, int, int]] = set()
    for component in sorted_components:
        kept.update(component)
        if len(kept) >= keep_target:
            break
    if len(kept) == total:
        return cloud, {
            "connected_component_count": len(components),
            "kept_component_count": len(sorted_components),
            "kept_voxel_count": total,
            "removed_voxel_count": 0,
        }
    point_voxels = np.floor(
        (cloud.points_world_m - occupancy.bbox_world_min_m) / voxel_size_m
    ).astype(np.int32)
    mask = np.asarray([_voxel_tuple(row) in kept for row in point_voxels], dtype=np.bool_)
    if int(mask.sum()) < max(1, math.ceil(0.5 * cloud.points_world_m.shape[0])):
        cleanup_reasons.append("voxel component cleanup skipped to avoid dropping >50% of points")
        return cloud, {
            "connected_component_count": len(components),
            "kept_component_count": len(components),
            "kept_voxel_count": total,
            "removed_voxel_count": 0,
        }
    return _filter_cloud(cloud, mask), {
        "connected_component_count": len(components),
        "kept_component_count": sum(1 for component in components if component & kept),
        "kept_voxel_count": len(kept),
        "removed_voxel_count": total - len(kept),
    }


def _voxel_components(indices: NDArray[np.int32]) -> list[set[tuple[int, int, int]]]:
    occupied = {_voxel_tuple(row) for row in indices}
    components: list[set[tuple[int, int, int]]] = []
    while occupied:
        seed = occupied.pop()
        component = {seed}
        queue: deque[tuple[int, int, int]] = deque([seed])
        while queue:
            voxel = queue.popleft()
            for neighbor in _neighbors(voxel):
                if neighbor in occupied:
                    occupied.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _voxel_tuple(row: NDArray[np.int32]) -> tuple[int, int, int]:
    return (int(row[0]), int(row[1]), int(row[2]))


def _neighbors(voxel: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    i, j, k = voxel
    return (
        (i + 1, j, k),
        (i - 1, j, k),
        (i, j + 1, k),
        (i, j - 1, k),
        (i, j, k + 1),
        (i, j, k - 1),
    )


def _patch_best_json(path: Path, cleanup_payload: dict[str, object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(cleanup_payload)
    payload["physical_accuracy_claim"] = False
    payload["training_quality_claim"] = False
    write_json(path, payload)


def _append_best_map_failure(failure_points: list[FailurePoint], why: str) -> None:
    failure_points.append(
        FailurePoint(
            module="best_map_selection",
            code="best_map_unavailable",
            severity="warning",
            status="unavailable",
            why=why,
            input_missing="nonempty raw, optimized, or VGGT fallback map",
            artifact_path="world_map_best/world_map_manifest.json",
        )
    )


def _bbox_tuple(value: NDArray[np.float32] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))
