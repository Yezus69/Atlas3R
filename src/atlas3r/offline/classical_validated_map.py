"""Optional map filtering using aligned classical sparse geometry as a witness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.fused_world_map import FusedWorldMapOptions
from atlas3r.offline.fused_world_map_artifacts import (
    FusedPointCloud,
    bbox,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    write_map_artifacts,
    write_quality_and_manifest,
)
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.proposal_cache import ProposalCacheResult

CLASSICAL_VALIDATED_TRUTH_BOUNDARY: dict[str, object] = {
    "label_type": "unanchored_teacher_consensus_map",
    "measured_geometry": False,
    "observed_only": True,
    "predicted_completion": False,
    "hidden_geometry_measured": False,
    "metric_scale_source": "depth_pro_vggt_soft_metric_prior_classical_validated",
    "scale_status": "soft_metric_unanchored",
    "physical_accuracy_claim": False,
    "training_quality": False,
    "realtime_claim": False,
    "optimized_world_state": "diagnostic_depth_consistency_plus_classical_witness",
    "accuracy_report": False,
    "usable_for_training": False,
}


@dataclass(frozen=True)
class ClassicalValidatedMapResult:
    status: str
    reason: str
    accepted: bool = False
    selected_source: str = "none"
    retained_point_ratio: float = 0.0
    point_count: int = 0
    occupied_voxel_count: int = 0
    mesh_triangle_count: int = 0
    cloud: FusedPointCloud | None = None
    artifact_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "accepted": self.accepted,
            "selected_source": self.selected_source,
            "retained_point_ratio": self.retained_point_ratio,
            "point_count": self.point_count,
            "occupied_voxel_count": self.occupied_voxel_count,
            "observed_mesh_triangle_count": self.mesh_triangle_count,
            "artifact_paths": list(self.artifact_paths),
        }


def try_write_classical_validated_map(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    source_cloud: FusedPointCloud,
    trajectory: list[dict[str, object]],
    rejected_pose_count: int,
    map_options: FusedWorldMapOptions,
    selected_source: str,
    classical_comparison: object | None,
) -> ClassicalValidatedMapResult:
    if classical_comparison is None:
        return ClassicalValidatedMapResult(status="disabled", reason="classical comparison absent")
    if getattr(classical_comparison, "status", "unavailable") != "available":
        return ClassicalValidatedMapResult(
            status="unavailable", reason="classical comparison unavailable"
        )
    trajectory_status = str(getattr(classical_comparison, "trajectory_agreement_status", ""))
    map_status = str(getattr(classical_comparison, "map_agreement_status", ""))
    if trajectory_status not in {"agrees", "weak_agreement"} or map_status not in {
        "agrees",
        "weak_agreement",
    }:
        return ClassicalValidatedMapResult(
            status="skipped",
            reason=(
                "classical agreement not strong enough: "
                f"trajectory={trajectory_status}, map={map_status}"
            ),
        )
    aligned_path = getattr(classical_comparison, "aligned_sparse_points_path", None)
    if not isinstance(aligned_path, str):
        return ClassicalValidatedMapResult(
            status="unavailable", reason="aligned sparse points missing"
        )
    classical_points = _read_points(Path(run_dir) / aligned_path)
    if classical_points.shape[0] == 0 or source_cloud.points_world_m.shape[0] == 0:
        return ClassicalValidatedMapResult(
            status="unavailable", reason="empty classical or source map points"
        )
    mask = _bbox_validation_mask(
        source_cloud.points_world_m,
        classical_points,
        expansion_m=max(0.25, 5.0 * map_options.voxel_size_m),
    )
    retained = int(mask.sum())
    original = int(source_cloud.points_world_m.shape[0])
    retained_ratio = float(retained / original) if original else 0.0
    if retained_ratio < 0.5:
        return ClassicalValidatedMapResult(
            status="rejected",
            reason=(
                "anti-cheat rejected classical filter because it would keep <50% of best-map points"
            ),
            retained_point_ratio=retained_ratio,
        )
    if retained == original:
        return ClassicalValidatedMapResult(
            status="skipped",
            reason="classical bbox validation removed no points",
            retained_point_ratio=1.0,
        )
    filtered = _filter_cloud(source_cloud, mask)
    occupancy = build_sparse_occupancy(
        filtered.points_world_m,
        filtered.colors_u8,
        filtered.confidence,
        voxel_size_m=map_options.voxel_size_m,
    )
    mesh = build_observed_voxel_mesh(occupancy)
    if occupancy.occupied_voxel_count == 0 or mesh.triangle_count == 0:
        return ClassicalValidatedMapResult(
            status="rejected",
            reason="anti-cheat rejected classical filter because occupancy or mesh became empty",
            retained_point_ratio=retained_ratio,
        )
    root = Path(run_dir)
    paths = write_map_artifacts(
        root,
        cloud=filtered,
        occupancy=occupancy,
        mesh=mesh,
        trajectory=trajectory,
        write_observed_mesh=True,
        rejected_pose_count=rejected_pose_count,
        map_dir_name="world_map_classical_validated",
        truth_boundary=CLASSICAL_VALIDATED_TRUTH_BOUNDARY,
    )
    quality = write_quality_and_manifest(
        root,
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        cloud=filtered,
        occupancy=occupancy,
        mesh=mesh,
        trajectory_count=len(trajectory),
        write_observed_mesh=True,
        voxel_size_m=map_options.voxel_size_m,
        min_confidence=map_options.min_confidence,
        max_relative_disagreement=map_options.max_relative_disagreement,
        paths=paths,
        failure_reasons=[],
        rejected_pose_count=rejected_pose_count,
        map_dir_name="world_map_classical_validated",
        truth_boundary=CLASSICAL_VALIDATED_TRUTH_BOUNDARY,
    )
    _patch_validation_json(
        root / "world_map_classical_validated" / "world_map_manifest.json",
        selected_source,
        retained_ratio,
    )
    _patch_validation_json(
        root / "world_map_classical_validated" / "map_quality.json", selected_source, retained_ratio
    )
    artifact_paths = tuple(
        f"world_map_classical_validated/{name}"
        for name in (
            "world_map_manifest.json",
            "fused_points.npz",
            "fused_points.ply",
            "occupancy_grid.npz",
            "occupancy_grid_metadata.json",
            "observed_voxel_mesh.ply",
            "camera_trajectory.json",
            "map_quality.json",
            "map_quality.md",
        )
    )
    return ClassicalValidatedMapResult(
        status="available" if bool(quality["inspectable_map_available"]) else "unavailable",
        reason="classical bbox witness removed outlying map points without collapsing the map",
        accepted=bool(quality["inspectable_map_available"]),
        selected_source=selected_source,
        retained_point_ratio=retained_ratio,
        point_count=int(filtered.points_world_m.shape[0]),
        occupied_voxel_count=occupancy.occupied_voxel_count,
        mesh_triangle_count=mesh.triangle_count,
        cloud=filtered,
        artifact_paths=artifact_paths,
    )


def _bbox_validation_mask(
    map_points: NDArray[np.float32], classical_points: NDArray[np.float32], *, expansion_m: float
) -> NDArray[np.bool_]:
    classical_min, classical_max = bbox(classical_points)
    if classical_min is None or classical_max is None:
        return np.zeros((map_points.shape[0],), dtype=np.bool_)
    low = classical_min - expansion_m
    high = classical_max + expansion_m
    return cast(NDArray[np.bool_], np.all((map_points >= low) & (map_points <= high), axis=1))


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


def _read_points(path: Path) -> NDArray[np.float32]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["points_world_m"], dtype=np.float32)


def _patch_validation_json(path: Path, selected_source: str, retained_ratio: float) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "classical_validation_source": selected_source,
            "classical_validation_retained_point_ratio": retained_ratio,
            "physical_accuracy_claim": False,
            "training_quality_claim": False,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
