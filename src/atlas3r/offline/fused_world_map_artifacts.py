"""Artifact helpers for fused teacher-pseudo world maps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import write_json

TRUTH_BOUNDARY: dict[str, object] = {
    "label_type": "teacher_pseudo_fused_map",
    "measured_geometry": False,
    "observed_only": True,
    "predicted_completion": False,
    "hidden_geometry_measured": False,
    "metric_scale_source": "unanchored_vggt_depthpro_teacher_consensus",
    "physical_accuracy_claim": False,
    "training_quality": False,
    "realtime_claim": False,
    "optimized_world_state": False,
    "accuracy_report": False,
    "usable_for_training": False,
}

OPTIMIZED_TRUTH_BOUNDARY: dict[str, object] = {
    "label_type": "teacher_pseudo_optimized_map",
    "measured_geometry": False,
    "observed_only": True,
    "predicted_completion": False,
    "hidden_geometry_measured": False,
    "metric_scale_source": "unanchored_vggt_depthpro_teacher_consensus",
    "physical_accuracy_claim": False,
    "training_quality": False,
    "realtime_claim": False,
    "optimized_world_state": "diagnostic_depth_consistency_only",
    "accuracy_report": False,
    "usable_for_training": False,
}


@dataclass(frozen=True)
class FusedPointCloud:
    points_world_m: NDArray[np.float32]
    colors_u8: NDArray[np.uint8]
    confidence: NDArray[np.float32]
    source_frame_ids: NDArray[np.int64]
    source_keyframe_ids: NDArray[np.int64]
    depth_source_id: NDArray[np.int32]
    disagreement_rel: NDArray[np.float32]
    point_sigma_m: NDArray[np.float32]
    depth_source: str
    metric_scale_source: str
    valid_depth_ratio: float
    rejected_low_confidence_ratio: float
    rejected_high_disagreement_ratio: float
    mapped_disagreement_mean: float | None
    mapped_disagreement_p50: float | None
    mapped_disagreement_p95: float | None
    per_frame_point_counts: dict[str, int]


@dataclass
class PointChunks:
    points: list[NDArray[np.float32]] = field(default_factory=list)
    colors: list[NDArray[np.uint8]] = field(default_factory=list)
    confidence: list[NDArray[np.float32]] = field(default_factory=list)
    frame_ids: list[NDArray[np.int64]] = field(default_factory=list)
    keyframe_ids: list[NDArray[np.int64]] = field(default_factory=list)
    source_ids: list[NDArray[np.int32]] = field(default_factory=list)
    disagreement: list[NDArray[np.float32]] = field(default_factory=list)
    sigma: list[NDArray[np.float32]] = field(default_factory=list)

    def append(
        self,
        points: NDArray[np.float32],
        colors: NDArray[np.uint8],
        confidence: NDArray[np.float32],
        frame_ids: NDArray[np.int64],
        keyframe_ids: NDArray[np.int64],
        source_ids: NDArray[np.int32],
        disagreement: NDArray[np.float32],
        sigma: NDArray[np.float32],
    ) -> None:
        self.points.append(points.astype(np.float32))
        self.colors.append(colors.astype(np.uint8))
        self.confidence.append(confidence.astype(np.float32))
        self.frame_ids.append(frame_ids)
        self.keyframe_ids.append(keyframe_ids)
        self.source_ids.append(source_ids)
        self.disagreement.append(disagreement.astype(np.float32))
        self.sigma.append(sigma.astype(np.float32))

    def concatenate(self, depth_source: str, metric_scale_source: str) -> FusedPointCloud:
        if not self.points:
            return empty_cloud(depth_source)
        return FusedPointCloud(
            points_world_m=np.concatenate(self.points, axis=0).astype(np.float32),
            colors_u8=np.concatenate(self.colors, axis=0).astype(np.uint8),
            confidence=np.concatenate(self.confidence, axis=0).astype(np.float32),
            source_frame_ids=np.concatenate(self.frame_ids, axis=0).astype(np.int64),
            source_keyframe_ids=np.concatenate(self.keyframe_ids, axis=0).astype(np.int64),
            depth_source_id=np.concatenate(self.source_ids, axis=0).astype(np.int32),
            disagreement_rel=np.concatenate(self.disagreement, axis=0).astype(np.float32),
            point_sigma_m=np.concatenate(self.sigma, axis=0).astype(np.float32),
            depth_source=depth_source,
            metric_scale_source=metric_scale_source,
            valid_depth_ratio=0.0,
            rejected_low_confidence_ratio=0.0,
            rejected_high_disagreement_ratio=0.0,
            mapped_disagreement_mean=None,
            mapped_disagreement_p50=None,
            mapped_disagreement_p95=None,
            per_frame_point_counts={},
        )


@dataclass(frozen=True)
class SparseOccupancyGrid:
    voxel_indices_ijk: NDArray[np.int32]
    occupancy_count: NDArray[np.int32]
    confidence_mean: NDArray[np.float32]
    color_mean_u8: NDArray[np.uint8]
    bbox_world_min_m: NDArray[np.float32]
    voxel_size_m: float

    @property
    def occupied_voxel_count(self) -> int:
        return int(self.voxel_indices_ijk.shape[0])


@dataclass(frozen=True)
class ObservedVoxelMesh:
    vertices_world_m: NDArray[np.float32]
    colors_u8: NDArray[np.uint8]
    triangles: NDArray[np.uint32]

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


def build_sparse_occupancy(
    points_world_m: NDArray[np.float32],
    colors_u8: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    *,
    voxel_size_m: float,
) -> SparseOccupancyGrid:
    if voxel_size_m <= 0.0 or not np.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be positive and finite")
    if points_world_m.shape[0] == 0:
        return SparseOccupancyGrid(
            voxel_indices_ijk=np.zeros((0, 3), dtype=np.int32),
            occupancy_count=np.zeros((0,), dtype=np.int32),
            confidence_mean=np.zeros((0,), dtype=np.float32),
            color_mean_u8=np.zeros((0, 3), dtype=np.uint8),
            bbox_world_min_m=np.zeros((3,), dtype=np.float32),
            voxel_size_m=voxel_size_m,
        )
    bbox_min = points_world_m.min(axis=0).astype(np.float32)
    voxel_indices = np.floor((points_world_m - bbox_min) / voxel_size_m).astype(np.int32)
    unique, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int32)
    conf_sum = np.bincount(inverse, weights=confidence.astype(np.float64)).astype(np.float32)
    color_sums = np.zeros((unique.shape[0], 3), dtype=np.float64)
    np.add.at(color_sums, inverse, colors_u8.astype(np.float64))
    count_float = np.maximum(counts.astype(np.float32), 1.0)
    colors = np.clip(np.round(color_sums / count_float[:, None]), 0, 255).astype(np.uint8)
    return SparseOccupancyGrid(
        voxel_indices_ijk=cast(NDArray[np.int32], unique.astype(np.int32)),
        occupancy_count=counts,
        confidence_mean=(conf_sum / count_float).astype(np.float32),
        color_mean_u8=colors,
        bbox_world_min_m=bbox_min,
        voxel_size_m=voxel_size_m,
    )


def build_observed_voxel_mesh(grid: SparseOccupancyGrid) -> ObservedVoxelMesh:
    if grid.occupied_voxel_count == 0:
        return empty_mesh()
    occupied = {tuple(int(v) for v in row) for row in grid.voxel_indices_ijk.tolist()}
    vertices: list[list[float]] = []
    colors: list[list[int]] = []
    triangles: list[list[int]] = []
    for voxel_index, ijk_array in enumerate(grid.voxel_indices_ijk):
        ijk = tuple(int(v) for v in ijk_array.tolist())
        base = grid.bbox_world_min_m + grid.voxel_size_m * ijk_array.astype(np.float32)
        for direction, corners in _face_definitions():
            neighbor = (ijk[0] + direction[0], ijk[1] + direction[1], ijk[2] + direction[2])
            if neighbor in occupied:
                continue
            start = len(vertices)
            for corner in corners:
                vertex = base + grid.voxel_size_m * np.asarray(corner, dtype=np.float32)
                vertices.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
                color = grid.color_mean_u8[voxel_index].tolist()
                colors.append([int(color[0]), int(color[1]), int(color[2])])
            triangles.append([start, start + 1, start + 2])
            triangles.append([start, start + 2, start + 3])
    return ObservedVoxelMesh(
        vertices_world_m=np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        colors_u8=np.asarray(colors, dtype=np.uint8).reshape((-1, 3)),
        triangles=np.asarray(triangles, dtype=np.uint32).reshape((-1, 3)),
    )


def empty_mesh() -> ObservedVoxelMesh:
    return ObservedVoxelMesh(
        vertices_world_m=np.zeros((0, 3), dtype=np.float32),
        colors_u8=np.zeros((0, 3), dtype=np.uint8),
        triangles=np.zeros((0, 3), dtype=np.uint32),
    )


def empty_cloud(depth_source: str) -> FusedPointCloud:
    return FusedPointCloud(
        points_world_m=np.zeros((0, 3), dtype=np.float32),
        colors_u8=np.zeros((0, 3), dtype=np.uint8),
        confidence=np.zeros((0,), dtype=np.float32),
        source_frame_ids=np.zeros((0,), dtype=np.int64),
        source_keyframe_ids=np.zeros((0,), dtype=np.int64),
        depth_source_id=np.zeros((0,), dtype=np.int32),
        disagreement_rel=np.zeros((0,), dtype=np.float32),
        point_sigma_m=np.zeros((0,), dtype=np.float32),
        depth_source=depth_source,
        metric_scale_source=str(TRUTH_BOUNDARY["metric_scale_source"]),
        valid_depth_ratio=0.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=None,
        mapped_disagreement_p50=None,
        mapped_disagreement_p95=None,
        per_frame_point_counts={},
    )


def write_map_artifacts(
    run_dir: Path,
    *,
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    mesh: ObservedVoxelMesh,
    trajectory: list[dict[str, object]],
    write_observed_mesh: bool,
    rejected_pose_count: int,
    map_dir_name: str = "world_map",
    truth_boundary: dict[str, object] | None = None,
) -> dict[str, str]:
    truth = TRUTH_BOUNDARY if truth_boundary is None else truth_boundary
    map_dir = run_dir / map_dir_name
    map_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    _write_fused_points_npz(map_dir / "fused_points.npz", cloud, truth)
    paths["fused_points_npz"] = f"{map_dir_name}/fused_points.npz"
    if cloud.points_world_m.shape[0] > 0:
        _write_point_ply(map_dir / "fused_points.ply", cloud.points_world_m, cloud.colors_u8, truth)
        paths["fused_points_ply"] = f"{map_dir_name}/fused_points.ply"
    _write_occupancy_npz(map_dir / "occupancy_grid.npz", occupancy, truth)
    write_json(
        map_dir / "occupancy_grid_metadata.json",
        occupancy_metadata(occupancy, cloud, truth),
    )
    paths["occupancy_grid_npz"] = f"{map_dir_name}/occupancy_grid.npz"
    paths["occupancy_grid_metadata"] = f"{map_dir_name}/occupancy_grid_metadata.json"
    if write_observed_mesh:
        _write_mesh_ply(map_dir / "observed_voxel_mesh.ply", mesh, truth)
        paths["observed_voxel_mesh"] = f"{map_dir_name}/observed_voxel_mesh.ply"
    write_json(
        map_dir / "camera_trajectory.json",
        {
            "format_name": "atlas3r_camera_trajectory",
            "format_version": 1,
            "trajectory_count": len(trajectory),
            "rejected_nonfinite_pose_count": rejected_pose_count,
            "metric_scale_source": truth["metric_scale_source"],
            "truth_boundary": truth,
            "poses": trajectory,
        },
    )
    paths["camera_trajectory"] = f"{map_dir_name}/camera_trajectory.json"
    return paths


def write_quality_and_manifest(
    run_dir: Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    mesh: ObservedVoxelMesh,
    trajectory_count: int,
    write_observed_mesh: bool,
    voxel_size_m: float,
    min_confidence: float,
    max_relative_disagreement: float,
    paths: dict[str, str],
    failure_reasons: list[str],
    rejected_pose_count: int,
    map_dir_name: str = "world_map",
    truth_boundary: dict[str, object] | None = None,
) -> dict[str, object]:
    truth = TRUTH_BOUNDARY if truth_boundary is None else truth_boundary
    map_dir = run_dir / map_dir_name
    quality = map_quality_payload(
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory_count=trajectory_count,
        write_observed_mesh=write_observed_mesh,
        paths=paths,
        failure_reasons=failure_reasons,
        rejected_pose_count=rejected_pose_count,
        truth_boundary=truth,
    )
    write_json(map_dir / "map_quality.json", quality)
    (map_dir / "map_quality.md").write_text(map_quality_markdown(quality), encoding="utf-8")
    manifest = manifest_payload(
        cloud,
        occupancy,
        voxel_size_m=voxel_size_m,
        min_confidence=min_confidence,
        max_relative_disagreement=max_relative_disagreement,
        paths=paths,
        failure_reasons=failure_reasons,
        truth_boundary=truth,
    )
    write_json(map_dir / "world_map_manifest.json", manifest)
    return quality


def manifest_payload(
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    *,
    voxel_size_m: float,
    min_confidence: float,
    max_relative_disagreement: float,
    paths: dict[str, str],
    failure_reasons: list[str],
    truth_boundary: dict[str, object] | None = None,
) -> dict[str, object]:
    truth = TRUTH_BOUNDARY if truth_boundary is None else truth_boundary
    bbox_min, bbox_max = bbox(cloud.points_world_m)
    return {
        "format_name": "atlas3r_fused_world_map",
        "format_version": 1,
        "point_count": int(cloud.points_world_m.shape[0]),
        "source_keyframe_count": int(np.unique(cloud.source_keyframe_ids).shape[0])
        if cloud.source_keyframe_ids.size
        else 0,
        "depth_source": cloud.depth_source,
        "voxel_size_m": voxel_size_m,
        "min_confidence": min_confidence,
        "max_relative_disagreement": max_relative_disagreement,
        "bbox_world_min_m": None if bbox_min is None else bbox_min.tolist(),
        "bbox_world_max_m": None if bbox_max is None else bbox_max.tolist(),
        "occupied_voxel_count": occupancy.occupied_voxel_count,
        "metric_scale_source": truth["metric_scale_source"],
        "truth_boundary": truth,
        "failure_points": failure_reasons,
        "artifacts": paths,
    }


def map_quality_payload(
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    cloud: FusedPointCloud,
    occupancy: SparseOccupancyGrid,
    mesh: ObservedVoxelMesh,
    trajectory_count: int,
    write_observed_mesh: bool,
    paths: dict[str, str],
    failure_reasons: list[str],
    rejected_pose_count: int,
    truth_boundary: dict[str, object] | None = None,
) -> dict[str, object]:
    truth = TRUTH_BOUNDARY if truth_boundary is None else truth_boundary
    bbox_min, bbox_max = bbox(cloud.points_world_m)
    bbox_size = None if bbox_min is None or bbox_max is None else (bbox_max - bbox_min).tolist()
    inspectable = (
        "fused_points_ply" in paths
        and "occupancy_grid_npz" in paths
        and (not write_observed_mesh or "observed_voxel_mesh" in paths)
        and cloud.points_world_m.shape[0] > 0
        and occupancy.occupied_voxel_count > 0
        and (not write_observed_mesh or mesh.triangle_count > 0)
    )
    return {
        "format_name": "atlas3r_fused_map_quality_report",
        "format_version": 1,
        "input_source": input_path,
        "frames_decoded": len(frame_cache.records),
        "keyframes_selected": len(keyframes),
        "vggt_proposal_counts": {
            "cameras": len(proposal_cache.vggt_camera_records),
            "depths": len(proposal_cache.vggt_depth_records),
        },
        "depth_proposal_counts": {
            "cameras": len(proposal_cache.depth_pro_camera_records),
            "depths": len(proposal_cache.depth_pro_depth_records),
        },
        "depth_source_used": cloud.depth_source,
        "fused_point_count": int(cloud.points_world_m.shape[0]),
        "occupied_voxel_count": occupancy.occupied_voxel_count,
        "observed_mesh_vertex_count": int(mesh.vertices_world_m.shape[0]),
        "observed_mesh_triangle_count": mesh.triangle_count,
        "camera_trajectory_count": trajectory_count,
        "rejected_nonfinite_pose_count": rejected_pose_count,
        "bbox_size_m": bbox_size,
        "valid_depth_ratio": cloud.valid_depth_ratio,
        "rejected_low_confidence_pixel_ratio": cloud.rejected_low_confidence_ratio,
        "rejected_high_disagreement_pixel_ratio": cloud.rejected_high_disagreement_ratio,
        "mapped_disagreement": {
            "mean": cloud.mapped_disagreement_mean,
            "p50": cloud.mapped_disagreement_p50,
            "p95": cloud.mapped_disagreement_p95,
        },
        "scale_source": truth["metric_scale_source"],
        "physical_accuracy_claim": False,
        "training_quality_claim": False,
        "known_failure_points": failure_reasons,
        "inspectable_map_available": inspectable,
        "truth_boundary": truth,
        "artifacts": paths,
    }


def map_quality_markdown(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Fused World Map Quality Report",
            "",
            f"- Inspectable map available: {payload['inspectable_map_available']}",
            f"- Depth source used: {payload['depth_source_used']}",
            f"- Frames decoded: {payload['frames_decoded']}",
            f"- Keyframes selected: {payload['keyframes_selected']}",
            f"- Fused points: {payload['fused_point_count']}",
            f"- Occupied voxels: {payload['occupied_voxel_count']}",
            f"- Observed mesh vertices: {payload['observed_mesh_vertex_count']}",
            f"- Observed mesh triangles: {payload['observed_mesh_triangle_count']}",
            f"- Camera trajectory poses: {payload['camera_trajectory_count']}",
            f"- Valid depth ratio: {payload['valid_depth_ratio']}",
            f"- Rejected low-confidence ratio: {payload['rejected_low_confidence_pixel_ratio']}",
            "- Rejected high-disagreement ratio: "
            f"{payload['rejected_high_disagreement_pixel_ratio']}",
            f"- Physical accuracy claim: {payload['physical_accuracy_claim']}",
            f"- Training-quality claim: {payload['training_quality_claim']}",
            "",
            "This map is teacher-pseudo, observed-only, and not an accuracy or "
            "training-quality report.",
            "",
        ]
    )


def occupancy_metadata(
    occupancy: SparseOccupancyGrid,
    cloud: FusedPointCloud | None,
    truth_boundary: dict[str, object] | None = None,
) -> dict[str, object]:
    truth = TRUTH_BOUNDARY if truth_boundary is None else truth_boundary
    return {
        "format_name": "atlas3r_sparse_occupancy_grid",
        "format_version": 1,
        "occupied_voxel_count": occupancy.occupied_voxel_count,
        "voxel_size_m": occupancy.voxel_size_m,
        "bbox_world_min_m": occupancy.bbox_world_min_m.tolist(),
        "unknown_space_filled": False,
        "free_space_carving": False,
        "source_point_count": 0 if cloud is None else int(cloud.points_world_m.shape[0]),
        "truth_boundary": truth,
    }


def bbox(
    points: NDArray[np.float32],
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
    if points.shape[0] == 0:
        return None, None
    return points.min(axis=0).astype(np.float32), points.max(axis=0).astype(np.float32)


def _write_fused_points_npz(
    path: Path, cloud: FusedPointCloud, truth_boundary: dict[str, object]
) -> None:
    np.savez_compressed(
        path,
        points_world_m=cloud.points_world_m.astype(np.float32),
        colors_u8=cloud.colors_u8.astype(np.uint8),
        confidence=cloud.confidence.astype(np.float32),
        source_frame_ids=cloud.source_frame_ids.astype(np.int64),
        source_keyframe_ids=cloud.source_keyframe_ids.astype(np.int64),
        depth_source_id=cloud.depth_source_id.astype(np.int32),
        disagreement_rel=cloud.disagreement_rel.astype(np.float32),
        point_sigma_m=cloud.point_sigma_m.astype(np.float32),
        metadata_json=json.dumps(
            {
                "format_name": "atlas3r_fused_points",
                "format_version": 1,
                "point_count": int(cloud.points_world_m.shape[0]),
                "depth_source": cloud.depth_source,
                "metric_scale_source": cloud.metric_scale_source,
                "truth_boundary": truth_boundary,
            },
            sort_keys=True,
        ),
    )


def _write_occupancy_npz(
    path: Path, occupancy: SparseOccupancyGrid, truth_boundary: dict[str, object]
) -> None:
    np.savez_compressed(
        path,
        voxel_indices_ijk=occupancy.voxel_indices_ijk.astype(np.int32),
        occupancy_count=occupancy.occupancy_count.astype(np.int32),
        confidence_mean=occupancy.confidence_mean.astype(np.float32),
        color_mean_u8=occupancy.color_mean_u8.astype(np.uint8),
        bbox_world_min_m=occupancy.bbox_world_min_m.astype(np.float32),
        voxel_size_m=np.asarray([occupancy.voxel_size_m], dtype=np.float32),
        metadata_json=json.dumps(
            occupancy_metadata(occupancy, None, truth_boundary), sort_keys=True
        ),
    )


def _write_point_ply(
    path: Path,
    points_world_m: NDArray[np.float32],
    colors_u8: NDArray[np.uint8],
    truth_boundary: dict[str, object],
) -> None:
    metadata = json.dumps({"truth_boundary": truth_boundary}, sort_keys=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment atlas3r_metadata_json {metadata}",
        f"element vertex {points_world_m.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 0",
        "property list uchar uint vertex_indices",
        "end_header",
    ]
    for point, color in zip(points_world_m, colors_u8, strict=True):
        lines.append(
            f"{float(point[0]):.8g} {float(point[1]):.8g} {float(point[2]):.8g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mesh_ply(path: Path, mesh: ObservedVoxelMesh, truth_boundary: dict[str, object]) -> None:
    metadata = json.dumps(
        {"mesh_kind": "observed_voxel_mesh", "truth_boundary": truth_boundary},
        sort_keys=True,
    )
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment atlas3r_metadata_json {metadata}",
        f"element vertex {mesh.vertices_world_m.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {mesh.triangles.shape[0]}",
        "property list uchar uint vertex_indices",
        "end_header",
    ]
    for point, color in zip(mesh.vertices_world_m, mesh.colors_u8, strict=True):
        lines.append(
            f"{float(point[0]):.8g} {float(point[1]):.8g} {float(point[2]):.8g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}"
        )
    lines.extend(f"3 {int(a)} {int(b)} {int(c)}" for a, b, c in mesh.triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _face_definitions() -> tuple[
    tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...
]:
    return (
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    )
