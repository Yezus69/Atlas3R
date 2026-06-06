"""Import external ViPE artifacts into Atlas3R map artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import transform_points, unproject_depth, validate_T_A_B
from atlas3r.offline.fused_world_map_artifacts import (
    FusedPointCloud,
    ObservedVoxelMesh,
    PointChunks,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    write_map_artifacts,
)
from atlas3r.offline.vipe_artifact_io import (
    DenseSlamMap,
    DepthFrame,
    IntrinsicsTable,
    PoseTable,
    VipeImportError,
    indexed_frame_paths,
    load_depth_frames,
    load_intrinsics_table,
    load_pose_table,
    load_rgb_for_frame,
    load_slam_dense_map,
)
from atlas3r.offline.vipe_comparison import write_vipe_comparison
from atlas3r.offline.vipe_quality import write_vipe_quality_and_manifest

VIPE_DEPTH_SOURCE_ID = 6
VIPE_DEPTH_SOURCE = "vipe_dense_depth"
VIPE_SLAM_MAP_SOURCE = "vipe_slam_dense_map_fallback"
VIPE_METRIC_SCALE_SOURCE = "vipe_near_metric_teacher"
VIPE_TRUTH_BOUNDARY: dict[str, object] = {
    "label_type": "teacher_pseudo_vipe_near_metric",
    "measured_geometry": False,
    "observed_only": True,
    "predicted_completion": False,
    "hidden_geometry_measured": False,
    "metric_scale_source": VIPE_METRIC_SCALE_SOURCE,
    "scale_status": "teacher_near_metric_unanchored",
    "physical_accuracy_claim": False,
    "training_quality": False,
    "realtime_claim": False,
    "optimized_world_state": False,
    "accuracy_report": False,
    "usable_for_training": False,
}


@dataclass(frozen=True)
class VipeImportOptions:
    vipe_output: Path
    frames: Path
    output: Path
    artifact_name: str = "frames"
    point_stride: int = 8
    max_points: int = 2_000_000
    voxel_size_m: float = 0.05
    write_observed_mesh: bool = True
    previous_map: Path | None = None


@dataclass(frozen=True)
class VipeImportResult:
    output_dir: Path
    status: str
    point_count: int
    occupied_voxel_count: int
    mesh_triangle_count: int
    trajectory_count: int
    valid_depth_ratio: float
    map_quality_path: Path
    comparison_path: Path | None
    failure_reasons: tuple[str, ...]


def import_vipe_world_map(options: VipeImportOptions) -> VipeImportResult:
    _validate_options(options)
    failures: list[str] = []
    vipe_output = Path(options.vipe_output)
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    poses = load_pose_table(vipe_output, options.artifact_name)
    intrinsics = load_intrinsics_table(vipe_output, options.artifact_name)
    depth_frames = load_depth_frames(vipe_output, options.artifact_name)
    frame_paths = indexed_frame_paths(Path(options.frames))

    cloud = _build_vipe_cloud(
        depth_frames,
        poses=poses,
        intrinsics=intrinsics,
        frame_paths=frame_paths,
        point_stride=options.point_stride,
        max_points=options.max_points,
        failures=failures,
    )
    if cloud.points_world_m.shape[0] == 0:
        try:
            cloud = _cap_cloud(
                _cloud_from_slam_map(load_slam_dense_map(vipe_output, options.artifact_name)),
                options.max_points,
            )
            failures.append(
                "ViPE pose/depth lift produced no finite points; used finite dense SLAM map "
                "fallback without camera trajectory"
            )
        except VipeImportError as exc:
            failures.append(f"ViPE dense SLAM map fallback unavailable: {exc}")
    occupancy = build_sparse_occupancy(
        cloud.points_world_m,
        cloud.colors_u8,
        cloud.confidence,
        voxel_size_m=options.voxel_size_m,
    )
    mesh = build_observed_voxel_mesh(occupancy) if options.write_observed_mesh else _empty_mesh()
    trajectory, rejected_pose_count = _camera_trajectory(poses)
    paths = write_map_artifacts(
        output.parent,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory=trajectory,
        write_observed_mesh=options.write_observed_mesh,
        rejected_pose_count=rejected_pose_count,
        map_dir_name=output.name,
        truth_boundary=VIPE_TRUTH_BOUNDARY,
    )
    quality = write_vipe_quality_and_manifest(
        output,
        frames=options.frames,
        vipe_output=options.vipe_output,
        artifact_name=options.artifact_name,
        point_stride=options.point_stride,
        max_points=options.max_points,
        voxel_size_m=options.voxel_size_m,
        write_observed_mesh=options.write_observed_mesh,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory_count=len(trajectory),
        rejected_pose_count=rejected_pose_count,
        depth_frame_count=len(depth_frames),
        frame_count=len(frame_paths),
        paths=paths,
        failures=failures,
        truth_boundary=VIPE_TRUTH_BOUNDARY,
    )
    comparison_path = write_vipe_comparison(
        output, quality, options.previous_map, truth_boundary=VIPE_TRUTH_BOUNDARY
    )
    return VipeImportResult(
        output_dir=output,
        status="available" if bool(quality["inspectable_map_available"]) else "unavailable",
        point_count=int(cloud.points_world_m.shape[0]),
        occupied_voxel_count=occupancy.occupied_voxel_count,
        mesh_triangle_count=mesh.triangle_count,
        trajectory_count=len(trajectory),
        valid_depth_ratio=cloud.valid_depth_ratio,
        map_quality_path=output / "map_quality.json",
        comparison_path=comparison_path,
        failure_reasons=tuple(failures),
    )


def _build_vipe_cloud(
    depth_frames: tuple[DepthFrame, ...],
    *,
    poses: PoseTable,
    intrinsics: IntrinsicsTable,
    frame_paths: dict[int, Path],
    point_stride: int,
    max_points: int,
    failures: list[str],
) -> FusedPointCloud:
    chunks = PointChunks()
    sampled_pixels = 0
    valid_depth_pixels = 0
    per_frame_counts: dict[str, int] = {}
    missing_pose_or_intrinsics = 0
    rejected_by_reason: dict[str, int] = {}
    for ordinal, depth_frame in enumerate(depth_frames):
        frame_id = depth_frame.frame_id
        depth = depth_frame.depth_m.astype(np.float32)
        T_world_camera = poses.transform_for(frame_id, ordinal)
        K = intrinsics.K_for(frame_id, ordinal)
        if T_world_camera is None or K is None:
            missing_pose_or_intrinsics += 1
            continue
        if depth.ndim != 2:
            rejected_by_reason["depth must be a 2D array"] = (
                rejected_by_reason.get("depth must be a 2D array", 0) + 1
            )
            continue
        depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
        try:
            T_world_camera = validate_T_A_B(T_world_camera, "T_world_camera")
            _validate_K(K)
        except ValueError as exc:
            reason = str(exc)
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
            continue
        sample_mask = _sample_mask(depth_shape, point_stride)
        valid = sample_mask & np.isfinite(depth) & (depth > 0.0)
        sampled_pixels += int(sample_mask.sum())
        valid_depth_pixels += int(valid.sum())
        if not np.any(valid):
            continue
        try:
            rgb = load_rgb_for_frame(frame_paths, frame_id)
        except VipeImportError as exc:
            failures.append(f"frame {frame_id} color fallback: {exc}")
            rgb = np.full((depth_shape[0], depth_shape[1], 3), 128, dtype=np.uint8)
        points = _lift_points(K, depth, T_world_camera, valid)
        if not points.size:
            continue
        flat_valid = valid.reshape((-1,))
        count = int(points.shape[0])
        colors = _colors_for_depth(rgb, depth_shape)[flat_valid]
        confidence = np.ones((count,), dtype=np.float32)
        sampled_depth = depth.reshape((-1,))[flat_valid].astype(np.float32)
        sigma = np.maximum(0.02, 0.05 * sampled_depth).astype(np.float32)
        chunks.append(
            points,
            colors,
            confidence,
            np.full(count, frame_id, dtype=np.int64),
            np.full(count, frame_id, dtype=np.int64),
            np.full(count, VIPE_DEPTH_SOURCE_ID, dtype=np.int32),
            np.full(count, np.nan, dtype=np.float32),
            sigma,
        )
        per_frame_counts[str(frame_id)] = per_frame_counts.get(str(frame_id), 0) + count
    if missing_pose_or_intrinsics:
        failures.append(
            f"skipped {missing_pose_or_intrinsics} ViPE frames: missing pose or intrinsics"
        )
    for reason, count in rejected_by_reason.items():
        failures.append(f"skipped {count} ViPE frames: {reason}")
    cloud = _cap_cloud(chunks.concatenate(VIPE_DEPTH_SOURCE, VIPE_METRIC_SCALE_SOURCE), max_points)
    return FusedPointCloud(
        points_world_m=cloud.points_world_m,
        colors_u8=cloud.colors_u8,
        confidence=cloud.confidence,
        source_frame_ids=cloud.source_frame_ids,
        source_keyframe_ids=cloud.source_keyframe_ids,
        depth_source_id=cloud.depth_source_id,
        disagreement_rel=cloud.disagreement_rel,
        point_sigma_m=cloud.point_sigma_m,
        depth_source=VIPE_DEPTH_SOURCE,
        metric_scale_source=VIPE_METRIC_SCALE_SOURCE,
        valid_depth_ratio=float(valid_depth_pixels / sampled_pixels) if sampled_pixels else 0.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=None,
        mapped_disagreement_p50=None,
        mapped_disagreement_p95=None,
        per_frame_point_counts=per_frame_counts,
    )


def _cloud_from_slam_map(slam_map: DenseSlamMap) -> FusedPointCloud:
    points = slam_map.points_world_m.astype(np.float32)
    count = int(points.shape[0])
    confidence = np.ones((count,), dtype=np.float32)
    sigma = np.maximum(0.02, 0.05 * np.linalg.norm(points, axis=1)).astype(np.float32)
    return FusedPointCloud(
        points_world_m=points,
        colors_u8=slam_map.colors_u8.astype(np.uint8),
        confidence=confidence,
        source_frame_ids=slam_map.source_frame_ids.astype(np.int64),
        source_keyframe_ids=slam_map.source_frame_ids.astype(np.int64),
        depth_source_id=np.full(count, VIPE_DEPTH_SOURCE_ID, dtype=np.int32),
        disagreement_rel=np.full(count, np.nan, dtype=np.float32),
        point_sigma_m=sigma,
        depth_source=VIPE_SLAM_MAP_SOURCE,
        metric_scale_source=VIPE_METRIC_SCALE_SOURCE,
        valid_depth_ratio=0.0,
        rejected_low_confidence_ratio=0.0,
        rejected_high_disagreement_ratio=0.0,
        mapped_disagreement_mean=None,
        mapped_disagreement_p50=None,
        mapped_disagreement_p95=None,
        per_frame_point_counts=_per_frame_counts(slam_map.source_frame_ids),
    )


def _per_frame_counts(source_frame_ids: NDArray[np.int64]) -> dict[str, int]:
    valid = source_frame_ids[source_frame_ids >= 0]
    if not valid.size:
        return {}
    unique, counts = np.unique(valid, return_counts=True)
    return {str(int(frame_id)): int(count) for frame_id, count in zip(unique, counts, strict=True)}


def _camera_trajectory(poses: PoseTable) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    rejected = 0
    for frame_id, transform in zip(poses.frame_ids, poses.transforms, strict=True):
        try:
            T_world_camera = validate_T_A_B(transform, "T_world_camera")
        except ValueError:
            rejected += 1
            continue
        rows.append(
            {
                "frame_id": int(frame_id),
                "keyframe_id": int(frame_id),
                "timestamp_ns": None,
                "T_world_camera": T_world_camera.astype(float).tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].astype(float).tolist(),
                "pose_source": "vipe",
                "pose_confidence": 1.0,
                "metric_scale_source": VIPE_METRIC_SCALE_SOURCE,
                "pseudo_submap_id": 0,
            }
        )
    return rows, rejected


def _validate_K(K: NDArray[np.float32]) -> None:
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("K must be finite 3x3")
    if float(K[0, 0]) <= 0.0 or float(K[1, 1]) <= 0.0:
        raise ValueError("K focal lengths must be positive")


def _lift_points(
    K: NDArray[np.float32],
    depth_m: NDArray[np.float32],
    T_world_camera: NDArray[np.float32],
    lift_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    masked_depth = np.where(lift_mask, depth_m, 0.0).astype(np.float32)
    points_camera = unproject_depth(K, masked_depth).reshape((-1, 3))
    points_world = transform_points(T_world_camera, points_camera)
    points = points_world[lift_mask.reshape((-1,))]
    if points.size and not np.all(np.isfinite(points)):
        return np.zeros((0, 3), dtype=np.float32)
    return cast(NDArray[np.float32], points.astype(np.float32))


def _colors_for_depth(rgb_u8: NDArray[np.uint8], depth_shape: tuple[int, int]) -> NDArray[np.uint8]:
    height, width = depth_shape
    src_h, src_w = rgb_u8.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(np.int32), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(np.int32), 0, src_w - 1)
    return cast(NDArray[np.uint8], rgb_u8[y_idx[:, None], x_idx[None, :]].reshape((-1, 3)))


def _sample_mask(shape: tuple[int, int], stride: int) -> NDArray[np.bool_]:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[::stride, ::stride] = True
    return mask


def _cap_cloud(cloud: FusedPointCloud, max_points: int) -> FusedPointCloud:
    count = int(cloud.points_world_m.shape[0])
    if count <= max_points:
        return cloud
    keep = np.linspace(0, count - 1, max_points).round().astype(np.int64)
    return FusedPointCloud(
        points_world_m=cloud.points_world_m[keep],
        colors_u8=cloud.colors_u8[keep],
        confidence=cloud.confidence[keep],
        source_frame_ids=cloud.source_frame_ids[keep],
        source_keyframe_ids=cloud.source_keyframe_ids[keep],
        depth_source_id=cloud.depth_source_id[keep],
        disagreement_rel=cloud.disagreement_rel[keep],
        point_sigma_m=cloud.point_sigma_m[keep],
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


def _empty_mesh() -> ObservedVoxelMesh:
    return ObservedVoxelMesh(
        vertices_world_m=np.zeros((0, 3), dtype=np.float32),
        colors_u8=np.zeros((0, 3), dtype=np.uint8),
        triangles=np.zeros((0, 3), dtype=np.uint32),
    )


def _validate_options(options: VipeImportOptions) -> None:
    if options.point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if options.max_points <= 0:
        raise ValueError("max_points must be positive")
    if options.voxel_size_m <= 0.0 or not np.isfinite(options.voxel_size_m):
        raise ValueError("voxel_size_m must be positive and finite")


__all__ = [
    "VIPE_TRUTH_BOUNDARY",
    "VipeImportError",
    "VipeImportOptions",
    "VipeImportResult",
    "import_vipe_world_map",
]
