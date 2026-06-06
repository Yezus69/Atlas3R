"""Fused teacher-pseudo world-map export for offline world builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import transform_points, unproject_depth, validate_T_A_B
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.fused_world_map_artifacts import (
    TRUTH_BOUNDARY,
    FusedPointCloud,
    ObservedVoxelMesh,
    PointChunks,
    SparseOccupancyGrid,
    bbox,
    build_observed_voxel_mesh,
    build_sparse_occupancy,
    empty_cloud,
    empty_mesh,
    write_map_artifacts,
    write_quality_and_manifest,
)
from atlas3r.offline.keyframes import KeyframeRecord
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint

MapDepthSource = Literal["consensus", "vggt", "depth_pro_with_vggt_pose"]
_DEPTH_SOURCE_IDS = {
    "consensus": 1,
    "vggt": 2,
    "depth_pro_with_vggt_pose": 3,
    "debug_flat_depth": 4,
}


@dataclass(frozen=True)
class FusedWorldMapOptions:
    export_world_map: bool = False
    point_stride: int = 8
    max_points: int = 2_000_000
    min_confidence: float = 0.25
    max_relative_disagreement: float = 0.25
    voxel_size_m: float = 0.05
    depth_source: MapDepthSource | None = None
    write_observed_mesh: bool = False
    write_occupancy: bool = False


@dataclass(frozen=True)
class FusedWorldMapResult:
    status: str
    manifest_path: str | None = None
    camera_trajectory_path: str | None = None
    fused_points_npz_path: str | None = None
    fused_points_ply_path: str | None = None
    occupancy_grid_npz_path: str | None = None
    occupancy_grid_metadata_path: str | None = None
    observed_voxel_mesh_ply_path: str | None = None
    map_quality_json_path: str | None = None
    map_quality_markdown_path: str | None = None
    depth_source: str = "none"
    point_count: int = 0
    occupied_voxel_count: int = 0
    mesh_vertex_count: int = 0
    mesh_triangle_count: int = 0
    trajectory_count: int = 0
    inspectable_map_available: bool = False
    valid_depth_ratio: float = 0.0
    rejected_low_confidence_ratio: float = 0.0
    rejected_high_disagreement_ratio: float = 0.0
    bbox_world_min_m: tuple[float, float, float] | None = None
    bbox_world_max_m: tuple[float, float, float] | None = None
    failure_reasons: tuple[str, ...] = ()

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        paths = (
            self.manifest_path,
            self.camera_trajectory_path,
            self.fused_points_npz_path,
            self.fused_points_ply_path,
            self.occupancy_grid_npz_path,
            self.occupancy_grid_metadata_path,
            self.observed_voxel_mesh_ply_path,
            self.map_quality_json_path,
            self.map_quality_markdown_path,
        )
        return tuple(path for path in paths if path is not None)


def write_fused_world_map(
    run_dir: str | Path,
    *,
    input_path: str,
    frame_cache: FrameCacheResult,
    keyframes: tuple[KeyframeRecord, ...],
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    options: FusedWorldMapOptions,
    failure_points: list[FailurePoint],
) -> FusedWorldMapResult:
    if not options.export_world_map:
        return FusedWorldMapResult(status="disabled")
    _validate_options(options)
    root = Path(run_dir)
    (root / "world_map").mkdir(parents=True, exist_ok=True)
    failure_reasons: list[str] = []
    trajectory, rejected_pose_count = _camera_trajectory(frame_cache, proposal_cache)
    cloud = build_fused_point_cloud(
        root,
        frame_cache=frame_cache,
        proposal_cache=proposal_cache,
        disagreement=disagreement,
        options=options,
        failure_reasons=failure_reasons,
    )
    if cloud.points_world_m.shape[0] == 0:
        _append_map_failure(failure_points, failure_reasons)
    occupancy = build_sparse_occupancy(
        cloud.points_world_m,
        cloud.colors_u8,
        cloud.confidence,
        voxel_size_m=options.voxel_size_m,
    )
    mesh = build_observed_voxel_mesh(occupancy) if options.write_observed_mesh else empty_mesh()
    paths = write_map_artifacts(
        root,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory=trajectory,
        write_observed_mesh=options.write_observed_mesh,
        rejected_pose_count=rejected_pose_count,
    )
    quality = write_quality_and_manifest(
        root,
        input_path=input_path,
        frame_cache=frame_cache,
        keyframes=keyframes,
        proposal_cache=proposal_cache,
        cloud=cloud,
        occupancy=occupancy,
        mesh=mesh,
        trajectory_count=len(trajectory),
        write_observed_mesh=options.write_observed_mesh,
        voxel_size_m=options.voxel_size_m,
        min_confidence=options.min_confidence,
        max_relative_disagreement=options.max_relative_disagreement,
        paths=paths,
        failure_reasons=failure_reasons,
        rejected_pose_count=rejected_pose_count,
    )
    bbox_min, bbox_max = bbox(cloud.points_world_m)
    return FusedWorldMapResult(
        status="available" if cloud.points_world_m.shape[0] > 0 else "unavailable",
        manifest_path="world_map/world_map_manifest.json",
        camera_trajectory_path="world_map/camera_trajectory.json",
        fused_points_npz_path="world_map/fused_points.npz",
        fused_points_ply_path=paths.get("fused_points_ply"),
        occupancy_grid_npz_path="world_map/occupancy_grid.npz",
        occupancy_grid_metadata_path="world_map/occupancy_grid_metadata.json",
        observed_voxel_mesh_ply_path=paths.get("observed_voxel_mesh"),
        map_quality_json_path="world_map/map_quality.json",
        map_quality_markdown_path="world_map/map_quality.md",
        depth_source=cloud.depth_source,
        point_count=int(cloud.points_world_m.shape[0]),
        occupied_voxel_count=occupancy.occupied_voxel_count,
        mesh_vertex_count=int(mesh.vertices_world_m.shape[0]),
        mesh_triangle_count=mesh.triangle_count,
        trajectory_count=len(trajectory),
        inspectable_map_available=bool(quality["inspectable_map_available"]),
        valid_depth_ratio=cloud.valid_depth_ratio,
        rejected_low_confidence_ratio=cloud.rejected_low_confidence_ratio,
        rejected_high_disagreement_ratio=cloud.rejected_high_disagreement_ratio,
        bbox_world_min_m=_bbox_tuple(bbox_min),
        bbox_world_max_m=_bbox_tuple(bbox_max),
        failure_reasons=tuple(failure_reasons),
    )


def build_fused_point_cloud(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    options: FusedWorldMapOptions,
    failure_reasons: list[str] | None = None,
) -> FusedPointCloud:
    failures = failure_reasons if failure_reasons is not None else []
    selected = _select_depth_records(proposal_cache, disagreement, options, failures)
    if selected is None:
        return empty_cloud("none")
    depth_source, depth_records, depth_arrays = selected
    frames_by_id = {frame.frame_id: frame for frame in frame_cache.frames}
    rel_maps = _load_relative_disagreement_maps(Path(run_dir), disagreement)
    vggt_cameras_by_depth = {
        str(record.get("depth_key")): record for record in proposal_cache.vggt_camera_records
    }
    vggt_cameras_by_frame = {
        int(str(record["frame_id"])): record for record in proposal_cache.vggt_camera_records
    }
    depth_pro_cameras_by_frame = {
        int(str(record["frame_id"])): record for record in proposal_cache.depth_pro_camera_records
    }
    chunks = PointChunks()
    valid_depth_pixels = 0
    sampled_pixels = 0
    low_conf_rejected = 0
    high_disagreement_rejected = 0
    mapped_disagreements: list[NDArray[np.float32]] = []
    metric_scale_source = str(TRUTH_BOUNDARY["metric_scale_source"])
    per_frame_point_counts: dict[str, int] = {}
    for record in depth_records:
        frame_id = int(str(record["frame_id"]))
        frame = frames_by_id.get(frame_id)
        if frame is None:
            continue
        try:
            arrays = _map_proposal_arrays(
                record,
                depth_source=depth_source,
                depth_arrays=depth_arrays,
                vggt_cameras_by_depth=vggt_cameras_by_depth,
                vggt_cameras_by_frame=vggt_cameras_by_frame,
                depth_pro_cameras_by_frame=depth_pro_cameras_by_frame,
            )
        except (KeyError, ValueError) as exc:
            failures.append(f"rejected frame {frame_id} map proposal: {exc}")
            continue
        depth_shape = _array_shape(arrays.depth_m)
        sample_mask = _sample_mask(depth_shape, options.point_stride)
        valid = sample_mask & arrays.valid_mask & np.isfinite(arrays.depth_m) & (arrays.depth_m > 0)
        sampled_pixels += int(sample_mask.sum())
        valid_depth_pixels += int(valid.sum())
        confidence_ok = arrays.confidence >= options.min_confidence
        low_conf_rejected += int((valid & ~confidence_ok).sum())
        rel = _relative_disagreement_for_frame(rel_maps, frame_id, depth_shape)
        disagreement_ok = np.ones(depth_shape, dtype=np.bool_)
        if rel is not None:
            finite_rel = np.isfinite(rel)
            disagreement_ok = ~finite_rel | (rel <= options.max_relative_disagreement)
            high_disagreement_rejected += int((valid & ~disagreement_ok).sum())
        lift_mask = valid & confidence_ok & disagreement_ok
        if not np.any(lift_mask):
            continue
        points = _lift_masked_points(arrays.K, arrays.depth_m, arrays.T_world_camera, lift_mask)
        if not points.size:
            continue
        flat_mask = lift_mask.reshape((-1,))
        colors = _colors_for_depth(frame.rgb_u8, depth_shape)[flat_mask]
        confidence = arrays.confidence.reshape((-1,))[flat_mask].astype(np.float32)
        sigma = arrays.sigma_m.reshape((-1,))[flat_mask].astype(np.float32)
        rel_flat = (
            np.full(confidence.shape, np.nan, dtype=np.float32)
            if rel is None
            else rel.reshape((-1,))[flat_mask].astype(np.float32)
        )
        if rel_flat.size and np.any(np.isfinite(rel_flat)):
            mapped_disagreements.append(rel_flat[np.isfinite(rel_flat)])
        count = int(points.shape[0])
        chunks.append(
            points,
            colors,
            confidence,
            np.full(count, frame_id, dtype=np.int64),
            np.full(count, int(str(record["keyframe_index"])), dtype=np.int64),
            np.full(count, _DEPTH_SOURCE_IDS[depth_source], dtype=np.int32),
            rel_flat,
            sigma,
        )
        metric_scale_source = str(record.get("metric_scale_source", metric_scale_source))
        per_frame_point_counts[str(frame_id)] = per_frame_point_counts.get(str(frame_id), 0) + count
    cloud = _cap_cloud(chunks.concatenate(depth_source, metric_scale_source), options.max_points)
    return FusedPointCloud(
        points_world_m=cloud.points_world_m,
        colors_u8=cloud.colors_u8,
        confidence=cloud.confidence,
        source_frame_ids=cloud.source_frame_ids,
        source_keyframe_ids=cloud.source_keyframe_ids,
        depth_source_id=cloud.depth_source_id,
        disagreement_rel=cloud.disagreement_rel,
        point_sigma_m=cloud.point_sigma_m,
        depth_source=cloud.depth_source,
        metric_scale_source=cloud.metric_scale_source,
        valid_depth_ratio=float(valid_depth_pixels / sampled_pixels) if sampled_pixels else 0.0,
        rejected_low_confidence_ratio=float(low_conf_rejected / valid_depth_pixels)
        if valid_depth_pixels
        else 0.0,
        rejected_high_disagreement_ratio=float(high_disagreement_rejected / valid_depth_pixels)
        if valid_depth_pixels
        else 0.0,
        mapped_disagreement_mean=_mean_or_none(mapped_disagreements),
        mapped_disagreement_p50=_percentile_or_none(mapped_disagreements, 50),
        mapped_disagreement_p95=_percentile_or_none(mapped_disagreements, 95),
        per_frame_point_counts=per_frame_point_counts,
    )


@dataclass(frozen=True)
class _MapProposalArrays:
    depth_m: NDArray[np.float32]
    sigma_m: NDArray[np.float32]
    confidence: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    K: NDArray[np.float32]
    T_world_camera: NDArray[np.float32]


def _select_depth_records(
    proposal_cache: ProposalCacheResult,
    disagreement: DisagreementResult | None,
    options: FusedWorldMapOptions,
    failures: list[str],
) -> tuple[str, tuple[dict[str, object], ...], dict[str, NDArray[np.float32]]] | None:
    has_vggt_pose = bool(proposal_cache.vggt_camera_records)
    has_vggt_depth = bool(proposal_cache.vggt_depth_records)
    has_depth_pro = bool(proposal_cache.depth_pro_depth_records)
    has_consensus = bool(disagreement is not None and disagreement.consensus_depth_records)
    if proposal_cache.debug_depth_records and not has_vggt_pose and options.depth_source is None:
        return "debug_flat_depth", proposal_cache.debug_depth_records, proposal_cache.depth_arrays
    requested = options.depth_source
    if requested == "vggt":
        if has_vggt_pose and has_vggt_depth:
            return "vggt", proposal_cache.vggt_depth_records, proposal_cache.depth_arrays
        failures.append("requested VGGT map source but VGGT pose/depth proposals are unavailable")
        return None
    if requested == "depth_pro_with_vggt_pose":
        if has_vggt_pose and has_depth_pro:
            return (
                "depth_pro_with_vggt_pose",
                proposal_cache.depth_pro_depth_records,
                proposal_cache.depth_arrays,
            )
        failures.append("Depth Pro map source requires VGGT global pose proposals")
        return None
    if requested == "consensus" and has_consensus and has_vggt_pose and disagreement is not None:
        return (
            "consensus",
            disagreement.consensus_depth_records,
            disagreement.consensus_depth_arrays,
        )
    if requested == "consensus" and not has_consensus:
        failures.append("consensus depth unavailable; falling back to VGGT depth when possible")
    if requested is None and has_consensus and has_vggt_pose and disagreement is not None:
        return (
            "consensus",
            disagreement.consensus_depth_records,
            disagreement.consensus_depth_arrays,
        )
    if has_vggt_pose and has_vggt_depth:
        return "vggt", proposal_cache.vggt_depth_records, proposal_cache.depth_arrays
    if has_depth_pro and not has_vggt_pose:
        failures.append("Depth Pro proposals do not include global pose; no global map was created")
    else:
        failures.append("no usable depth plus T_world_camera proposal exists for map export")
    return None


def _map_proposal_arrays(
    record: dict[str, object],
    *,
    depth_source: str,
    depth_arrays: dict[str, NDArray[np.float32]],
    vggt_cameras_by_depth: dict[str, dict[str, object]],
    vggt_cameras_by_frame: dict[int, dict[str, object]],
    depth_pro_cameras_by_frame: dict[int, dict[str, object]],
) -> _MapProposalArrays:
    if depth_source == "debug_flat_depth":
        camera = _require_dict(record["camera"])
        height = int(str(camera["height"]))
        width = int(str(camera["width"]))
        depth = np.full((height, width), float(str(record["constant_depth_m"])), np.float32)
        return _MapProposalArrays(
            depth_m=depth,
            sigma_m=np.full(depth.shape, float(str(record["depth_sigma_m"])), np.float32),
            confidence=np.full(depth.shape, float(str(record["confidence"])), np.float32),
            valid_mask=np.ones(depth.shape, dtype=np.bool_),
            K=np.asarray(record["K"], dtype=np.float32),
            T_world_camera=validate_T_A_B(record["T_world_camera"], "T_world_camera"),
        )
    frame_id = int(str(record["frame_id"]))
    camera = (
        vggt_cameras_by_frame[frame_id]
        if depth_source in {"consensus", "depth_pro_with_vggt_pose"}
        else vggt_cameras_by_depth[str(record["depth_key"])]
    )
    depth = depth_arrays[str(record["depth_key"])].astype(np.float32)
    sigma = depth_arrays[str(record["depth_sigma_key"])].astype(np.float32)
    confidence = depth_arrays[str(record["confidence_key"])].astype(np.float32)
    valid = depth_arrays[str(record["valid_mask_key"])] > 0.0
    K = _intrinsics_for_source(
        frame_id,
        depth_source=depth_source,
        vggt_camera=camera,
        depth_pro_cameras_by_frame=depth_pro_cameras_by_frame,
        target_shape=_array_shape(depth),
    )
    T_world_camera = validate_T_A_B(camera["T_world_camera"], "T_world_camera")
    if depth.shape != sigma.shape or depth.shape != confidence.shape or depth.shape != valid.shape:
        raise ValueError("depth, sigma, confidence, and valid mask shapes must match")
    return _MapProposalArrays(depth, sigma, confidence, valid, K, T_world_camera)


def _intrinsics_for_source(
    frame_id: int,
    *,
    depth_source: str,
    vggt_camera: dict[str, object],
    depth_pro_cameras_by_frame: dict[int, dict[str, object]],
    target_shape: tuple[int, int],
) -> NDArray[np.float32]:
    if depth_source == "depth_pro_with_vggt_pose":
        camera = depth_pro_cameras_by_frame.get(frame_id)
        if camera is not None and camera.get("K") is not None:
            K = np.asarray(camera["K"], dtype=np.float32)
            if K.shape == (3, 3) and np.all(np.isfinite(K)):
                return K
    K = np.asarray(vggt_camera["K"], dtype=np.float32)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("K must be finite 3x3")
    return _scale_K_for_shape(K, _shape_from_record(vggt_camera), target_shape)


def _scale_K_for_shape(
    K: NDArray[np.float32], source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> NDArray[np.float32]:
    if source_shape == target_shape:
        return K.astype(np.float32)
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    scaled = K.astype(np.float32).copy()
    scaled[0, :] *= float(dst_w) / max(1.0, float(src_w))
    scaled[1, :] *= float(dst_h) / max(1.0, float(src_h))
    scaled[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return scaled


def _shape_from_record(record: dict[str, object]) -> tuple[int, int]:
    shape = record.get("depth_shape")
    if isinstance(shape, list) and len(shape) == 2:
        return int(str(shape[0])), int(str(shape[1]))
    return 1, 1


def _lift_masked_points(
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


def _load_relative_disagreement_maps(
    run_dir: Path, disagreement: DisagreementResult | None
) -> dict[int, NDArray[np.float32]]:
    if disagreement is None or disagreement.maps_npz_path is None:
        return {}
    path = run_dir / disagreement.maps_npz_path
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        frame_ids = payload["frame_ids"].astype(np.int32)
        rel = payload["rel_depth_diff"].astype(np.float32)
    return {int(frame_id): rel[index] for index, frame_id in enumerate(frame_ids.tolist())}


def _relative_disagreement_for_frame(
    rel_maps: dict[int, NDArray[np.float32]], frame_id: int, shape: tuple[int, int]
) -> NDArray[np.float32] | None:
    rel = rel_maps.get(frame_id)
    if rel is None:
        return None
    return rel if rel.shape == shape else _resize_nearest(rel, shape)


def _camera_trajectory(
    frame_cache: FrameCacheResult, proposal_cache: ProposalCacheResult
) -> tuple[list[dict[str, object]], int]:
    frame_records = {record.frame_id: record for record in frame_cache.records}
    rows: list[dict[str, object]] = []
    rejected = 0
    records = (
        proposal_cache.vggt_camera_records
        if proposal_cache.vggt_camera_records
        else proposal_cache.debug_depth_records
    )
    for record in records:
        frame_id = int(str(record["frame_id"]))
        try:
            T_world_camera = validate_T_A_B(record["T_world_camera"], "T_world_camera")
        except (KeyError, ValueError):
            rejected += 1
            continue
        frame = frame_records.get(frame_id)
        rows.append(
            {
                "frame_id": frame_id,
                "keyframe_id": int(str(record["keyframe_index"])),
                "timestamp_ns": None if frame is None else frame.timestamp_ns,
                "T_world_camera": T_world_camera.astype(float).tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].astype(float).tolist(),
                "pose_source": str(record.get("pose_source", record.get("teacher_name", "debug"))),
                "pose_confidence": _float_value(record.get("pose_confidence", 1.0)),
                "metric_scale_source": str(record.get("metric_scale_source", "unknown")),
                "pseudo_submap_id": int(str(record.get("pseudo_submap_id", 0))),
            }
        )
    return rows, rejected


def _append_map_failure(failure_points: list[FailurePoint], failure_reasons: list[str]) -> None:
    why = "; ".join(failure_reasons) if failure_reasons else "fused world map is empty"
    failure_points.append(
        FailurePoint(
            module="fused_world_map",
            code="fused_map_empty",
            severity="warning",
            status="unavailable",
            why=why,
            input_missing="VGGT pose plus usable depth proposals",
            future_module="provide VGGT pose/depth or debug flat-depth mode",
            artifact_path="world_map/world_map_manifest.json",
        )
    )


def _sample_mask(shape: tuple[int, int], stride: int) -> NDArray[np.bool_]:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[::stride, ::stride] = True
    return mask


def _colors_for_depth(rgb_u8: NDArray[np.uint8], depth_shape: tuple[int, int]) -> NDArray[np.uint8]:
    height, width = depth_shape
    src_h, src_w = rgb_u8.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(np.int32), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(np.int32), 0, src_w - 1)
    return cast(NDArray[np.uint8], rgb_u8[y_idx[:, None], x_idx[None, :]].reshape((-1, 3)))


def _resize_nearest(array: NDArray[np.float32], shape: tuple[int, int]) -> NDArray[np.float32]:
    y_idx = np.clip(
        np.round(np.linspace(0, array.shape[0] - 1, shape[0])).astype(np.int32),
        0,
        array.shape[0] - 1,
    )
    x_idx = np.clip(
        np.round(np.linspace(0, array.shape[1] - 1, shape[1])).astype(np.int32),
        0,
        array.shape[1] - 1,
    )
    return cast(NDArray[np.float32], array[y_idx[:, None], x_idx[None, :]].astype(np.float32))


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


def _mean_or_none(chunks: list[NDArray[np.float32]]) -> float | None:
    values = _concat_values(chunks)
    return None if values.size == 0 else float(values.mean())


def _percentile_or_none(chunks: list[NDArray[np.float32]], percentile: float) -> float | None:
    values = _concat_values(chunks)
    return None if values.size == 0 else float(np.percentile(values, percentile))


def _concat_values(chunks: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0,), np.float32)


def _require_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))


def _array_shape(array: NDArray[np.float32]) -> tuple[int, int]:
    return int(array.shape[0]), int(array.shape[1])


def _bbox_tuple(value: NDArray[np.float32] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _validate_options(options: FusedWorldMapOptions) -> None:
    if options.point_stride <= 0:
        raise ValueError("map point stride must be positive")
    if options.max_points <= 0:
        raise ValueError("map max points must be positive")
    if not np.isfinite(options.min_confidence):
        raise ValueError("map min confidence must be finite")
    if not np.isfinite(options.max_relative_disagreement):
        raise ValueError("map max relative disagreement must be finite")
    if options.voxel_size_m <= 0.0 or not np.isfinite(options.voxel_size_m):
        raise ValueError("map voxel size must be positive and finite")


__all__ = [
    "FusedPointCloud",
    "FusedWorldMapOptions",
    "FusedWorldMapResult",
    "ObservedVoxelMesh",
    "SparseOccupancyGrid",
    "build_fused_point_cloud",
    "build_observed_voxel_mesh",
    "build_sparse_occupancy",
    "write_fused_world_map",
]
