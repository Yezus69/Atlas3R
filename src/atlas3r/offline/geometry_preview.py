"""Geometry preview and pixel/depth lifting for offline traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts import COORDINATE_FRAME_NAME, MapArtifact, TruthBoundary
from atlas3r.contracts.coordinates import transform_points, unproject_depth, validate_T_A_B
from atlas3r.mapping import read_npz_artifact_metadata, write_mesh_ply
from atlas3r.offline.disagreement import DisagreementResult
from atlas3r.offline.frame_cache import FrameCacheResult
from atlas3r.offline.proposal_cache import ProposalCacheResult
from atlas3r.offline.run_manifest import FailurePoint


@dataclass(frozen=True)
class GeometryPreviewResult:
    status: str
    geometry_npz_path: str
    geometry_ply_path: str | None
    point_count: int
    observed_only: bool
    predicted_completion: bool
    measured_geometry: bool
    metric_scale_source: str
    source_teacher: str = "none"
    valid_point_ratio: float = 0.0
    per_frame_point_counts: dict[str, int] = field(default_factory=dict)


def lift_depth_to_world_points(
    K: NDArray[np.float32],
    depth_m: NDArray[np.float32],
    T_world_camera: NDArray[np.float32],
) -> NDArray[np.float32]:
    points_camera = unproject_depth(K, depth_m).reshape((-1, 3))
    return transform_points(T_world_camera, points_camera)


def write_geometry_preview(
    run_dir: str | Path,
    *,
    frame_cache: FrameCacheResult,
    proposal_cache: ProposalCacheResult,
    write_ply: bool,
    failure_points: list[FailurePoint],
    disagreement: DisagreementResult | None = None,
) -> GeometryPreviewResult:
    root = Path(run_dir)
    npz_path = root / "geometry" / "geometry_preview.npz"
    consensus_records = () if disagreement is None else disagreement.consensus_depth_records
    if not proposal_cache.geometry_depth_records and not consensus_records:
        metadata = {
            "status": "unavailable",
            "why": "no usable depth, intrinsics, and T_world_camera proposal exists",
            "observed_only": True,
            "predicted_completion": False,
            "measured_geometry": False,
            "metric_scale_source": "unknown",
            "source_teacher": "none",
        }
        _write_geometry_npz(
            npz_path,
            _empty_points(),
            _empty_colors(),
            _empty_float(),
            (),
            (),
            (),
            metadata,
        )
        failure_points.append(
            FailurePoint(
                module="geometry_lifter",
                code="geometry_inputs_missing",
                severity="warning",
                status="unavailable",
                why=str(metadata["why"]),
                input_missing="depth + K + T_world_camera",
                future_module="real teacher geometry proposal path",
                artifact_path="geometry/geometry_preview.npz",
            )
        )
        return GeometryPreviewResult(
            status="unavailable",
            geometry_npz_path="geometry/geometry_preview.npz",
            geometry_ply_path=None,
            point_count=0,
            observed_only=True,
            predicted_completion=False,
            measured_geometry=False,
            metric_scale_source="unknown",
        )
    return _write_teacher_geometry_preview(
        root,
        npz_path=npz_path,
        frame_cache=frame_cache,
        proposal_cache=proposal_cache,
        write_ply=write_ply,
        failure_points=failure_points,
        disagreement=disagreement,
    )


def _write_teacher_geometry_preview(
    run_dir: Path,
    *,
    npz_path: Path,
    frame_cache: FrameCacheResult,
    proposal_cache: ProposalCacheResult,
    write_ply: bool,
    failure_points: list[FailurePoint],
    disagreement: DisagreementResult | None,
) -> GeometryPreviewResult:
    frames_by_id = {frame.frame_id: frame for frame in frame_cache.frames}
    vggt_cameras_by_depth = {
        str(record.get("depth_key")): record for record in proposal_cache.vggt_camera_records
    }
    vggt_cameras_by_frame = {
        int(str(record["frame_id"])): record for record in proposal_cache.vggt_camera_records
    }
    use_consensus_depth = bool(
        disagreement is not None
        and disagreement.consensus_depth_records
        and proposal_cache.vggt_camera_records
    )
    depth_records = (
        disagreement.consensus_depth_records
        if use_consensus_depth and disagreement is not None
        else proposal_cache.geometry_depth_records
    )
    consensus_arrays = (
        disagreement.consensus_depth_arrays
        if use_consensus_depth and disagreement is not None
        else {}
    )
    point_chunks: list[NDArray[np.float32]] = []
    color_chunks: list[NDArray[np.uint8]] = []
    uncertainty_chunks: list[NDArray[np.float32]] = []
    frame_id_chunks: list[NDArray[np.int32]] = []
    submap_chunks: list[NDArray[np.int32]] = []
    source_teacher_chunks: list[str] = []
    per_frame_point_counts: dict[str, int] = {}
    measured_geometry = False
    metric_scale_source = "unknown"
    source_teacher = (
        "vggt_pose_consensus_depth_diagnostic"
        if use_consensus_depth
        else proposal_cache.geometry_source
    )
    valid_points = 0
    total_sampled = 0
    for proposal in depth_records:
        teacher = str(proposal.get("teacher_name", "unknown"))
        frame_id = int(str(proposal["frame_id"]))
        if frame_id not in frames_by_id:
            continue
        try:
            frame = frames_by_id[frame_id]
            depth, sigma, confidence, valid_mask, K, T_world_camera, submap_id = _proposal_arrays(
                proposal,
                proposal_cache,
                vggt_cameras_by_depth,
                vggt_cameras_by_frame,
                consensus_arrays,
            )
        except (KeyError, ValueError) as exc:
            failure_points.append(
                FailurePoint(
                    module="geometry_lifter",
                    code="teacher_geometry_rejected",
                    severity="warning",
                    status="rejected",
                    why=f"rejected {teacher} proposal for frame {frame_id}: {exc}",
                    input_missing="valid finite teacher depth/K/T_world_camera",
                    artifact_path="geometry/geometry_preview.npz",
                )
            )
            continue
        sample_mask = _sample_mask(depth.shape[0], depth.shape[1])
        lift_mask = sample_mask & valid_mask & np.isfinite(depth) & (depth > 0.0)
        total_sampled += int(sample_mask.sum())
        if not np.any(lift_mask):
            continue
        masked_depth = np.where(lift_mask, depth, 0.0).astype(np.float32)
        points_world = lift_depth_to_world_points(K, masked_depth, T_world_camera).reshape((-1, 3))
        flat_mask = lift_mask.reshape((-1,))
        points = points_world[flat_mask]
        if points.size and not np.all(np.isfinite(points)):
            continue
        colors = _sample_colors_for_depth(frame.rgb_u8, (int(depth.shape[0]), int(depth.shape[1])))[
            flat_mask
        ]
        uncertainty = sigma.reshape((-1,))[flat_mask]
        point_count = int(points.shape[0])
        point_chunks.append(points.astype(np.float32))
        color_chunks.append(colors.astype(np.uint8))
        uncertainty_chunks.append(uncertainty.astype(np.float32))
        frame_id_chunks.append(np.full(point_count, frame_id, np.int32))
        submap_chunks.append(np.full(point_count, submap_id, np.int32))
        source_teacher_chunks.extend([teacher] * point_count)
        per_frame_point_counts[str(frame_id)] = (
            per_frame_point_counts.get(str(frame_id), 0) + point_count
        )
        measured_geometry = measured_geometry or bool(proposal.get("measured_geometry", False))
        metric_scale_source = str(proposal.get("metric_scale_source", metric_scale_source))
        valid_points += point_count
        _ = confidence
    points_all = np.concatenate(point_chunks, axis=0) if point_chunks else _empty_points()
    colors_all = np.concatenate(color_chunks, axis=0) if color_chunks else _empty_colors()
    uncertainty_all = (
        np.concatenate(uncertainty_chunks, axis=0) if uncertainty_chunks else _empty_float()
    )
    frame_ids = (
        np.concatenate(frame_id_chunks, axis=0) if frame_id_chunks else np.zeros((0,), np.int32)
    )
    submap_ids = (
        np.concatenate(submap_chunks, axis=0) if submap_chunks else np.zeros((0,), np.int32)
    )
    if points_all.shape[0] == 0:
        failure_points.append(
            FailurePoint(
                module="geometry_lifter",
                code="teacher_geometry_empty",
                severity="warning",
                status="unavailable",
                why="teacher proposals existed but no valid finite positive depth samples survived",
                input_missing="valid teacher depth samples",
                artifact_path="geometry/geometry_preview.npz",
            )
        )
    valid_ratio = float(valid_points / total_sampled) if total_sampled else 0.0
    label_type = (
        "teacher_pseudo" if _is_teacher_pseudo_source(source_teacher) else "debug_synthetic"
    )
    metadata: dict[str, object] = {
        "status": "partial" if points_all.shape[0] else "unavailable",
        "why": (
            "VGGT pose with diagnostic VGGT/Depth Pro consensus depth preview"
            if source_teacher == "vggt_pose_consensus_depth_diagnostic"
            else "VGGT teacher-proposed geometry preview"
            if source_teacher == "vggt"
            else "debug-only flat-depth geometry preview"
        ),
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "measured_geometry": measured_geometry,
        "metric_scale_source": metric_scale_source,
        "coordinate_frame": COORDINATE_FRAME_NAME,
        "coordinate_convention": COORDINATE_FRAME_NAME,
        "point_count": int(points_all.shape[0]),
        "valid_point_ratio": valid_ratio,
        "per_frame_point_counts": per_frame_point_counts,
        "source_teacher": source_teacher,
        "label_type": label_type,
        "accuracy_report": False,
        "realtime_claim": False,
        "usable_for_training": False,
        "pseudo_submap_ids": sorted(set(int(item) for item in submap_ids.tolist())),
    }
    _write_geometry_npz(
        npz_path,
        points_all,
        colors_all,
        uncertainty_all,
        tuple(frame_ids.tolist()),
        tuple(source_teacher_chunks),
        tuple(submap_ids.tolist()),
        metadata,
    )
    ply_rel = _write_ply_if_requested(
        run_dir,
        write_ply=write_ply,
        points=points_all,
        uncertainty=uncertainty_all,
        frame_ids=frame_ids,
        source_teacher=source_teacher,
        label_type=label_type,
        metric_scale_source=metric_scale_source,
        measured_geometry=measured_geometry,
    )
    return GeometryPreviewResult(
        status=str(metadata["status"]),
        geometry_npz_path="geometry/geometry_preview.npz",
        geometry_ply_path=ply_rel,
        point_count=int(points_all.shape[0]),
        observed_only=True,
        predicted_completion=False,
        measured_geometry=measured_geometry,
        metric_scale_source=metric_scale_source,
        source_teacher=source_teacher,
        valid_point_ratio=valid_ratio,
        per_frame_point_counts=per_frame_point_counts,
    )


def _proposal_arrays(
    proposal: dict[str, object],
    proposal_cache: ProposalCacheResult,
    vggt_cameras_by_depth: dict[str, dict[str, object]],
    vggt_cameras_by_frame: dict[int, dict[str, object]],
    consensus_arrays: dict[str, NDArray[np.float32]],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.float32],
    int,
]:
    if proposal.get("teacher_name") == "debug_flat_depth":
        height = int(str(_camera_dict(proposal)["height"]))
        width = int(str(_camera_dict(proposal)["width"]))
        depth = np.full((height, width), float(str(proposal["constant_depth_m"])), np.float32)
        sigma = np.full((height, width), float(str(proposal["depth_sigma_m"])), np.float32)
        confidence = np.full((height, width), float(str(proposal["confidence"])), np.float32)
        valid = np.ones((height, width), dtype=np.bool_)
        K = np.asarray(proposal["K"], dtype=np.float32)
        T_world_camera = validate_T_A_B(proposal["T_world_camera"], "T_world_camera")
        return depth, sigma, confidence, valid, K, T_world_camera, 0
    if proposal.get("teacher_name") == "consensus_preview":
        frame_id = int(str(proposal["frame_id"]))
        camera = vggt_cameras_by_frame[frame_id]
        depth = consensus_arrays[str(proposal["depth_key"])].astype(np.float32)
        sigma = consensus_arrays[str(proposal["depth_sigma_key"])].astype(np.float32)
        confidence = consensus_arrays[str(proposal["confidence_key"])].astype(np.float32)
        valid = consensus_arrays[str(proposal["valid_mask_key"])] > 0.0
        K = np.asarray(camera["K"], dtype=np.float32)
        T_world_camera = validate_T_A_B(camera["T_world_camera"], "T_world_camera")
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError("K must be finite 3x3")
        return (
            depth,
            sigma,
            confidence,
            valid,
            K,
            T_world_camera,
            int(str(camera["pseudo_submap_id"])),
        )
    depth_key = str(proposal["depth_key"])
    camera = vggt_cameras_by_depth[depth_key]
    depth = proposal_cache.depth_arrays[depth_key].astype(np.float32)
    sigma = proposal_cache.depth_arrays[str(proposal["depth_sigma_key"])].astype(np.float32)
    confidence = proposal_cache.depth_arrays[str(proposal["confidence_key"])].astype(np.float32)
    valid = proposal_cache.depth_arrays[str(proposal["valid_mask_key"])] > 0.0
    K = np.asarray(camera["K"], dtype=np.float32)
    T_world_camera = validate_T_A_B(camera["T_world_camera"], "T_world_camera")
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("K must be finite 3x3")
    if depth.shape != sigma.shape or depth.shape != confidence.shape or depth.shape != valid.shape:
        raise ValueError("depth, sigma, confidence, and valid mask shapes must match")
    return depth, sigma, confidence, valid, K, T_world_camera, int(str(camera["pseudo_submap_id"]))


def _is_teacher_pseudo_source(source_teacher: str) -> bool:
    return source_teacher in {"vggt", "vggt_pose_consensus_depth_diagnostic"}


def _camera_dict(proposal: dict[str, object]) -> dict[str, object]:
    camera = proposal["camera"]
    if not isinstance(camera, dict):
        raise ValueError("debug proposal camera must be a dict")
    return camera


def _sample_colors_for_depth(
    rgb_u8: NDArray[np.uint8], depth_shape: tuple[int, int]
) -> NDArray[np.uint8]:
    height, width = depth_shape
    src_h, src_w = rgb_u8.shape[:2]
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(np.int32), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(np.int32), 0, src_w - 1)
    return cast(NDArray[np.uint8], rgb_u8[y_idx[:, None], x_idx[None, :]].reshape((-1, 3)))


def _write_ply_if_requested(
    run_dir: Path,
    *,
    write_ply: bool,
    points: NDArray[np.float32],
    uncertainty: NDArray[np.float32],
    frame_ids: NDArray[np.int32],
    source_teacher: str,
    label_type: str,
    metric_scale_source: str,
    measured_geometry: bool,
) -> str | None:
    if not write_ply or points.shape[0] == 0:
        return None
    ply_path = run_dir / "geometry" / "geometry_preview.ply"
    truth = TruthBoundary(
        label_type="teacher_pseudo" if label_type == "teacher_pseudo" else "debug_synthetic",
        metric_scale_source=metric_scale_source,
        measured_geometry=measured_geometry,
        observed_only=True,
        notes=(
            "VGGT teacher-proposed point preview; not measured geometry."
            if source_teacher == "vggt"
            else "Diagnostic consensus depth preview lifted with VGGT poses; not optimized."
            if source_teacher == "vggt_pose_consensus_depth_diagnostic"
            else "Debug flat-depth preview; not teacher-measured geometry."
        ),
    )
    artifact = MapArtifact(
        artifact_type="mesh",
        path="geometry/geometry_preview.ply",
        coordinate_frame=COORDINATE_FRAME_NAME,
        source_frame_ids=tuple(sorted(set(int(item) for item in frame_ids.tolist()))),
        voxel_size_m=None,
        observed_coverage_estimate=0.0,
        mean_uncertainty_m=float(uncertainty.mean()) if uncertainty.size else 0.0,
        p95_uncertainty_m=float(np.percentile(uncertainty, 95)) if uncertainty.size else 0.0,
        truth_boundary=truth,
        metadata={"geometry_kind": "point_preview", "faces": 0, "source_teacher": source_teacher},
    )
    write_mesh_ply(
        ply_path,
        vertices_world_m=points.astype(np.float32),
        triangles=np.zeros((0, 3), dtype=np.uint32),
        artifact=artifact,
    )
    return "geometry/geometry_preview.ply"


def read_geometry_metadata(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("geometry metadata must be a JSON object")
    return metadata


def read_geometry_artifact_metadata(path: str | Path) -> dict[str, object]:
    metadata = read_npz_artifact_metadata(path)
    return dict(metadata)


def _write_geometry_npz(
    path: Path,
    points_world_m: NDArray[np.float32],
    colors_u8: NDArray[np.uint8],
    uncertainty_m: NDArray[np.float32],
    frame_ids: tuple[int, ...],
    source_teachers: tuple[str, ...],
    pseudo_submap_ids: tuple[int, ...],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points_world_m=points_world_m.astype(np.float32),
        colors_u8=colors_u8.astype(np.uint8),
        uncertainty_m=uncertainty_m.astype(np.float32),
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        source_teachers=np.asarray(source_teachers, dtype="<U32"),
        pseudo_submap_ids=np.asarray(pseudo_submap_ids, dtype=np.int32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _sample_mask(height: int, width: int) -> NDArray[np.bool_]:
    stride = max(1, min(height, width) // 24)
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[::stride, ::stride] = True
    return mask


def _empty_points() -> NDArray[np.float32]:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_colors() -> NDArray[np.uint8]:
    return np.zeros((0, 3), dtype=np.uint8)


def _empty_float() -> NDArray[np.float32]:
    return np.zeros((0,), dtype=np.float32)
