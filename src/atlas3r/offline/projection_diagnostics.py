"""Projection residual diagnostics for offline teacher-pseudo maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.coordinates import (
    invert_T_A_B,
    transform_points,
    unproject_depth,
    validate_T_A_B,
)
from atlas3r.offline.proposal_cache import ProposalCacheResult


@dataclass(frozen=True)
class ProjectionDiagnostics:
    projection_count: int
    residual_mean_m: float
    residual_p95_m: float
    residuals_m: NDArray[np.float32]
    frame_i: NDArray[np.int32]
    frame_j: NDArray[np.int32]


def compute_projection_consistency(
    *,
    proposal_cache: ProposalCacheResult,
    depth_records: tuple[dict[str, object], ...],
    depth_arrays: dict[str, NDArray[np.float32]],
    focal_scale: float = 1.0,
    max_neighbor_pairs: int = 3,
    min_overlap_pixels: int = 512,
) -> ProjectionDiagnostics:
    cameras = {
        int(str(record["frame_id"])): record for record in proposal_cache.vggt_camera_records
    }
    if not cameras or not depth_records:
        return empty_projection_diagnostics()
    records = sorted(depth_records, key=lambda item: int(str(item["keyframe_index"])))
    residual_chunks: list[NDArray[np.float32]] = []
    frame_i_chunks: list[NDArray[np.int32]] = []
    frame_j_chunks: list[NDArray[np.int32]] = []
    for index, source in enumerate(records):
        for target in records[index + 1 : index + 1 + max(1, max_neighbor_pairs)]:
            residuals = _projection_residual_pair(
                source,
                target,
                cameras=cameras,
                depth_arrays=depth_arrays,
                focal_scale=focal_scale,
            )
            if residuals.shape[0] < min_overlap_pixels:
                continue
            src_id = int(str(source["frame_id"]))
            dst_id = int(str(target["frame_id"]))
            residual_chunks.append(residuals)
            frame_i_chunks.append(np.full(residuals.shape[0], src_id, dtype=np.int32))
            frame_j_chunks.append(np.full(residuals.shape[0], dst_id, dtype=np.int32))
    if not residual_chunks:
        return empty_projection_diagnostics()
    residual_all = np.concatenate(residual_chunks).astype(np.float32)
    return ProjectionDiagnostics(
        projection_count=int(residual_all.shape[0]),
        residual_mean_m=float(residual_all.mean()),
        residual_p95_m=float(np.percentile(residual_all, 95)),
        residuals_m=residual_all,
        frame_i=np.concatenate(frame_i_chunks).astype(np.int32),
        frame_j=np.concatenate(frame_j_chunks).astype(np.int32),
    )


def empty_projection_diagnostics() -> ProjectionDiagnostics:
    return ProjectionDiagnostics(
        projection_count=0,
        residual_mean_m=0.0,
        residual_p95_m=0.0,
        residuals_m=np.zeros((0,), dtype=np.float32),
        frame_i=np.zeros((0,), dtype=np.int32),
        frame_j=np.zeros((0,), dtype=np.int32),
    )


def _projection_residual_pair(
    source: dict[str, object],
    target: dict[str, object],
    *,
    cameras: dict[int, dict[str, object]],
    depth_arrays: dict[str, NDArray[np.float32]],
    focal_scale: float,
) -> NDArray[np.float32]:
    src_frame = int(str(source["frame_id"]))
    dst_frame = int(str(target["frame_id"]))
    src_camera = cameras.get(src_frame)
    dst_camera = cameras.get(dst_frame)
    if src_camera is None or dst_camera is None:
        return np.zeros((0,), dtype=np.float32)
    src_depth = depth_arrays[str(source["depth_key"])].astype(np.float32)
    dst_depth = depth_arrays[str(target["depth_key"])].astype(np.float32)
    src_valid = depth_arrays[str(source["valid_mask_key"])] > 0.0
    dst_valid = depth_arrays[str(target["valid_mask_key"])] > 0.0
    src_K = _scaled_K(src_camera, src_depth.shape, focal_scale)
    dst_K = _scaled_K(dst_camera, dst_depth.shape, focal_scale)
    src_T = validate_T_A_B(src_camera["T_world_camera"], "T_world_camera")
    dst_T = validate_T_A_B(dst_camera["T_world_camera"], "T_world_camera")
    stride = max(1, min(src_depth.shape) // 48)
    sample = np.zeros(src_depth.shape, dtype=np.bool_)
    sample[::stride, ::stride] = True
    lift = sample & src_valid & np.isfinite(src_depth) & (src_depth > 0.0)
    if not np.any(lift):
        return np.zeros((0,), dtype=np.float32)
    points_camera = unproject_depth(src_K, np.where(lift, src_depth, 0.0)).reshape((-1, 3))
    points_world = transform_points(src_T, points_camera)
    points_dst = transform_points(invert_T_A_B(dst_T), points_world)
    points_dst = points_dst[lift.reshape((-1,))]
    positive = points_dst[:, 2] > 0.0
    if not np.any(positive):
        return np.zeros((0,), dtype=np.float32)
    points_dst = points_dst[positive]
    pixels_h = points_dst @ dst_K.T
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-6)
    u = np.round(pixels[:, 0]).astype(np.int32)
    v = np.round(pixels[:, 1]).astype(np.int32)
    inside = (u >= 0) & (v >= 0) & (u < dst_depth.shape[1]) & (v < dst_depth.shape[0])
    if not np.any(inside):
        return np.zeros((0,), dtype=np.float32)
    u = u[inside]
    v = v[inside]
    z = points_dst[:, 2][inside]
    valid = dst_valid[v, u] & np.isfinite(dst_depth[v, u]) & (dst_depth[v, u] > 0.0)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float32)
    return cast(NDArray[np.float32], np.abs(z[valid] - dst_depth[v, u][valid]).astype(np.float32))


def _scaled_K(
    camera: dict[str, object], target_shape: tuple[int, ...], focal_scale: float
) -> NDArray[np.float32]:
    K = np.asarray(camera["K"], dtype=np.float32)
    shape = camera.get("depth_shape")
    if isinstance(shape, list) and len(shape) == 2:
        src_h, src_w = int(str(shape[0])), int(str(shape[1]))
    else:
        src_h, src_w = int(target_shape[0]), int(target_shape[1])
    dst_h, dst_w = int(target_shape[0]), int(target_shape[1])
    scaled = K.copy()
    scaled[0, :] *= float(dst_w) / max(float(src_w), 1.0)
    scaled[1, :] *= float(dst_h) / max(float(src_h), 1.0)
    scaled[0, 0] *= focal_scale
    scaled[1, 1] *= focal_scale
    scaled[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return scaled.astype(np.float32)
