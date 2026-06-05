"""Projection math helpers for sparse block TSDF integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import FLOAT32, FLOAT64
from atlas3r.mapping.observations import DepthObservation
from atlas3r.pose.transforms import invert_transform, transform_points


@dataclass(frozen=True)
class SparseSurfaceSamples:
    points_world_m: npt.NDArray[np.float64]
    valid_depth_sample_count: int


@dataclass(frozen=True)
class SparseProjectedUpdates:
    voxel_coords_xyz: npt.NDArray[np.int64]
    tsdf: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]


def sparse_surface_samples(
    observation: DepthObservation,
    *,
    pixel_stride: int,
) -> SparseSurfaceSamples:
    rows = np.arange(0, observation.camera.height, pixel_stride, dtype=np.int64)
    cols = np.arange(0, observation.camera.width, pixel_stride, dtype=np.int64)
    grid_v, grid_u = np.meshgrid(rows, cols, indexing="ij")
    pixel_v = grid_v.reshape(-1)
    pixel_u = grid_u.reshape(-1)
    depth = observation.depth_m[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    confidence = observation.confidence[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence) & (confidence > 0.0)
    if observation.static_mask is not None:
        static_values = observation.static_mask[pixel_v, pixel_u]
        valid &= (
            static_values if static_values.dtype == np.bool_ else np.asarray(static_values) > 0.5
        )
    if not np.any(valid):
        return SparseSurfaceSamples(np.empty((0, 3), dtype=FLOAT64), 0)
    pixel_u = pixel_u[valid]
    pixel_v = pixel_v[valid]
    depth = depth[valid]
    K = observation.camera.K.astype(FLOAT64, copy=False)
    x_camera = (pixel_u.astype(FLOAT64) - K[0, 2]) * depth / K[0, 0]
    y_camera = (pixel_v.astype(FLOAT64) - K[1, 2]) * depth / K[1, 1]
    points_camera = np.stack([x_camera, y_camera, depth], axis=1)
    return SparseSurfaceSamples(
        transform_points(observation.pose.T_world_camera, points_camera),
        int(points_camera.shape[0]),
    )


def sparse_candidate_offsets(
    *,
    voxel_size_m: float,
    truncation_distance_m: float,
) -> npt.NDArray[np.int64]:
    radius_voxels = int(np.ceil(truncation_distance_m / voxel_size_m))
    values = np.arange(-radius_voxels, radius_voxels + 1, dtype=np.int64)
    offsets = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = np.linalg.norm(offsets.astype(FLOAT64), axis=1) * voxel_size_m
    return cast(npt.NDArray[np.int64], offsets[radius <= truncation_distance_m + voxel_size_m])


def sparse_candidate_voxel_coords(
    points_world_m: npt.NDArray[np.float64],
    *,
    voxel_size_m: float,
    offsets_xyz: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    base_coords = np.floor(points_world_m / voxel_size_m).astype(np.int64)
    coords = (base_coords[:, None, :] + offsets_xyz[None, :, :]).reshape(-1, 3)
    return cast(npt.NDArray[np.int64], np.unique(coords, axis=0))


def project_sparse_candidates(
    *,
    observation: DepthObservation,
    voxel_coords_xyz: npt.NDArray[np.int64],
    voxel_size_m: float,
    truncation_distance_m: float,
) -> SparseProjectedUpdates:
    if voxel_coords_xyz.size == 0:
        return empty_sparse_updates()
    centers_world_m = voxel_centers_from_sparse_coords(voxel_coords_xyz, voxel_size_m)
    centers_camera_m = transform_points(
        invert_transform(observation.pose.T_world_camera),
        centers_world_m,
    )
    z_camera_m = centers_camera_m[:, 2]
    valid_z = z_camera_m > 0.0
    if not np.any(valid_z):
        return empty_sparse_updates()
    indices = np.flatnonzero(valid_z)
    valid_points = centers_camera_m[indices]
    K = observation.camera.K.astype(FLOAT64, copy=False)
    projected_u = K[0, 0] * valid_points[:, 0] / valid_points[:, 2] + K[0, 2]
    projected_v = K[1, 1] * valid_points[:, 1] / valid_points[:, 2] + K[1, 2]
    pixel_u = np.rint(projected_u).astype(np.int64)
    pixel_v = np.rint(projected_v).astype(np.int64)
    inside = (
        (pixel_u >= 0)
        & (pixel_u < observation.camera.width)
        & (pixel_v >= 0)
        & (pixel_v < observation.camera.height)
    )
    if not np.any(inside):
        return empty_sparse_updates()
    indices = indices[inside]
    pixel_u = pixel_u[inside]
    pixel_v = pixel_v[inside]
    z_camera_m = valid_points[inside, 2]
    measured_depth_m = observation.depth_m[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    signed_distance_m = measured_depth_m - z_camera_m
    confidence = observation.confidence[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    sigma = observation.depth_sigma_m[pixel_v, pixel_u].astype(FLOAT64, copy=False)
    valid = (
        np.isfinite(measured_depth_m)
        & (measured_depth_m > 0.0)
        & (np.abs(signed_distance_m) <= truncation_distance_m)
        & np.isfinite(confidence)
        & (confidence > 0.0)
        & np.isfinite(sigma)
    )
    if not np.any(valid):
        return empty_sparse_updates()
    weights = confidence[valid] / (1.0 + sigma[valid] / voxel_size_m)
    positive = weights > 0.0
    if not np.any(positive):
        return empty_sparse_updates()
    valid_indices = indices[valid][positive]
    normalized_tsdf = np.clip(
        signed_distance_m[valid][positive] / truncation_distance_m,
        -1.0,
        1.0,
    )
    return SparseProjectedUpdates(
        voxel_coords_xyz=voxel_coords_xyz[valid_indices],
        tsdf=normalized_tsdf.astype(FLOAT32),
        weight=weights[positive].astype(FLOAT32),
    )


def empty_sparse_updates() -> SparseProjectedUpdates:
    return SparseProjectedUpdates(
        voxel_coords_xyz=np.empty((0, 3), dtype=np.int64),
        tsdf=np.empty((0,), dtype=FLOAT32),
        weight=np.empty((0,), dtype=FLOAT32),
    )


def voxel_centers_from_sparse_coords(
    voxel_coords_xyz: npt.NDArray[np.integer[Any]],
    voxel_size_m: float,
) -> npt.NDArray[np.float64]:
    return (voxel_coords_xyz.astype(FLOAT64, copy=False) + 0.5) * voxel_size_m


__all__ = [
    "SparseProjectedUpdates",
    "SparseSurfaceSamples",
    "project_sparse_candidates",
    "sparse_candidate_offsets",
    "sparse_candidate_voxel_coords",
    "sparse_surface_samples",
    "voxel_centers_from_sparse_coords",
]
