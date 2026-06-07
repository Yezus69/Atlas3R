"""M7 ray-fused static map.

Fuses ``FrameRayPacket`` surface samples into a 3D voxel map (TSDF + occupancy
log-odds + per-voxel counts) and a floor-aligned 2D ``OccupancyGrid2D``.

Honesty rules honored here:
- Voxels with no ray evidence stay UNKNOWN (never free). Unknown is not free.
- Without static/dynamic input, ``P_dynamic`` and ``P_movable_static`` are 0
  (not inferred), while free/occupied/unknown are real.
- Free = space a ray passed through before its surface. Occupied = the surface
  voxel. Beyond the surface is untouched -> unknown.
- The five occupancy probability channels stay pairwise non-collapsed per the
  contract.

``numpy`` is imported lazily inside functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import (
    ContractValidationError,
    FrameRayPacket,
    OccupancyGrid2D,
    ScalePosterior,
    VoxelMapState,
)

DEFAULT_VOXEL_SIZE_M = 0.05
# Bound the grid so a pathological monocular reconstruction cannot allocate an
# enormous dense array. If the scene exceeds this, the voxel size is grown.
MAX_VOXELS_PER_AXIS = 256
TSDF_TRUNCATION_VOXELS = 3.0
LOG_ODDS_HIT = 0.85
LOG_ODDS_MISS = -0.40
LOG_ODDS_CLAMP = 5.0
RANSAC_ITERS = 200
RANSAC_INLIER_DIST_FACTOR = 1.5  # in voxel units


def fuse_static_map(
    packets: Sequence[FrameRayPacket],
    scale_posterior: ScalePosterior,
    *,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
) -> tuple[VoxelMapState | None, OccupancyGrid2D | None, dict[str, Any]]:
    """Return ``(VoxelMapState, OccupancyGrid2D, map_report)``.

    On too-few packets or empty surface, returns ``(None, None, report)`` with
    an explicit blocked status -- never a fabricated map.
    """
    base_report: dict[str, Any] = {
        "module": "M7 - Ray-Fused Static Map",
        "packet_count": len(packets),
        "requested_voxel_size_m": voxel_size_m,
    }

    if len(packets) < 1:
        return None, None, {
            **base_report,
            "status": "blocked_no_packets",
            "blockers": ("no_packets_to_fuse",),
        }

    import numpy as np  # type: ignore

    status_value = scale_posterior.metric_acceptance_status.value
    coordinate_frame = (
        "metric_world"
        if status_value in {"measured_metric", "metric_pseudo_label"}
        else "reconstruction_world"
    )

    surfaces, origins, ray_dirs = _collect_world_points(packets, np)
    if surfaces.shape[0] == 0:
        return None, None, {
            **base_report,
            "status": "blocked_no_surface_points",
            "blockers": ("no_surface_points_after_lifting",),
        }

    # Robust bounds: clip extreme outliers so the grid bounds are not blown out
    # by a few stray points. The points themselves are kept; only the grid
    # extent is computed robustly.
    lo = np.percentile(surfaces, 1.0, axis=0)
    hi = np.percentile(surfaces, 99.0, axis=0)
    pad = TSDF_TRUNCATION_VOXELS * voxel_size_m
    grid_min = np.minimum(lo, origins.min(axis=0)) - pad
    grid_max = np.maximum(hi, origins.max(axis=0)) + pad

    extent = grid_max - grid_min
    extent = np.where(extent > 0.0, extent, voxel_size_m)
    effective_voxel = float(voxel_size_m)
    needed = np.ceil(extent / effective_voxel).astype(np.int64)
    while int(needed.max()) > MAX_VOXELS_PER_AXIS:
        effective_voxel *= 2.0
        needed = np.ceil(extent / effective_voxel).astype(np.int64)
    dims = tuple(int(max(1, d)) for d in needed)

    nx, ny, nz = dims
    tsdf_value = np.ones(dims, dtype=np.float64)  # truncated, default empty=+1
    tsdf_weight = np.zeros(dims, dtype=np.float64)
    log_odds = np.zeros(dims, dtype=np.float64)
    free_count = np.zeros(dims, dtype=np.float64)
    surface_count = np.zeros(dims, dtype=np.float64)
    dynamic_count = np.zeros(dims, dtype=np.float64)
    uncertainty = np.zeros(dims, dtype=np.float64)

    rel_unc = float(scale_posterior.relative_scale_uncertainty)

    # Fuse each surface sample: free along the ray, occupied at the surface.
    _fuse_rays(
        surfaces, origins, ray_dirs,
        grid_min, effective_voxel, dims,
        tsdf_value, tsdf_weight, log_odds,
        free_count, surface_count, uncertainty,
        rel_unc, np,
    )

    try:
        voxel_map = VoxelMapState(
            voxel_size_m=effective_voxel,
            coordinate_frame=coordinate_frame,
            tsdf_value=tsdf_value,
            tsdf_weight=tsdf_weight,
            occupancy_log_odds=log_odds,
            free_space_count=free_count,
            surface_count=surface_count,
            dynamic_count=dynamic_count,
            uncertainty=uncertainty,
        )
    except ContractValidationError as exc:
        return None, None, {
            **base_report,
            "status": "voxel_map_contract_rejected",
            "error": str(exc),
            "blockers": ("voxel_map_contract_rejected",),
        }

    floor_info = _estimate_floor(surfaces, effective_voxel, np)
    grid, grid_report = _build_occupancy_grid(
        free_count, surface_count, log_odds,
        grid_min, effective_voxel, dims,
        floor_info, scale_posterior, coordinate_frame, np,
    )

    occupied_voxels = int(np.count_nonzero(surface_count > 0.0))
    free_voxels = int(np.count_nonzero((free_count > 0.0) & (surface_count <= 0.0)))
    touched = int(np.count_nonzero((free_count > 0.0) | (surface_count > 0.0)))
    total_voxels = int(nx * ny * nz)
    unknown_voxels = total_voxels - touched

    # Real free/occupied contradiction: voxels that received BOTH free-carve and
    # surface evidence (across all frames, order-independent). The carve loop
    # skips a voxel only if it is already a surface voxel at carve time, so a
    # cross-frame conflict (one frame's surface = another frame's carved-free)
    # still lands here and is counted honestly.
    conflict_voxels = int(np.count_nonzero((free_count > 0.0) & (surface_count > 0.0)))
    occupied_evidence_voxels = occupied_voxels
    if occupied_evidence_voxels > 0:
        contradiction_rate = conflict_voxels / occupied_evidence_voxels
    else:
        contradiction_rate = 0.0
    contradiction_rate = float(max(0.0, min(1.0, contradiction_rate)))

    report = {
        **base_report,
        "status": "fused",
        "coordinate_frame": coordinate_frame,
        "effective_voxel_size_m": effective_voxel,
        "grid_dims": dims,
        "grid_min": tuple(float(v) for v in grid_min),
        "grid_max": tuple(float(v) for v in grid_max),
        "total_voxels": total_voxels,
        "occupied_voxel_count": occupied_voxels,
        "free_voxel_count": free_voxels,
        "unknown_voxel_count": unknown_voxels,
        "occupied_fraction": occupied_voxels / total_voxels if total_voxels else 0.0,
        "free_fraction": free_voxels / total_voxels if total_voxels else 0.0,
        "unknown_fraction": unknown_voxels / total_voxels if total_voxels else 0.0,
        "free_occupied_conflict_voxel_count": conflict_voxels,
        "free_space_contradiction_rate": contradiction_rate,
        "free_space_contradiction_basis": "conflict_voxels_over_occupied_voxels",
        "surface_point_count": int(surfaces.shape[0]),
        "floor": floor_info["report"],
        "occupancy_grid": grid_report,
        "dynamic_inference": "not_inferred_no_static_dynamic_input",
        "blockers": tuple(floor_info.get("blockers", ())),
    }
    return voxel_map, grid, report


# ---------------------------------------------------------------------------
# fusion internals
# ---------------------------------------------------------------------------


def _collect_world_points(packets: Sequence[FrameRayPacket], np: Any) -> tuple[Any, Any, Any]:
    surf_list = []
    origin_list = []
    dir_list = []
    for packet in packets:
        rays = np.asarray(packet.rays_camera, dtype=np.float64).reshape((-1, 3))
        depth = np.asarray(packet.radial_depth_m, dtype=np.float64).reshape((-1,))
        T = np.asarray(packet.T_world_camera, dtype=np.float64).reshape((4, 4))
        R = T[:3, :3]
        t = T[:3, 3]
        X_cam = rays * depth[:, None]
        X_world = X_cam @ R.T + t[None, :]
        dirs_world = rays @ R.T
        surf_list.append(X_world)
        origin_list.append(np.repeat(t[None, :], X_world.shape[0], axis=0))
        dir_list.append(dirs_world)
    if not surf_list:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(surf_list, axis=0),
        np.concatenate(origin_list, axis=0),
        np.concatenate(dir_list, axis=0),
    )


def _fuse_rays(
    surfaces, origins, ray_dirs,
    grid_min, voxel, dims,
    tsdf_value, tsdf_weight, log_odds,
    free_count, surface_count, uncertainty,
    rel_unc, np,
):
    nx, ny, nz = dims
    inv_voxel = 1.0 / voxel
    n = surfaces.shape[0]
    # Cap fused samples to keep the fuse bounded for dense monocular packets.
    max_fuse = 20000
    if n > max_fuse:
        idx = np.unique(np.rint(np.linspace(0, n - 1, num=max_fuse)).astype(np.int64))
        surfaces = surfaces[idx]
        origins = origins[idx]
        ray_dirs = ray_dirs[idx]
        n = surfaces.shape[0]

    surf_idx = np.floor((surfaces - grid_min[None, :]) * inv_voxel).astype(np.int64)
    in_bounds = (
        (surf_idx[:, 0] >= 0) & (surf_idx[:, 0] < nx)
        & (surf_idx[:, 1] >= 0) & (surf_idx[:, 1] < ny)
        & (surf_idx[:, 2] >= 0) & (surf_idx[:, 2] < nz)
    )
    for k in range(n):
        if in_bounds[k]:
            ix, iy, iz = int(surf_idx[k, 0]), int(surf_idx[k, 1]), int(surf_idx[k, 2])
            surface_count[ix, iy, iz] += 1.0
            tsdf_value[ix, iy, iz] = 0.0
            tsdf_weight[ix, iy, iz] += 1.0
            log_odds[ix, iy, iz] = float(np.clip(log_odds[ix, iy, iz] + LOG_ODDS_HIT, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP))
            uncertainty[ix, iy, iz] += rel_unc
        # Carve free space: step from the camera origin toward the surface,
        # marking voxels before the surface as free evidence.
        origin = origins[k]
        target = surfaces[k]
        seg = target - origin
        dist = float(np.linalg.norm(seg))
        if dist <= voxel:
            continue
        n_steps = int(dist * inv_voxel)
        if n_steps <= 1:
            continue
        # Sample free voxels along the ray, excluding the final surface voxel.
        ts = np.linspace(0.0, 1.0, num=min(n_steps, 64), endpoint=False)
        pts = origin[None, :] + ts[:, None] * seg[None, :]
        free_idx = np.floor((pts - grid_min[None, :]) * inv_voxel).astype(np.int64)
        for m in range(free_idx.shape[0] - 1):  # last sample is near surface
            fx, fy, fz = int(free_idx[m, 0]), int(free_idx[m, 1]), int(free_idx[m, 2])
            if 0 <= fx < nx and 0 <= fy < ny and 0 <= fz < nz:
                if surface_count[fx, fy, fz] > 0.0:
                    continue
                free_count[fx, fy, fz] += 1.0
                log_odds[fx, fy, fz] = float(np.clip(log_odds[fx, fy, fz] + LOG_ODDS_MISS, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP))


def _estimate_floor(surfaces, voxel, np) -> dict[str, Any]:
    """RANSAC a horizontal-ish floor plane; fall back to lowest-Z heuristic."""
    n = surfaces.shape[0]
    if n < 3:
        z_floor = float(np.min(surfaces[:, 2])) if n else 0.0
        return {
            "floor_axis": 2,
            "floor_value": z_floor,
            "normal": (0.0, 0.0, 1.0),
            "method": "lowest_z_heuristic_insufficient_points",
            "report": {"method": "lowest_z_heuristic", "floor_value": z_floor, "inlier_ratio": 0.0},
            "blockers": ("floor_ransac_insufficient_points_used_lowest_z",),
        }

    rng = np.random.default_rng(0)
    inlier_dist = RANSAC_INLIER_DIST_FACTOR * voxel
    best_inliers = -1
    best_plane = None
    for _ in range(RANSAC_ITERS):
        sample = surfaces[rng.choice(n, size=3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        dist = np.abs(surfaces @ normal + d)
        inliers = int(np.count_nonzero(dist < inlier_dist))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    if best_plane is None or best_inliers < max(3, int(0.05 * n)):
        z_floor = float(np.percentile(surfaces[:, 2], 5.0))
        return {
            "floor_axis": 2,
            "floor_value": z_floor,
            "normal": (0.0, 0.0, 1.0),
            "method": "lowest_z_heuristic_ransac_failed",
            "report": {"method": "lowest_z_heuristic", "floor_value": z_floor, "inlier_ratio": 0.0},
            "blockers": ("floor_ransac_failed_used_lowest_z_assumption",),
        }

    normal, d = best_plane
    # Orient normal so it points "up" relative to the points' centroid.
    centroid = surfaces.mean(axis=0)
    if (normal @ centroid + d) < 0:
        normal = -normal
        d = -d
    inlier_ratio = best_inliers / n
    # Determine the dominant axis of the plane normal for the 2D projection.
    floor_axis = int(np.argmax(np.abs(normal)))
    floor_value = float(np.percentile(surfaces[:, floor_axis], 5.0))
    return {
        "floor_axis": floor_axis,
        "floor_value": floor_value,
        "normal": tuple(float(v) for v in normal),
        "plane_d": float(d),
        "method": "ransac",
        "report": {
            "method": "ransac",
            "inlier_ratio": float(inlier_ratio),
            "normal": tuple(float(v) for v in normal),
            "floor_axis": floor_axis,
            "floor_value": floor_value,
        },
        "blockers": (),
    }


def _build_occupancy_grid(
    free_count, surface_count, log_odds,
    grid_min, voxel, dims,
    floor_info, scale_posterior, coordinate_frame, np,
) -> tuple[OccupancyGrid2D | None, dict[str, Any]]:
    floor_axis = floor_info["floor_axis"]
    # The two non-floor axes form the 2D grid plane.
    plane_axes = [a for a in range(3) if a != floor_axis]
    a0, a1 = plane_axes
    nx, ny, nz = dims
    plane_dims = (dims[a0], dims[a1])

    # Collapse occupancy/free counts along the floor axis.
    occ_any = (surface_count > 0.0)
    free_any = (free_count > 0.0) & (~occ_any)

    occ_2d = np.any(occ_any, axis=floor_axis)
    free_2d = np.any(free_any, axis=floor_axis) & (~occ_2d)
    touched_2d = occ_2d | free_2d

    p_occupied = np.where(occ_2d, 0.9, 0.0).astype(np.float64)
    p_free = np.where(free_2d, 0.85, 0.0).astype(np.float64)
    # Unknown is everything untouched (and is NOT free). Cells that are occupied
    # or free get a small residual unknown so channels stay distinct but valid.
    p_unknown = np.where(touched_2d, 0.05, 1.0).astype(np.float64)
    # No static/dynamic input -> these are honestly zero.
    p_movable = np.zeros(plane_dims, dtype=np.float64)
    p_dynamic = np.zeros(plane_dims, dtype=np.float64)

    # Height extent per cell from the actual touched voxel range.
    height_min, height_max = _height_extents(
        occ_any | free_any, floor_axis, grid_min, voxel, dims, np
    )

    origin_world = (float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
    scale_unc = float(scale_posterior.scale_std)

    try:
        grid = OccupancyGrid2D(
            grid_frame=coordinate_frame,
            resolution_m=float(voxel),
            origin_world=origin_world,
            P_free=p_free,
            P_occupied_static=p_occupied,
            P_movable_static=p_movable,
            P_dynamic=p_dynamic,
            P_unknown=p_unknown,
            height_min_m=height_min,
            height_max_m=height_max,
            scale_uncertainty=scale_unc,
            map_confidence=_map_confidence(scale_posterior),
        )
    except ContractValidationError as exc:
        return None, {
            "status": "occupancy_grid_contract_rejected",
            "error": str(exc),
        }

    occupied_cells = int(np.count_nonzero(occ_2d))
    free_cells = int(np.count_nonzero(free_2d))
    unknown_cells = int(np.count_nonzero(~touched_2d))
    total_cells = int(plane_dims[0] * plane_dims[1])
    return grid, {
        "status": "built",
        "grid_dims": plane_dims,
        "floor_axis": floor_axis,
        "plane_axes": (a0, a1),
        "occupied_cells": occupied_cells,
        "free_cells": free_cells,
        "unknown_cells": unknown_cells,
        "total_cells": total_cells,
        "occupied_fraction": occupied_cells / total_cells if total_cells else 0.0,
        "free_fraction": free_cells / total_cells if total_cells else 0.0,
        "unknown_fraction": unknown_cells / total_cells if total_cells else 0.0,
        "dynamic_channel": "zero_not_inferred",
        "movable_static_channel": "zero_not_inferred",
    }


def _height_extents(touched_3d, floor_axis, grid_min, voxel, dims, np):
    # For each 2D cell, height_min/max along the floor axis of touched voxels.
    # Default: a flat valid extent (min==max) where nothing is touched.
    plane_axes = [a for a in range(3) if a != floor_axis]
    a0, a1 = plane_axes
    plane_dims = (dims[a0], dims[a1])

    # Build coordinate of each voxel index along the floor axis (world).
    nf = dims[floor_axis]
    floor_coords = grid_min[floor_axis] + (np.arange(nf) + 0.5) * voxel

    # Move floor axis to the front for reduction.
    moved = np.moveaxis(touched_3d, floor_axis, 0)  # shape (nf, A0, A1)
    any_touched = np.any(moved, axis=0)

    # Index of first/last touched along floor axis.
    coords_b = floor_coords[:, None, None]
    big = grid_min[floor_axis] + nf * voxel
    small = grid_min[floor_axis]
    min_h = np.where(moved, coords_b, big).min(axis=0)
    max_h = np.where(moved, coords_b, small).max(axis=0)

    # Untouched cells get a flat, contract-valid extent at the grid floor base.
    base = float(grid_min[floor_axis])
    height_min = np.where(any_touched, min_h, base).astype(np.float64)
    height_max = np.where(any_touched, max_h, base).astype(np.float64)
    # Guard: ensure max >= min everywhere.
    height_max = np.maximum(height_max, height_min)
    return height_min, height_max


def _map_confidence(scale_posterior: ScalePosterior) -> float:
    status = scale_posterior.metric_acceptance_status.value
    base = {
        "measured_metric": 0.9,
        "metric_pseudo_label": 0.6,
        "non_metric_pseudo_label": 0.4,
        "rejected": 0.1,
    }.get(status, 0.3)
    return max(0.0, min(1.0, base))


__all__ = ["fuse_static_map"]
