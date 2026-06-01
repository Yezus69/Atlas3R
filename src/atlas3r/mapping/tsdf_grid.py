"""Public TSDF grid geometry helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def compute_tsdf_grid_shape(
    grid_min_world_m: npt.NDArray[Any],
    grid_max_world_m: npt.NDArray[Any],
    voxel_size_m: float,
) -> tuple[int, int, int]:
    """Return the deterministic XYZ voxel shape for world-space grid bounds."""
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    grid_min = np.asarray(grid_min_world_m, dtype=np.float64)
    grid_max = np.asarray(grid_max_world_m, dtype=np.float64)
    extent = grid_max - grid_min
    if extent.shape != (3,) or np.any(extent <= 0.0):
        raise ValueError("grid bounds: max corner must be greater than min corner")
    shape = np.ceil((extent / voxel_size_m) - 1e-9).astype(np.int64)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def voxel_centers_world(
    grid_min_world_m: npt.NDArray[Any],
    shape_xyz: tuple[int, int, int],
    voxel_size_m: float,
) -> npt.NDArray[np.float64]:
    """Return voxel centers as an Nx3 world-space array."""
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m: must be positive")
    nx, ny, nz = shape_xyz
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("shape_xyz: all dimensions must be positive")
    grid_min = np.asarray(grid_min_world_m, dtype=np.float64)
    if grid_min.shape != (3,):
        raise ValueError("grid_min_world_m: must be a 3-vector")
    x = grid_min[0] + (np.arange(nx, dtype=np.float64) + 0.5) * voxel_size_m
    y = grid_min[1] + (np.arange(ny, dtype=np.float64) + 0.5) * voxel_size_m
    z = grid_min[2] + (np.arange(nz, dtype=np.float64) + 0.5) * voxel_size_m
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


__all__ = [
    "compute_tsdf_grid_shape",
    "voxel_centers_world",
]
