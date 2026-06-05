"""Artifact writers for sparse block TSDF diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import TSDFSurface
from atlas3r.mapping.sparse_tsdf import SparseBlockTSDFMapper


def write_sparse_tsdf_outputs(
    output: Path,
    *,
    mapper: SparseBlockTSDFMapper,
    surface: TSDFSurface,
    metrics: dict[str, object],
) -> Path:
    """Write sparse state, surface arrays, metadata, and metrics."""

    output.mkdir(parents=True, exist_ok=True)
    voxel_coords, tsdf, weight = mapper.active_voxel_arrays()
    np.savez(
        output / "sparse_tsdf_state.npz",
        block_coords_xyz=mapper.block_coordinates(),
        voxel_coords_xyz=voxel_coords,
        tsdf=tsdf,
        weight=weight,
        voxel_size_m=np.asarray(surface.metadata["voxel_size_m"], dtype=np.float32),
        truncation_distance_m=np.asarray(
            surface.metadata["truncation_distance_m"],
            dtype=np.float32,
        ),
        block_size_voxels=np.asarray(surface.metadata["block_size_voxels"], dtype=np.int32),
    )
    np.savez(
        output / "surface_points.npz",
        points_world_m=surface.points_world_m,
        confidence=surface.confidence,
        uncertainty_m=surface.uncertainty_m,
        voxel_indices_xyz=surface.voxel_indices_xyz,
    )
    _write_json(output / "metadata.json", surface.metadata)
    _write_json(output / "metrics.json", metrics)
    return output


def sparse_state_array_bytes(mapper: SparseBlockTSDFMapper, surface: TSDFSurface) -> int:
    """Return deterministic in-memory sparse state and final surface array bytes."""

    voxel_coords, tsdf, weight = mapper.active_voxel_arrays()
    return int(
        mapper.block_coordinates().nbytes
        + voxel_coords.nbytes
        + tsdf.nbytes
        + weight.nbytes
        + surface.points_world_m.nbytes
        + surface.confidence.nbytes
        + surface.uncertainty_m.nbytes
    )


def _write_json(path: Path, record: dict[str, Any] | npt.NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "sparse_state_array_bytes",
    "write_sparse_tsdf_outputs",
]
