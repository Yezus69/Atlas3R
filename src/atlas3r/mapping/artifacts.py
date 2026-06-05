"""Small artifact writers for inspectable offline map outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atlas3r.contracts.geometry import MapArtifact


def write_voxel_npz(
    path: str | Path,
    occupancy: NDArray[np.float32],
    uncertainty_m: NDArray[np.float32],
    artifact: MapArtifact,
) -> None:
    if occupancy.shape != uncertainty_m.shape:
        raise ValueError("occupancy and uncertainty_m shapes must match")
    if np.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("occupancy values must be in [0, 1]")
    if np.any(uncertainty_m < 0):
        raise ValueError("uncertainty_m must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        occupancy=occupancy.astype(np.float32),
        uncertainty_m=uncertainty_m.astype(np.float32),
        metadata_json=json.dumps(artifact.to_dict(), sort_keys=True),
    )


def read_npz_artifact_metadata(path: str | Path) -> Mapping[str, object]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata_json = str(payload["metadata_json"].item())
    metadata = json.loads(metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata must be a JSON object")
    return metadata


def write_mesh_ply(
    path: str | Path,
    vertices_world_m: NDArray[np.float32],
    triangles: NDArray[np.uint32],
    artifact: MapArtifact,
) -> None:
    if vertices_world_m.ndim != 2 or vertices_world_m.shape[1] != 3:
        raise ValueError("vertices_world_m must be shaped N,3")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be shaped M,3")
    if vertices_world_m.size and not np.all(np.isfinite(vertices_world_m)):
        raise ValueError("vertices_world_m must be finite")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(artifact.to_dict(), sort_keys=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment atlas3r_metadata_json {metadata_json}",
        f"element vertex {vertices_world_m.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {triangles.shape[0]}",
        "property list uchar uint vertex_indices",
        "end_header",
    ]
    lines.extend(f"{float(x):.8g} {float(y):.8g} {float(z):.8g}" for x, y, z in vertices_world_m)
    lines.extend(f"3 {int(a)} {int(b)} {int(c)}" for a, b, c in triangles)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
