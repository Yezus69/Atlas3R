"""Optional real triangle-mesh extraction from a CPU TSDF volume."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.cpu_tsdf import TSDFSurface, TSDFVolume

MESH_INSTALL_HINT = "Install optional mesh support with: python -m pip install -e .[mesh]"


def mesh_dependency_available() -> bool:
    """Return whether scikit-image marching cubes can be imported."""

    return importlib.util.find_spec("skimage") is not None


def export_tsdf_triangle_mesh(
    output_dir: str | Path,
    *,
    volume: TSDFVolume,
    surface: TSDFSurface,
    export_mode: str,
    force_dependency_missing: bool = False,
) -> dict[str, object]:
    """Export a real marching-cubes triangle mesh when requested and available."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if export_mode not in {"off", "auto", "required"}:
        raise ValueError("export_mode: must be off, auto, or required")
    if export_mode == "off":
        return _write_status(
            output,
            {
                "export_mode": export_mode,
                "format_name": "atlas3r_phase6a_mesh_status",
                "format_version": 1,
                "mesh_exported": False,
                "reason": "mesh export disabled",
            },
        )
    if force_dependency_missing or not mesh_dependency_available():
        status = {
            "export_mode": export_mode,
            "format_name": "atlas3r_phase6a_mesh_status",
            "format_version": 1,
            "install_hint": MESH_INSTALL_HINT,
            "mesh_exported": False,
            "reason": "optional dependency scikit-image is not available",
        }
        if export_mode == "required":
            _write_status(output, status)
            raise ValueError(status["reason"])
        return _write_status(output, status)
    vertices, faces, normals = _marching_cubes(volume)
    mesh_path = output / "mesh.obj"
    _write_obj(mesh_path, vertices, faces, normals)
    return _write_status(
        output,
        {
            "export_mode": export_mode,
            "face_count": int(faces.shape[0]),
            "format_name": "atlas3r_phase6a_mesh_status",
            "format_version": 1,
            "mesh_exported": True,
            "mesh_path": mesh_path.name,
            "mesh_type": "triangle_mesh_obj",
            "source_frame_ids": list(volume.source_frame_ids),
            "surface_point_count": int(surface.points_world_m.shape[0]),
            "vertex_count": int(vertices.shape[0]),
            "voxel_size_m": volume.voxel_size_m,
        },
    )


def _marching_cubes(
    volume: TSDFVolume,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    measure = importlib.import_module("skimage.measure")
    mask = volume.weight > 0.0
    if not np.any(mask):
        raise ValueError("TSDF mesh extraction requires observed voxels")
    verts, faces, normals, _values = measure.marching_cubes(
        volume.tsdf.astype(np.float32, copy=False),
        level=0.0,
        spacing=(volume.voxel_size_m, volume.voxel_size_m, volume.voxel_size_m),
        mask=mask,
    )
    vertices_world = verts.astype(np.float32, copy=False)
    vertices_world += volume.grid_min_corner_world_m[np.newaxis, :]
    vertices_world += np.float32(0.5 * volume.voxel_size_m)
    return (
        cast(npt.NDArray[np.float32], vertices_world.astype(np.float32, copy=False)),
        cast(npt.NDArray[np.int32], faces.astype(np.int32, copy=False)),
        cast(npt.NDArray[np.float32], normals.astype(np.float32, copy=False)),
    )


def _write_obj(
    path: Path,
    vertices: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
    normals: npt.NDArray[np.float32],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Atlas3R Phase 6A diagnostic marching-cubes mesh\n")
        for vertex in vertices:
            handle.write(
                f"v {float(vertex[0]):.7g} {float(vertex[1]):.7g} {float(vertex[2]):.7g}\n"
            )
        for normal in normals:
            handle.write(
                f"vn {float(normal[0]):.7g} {float(normal[1]):.7g} {float(normal[2]):.7g}\n"
            )
        for face in faces:
            a, b, c = (int(index) + 1 for index in face)
            handle.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")


def _write_status(output: Path, status: dict[str, Any]) -> dict[str, object]:
    with (output / "mesh_status.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return dict(status)


__all__ = [
    "MESH_INSTALL_HINT",
    "export_tsdf_triangle_mesh",
    "mesh_dependency_available",
]
