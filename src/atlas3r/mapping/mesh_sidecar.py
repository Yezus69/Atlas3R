"""Dependency-free MeshChunk sidecars for CPU TSDF surface artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import MeshChunk, ScaleSource, SurfaceSource
from atlas3r.mapping.cpu_tsdf import FLOAT32, TSDFSurface

MESH_SIDECAR_FORMAT_NAME = "atlas3r_tsdf_surface_mesh_chunk_sidecar"
MESH_SIDECAR_FORMAT_VERSION = 1
MESH_SIDECAR_FILENAME = "mesh_chunk_sidecar.json"
DEFAULT_MAX_SURFACE_SAMPLES = 2048
_SCALE_SOURCE_VALUES = {item.value for item in ScaleSource}


def mesh_chunk_from_tsdf_surface(
    surface: TSDFSurface,
    *,
    chunk_id: str = "cpu_tsdf_surface_reference",
    version: int = 1,
    max_surface_samples: int = DEFAULT_MAX_SURFACE_SAMPLES,
) -> MeshChunk:
    """Build a contract-valid low-fidelity MeshChunk from observed TSDF samples."""
    build = _build_mesh_sidecar(surface, chunk_id, version, max_surface_samples)
    return build.mesh_chunk


def tsdf_surface_mesh_sidecar_record(
    surface: TSDFSurface,
    *,
    chunk_id: str = "cpu_tsdf_surface_reference",
    version: int = 1,
    max_surface_samples: int = DEFAULT_MAX_SURFACE_SAMPLES,
) -> dict[str, Any]:
    """Return the deterministic JSON record for a TSDF surface MeshChunk sidecar."""
    build = _build_mesh_sidecar(surface, chunk_id, version, max_surface_samples)
    return {
        "format_name": MESH_SIDECAR_FORMAT_NAME,
        "format_version": MESH_SIDECAR_FORMAT_VERSION,
        "mesh_chunk": _mesh_chunk_record(build.mesh_chunk),
        "metadata": build.metadata,
        "sample_attributes": build.sample_attributes,
    }


def write_tsdf_surface_mesh_sidecar(
    surface: TSDFSurface,
    path: str | Path,
    *,
    chunk_id: str = "cpu_tsdf_surface_reference",
    version: int = 1,
    max_surface_samples: int = DEFAULT_MAX_SURFACE_SAMPLES,
) -> Path:
    """Write a deterministic JSON MeshChunk sidecar for an in-memory TSDF surface."""
    sidecar_path = Path(path)
    record = tsdf_surface_mesh_sidecar_record(
        surface,
        chunk_id=chunk_id,
        version=version,
        max_surface_samples=max_surface_samples,
    )
    _write_json(sidecar_path, record)
    return sidecar_path


def write_tsdf_surface_mesh_sidecar_from_artifacts(
    output_folder: str | Path,
    *,
    sidecar_name: str = MESH_SIDECAR_FILENAME,
    chunk_id: str = "cpu_tsdf_surface_reference",
    version: int = 1,
    max_surface_samples: int = DEFAULT_MAX_SURFACE_SAMPLES,
) -> Path:
    """Load TSDF smoke artifacts and write a path-named MeshChunk sidecar."""
    output_path = Path(output_folder)
    surface = load_tsdf_surface_artifacts(output_path)
    return write_tsdf_surface_mesh_sidecar(
        surface,
        output_path / sidecar_name,
        chunk_id=chunk_id,
        version=version,
        max_surface_samples=max_surface_samples,
    )


def load_tsdf_surface_artifacts(output_folder: str | Path) -> TSDFSurface:
    """Load `surface_points.npz` and `metadata.json` as a validated TSDFSurface."""
    output_path = Path(output_folder)
    surface_path = output_path / "surface_points.npz"
    metadata_path = output_path / "metadata.json"
    metadata = _read_json_object(metadata_path)
    arrays = _read_surface_npz(surface_path)
    surface = TSDFSurface(
        points_world_m=arrays["points_world_m"],
        confidence=arrays["confidence"],
        uncertainty_m=arrays["uncertainty_m"],
        voxel_indices_xyz=arrays["voxel_indices_xyz"],
        metadata=metadata,
    )
    _validate_surface(surface, output_path)
    return surface


def load_tsdf_surface_mesh_sidecar(path: str | Path) -> MeshChunk:
    """Load and validate a TSDF surface MeshChunk sidecar JSON file."""
    sidecar_path = Path(path)
    record = _read_json_object(sidecar_path)
    if record.get("format_name") != MESH_SIDECAR_FORMAT_NAME:
        raise ValueError(f"{sidecar_path}: unexpected MeshChunk sidecar format_name")
    if int(record.get("format_version", -1)) != MESH_SIDECAR_FORMAT_VERSION:
        raise ValueError(f"{sidecar_path}: unsupported MeshChunk sidecar format_version")
    mesh_record = _dict_field(record, "mesh_chunk", sidecar_path)
    try:
        return _mesh_chunk_from_record(mesh_record)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sidecar_path}: malformed mesh_chunk sidecar: {exc}") from exc


class _MeshSidecarBuild:
    def __init__(
        self,
        *,
        mesh_chunk: MeshChunk,
        metadata: dict[str, Any],
        sample_attributes: dict[str, Any],
    ) -> None:
        self.mesh_chunk = mesh_chunk
        self.metadata = metadata
        self.sample_attributes = sample_attributes


def _build_mesh_sidecar(
    surface: TSDFSurface,
    chunk_id: str,
    version: int,
    max_surface_samples: int,
) -> _MeshSidecarBuild:
    _validate_surface(surface, Path("TSDFSurface"))
    if max_surface_samples <= 0:
        raise ValueError("max_surface_samples: must be positive")

    metadata = surface.metadata
    voxel_size_m = _metadata_float(metadata, "voxel_size_m", Path("TSDFSurface"))
    scale_source = str(_required_metadata(metadata, "metric_scale_source", Path("TSDFSurface")))
    if scale_source not in _SCALE_SOURCE_VALUES:
        raise ValueError(
            "TSDFSurface.metadata.metric_scale_source: must be a MeshChunk scale_source value"
        )
    source_frame_ids = _metadata_int_list(metadata, "source_frame_ids", Path("TSDFSurface"))
    sample_indices = _sample_indices(surface.points_world_m.shape[0], max_surface_samples)
    sampled_points = surface.points_world_m[sample_indices]
    vertices, faces = _triangle_markers(sampled_points, voxel_size_m)
    face_confidence = surface.confidence[sample_indices]
    face_uncertainty_m = surface.uncertainty_m[sample_indices]
    mean_uncertainty_m, p95_uncertainty_m = _mesh_uncertainty_values(surface)

    flags = [
        "phase_1e_cpu_tsdf_mesh_sidecar",
        "low_fidelity_reference_only",
        "observed_surface_samples",
        "not_completed_surface",
        "not_accuracy_report",
    ]
    mesh_chunk = MeshChunk(
        chunk_id=chunk_id,
        version=version,
        T_world_chunk=np.eye(4, dtype=FLOAT32),
        vertices_m=vertices,
        faces=faces,
        normals=None,
        colors=None,
        uvs=None,
        object_id_per_face=np.full(faces.shape[0], -1, dtype=np.int32),
        surface_source_per_face=np.full(
            faces.shape[0], int(SurfaceSource.OBSERVED_SURFACE), dtype=np.int8
        ),
        voxel_size_m=voxel_size_m,
        mean_uncertainty_m=mean_uncertainty_m,
        p95_uncertainty_m=p95_uncertainty_m,
        source_frame_ids=source_frame_ids,
        scale_source=scale_source,
        flags=flags,
    )
    sidecar_metadata = _sidecar_metadata(
        surface=surface,
        sample_indices=sample_indices,
        marker_edge_m=voxel_size_m * 0.32,
        max_surface_samples=max_surface_samples,
        flags=flags,
        mean_uncertainty_m=mean_uncertainty_m,
        p95_uncertainty_m=p95_uncertainty_m,
    )
    sample_attributes = {
        "source_surface_sample_indices": _json_array(sample_indices),
        "source_voxel_indices_xyz": _json_array(surface.voxel_indices_xyz[sample_indices]),
        "face_confidence": _json_array(face_confidence),
        "face_uncertainty_m": _json_array(face_uncertainty_m),
        "vertex_confidence": _json_array(np.repeat(face_confidence, 3).astype(FLOAT32)),
        "vertex_uncertainty_m": _json_array(np.repeat(face_uncertainty_m, 3).astype(FLOAT32)),
    }
    return _MeshSidecarBuild(
        mesh_chunk=mesh_chunk,
        metadata=sidecar_metadata,
        sample_attributes=sample_attributes,
    )


def _validate_surface(surface: TSDFSurface, location: Path) -> None:
    points = np.asarray(surface.points_world_m)
    confidence = np.asarray(surface.confidence)
    uncertainty = np.asarray(surface.uncertainty_m)
    voxel_indices = np.asarray(surface.voxel_indices_xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{location / 'surface_points.npz'}: points_world_m must have shape Nx3")
    if not np.issubdtype(points.dtype, np.number) or not np.all(np.isfinite(points)):
        raise ValueError(f"{location / 'surface_points.npz'}: points_world_m must be finite")
    if confidence.shape != (points.shape[0],):
        raise ValueError(
            f"{location / 'surface_points.npz'}: confidence must have length matching points"
        )
    if (
        not np.issubdtype(confidence.dtype, np.number)
        or not np.all(np.isfinite(confidence))
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise ValueError(
            f"{location / 'surface_points.npz'}: confidence must contain values in [0, 1]"
        )
    if uncertainty.shape != (points.shape[0],):
        raise ValueError(
            f"{location / 'surface_points.npz'}: uncertainty_m must have length matching points"
        )
    if (
        not np.issubdtype(uncertainty.dtype, np.number)
        or not np.all(np.isfinite(uncertainty))
        or np.any(uncertainty < 0.0)
    ):
        raise ValueError(f"{location / 'surface_points.npz'}: uncertainty_m must be non-negative")
    if voxel_indices.shape != points.shape or not np.issubdtype(voxel_indices.dtype, np.integer):
        raise ValueError(
            f"{location / 'surface_points.npz'}: voxel_indices_xyz must have shape Nx3"
        )
    metadata_path = location / "metadata.json"
    _required_metadata(surface.metadata, "coordinate_frame", metadata_path)
    scale_source = str(_required_metadata(surface.metadata, "metric_scale_source", metadata_path))
    if scale_source not in _SCALE_SOURCE_VALUES:
        raise ValueError(
            f"{metadata_path}: metadata field metric_scale_source must be a MeshChunk "
            "scale_source value"
        )
    _metadata_int_list(surface.metadata, "source_frame_ids", metadata_path)
    _metadata_float(surface.metadata, "voxel_size_m", metadata_path)
    observed_coverage = float(
        _required_metadata(surface.metadata, "observed_coverage_estimate", metadata_path)
    )
    if not np.isfinite(observed_coverage) or observed_coverage < 0.0 or observed_coverage > 1.0:
        raise ValueError(
            f"{metadata_path}: metadata field observed_coverage_estimate must be in [0, 1]"
        )


def _triangle_markers(
    points_world_m: npt.NDArray[np.float32],
    voxel_size_m: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    offsets = np.array(
        [
            [-0.16, -0.16, 0.0],
            [0.16, -0.16, 0.0],
            [-0.16, 0.16, 0.0],
        ],
        dtype=FLOAT32,
    )
    vertices = (points_world_m[:, None, :] + offsets[None, :, :] * voxel_size_m).reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)
    return vertices.astype(FLOAT32), faces


def _sample_indices(surface_count: int, max_surface_samples: int) -> npt.NDArray[np.int64]:
    if surface_count <= max_surface_samples:
        return np.arange(surface_count, dtype=np.int64)
    step = float(surface_count) / float(max_surface_samples)
    return cast(
        npt.NDArray[np.int64],
        np.floor((np.arange(max_surface_samples, dtype=np.float64) + 0.5) * step).astype(np.int64),
    )


def _mesh_uncertainty_values(surface: TSDFSurface) -> tuple[float, float]:
    summary = surface.metadata.get("uncertainty_summary_m")
    if isinstance(summary, dict) and "mean" in summary and "p95" in summary:
        return float(summary["mean"]), float(summary["p95"])
    uncertainty = surface.uncertainty_m.astype(np.float64, copy=False)
    return float(np.mean(uncertainty)), float(np.percentile(uncertainty, 95.0))


def _sidecar_metadata(
    *,
    surface: TSDFSurface,
    sample_indices: npt.NDArray[np.int64],
    marker_edge_m: float,
    max_surface_samples: int,
    flags: list[str],
    mean_uncertainty_m: float,
    p95_uncertainty_m: float,
) -> dict[str, Any]:
    metadata = surface.metadata
    emitted_confidence = surface.confidence[sample_indices]
    emitted_uncertainty = surface.uncertainty_m[sample_indices]
    return {
        "artifact_type": "phase_1e_cpu_tsdf_surface_mesh_chunk_sidecar",
        "source_surface_artifact_type": metadata.get("artifact_type"),
        "source_surface_metadata": metadata,
        "coordinate_frame": metadata["coordinate_frame"],
        "unit_scale": "meters",
        "metric_scale_source": metadata["metric_scale_source"],
        "source_frame_ids": metadata["source_frame_ids"],
        "voxel_size_m": metadata["voxel_size_m"],
        "observed_coverage_estimate": metadata["observed_coverage_estimate"],
        "surface_coverage_estimate": metadata.get("surface_coverage_estimate"),
        "source_surface_sample_count": int(surface.points_world_m.shape[0]),
        "emitted_surface_sample_count": int(sample_indices.shape[0]),
        "max_surface_samples": int(max_surface_samples),
        "triangle_marker_edge_m": float(marker_edge_m),
        "confidence_summary": _numeric_summary(emitted_confidence),
        "uncertainty_summary_m": {
            "mean": mean_uncertainty_m,
            "p95": p95_uncertainty_m,
            "emitted": _numeric_summary(emitted_uncertainty),
        },
        "accuracy_report_path": None,
        "accuracy_note": (
            "This MeshChunk sidecar is an observed-sample, low-fidelity reference artifact "
            "for smoke testing, not an accuracy report."
        ),
        "flags": flags,
    }


def _numeric_summary(array: npt.NDArray[np.float32]) -> dict[str, float]:
    values = array.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _read_surface_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required surface artifact")
    try:
        with np.load(path) as surface_file:
            arrays: dict[str, npt.NDArray[Any]] = {}
            for key in (
                "points_world_m",
                "confidence",
                "uncertainty_m",
                "voxel_indices_xyz",
            ):
                if key not in surface_file.files:
                    raise ValueError(f"{path}: missing required array {key}")
                arrays[key] = np.asarray(surface_file[key])
    except (OSError, ValueError) as exc:
        if str(exc).startswith(str(path)):
            raise
        raise ValueError(f"{path}: invalid surface artifact: {exc}") from exc
    return arrays


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required metadata artifact")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], data)


def _required_metadata(metadata: dict[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in metadata:
        raise ValueError(f"{path}: missing required metadata field {field_name}")
    return metadata[field_name]


def _metadata_float(metadata: dict[str, Any], field_name: str, path: Path) -> float:
    value = float(_required_metadata(metadata, field_name, path))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{path}: metadata field {field_name} must be positive")
    return value


def _metadata_int_list(metadata: dict[str, Any], field_name: str, path: Path) -> list[int]:
    value = _required_metadata(metadata, field_name, path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: metadata field {field_name} must be a non-empty list")
    return [int(item) for item in value]


def _mesh_chunk_record(chunk: MeshChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "version": chunk.version,
        "T_world_chunk": _json_array(chunk.T_world_chunk),
        "vertices_m": _json_array(chunk.vertices_m),
        "faces": _json_array(chunk.faces),
        "normals": _json_array(chunk.normals) if chunk.normals is not None else None,
        "colors": _json_array(chunk.colors) if chunk.colors is not None else None,
        "uvs": _json_array(chunk.uvs) if chunk.uvs is not None else None,
        "object_id_per_face": _json_array(chunk.object_id_per_face)
        if chunk.object_id_per_face is not None
        else None,
        "surface_source_per_face": _json_array(chunk.surface_source_per_face)
        if chunk.surface_source_per_face is not None
        else None,
        "voxel_size_m": chunk.voxel_size_m,
        "mean_uncertainty_m": chunk.mean_uncertainty_m,
        "p95_uncertainty_m": chunk.p95_uncertainty_m,
        "source_frame_ids": chunk.source_frame_ids,
        "scale_source": chunk.scale_source,
        "flags": chunk.flags,
    }


def _mesh_chunk_from_record(record: dict[str, Any]) -> MeshChunk:
    return MeshChunk(
        chunk_id=str(record["chunk_id"]),
        version=int(record["version"]),
        T_world_chunk=np.asarray(record["T_world_chunk"], dtype=FLOAT32),
        vertices_m=np.asarray(record["vertices_m"], dtype=FLOAT32),
        faces=np.asarray(record["faces"], dtype=np.int32),
        normals=_optional_array(record, "normals", FLOAT32),
        colors=_optional_array(record, "colors", np.uint8),
        uvs=_optional_array(record, "uvs", FLOAT32),
        object_id_per_face=_optional_array(record, "object_id_per_face", np.int32),
        surface_source_per_face=_optional_array(record, "surface_source_per_face", np.int8),
        voxel_size_m=float(record["voxel_size_m"]),
        mean_uncertainty_m=float(record["mean_uncertainty_m"]),
        p95_uncertainty_m=float(record["p95_uncertainty_m"]),
        source_frame_ids=[int(item) for item in record["source_frame_ids"]],
        scale_source=str(record["scale_source"]),
        flags=[str(item) for item in record["flags"]],
    )


def _dict_field(record: dict[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    if field_name not in record:
        raise ValueError(f"{path}: missing field {field_name}")
    value = record[field_name]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: field {field_name} must be an object")
    return cast(dict[str, Any], value)


def _optional_array(
    record: dict[str, Any], field_name: str, dtype: npt.DTypeLike
) -> npt.NDArray[Any] | None:
    value = record[field_name]
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _json_array(array: npt.NDArray[Any]) -> list[Any]:
    return cast(list[Any], array.tolist())


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "DEFAULT_MAX_SURFACE_SAMPLES",
    "MESH_SIDECAR_FILENAME",
    "MESH_SIDECAR_FORMAT_NAME",
    "MESH_SIDECAR_FORMAT_VERSION",
    "load_tsdf_surface_artifacts",
    "load_tsdf_surface_mesh_sidecar",
    "mesh_chunk_from_tsdf_surface",
    "tsdf_surface_mesh_sidecar_record",
    "write_tsdf_surface_mesh_sidecar",
    "write_tsdf_surface_mesh_sidecar_from_artifacts",
]
