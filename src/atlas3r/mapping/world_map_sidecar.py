"""Dependency-free WorldMap sidecars for CPU TSDF MeshChunk artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import MeshChunk, WorldMap
from atlas3r.mapping.mesh_sidecar import (
    MESH_SIDECAR_FILENAME,
    load_tsdf_surface_mesh_sidecar,
)

WORLD_MAP_SIDECAR_FORMAT_NAME = "atlas3r_tsdf_world_map_sidecar"
WORLD_MAP_SIDECAR_FORMAT_VERSION = 1
WORLD_MAP_SIDECAR_FILENAME = "world_map_sidecar.json"
DEFAULT_WORLD_MAP_ID = "phase_2a_cpu_tsdf_world_map_reference"
DETERMINISTIC_CREATED_AT_NS = 0
REQUIRED_TRUTH_BOUNDARY_FLAGS = (
    "low_fidelity_reference_only",
    "observed_surface_samples",
    "not_completed_surface",
    "not_accuracy_report",
)


def world_map_from_mesh_sidecar(
    mesh_sidecar_path: str | Path,
    *,
    map_id: str = DEFAULT_WORLD_MAP_ID,
    created_at_ns: int = DETERMINISTIC_CREATED_AT_NS,
) -> WorldMap:
    """Load a Phase 1E MeshChunk sidecar and wrap it in a contract-valid WorldMap."""
    build = _build_world_map_sidecar(
        Path(mesh_sidecar_path),
        map_id=map_id,
        created_at_ns=created_at_ns,
    )
    return build.world_map


def tsdf_world_map_sidecar_record(
    mesh_sidecar_path: str | Path,
    *,
    map_id: str = DEFAULT_WORLD_MAP_ID,
    created_at_ns: int = DETERMINISTIC_CREATED_AT_NS,
) -> dict[str, Any]:
    """Return the deterministic JSON record for a CPU TSDF WorldMap sidecar."""
    build = _build_world_map_sidecar(
        Path(mesh_sidecar_path),
        map_id=map_id,
        created_at_ns=created_at_ns,
    )
    return {
        "format_name": WORLD_MAP_SIDECAR_FORMAT_NAME,
        "format_version": WORLD_MAP_SIDECAR_FORMAT_VERSION,
        "world_map": _world_map_record(build.world_map),
        "metadata": build.metadata,
    }


def write_tsdf_world_map_sidecar_from_mesh_sidecar(
    mesh_sidecar_path: str | Path,
    world_map_sidecar_path: str | Path,
    *,
    map_id: str = DEFAULT_WORLD_MAP_ID,
    created_at_ns: int = DETERMINISTIC_CREATED_AT_NS,
) -> Path:
    """Write a deterministic WorldMap sidecar from an existing MeshChunk sidecar."""
    output_path = Path(world_map_sidecar_path)
    record = tsdf_world_map_sidecar_record(
        mesh_sidecar_path,
        map_id=map_id,
        created_at_ns=created_at_ns,
    )
    _write_json(output_path, record)
    return output_path


def write_tsdf_world_map_sidecar_from_artifacts(
    output_folder: str | Path,
    *,
    mesh_sidecar_name: str = MESH_SIDECAR_FILENAME,
    sidecar_name: str = WORLD_MAP_SIDECAR_FILENAME,
    map_id: str = DEFAULT_WORLD_MAP_ID,
    created_at_ns: int = DETERMINISTIC_CREATED_AT_NS,
) -> Path:
    """Load `<output>/mesh_chunk_sidecar.json` and write `world_map_sidecar.json`."""
    output_path = Path(output_folder)
    mesh_sidecar_path = output_path / mesh_sidecar_name
    if not mesh_sidecar_path.is_file():
        raise ValueError(f"{mesh_sidecar_path}: missing required MeshChunk sidecar")
    return write_tsdf_world_map_sidecar_from_mesh_sidecar(
        mesh_sidecar_path,
        output_path / sidecar_name,
        map_id=map_id,
        created_at_ns=created_at_ns,
    )


def load_tsdf_world_map_sidecar(path: str | Path) -> WorldMap:
    """Load and validate a deterministic TSDF WorldMap sidecar JSON file."""
    sidecar_path = Path(path)
    record = _read_json_object(sidecar_path, missing_kind="WorldMap sidecar")
    if record.get("format_name") != WORLD_MAP_SIDECAR_FORMAT_NAME:
        raise ValueError(f"{sidecar_path}: unexpected WorldMap sidecar format_name")
    if int(record.get("format_version", -1)) != WORLD_MAP_SIDECAR_FORMAT_VERSION:
        raise ValueError(f"{sidecar_path}: unsupported WorldMap sidecar format_version")
    world_map_record = _dict_field(record, "world_map", sidecar_path)
    metadata = _dict_field(record, "metadata", sidecar_path)
    try:
        world_map = _world_map_from_record(world_map_record, sidecar_path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sidecar_path}: malformed world_map sidecar: {exc}") from exc
    if metadata != world_map.metadata:
        raise ValueError(f"{sidecar_path}: top-level metadata must match world_map.metadata")
    return world_map


def format_world_map_sidecar_inspection(path: str | Path) -> str:
    """Validate a WorldMap sidecar and return deterministic inspection JSON."""
    sidecar_path = Path(path)
    world_map = load_tsdf_world_map_sidecar(sidecar_path)
    record = _read_json_object(sidecar_path, missing_kind="WorldMap sidecar")
    metadata = _dict_field(record, "metadata", sidecar_path)
    inspection = {
        "format_name": record["format_name"],
        "format_version": record["format_version"],
        "map_id": world_map.map_id,
        "world_frame_name": world_map.world_frame_name,
        "mesh_chunk_ids": sorted(world_map.mesh_chunks),
        "object_count": len(world_map.objects),
        "keyframe_count": len(world_map.keyframes),
        "scale_source": world_map.scale_source,
        "global_confidence": world_map.global_confidence,
        "coordinate_frame": metadata["coordinate_frame"],
        "source_frame_ids": metadata["source_frame_ids"],
        "voxel_size_m": metadata["voxel_size_m"],
        "observed_coverage_estimate": metadata["observed_coverage_estimate"],
        "mean_uncertainty_m": metadata["mean_uncertainty_m"],
        "p95_uncertainty_m": metadata["p95_uncertainty_m"],
        "flags": metadata["flags"],
        "accuracy_report": False,
        "accuracy_note": metadata["accuracy_note"],
    }
    return json.dumps(inspection, indent=2, sort_keys=True) + "\n"


class _WorldMapSidecarBuild:
    def __init__(self, *, world_map: WorldMap, metadata: dict[str, Any]) -> None:
        self.world_map = world_map
        self.metadata = metadata


def _build_world_map_sidecar(
    mesh_sidecar_path: Path,
    *,
    map_id: str,
    created_at_ns: int,
) -> _WorldMapSidecarBuild:
    source_record = _read_json_object(mesh_sidecar_path, missing_kind="MeshChunk sidecar")
    mesh_chunk = load_tsdf_surface_mesh_sidecar(mesh_sidecar_path)
    source_metadata = _dict_field(source_record, "metadata", mesh_sidecar_path)
    _validate_mesh_truth_flags(mesh_chunk, mesh_sidecar_path)
    map_metadata = _map_metadata(mesh_chunk, source_record, source_metadata, mesh_sidecar_path)
    world_map = WorldMap(
        map_id=map_id,
        world_frame_name=str(map_metadata["coordinate_frame"]),
        created_at_ns=created_at_ns,
        mesh_chunks={mesh_chunk.chunk_id: mesh_chunk},
        objects={},
        keyframes={},
        scale_source=mesh_chunk.scale_source,
        global_confidence=float(map_metadata["global_confidence"]),
        metadata=map_metadata,
    )
    return _WorldMapSidecarBuild(world_map=world_map, metadata=map_metadata)


def _map_metadata(
    mesh_chunk: MeshChunk,
    source_record: dict[str, Any],
    source_metadata: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    coordinate_frame = _required_str(source_metadata, "coordinate_frame", path)
    metric_scale_source = _required_str(source_metadata, "metric_scale_source", path)
    if metric_scale_source != mesh_chunk.scale_source:
        raise ValueError(f"{path}: mesh_chunk scale_source does not match metadata")
    source_frame_ids = _required_int_list(source_metadata, "source_frame_ids", path)
    if source_frame_ids != mesh_chunk.source_frame_ids:
        raise ValueError(f"{path}: mesh_chunk source_frame_ids do not match metadata")
    voxel_size_m = _required_positive_float(source_metadata, "voxel_size_m", path)
    if not np.isclose(voxel_size_m, mesh_chunk.voxel_size_m):
        raise ValueError(f"{path}: mesh_chunk voxel_size_m does not match metadata")
    observed_coverage = _required_confidence(source_metadata, "observed_coverage_estimate", path)
    global_confidence = _global_confidence(source_metadata, path)
    flags = _stable_flags(
        [
            "phase_2a_cpu_tsdf_world_map_sidecar",
            *mesh_chunk.flags,
            "world_map_assembly_only",
            "no_object_meshes",
        ]
    )
    return {
        "artifact_type": "phase_2a_cpu_tsdf_world_map_sidecar",
        "source_mesh_sidecar_file": path.name,
        "source_mesh_sidecar_format_name": source_record.get("format_name"),
        "source_mesh_sidecar_format_version": source_record.get("format_version"),
        "source_mesh_metadata": source_metadata,
        "mesh_chunk_ids": [mesh_chunk.chunk_id],
        "mesh_chunk_count": 1,
        "object_count": 0,
        "object_meshes_invented": False,
        "keyframe_count": 0,
        "coordinate_frame": coordinate_frame,
        "unit_scale": "meters",
        "metric_scale_source": metric_scale_source,
        "source_frame_ids": source_frame_ids,
        "voxel_size_m": voxel_size_m,
        "observed_coverage_estimate": observed_coverage,
        "surface_coverage_estimate": source_metadata.get("surface_coverage_estimate"),
        "global_confidence": global_confidence,
        "confidence_summary": _dict_field(source_metadata, "confidence_summary", path),
        "mean_uncertainty_m": mesh_chunk.mean_uncertainty_m,
        "p95_uncertainty_m": mesh_chunk.p95_uncertainty_m,
        "uncertainty_summary_m": source_metadata.get("uncertainty_summary_m"),
        "accuracy_report_path": None,
        "accuracy_note": (
            "This WorldMap sidecar wraps observed CPU TSDF MeshChunk smoke output for "
            "pipeline testing only; it is not an accuracy report and contains no "
            "completed or hidden geometry."
        ),
        "flags": flags,
    }


def _validate_mesh_truth_flags(mesh_chunk: MeshChunk, path: Path) -> None:
    missing = [flag for flag in REQUIRED_TRUTH_BOUNDARY_FLAGS if flag not in mesh_chunk.flags]
    if missing:
        raise ValueError(
            f"{path}: mesh_chunk flags missing truth-boundary flags: {', '.join(missing)}"
        )
    if "completed_surface" in mesh_chunk.flags:
        raise ValueError(f"{path}: mesh_chunk flags must not mark completed_surface as measured")


def _global_confidence(metadata: dict[str, Any], path: Path) -> float:
    confidence_summary = _dict_field(metadata, "confidence_summary", path)
    value = float(_required(confidence_summary, "mean", path))
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{path}: confidence_summary.mean must be in [0, 1]")
    return value


def _world_map_record(world_map: WorldMap) -> dict[str, Any]:
    return {
        "map_id": world_map.map_id,
        "world_frame_name": world_map.world_frame_name,
        "created_at_ns": world_map.created_at_ns,
        "mesh_chunks": {
            chunk_id: _mesh_chunk_record(chunk)
            for chunk_id, chunk in sorted(world_map.mesh_chunks.items())
        },
        "objects": {},
        "keyframes": {},
        "scale_source": world_map.scale_source,
        "global_confidence": world_map.global_confidence,
        "metadata": world_map.metadata,
    }


def _world_map_from_record(record: dict[str, Any], path: Path) -> WorldMap:
    mesh_chunk_records = _dict_field(record, "mesh_chunks", path)
    mesh_chunks = {
        str(chunk_id): _mesh_chunk_from_record(_dict_value(chunk_record, "mesh_chunks", path), path)
        for chunk_id, chunk_record in mesh_chunk_records.items()
    }
    objects = _dict_field(record, "objects", path)
    if objects:
        raise ValueError("objects must be empty for Phase 2A WorldMap sidecars")
    keyframes = _dict_field(record, "keyframes", path)
    if keyframes:
        raise ValueError("keyframes must be empty for Phase 2A WorldMap sidecars")
    return WorldMap(
        map_id=str(_required(record, "map_id", path)),
        world_frame_name=str(_required(record, "world_frame_name", path)),
        created_at_ns=int(_required(record, "created_at_ns", path)),
        mesh_chunks=mesh_chunks,
        objects={},
        keyframes={},
        scale_source=str(_required(record, "scale_source", path)),
        global_confidence=float(_required(record, "global_confidence", path)),
        metadata=_dict_field(record, "metadata", path),
    )


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


def _mesh_chunk_from_record(record: dict[str, Any], path: Path) -> MeshChunk:
    try:
        return MeshChunk(
            chunk_id=str(_required(record, "chunk_id", path)),
            version=int(_required(record, "version", path)),
            T_world_chunk=np.asarray(_required(record, "T_world_chunk", path), dtype=np.float32),
            vertices_m=np.asarray(_required(record, "vertices_m", path), dtype=np.float32),
            faces=np.asarray(_required(record, "faces", path), dtype=np.int32),
            normals=_optional_array(record, "normals", path, np.float32),
            colors=_optional_array(record, "colors", path, np.uint8),
            uvs=_optional_array(record, "uvs", path, np.float32),
            object_id_per_face=_optional_array(record, "object_id_per_face", path, np.int32),
            surface_source_per_face=_optional_array(
                record,
                "surface_source_per_face",
                path,
                np.int8,
            ),
            voxel_size_m=float(_required(record, "voxel_size_m", path)),
            mean_uncertainty_m=float(_required(record, "mean_uncertainty_m", path)),
            p95_uncertainty_m=float(_required(record, "p95_uncertainty_m", path)),
            source_frame_ids=[int(item) for item in _required(record, "source_frame_ids", path)],
            scale_source=str(_required(record, "scale_source", path)),
            flags=[str(item) for item in _required(record, "flags", path)],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed MeshChunk in WorldMap sidecar: {exc}") from exc


def _read_json_object(path: Path, *, missing_kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required {missing_kind}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], data)


def _required(record: dict[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in record:
        raise ValueError(f"{path}: missing field {field_name}")
    return record[field_name]


def _dict_field(record: dict[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    value = _required(record, field_name, path)
    return _dict_value(value, field_name, path)


def _dict_value(value: Any, field_name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: field {field_name} must be an object")
    return cast(dict[str, Any], value)


def _required_str(record: dict[str, Any], field_name: str, path: Path) -> str:
    value = _required(record, field_name, path)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: field {field_name} must be a non-empty string")
    return value


def _required_int_list(record: dict[str, Any], field_name: str, path: Path) -> list[int]:
    value = _required(record, field_name, path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: field {field_name} must be a non-empty list")
    return [int(item) for item in value]


def _required_positive_float(record: dict[str, Any], field_name: str, path: Path) -> float:
    value = float(_required(record, field_name, path))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{path}: field {field_name} must be positive")
    return value


def _required_confidence(record: dict[str, Any], field_name: str, path: Path) -> float:
    value = float(_required(record, field_name, path))
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{path}: field {field_name} must be in [0, 1]")
    return value


def _optional_array(
    record: dict[str, Any],
    field_name: str,
    path: Path,
    dtype: npt.DTypeLike,
) -> npt.NDArray[Any] | None:
    value = _required(record, field_name, path)
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _stable_flags(flags: list[str]) -> list[str]:
    stable: list[str] = []
    for flag in flags:
        if flag not in stable:
            stable.append(flag)
    return stable


def _json_array(array: npt.NDArray[Any]) -> list[Any]:
    return cast(list[Any], array.tolist())


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "DEFAULT_WORLD_MAP_ID",
    "DETERMINISTIC_CREATED_AT_NS",
    "REQUIRED_TRUTH_BOUNDARY_FLAGS",
    "WORLD_MAP_SIDECAR_FILENAME",
    "WORLD_MAP_SIDECAR_FORMAT_NAME",
    "WORLD_MAP_SIDECAR_FORMAT_VERSION",
    "format_world_map_sidecar_inspection",
    "load_tsdf_world_map_sidecar",
    "tsdf_world_map_sidecar_record",
    "world_map_from_mesh_sidecar",
    "write_tsdf_world_map_sidecar_from_artifacts",
    "write_tsdf_world_map_sidecar_from_mesh_sidecar",
]
