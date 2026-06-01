"""Validation helpers for CPU TSDF output folder inspection."""

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
from atlas3r.mapping.world_map_sidecar import (
    REQUIRED_TRUTH_BOUNDARY_FLAGS,
    load_tsdf_world_map_sidecar,
)


def surface_record(
    *,
    metadata: dict[str, Any],
    point_count: int,
    bounds_summary: dict[str, list[float]],
    confidence_summary: dict[str, float],
    uncertainty_summary: dict[str, float],
    metadata_path: Path,
) -> dict[str, Any]:
    metadata_uncertainty = _dict_field(metadata, "uncertainty_summary_m", metadata_path)
    for field_name in ("mean", "p95"):
        _require_close(
            float(metadata_uncertainty[field_name]),
            uncertainty_summary[field_name],
            metadata_path,
            f"uncertainty_summary_m.{field_name}",
            "surface_points.npz uncertainty_m",
        )
    return {
        "artifact_type": str(_required(metadata, "artifact_type", metadata_path)),
        "coordinate_frame": _required_str(metadata, "coordinate_frame", metadata_path),
        "metric_scale_source": _required_str(metadata, "metric_scale_source", metadata_path),
        "source_frame_ids": _required_int_list(metadata, "source_frame_ids", metadata_path),
        "voxel_size_m": _required_positive_float(metadata, "voxel_size_m", metadata_path),
        "observed_coverage_estimate": _required_confidence(
            metadata,
            "observed_coverage_estimate",
            metadata_path,
        ),
        "surface_coverage_estimate": _optional_confidence(
            metadata,
            "surface_coverage_estimate",
            metadata_path,
        ),
        "point_count": point_count,
        "bounds_min_m": bounds_summary["bounds_min_m"],
        "bounds_max_m": bounds_summary["bounds_max_m"],
        "confidence_summary": confidence_summary,
        "uncertainty_summary_m": uncertainty_summary,
        "flags": [str(item) for item in metadata.get("flags", [])],
        "accuracy_report_path": metadata.get("accuracy_report_path"),
        "accuracy_note": str(
            metadata.get(
                "accuracy_note",
                "CPU TSDF surface inspection is not an accuracy report.",
            )
        ),
    }


def inspect_mesh_sidecar(
    path: Path,
    surface: dict[str, Any],
    surface_confidence: npt.NDArray[np.float32],
) -> dict[str, Any]:
    record = _read_json_object(path, missing_kind="MeshChunk sidecar")
    mesh = load_tsdf_surface_mesh_sidecar(path)
    metadata = _dict_field(record, "metadata", path)
    _validate_required_flags(mesh.flags, path, "mesh_chunk flags")
    _validate_required_flags([str(item) for item in metadata.get("flags", [])], path, "metadata")
    _cross_check_surface_fields(metadata, surface, path)
    _require_equal(
        mesh.scale_source, surface["metric_scale_source"], path, "mesh_chunk scale_source"
    )
    _require_equal(
        mesh.source_frame_ids, surface["source_frame_ids"], path, "mesh_chunk source_frame_ids"
    )
    _require_close(
        mesh.voxel_size_m,
        surface["voxel_size_m"],
        path,
        "mesh_chunk voxel_size_m",
        "metadata.json voxel_size_m",
    )
    _require_close(
        mesh.mean_uncertainty_m,
        surface["uncertainty_summary_m"]["mean"],
        path,
        "mesh_chunk mean_uncertainty_m",
        "metadata.json uncertainty_summary_m.mean",
    )
    _require_close(
        mesh.p95_uncertainty_m,
        surface["uncertainty_summary_m"]["p95"],
        path,
        "mesh_chunk p95_uncertainty_m",
        "metadata.json uncertainty_summary_m.p95",
    )
    _cross_check_mesh_confidence_summary(metadata, surface_confidence, path)
    return _mesh_inspection_record(record, mesh, metadata)


def inspect_world_map_sidecar(
    path: Path,
    surface: dict[str, Any],
    mesh_record: dict[str, Any] | None,
) -> dict[str, Any]:
    record = _read_json_object(path, missing_kind="WorldMap sidecar")
    world_map = load_tsdf_world_map_sidecar(path)
    metadata = _dict_field(record, "metadata", path)
    _validate_required_flags([str(item) for item in metadata.get("flags", [])], path, "metadata")
    for chunk_id, chunk in world_map.mesh_chunks.items():
        _validate_required_flags(chunk.flags, path, f"world_map mesh_chunk {chunk_id} flags")
    _cross_check_surface_fields(metadata, surface, path)
    _require_equal(
        world_map.world_frame_name,
        surface["coordinate_frame"],
        path,
        "world_map world_frame_name",
    )
    _require_equal(
        world_map.scale_source, surface["metric_scale_source"], path, "world_map scale_source"
    )
    _require_close(
        world_map.global_confidence,
        float(_dict_field(metadata, "confidence_summary", path)["mean"]),
        path,
        "world_map global_confidence",
        "metadata confidence_summary.mean",
    )
    _require_close(
        float(metadata["mean_uncertainty_m"]),
        surface["uncertainty_summary_m"]["mean"],
        path,
        "mean_uncertainty_m",
        "metadata.json uncertainty_summary_m.mean",
    )
    _require_close(
        float(metadata["p95_uncertainty_m"]),
        surface["uncertainty_summary_m"]["p95"],
        path,
        "p95_uncertainty_m",
        "metadata.json uncertainty_summary_m.p95",
    )
    if mesh_record is not None:
        _cross_check_world_map_mesh_record(metadata, mesh_record, path)
    return _world_map_inspection_record(record, world_map, metadata)


def metrics_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": "metrics.json", "present": False}
    metrics = _read_json_object(path, missing_kind="metrics artifact")
    if bool(metrics.get("accuracy_report", False)):
        raise ValueError(f"{path}: metrics artifact must not be marked as an accuracy report")
    return {
        "path": "metrics.json",
        "present": True,
        "metric_family": metrics.get("metric_family"),
        "accuracy_report": False,
    }


def surface_bounds_summary(surface_points: npt.NDArray[np.float32]) -> dict[str, list[float]]:
    values = surface_points.astype(np.float64, copy=False)
    return {
        "bounds_min_m": [float(value) for value in values.min(axis=0).tolist()],
        "bounds_max_m": [float(value) for value in values.max(axis=0).tolist()],
    }


def numeric_summary(array: npt.NDArray[np.float32]) -> dict[str, float]:
    values = array.astype(np.float64, copy=False)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def truth_boundary(flags: list[str], metrics: dict[str, Any]) -> dict[str, Any]:
    flag_text = " ".join(flags)
    return {
        "low_fidelity_reference_only": "low_fidelity_reference_only" in flags
        or "reference_only" in flag_text,
        "observed_only": "observed_surface_samples" in flags or "observed_surface_points" in flags,
        "not_completed": "not_completed_surface" in flags or "completed_surface" not in flags,
        "not_accuracy_report": "not_accuracy_report" in flags
        or not bool(metrics.get("accuracy_report", False)),
        "accuracy_report": False,
        "flags": flags,
    }


def validate_complete_truth_boundary(inspection: dict[str, Any], output_path: Path) -> None:
    truth = inspection["truth_boundary"]
    if not truth["low_fidelity_reference_only"]:
        raise ValueError(
            f"{output_path / 'metadata.json'}: missing low-fidelity reference boundary"
        )
    if not truth["observed_only"]:
        raise ValueError(f"{output_path / 'metadata.json'}: missing observed-only boundary")
    if not truth["not_completed"]:
        raise ValueError(
            f"{output_path / 'metadata.json'}: output must not mark completed geometry"
        )
    if not truth["not_accuracy_report"]:
        raise ValueError(f"{output_path / 'metadata.json'}: output must not be an accuracy report")


def artifact_presence(
    filename: str,
    record: dict[str, Any] | None,
    id_field: str,
) -> dict[str, Any]:
    if record is None:
        return {"path": filename, "present": False}
    return {"path": filename, "present": True, id_field: record[id_field]}


def stable_flags(flags: list[str]) -> list[str]:
    stable: list[str] = []
    for flag in flags:
        if flag not in stable:
            stable.append(flag)
    return stable


def _mesh_inspection_record(
    record: dict[str, Any],
    mesh: MeshChunk,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_name": record["format_name"],
        "format_version": record["format_version"],
        "chunk_id": mesh.chunk_id,
        "version": mesh.version,
        "vertex_count": int(mesh.vertices_m.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "coordinate_frame": metadata["coordinate_frame"],
        "source_frame_ids": metadata["source_frame_ids"],
        "metric_scale_source": metadata["metric_scale_source"],
        "voxel_size_m": metadata["voxel_size_m"],
        "observed_coverage_estimate": metadata["observed_coverage_estimate"],
        "surface_coverage_estimate": metadata.get("surface_coverage_estimate"),
        "source_surface_sample_count": metadata["source_surface_sample_count"],
        "emitted_surface_sample_count": metadata["emitted_surface_sample_count"],
        "confidence_summary": metadata["confidence_summary"],
        "mean_uncertainty_m": mesh.mean_uncertainty_m,
        "p95_uncertainty_m": mesh.p95_uncertainty_m,
        "uncertainty_summary_m": metadata["uncertainty_summary_m"],
        "flags": mesh.flags,
        "accuracy_report": False,
        "accuracy_note": metadata["accuracy_note"],
    }


def _world_map_inspection_record(
    record: dict[str, Any],
    world_map: WorldMap,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_name": record["format_name"],
        "format_version": record["format_version"],
        "map_id": world_map.map_id,
        "world_frame_name": world_map.world_frame_name,
        "mesh_chunk_ids": sorted(world_map.mesh_chunks),
        "mesh_chunk_count": len(world_map.mesh_chunks),
        "object_count": len(world_map.objects),
        "keyframe_count": len(world_map.keyframes),
        "coordinate_frame": metadata["coordinate_frame"],
        "source_frame_ids": metadata["source_frame_ids"],
        "metric_scale_source": metadata["metric_scale_source"],
        "voxel_size_m": metadata["voxel_size_m"],
        "observed_coverage_estimate": metadata["observed_coverage_estimate"],
        "surface_coverage_estimate": metadata.get("surface_coverage_estimate"),
        "global_confidence": world_map.global_confidence,
        "confidence_summary": metadata["confidence_summary"],
        "mean_uncertainty_m": metadata["mean_uncertainty_m"],
        "p95_uncertainty_m": metadata["p95_uncertainty_m"],
        "flags": metadata["flags"],
        "accuracy_report": False,
        "accuracy_note": metadata["accuracy_note"],
    }


def _cross_check_surface_fields(
    record: dict[str, Any], surface: dict[str, Any], path: Path
) -> None:
    for field_name in ("coordinate_frame", "metric_scale_source", "source_frame_ids"):
        _require_equal(record[field_name], surface[field_name], path, field_name)
    for field_name in ("voxel_size_m", "observed_coverage_estimate"):
        _require_close(
            float(record[field_name]), surface[field_name], path, field_name, "metadata.json"
        )
    if (
        surface["surface_coverage_estimate"] is not None
        and record.get("surface_coverage_estimate") is not None
    ):
        _require_close(
            float(record["surface_coverage_estimate"]),
            surface["surface_coverage_estimate"],
            path,
            "surface_coverage_estimate",
            "metadata.json",
        )


def _cross_check_mesh_confidence_summary(
    metadata: dict[str, Any],
    surface_confidence: npt.NDArray[np.float32],
    path: Path,
) -> None:
    source_count = int(metadata["source_surface_sample_count"])
    max_samples = int(metadata["max_surface_samples"])
    if source_count != int(surface_confidence.shape[0]):
        raise ValueError(f"{path}: source_surface_sample_count does not match surface_points.npz")
    sample_indices = _sample_indices(source_count, max_samples)
    expected = numeric_summary(surface_confidence[sample_indices])
    observed = _dict_field(metadata, "confidence_summary", path)
    for field_name, expected_value in expected.items():
        _require_close(
            float(observed[field_name]),
            expected_value,
            path,
            f"confidence_summary.{field_name}",
            "surface_points.npz confidence",
        )


def _cross_check_world_map_mesh_record(
    world_metadata: dict[str, Any],
    mesh_record: dict[str, Any],
    path: Path,
) -> None:
    source_metadata = _dict_field(world_metadata, "source_mesh_metadata", path)
    for field_name in (
        "coordinate_frame",
        "metric_scale_source",
        "source_frame_ids",
        "voxel_size_m",
        "observed_coverage_estimate",
        "confidence_summary",
        "mean_uncertainty_m",
        "p95_uncertainty_m",
    ):
        world_value = (
            source_metadata[field_name]
            if field_name in source_metadata
            else _required(world_metadata, field_name, path)
        )
        mesh_value = _required(mesh_record, field_name, path)
        if isinstance(world_value, float) or isinstance(mesh_value, float):
            _require_close(
                float(world_value), float(mesh_value), path, field_name, MESH_SIDECAR_FILENAME
            )
        else:
            _require_equal(world_value, mesh_value, path, field_name)
    _require_equal(
        world_metadata["mesh_chunk_ids"],
        [mesh_record["chunk_id"]],
        path,
        "mesh_chunk_ids",
    )


def _validate_required_flags(flags: list[str], path: Path, field_name: str) -> None:
    missing = [flag for flag in REQUIRED_TRUTH_BOUNDARY_FLAGS if flag not in flags]
    if missing:
        raise ValueError(f"{path}: {field_name} missing truth-boundary flags: {', '.join(missing)}")
    if "completed_surface" in flags:
        raise ValueError(f"{path}: {field_name} must not mark completed_surface as measured")


def _sample_indices(surface_count: int, max_surface_samples: int) -> npt.NDArray[np.int64]:
    if surface_count <= max_surface_samples:
        return np.arange(surface_count, dtype=np.int64)
    step = float(surface_count) / float(max_surface_samples)
    return cast(
        npt.NDArray[np.int64],
        np.floor((np.arange(max_surface_samples, dtype=np.float64) + 0.5) * step).astype(np.int64),
    )


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


def _optional_confidence(record: dict[str, Any], field_name: str, path: Path) -> float | None:
    if field_name not in record or record[field_name] is None:
        return None
    return _required_confidence(record, field_name, path)


def _require_equal(left: Any, right: Any, path: Path, field_name: str) -> None:
    if left != right:
        raise ValueError(f"{path}: {field_name} mismatch with metadata.json")


def _require_close(
    left: float,
    right: float,
    path: Path,
    field_name: str,
    expected_source: str,
) -> None:
    if not np.isclose(float(left), float(right), rtol=1e-6, atol=1e-6):
        raise ValueError(f"{path}: {field_name} mismatch with {expected_source}")


__all__ = [
    "artifact_presence",
    "inspect_mesh_sidecar",
    "inspect_world_map_sidecar",
    "metrics_record",
    "numeric_summary",
    "stable_flags",
    "surface_bounds_summary",
    "surface_record",
    "truth_boundary",
    "validate_complete_truth_boundary",
]
