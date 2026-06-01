"""Teacher prediction cache read/write helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.api import FramePrediction
from atlas3r.api.validation import validate_mapping, validate_nonempty_str
from atlas3r.io._teacher_cache_validation import validate_cache_directory
from atlas3r.io.teacher_cache_schema import (
    CACHE_FORMAT_NAME,
    CACHE_FORMAT_VERSION,
    FRAME_SUMMARIES_PATH,
    OPTIONAL_ARRAY_KEYS,
)
from atlas3r.models.adapters.contracts import AdapterCapabilities, AdapterStatus, TeacherPrediction


@dataclass(frozen=True)
class LoadedTeacherPredictionCache:
    """Validated metadata and per-frame summaries from a teacher cache."""

    root: Path
    metadata: dict[str, Any]
    frame_summaries: tuple[dict[str, Any], ...]
    adapter_status: AdapterStatus

    @property
    def frame_count(self) -> int:
        return len(self.frame_summaries)


def write_teacher_prediction_cache(
    prediction: TeacherPrediction,
    output_dir: str | Path,
    adapter_status: AdapterStatus,
) -> tuple[Path, ...]:
    """Write deterministic teacher prediction metadata and frame summaries."""
    if adapter_status.name != prediction.adapter_name:
        raise ValueError("adapter_status.name: must match prediction.adapter_name")
    if adapter_status.capabilities != prediction.capabilities:
        raise ValueError("adapter_status.capabilities: must match prediction.capabilities")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame_predictions = tuple(
        sorted(prediction.frame_predictions, key=lambda item: item.pose.frame_id)
    )
    coordinate_frame = validate_nonempty_str(
        "metadata.coordinate_frame",
        str(prediction.metadata.get("coordinate_frame", "world")),
    )
    frame_summaries = tuple(
        _frame_summary_from_prediction(frame_prediction, coordinate_frame)
        for frame_prediction in frame_predictions
    )
    metadata = _metadata_from_prediction(
        prediction,
        adapter_status,
        coordinate_frame=coordinate_frame,
        frame_summaries=frame_summaries,
    )

    metadata_path = root / "metadata.json"
    frame_summaries_path = root / FRAME_SUMMARIES_PATH
    _write_json(metadata_path, metadata)
    _write_jsonl(frame_summaries_path, frame_summaries)
    validate_teacher_prediction_cache(root)
    return (metadata_path, frame_summaries_path)


def load_teacher_prediction_cache(path: str | Path) -> LoadedTeacherPredictionCache:
    """Load and validate a teacher prediction cache without loading tensor arrays."""
    return validate_teacher_prediction_cache(path)


def validate_teacher_prediction_cache(path: str | Path) -> LoadedTeacherPredictionCache:
    """Validate cache metadata and frame summaries with path-named errors."""
    root, metadata, validated_summaries, adapter_status = validate_cache_directory(path)
    return LoadedTeacherPredictionCache(
        root=root,
        metadata=metadata,
        frame_summaries=validated_summaries,
        adapter_status=adapter_status,
    )


def _metadata_from_prediction(
    prediction: TeacherPrediction,
    adapter_status: AdapterStatus,
    *,
    coordinate_frame: str,
    frame_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    frame_ids = [int(summary["frame_id"]) for summary in frame_summaries]
    scale_sources = sorted({str(summary["scale_source"]) for summary in frame_summaries})
    return {
        "format_name": CACHE_FORMAT_NAME,
        "format_version": CACHE_FORMAT_VERSION,
        "adapter": _adapter_status_to_record(adapter_status),
        "prediction_metadata": _json_safe_mapping("prediction_metadata", prediction.metadata),
        "coordinate_frame": coordinate_frame,
        "coordinate_convention": {
            "units": "meters",
            "camera_frame": "x_right_y_down_z_forward",
            "transform": "T_world_camera maps camera points into world",
        },
        "frame_count": len(frame_summaries),
        "frame_ids": frame_ids,
        "scale_sources": scale_sources,
        "frame_summaries_path": FRAME_SUMMARIES_PATH,
        "arrays": {
            "stored": False,
            "directory": None,
            "optional_npz_keys": list(OPTIONAL_ARRAY_KEYS),
        },
    }


def _frame_summary_from_prediction(
    frame_prediction: FramePrediction,
    coordinate_frame: str,
) -> dict[str, Any]:
    pose = frame_prediction.pose
    camera = frame_prediction.camera
    return {
        "frame_id": pose.frame_id,
        "timestamp_ns": pose.timestamp_ns,
        "coordinate_frame": coordinate_frame,
        "scale_source": pose.scale_source,
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "confidence": camera.confidence,
            "source": camera.source,
        },
        "pose": {
            "tracking_state": pose.tracking_state,
            "confidence": pose.confidence,
            "scale_source": pose.scale_source,
            "has_covariance_6x6": pose.covariance_6x6 is not None,
            "uncertainty": _pose_uncertainty_summary(pose.covariance_6x6),
        },
        "confidence_summary": _numeric_summary(frame_prediction.confidence),
        "uncertainty_summary": _numeric_summary(frame_prediction.depth_sigma_m),
        "depth_summary": _numeric_summary(frame_prediction.depth_m),
        "tensor_shapes": {
            "depth_m": _shape_dtype(frame_prediction.depth_m),
            "depth_sigma_m": _shape_dtype(frame_prediction.depth_sigma_m),
            "normal_camera": _shape_dtype(frame_prediction.normal_camera),
            "point_world": _shape_dtype(frame_prediction.point_world),
            "confidence": _shape_dtype(frame_prediction.confidence),
            "static_mask": _shape_dtype(frame_prediction.static_mask),
            "object_embeddings": _optional_shape_dtype(frame_prediction.object_embeddings),
            "object_mask_logits": _optional_shape_dtype(frame_prediction.object_mask_logits),
        },
        "dense_matches": _dense_match_summary(frame_prediction),
        "arrays_path": None,
    }


def _adapter_status_to_record(status: AdapterStatus) -> dict[str, Any]:
    return {
        "name": status.name,
        "display_name": status.display_name,
        "availability": status.availability,
        "capabilities": _capabilities_to_record(status.capabilities),
        "install_hint": status.install_hint,
        "reason": status.reason,
    }


def _capabilities_to_record(capabilities: AdapterCapabilities) -> dict[str, Any]:
    return {
        "predicts_camera": capabilities.predicts_camera,
        "predicts_pose": capabilities.predicts_pose,
        "predicts_depth": capabilities.predicts_depth,
        "predicts_normals": capabilities.predicts_normals,
        "predicts_points": capabilities.predicts_points,
        "predicts_dense_matches": capabilities.predicts_dense_matches,
        "predicts_objects": capabilities.predicts_objects,
        "supports_batch": capabilities.supports_batch,
        "supports_streaming": capabilities.supports_streaming,
        "notes": list(capabilities.notes),
    }


def _pose_uncertainty_summary(covariance_6x6: npt.NDArray[Any] | None) -> dict[str, Any]:
    if covariance_6x6 is None:
        return {
            "covariance_6x6_present": False,
            "translation_std_m": None,
            "rotation_std": None,
            "max_std": None,
        }
    diagonal = np.diag(covariance_6x6.astype(np.float64, copy=False))
    std = np.sqrt(np.maximum(diagonal, 0.0))
    return {
        "covariance_6x6_present": True,
        "translation_std_m": [float(value) for value in std[:3]],
        "rotation_std": [float(value) for value in std[3:]],
        "max_std": float(np.max(std)),
    }


def _dense_match_summary(frame_prediction: FramePrediction) -> dict[str, Any] | None:
    dense_matches = frame_prediction.dense_matches
    if dense_matches is None:
        return None
    return {
        "source_frame_id": dense_matches.source_frame_id,
        "target_frame_id": dense_matches.target_frame_id,
        "match_count": int(dense_matches.confidence.shape[0]),
        "confidence_summary": _numeric_summary(dense_matches.confidence),
    }


def _numeric_summary(values: npt.NDArray[Any]) -> dict[str, Any]:
    array = np.asarray(values)
    flat = array.astype(np.float64, copy=False).reshape(-1)
    if flat.size == 0:
        return {
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "count": int(flat.size),
        "min": float(np.min(flat)),
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


def _shape_dtype(values: npt.NDArray[Any]) -> dict[str, Any]:
    array = np.asarray(values)
    return {"shape": [int(dim) for dim in array.shape], "dtype": str(array.dtype)}


def _optional_shape_dtype(values: npt.NDArray[Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    return _shape_dtype(values)


def _json_safe_mapping(field_name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    validate_mapping(field_name, value)
    return {str(key): _json_safe(f"{field_name}.{key}", item) for key, item in value.items()}


def _json_safe(field_name: str, value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return _json_safe_mapping(field_name, value)
    if isinstance(value, tuple | list):
        return [_json_safe(f"{field_name}[]", item) for item in value]
    raise ValueError(f"{field_name}: must be JSON-serializable")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


__all__ = [
    "CACHE_FORMAT_NAME",
    "CACHE_FORMAT_VERSION",
    "FRAME_SUMMARIES_PATH",
    "LoadedTeacherPredictionCache",
    "load_teacher_prediction_cache",
    "validate_teacher_prediction_cache",
    "write_teacher_prediction_cache",
]
