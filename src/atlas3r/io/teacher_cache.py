"""Teacher prediction cache read/write helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api import FramePrediction
from atlas3r.api.validation import validate_mapping, validate_nonempty_str
from atlas3r.io._teacher_cache_validation import (
    load_validated_array_payload,
    validate_cache_directory,
)
from atlas3r.io.teacher_cache_schema import (
    ARRAYS_DIRECTORY,
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
    *,
    store_arrays: bool = False,
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
        _frame_summary_from_prediction(
            frame_prediction,
            coordinate_frame,
            arrays_path=_array_payload_relative_path(frame_prediction.pose.frame_id)
            if store_arrays
            else None,
        )
        for frame_prediction in frame_predictions
    )
    metadata = _metadata_from_prediction(
        prediction,
        adapter_status,
        coordinate_frame=coordinate_frame,
        frame_summaries=frame_summaries,
        store_arrays=store_arrays,
    )

    metadata_path = root / "metadata.json"
    frame_summaries_path = root / FRAME_SUMMARIES_PATH
    written_paths: list[Path] = [metadata_path, frame_summaries_path]
    if store_arrays:
        arrays_dir = root / ARRAYS_DIRECTORY
        arrays_dir.mkdir(parents=True, exist_ok=True)
        for frame_prediction in frame_predictions:
            payload_path = root / _array_payload_relative_path(frame_prediction.pose.frame_id)
            _write_array_payload(payload_path, frame_prediction)
            written_paths.append(payload_path)
    _write_json(metadata_path, metadata)
    _write_jsonl(frame_summaries_path, frame_summaries)
    validate_teacher_prediction_cache(root)
    return tuple(written_paths)


def load_teacher_prediction_cache(path: str | Path) -> LoadedTeacherPredictionCache:
    """Load and validate teacher cache metadata, summaries, and declared payloads."""
    return validate_teacher_prediction_cache(path)


def load_teacher_prediction_array_payload(
    path: str | Path,
    frame_id: int,
) -> dict[str, npt.NDArray[Any]]:
    """Load a validated full-array payload for one cached frame."""
    cache = validate_teacher_prediction_cache(path)
    for summary in cache.frame_summaries:
        if int(summary["frame_id"]) == int(frame_id):
            return load_validated_array_payload(cache.root, summary)
    raise ValueError(f"{cache.root / 'metadata.json'}: frame_id {frame_id} not found")


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
    store_arrays: bool,
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
            "stored": store_arrays,
            "directory": ARRAYS_DIRECTORY if store_arrays else None,
            "optional_npz_keys": list(OPTIONAL_ARRAY_KEYS),
        },
    }


def _frame_summary_from_prediction(
    frame_prediction: FramePrediction,
    coordinate_frame: str,
    *,
    arrays_path: str | None,
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
            "K": _json_array(camera.K),
            "distortion_model": camera.distortion_model,
            "distortion_params": _optional_json_array(camera.distortion_params),
            "rolling_shutter_row_time_s": camera.rolling_shutter_row_time_s,
            "confidence": camera.confidence,
            "source": camera.source,
        },
        "pose": {
            "T_world_camera": _json_array(pose.T_world_camera),
            "q_world_camera_xyzw": _json_array(pose.q_world_camera_xyzw),
            "camera_center_world_m": _json_array(pose.camera_center_world_m),
            "covariance_6x6": _optional_json_array(pose.covariance_6x6),
            "tracking_state": pose.tracking_state,
            "confidence": pose.confidence,
            "scale_source": pose.scale_source,
            "has_covariance_6x6": pose.covariance_6x6 is not None,
            "uncertainty": _pose_uncertainty_summary(pose.covariance_6x6),
            "diagnostics": _json_safe_mapping("pose.diagnostics", pose.diagnostics),
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
        "arrays_path": arrays_path,
    }


def _array_payload_relative_path(frame_id: int) -> str:
    return f"{ARRAYS_DIRECTORY}/frame_{frame_id:06d}.npz"


def _write_array_payload(path: Path, frame_prediction: FramePrediction) -> None:
    payload = _array_payload_from_prediction(frame_prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _array_payload_from_prediction(
    frame_prediction: FramePrediction,
) -> dict[str, npt.NDArray[Any]]:
    payload: dict[str, npt.NDArray[Any]] = {
        "depth_m": np.asarray(frame_prediction.depth_m),
        "depth_sigma_m": np.asarray(frame_prediction.depth_sigma_m),
        "normal_camera": np.asarray(frame_prediction.normal_camera),
        "point_world": np.asarray(frame_prediction.point_world),
        "confidence": np.asarray(frame_prediction.confidence),
        "static_mask": np.asarray(frame_prediction.static_mask),
    }
    if frame_prediction.object_embeddings is not None:
        payload["object_embeddings"] = np.asarray(frame_prediction.object_embeddings)
    if frame_prediction.object_mask_logits is not None:
        payload["object_mask_logits"] = np.asarray(frame_prediction.object_mask_logits)
    _validate_array_payload_for_write(payload, frame_prediction)
    return payload


def _validate_array_payload_for_write(
    payload: Mapping[str, npt.NDArray[Any]],
    frame_prediction: FramePrediction,
) -> None:
    height, width = frame_prediction.camera.height, frame_prediction.camera.width
    expected_shapes = {
        "depth_m": (height, width),
        "depth_sigma_m": (height, width),
        "normal_camera": (height, width, 3),
        "point_world": (height, width, 3),
        "confidence": (height, width),
        "static_mask": (height, width),
    }
    for key, expected_shape in expected_shapes.items():
        array = payload[key]
        if array.shape != expected_shape:
            raise ValueError(f"{key}: payload shape {array.shape} must be {expected_shape}")
        _validate_finite_payload_array(key, array)
    if np.any(np.asarray(payload["depth_m"]) < 0.0):
        raise ValueError("depth_m: payload values must be non-negative")
    if np.any(np.asarray(payload["depth_sigma_m"]) < 0.0):
        raise ValueError("depth_sigma_m: payload values must be non-negative")
    confidence = np.asarray(payload["confidence"])
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence: payload values must be in [0, 1]")
    static_mask = np.asarray(payload["static_mask"])
    if static_mask.dtype != np.bool_ and np.any((static_mask < 0.0) | (static_mask > 1.0)):
        raise ValueError("static_mask: probability payload values must be in [0, 1]")
    for key in ("object_embeddings", "object_mask_logits"):
        if key not in payload:
            continue
        array = payload[key]
        if array.ndim < 2 or array.shape[:2] != (height, width):
            raise ValueError(f"{key}: payload first dimensions must be HxW")
        _validate_finite_payload_array(key, array)


def _validate_finite_payload_array(field_name: str, array: npt.NDArray[Any]) -> None:
    if array.dtype == np.bool_:
        if field_name == "static_mask":
            return
        raise ValueError(f"{field_name}: payload dtype must be numeric")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field_name}: payload dtype must be numeric or bool")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name}: payload values must be finite")


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


def _json_array(values: npt.NDArray[Any]) -> list[Any]:
    return cast(list[Any], np.asarray(values).tolist())


def _optional_json_array(values: npt.NDArray[Any] | None) -> list[Any] | None:
    if values is None:
        return None
    return _json_array(values)


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
    "load_teacher_prediction_array_payload",
    "load_teacher_prediction_cache",
    "validate_teacher_prediction_cache",
    "write_teacher_prediction_cache",
]
