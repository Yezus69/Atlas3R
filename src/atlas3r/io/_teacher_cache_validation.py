"""On-disk validation helpers for teacher prediction caches."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from atlas3r.api.contracts import SCALE_SOURCE_VALUES
from atlas3r.api.validation import (
    validate_choice,
    validate_confidence,
    validate_non_negative_int,
    validate_non_negative_scalar,
    validate_nonempty_str,
    validate_positive_int,
)
from atlas3r.io.teacher_cache_schema import (
    CACHE_FORMAT_NAME,
    CACHE_FORMAT_VERSION,
    OPTIONAL_ARRAY_KEYS,
)
from atlas3r.models.adapters.contracts import AdapterCapabilities, AdapterStatus


def validate_cache_directory(
    path: str | Path,
) -> tuple[Path, dict[str, Any], tuple[dict[str, Any], ...], AdapterStatus]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"{root}: expected a teacher prediction cache directory")
    metadata_path = root / "metadata.json"
    metadata = _read_json_object(metadata_path)
    adapter_status = _validate_metadata(metadata, metadata_path)
    frame_summaries_path = root / str(_required(metadata, "frame_summaries_path", metadata_path))
    frame_summaries = _read_jsonl(frame_summaries_path)
    validated_summaries = tuple(
        _validate_frame_summary(record, frame_summaries_path, line_number)
        for line_number, record in enumerate(frame_summaries, start=1)
    )
    _validate_cache_cross_refs(metadata, validated_summaries, metadata_path)
    return root, metadata, validated_summaries, adapter_status


def _validate_metadata(metadata: dict[str, Any], path: Path) -> AdapterStatus:
    if _required(metadata, "format_name", path) != CACHE_FORMAT_NAME:
        raise ValueError(f"{path}: format_name must be {CACHE_FORMAT_NAME}")
    if int(_required(metadata, "format_version", path)) != CACHE_FORMAT_VERSION:
        raise ValueError(f"{path}: format_version must be {CACHE_FORMAT_VERSION}")
    validate_nonempty_str("coordinate_frame", str(_required(metadata, "coordinate_frame", path)))
    validate_positive_int("frame_count", int(_required(metadata, "frame_count", path)))
    _int_list(metadata, "frame_ids", path)
    for scale_source in _str_list(metadata, "scale_sources", path):
        validate_choice("scale_sources", scale_source, SCALE_SOURCE_VALUES)
    validate_nonempty_str(
        "frame_summaries_path", str(_required(metadata, "frame_summaries_path", path))
    )
    _validate_arrays_metadata(_dict_field(metadata, "arrays", path), path)
    return _adapter_status_from_record(_dict_field(metadata, "adapter", path), path)


def _validate_arrays_metadata(record: Mapping[str, Any], path: Path) -> None:
    _bool_field(record, "stored", path)
    if _required(record, "directory", path) is not None:
        validate_nonempty_str("arrays.directory", str(_required(record, "directory", path)))
    keys = _str_list(record, "optional_npz_keys", path)
    missing = set(OPTIONAL_ARRAY_KEYS) - set(keys)
    if missing:
        raise ValueError(f"{path}: arrays.optional_npz_keys missing {sorted(missing)}")


def _validate_frame_summary(
    record: dict[str, Any],
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    location = Path(f"{path}:{line_number}")
    validate_non_negative_int("frame_id", int(_required(record, "frame_id", location)))
    validate_non_negative_int("timestamp_ns", int(_required(record, "timestamp_ns", location)))
    validate_nonempty_str("coordinate_frame", str(_required(record, "coordinate_frame", location)))
    validate_choice(
        "scale_source", _required(record, "scale_source", location), SCALE_SOURCE_VALUES
    )
    _validate_camera_summary(_dict_field(record, "camera", location), location)
    _validate_pose_summary(_dict_field(record, "pose", location), location)
    _validate_numeric_summary(
        _dict_field(record, "confidence_summary", location),
        location,
        field_name="confidence_summary",
        confidence=True,
    )
    _validate_numeric_summary(
        _dict_field(record, "uncertainty_summary", location),
        location,
        field_name="uncertainty_summary",
        non_negative=True,
    )
    _validate_numeric_summary(
        _dict_field(record, "depth_summary", location),
        location,
        field_name="depth_summary",
        non_negative=True,
    )
    _dict_field(record, "tensor_shapes", location)
    dense_matches = _required(record, "dense_matches", location)
    if dense_matches is not None and not isinstance(dense_matches, dict):
        raise ValueError(f"{location}: dense_matches must be an object or null")
    _required(record, "arrays_path", location)
    return record


def _validate_camera_summary(record: Mapping[str, Any], path: Path) -> None:
    validate_positive_int("camera.width", int(_required(record, "width", path)))
    validate_positive_int("camera.height", int(_required(record, "height", path)))
    validate_confidence("camera.confidence", float(_required(record, "confidence", path)))
    validate_nonempty_str("camera.source", str(_required(record, "source", path)))


def _validate_pose_summary(record: Mapping[str, Any], path: Path) -> None:
    validate_confidence("pose.confidence", float(_required(record, "confidence", path)))
    validate_choice(
        "pose.scale_source", _required(record, "scale_source", path), SCALE_SOURCE_VALUES
    )
    validate_nonempty_str("pose.tracking_state", str(_required(record, "tracking_state", path)))
    _bool_field(record, "has_covariance_6x6", path)
    uncertainty = _dict_field(record, "uncertainty", path)
    _bool_field(uncertainty, "covariance_6x6_present", path)


def _validate_numeric_summary(
    record: Mapping[str, Any],
    path: Path,
    *,
    field_name: str,
    confidence: bool = False,
    non_negative: bool = False,
) -> None:
    count = validate_non_negative_int(f"{field_name}.count", int(_required(record, "count", path)))
    _int_list(record, "shape", path)
    validate_nonempty_str(f"{field_name}.dtype", str(_required(record, "dtype", path)))
    for key in ("min", "mean", "p50", "p95", "max"):
        value = _required(record, key, path)
        if count == 0 and value is None:
            continue
        scalar = float(value)
        if confidence:
            validate_confidence(f"{field_name}.{key}", scalar)
        elif non_negative:
            validate_non_negative_scalar(f"{field_name}.{key}", scalar)


def _validate_cache_cross_refs(
    metadata: dict[str, Any],
    frame_summaries: Sequence[dict[str, Any]],
    metadata_path: Path,
) -> None:
    frame_count = int(_required(metadata, "frame_count", metadata_path))
    if frame_count != len(frame_summaries):
        raise ValueError(
            f"{metadata_path}: frame_count={frame_count} does not match "
            f"{len(frame_summaries)} frame summaries"
        )
    expected_frame_ids = _int_list(metadata, "frame_ids", metadata_path)
    actual_frame_ids = [int(summary["frame_id"]) for summary in frame_summaries]
    if expected_frame_ids != actual_frame_ids:
        raise ValueError(f"{metadata_path}: frame_ids do not match frame_summaries")
    coordinate_frame = str(_required(metadata, "coordinate_frame", metadata_path))
    for summary in frame_summaries:
        if summary["coordinate_frame"] != coordinate_frame:
            raise ValueError(f"{metadata_path}: frame summary coordinate_frame mismatch")


def _adapter_status_from_record(record: Mapping[str, Any], path: Path) -> AdapterStatus:
    capabilities = _capabilities_from_record(_dict_field(record, "capabilities", path), path)
    return AdapterStatus(
        name=str(_required(record, "name", path)),
        display_name=str(_required(record, "display_name", path)),
        availability=str(_required(record, "availability", path)),
        capabilities=capabilities,
        install_hint=_optional_str(record, "install_hint", path),
        reason=_optional_str(record, "reason", path),
    )


def _capabilities_from_record(record: Mapping[str, Any], path: Path) -> AdapterCapabilities:
    return AdapterCapabilities(
        predicts_camera=_bool_field(record, "predicts_camera", path),
        predicts_pose=_bool_field(record, "predicts_pose", path),
        predicts_depth=_bool_field(record, "predicts_depth", path),
        predicts_normals=_bool_field(record, "predicts_normals", path),
        predicts_points=_bool_field(record, "predicts_points", path),
        predicts_dense_matches=_bool_field(record, "predicts_dense_matches", path),
        predicts_objects=_bool_field(record, "predicts_objects", path),
        supports_batch=_bool_field(record, "supports_batch", path),
        supports_streaming=_bool_field(record, "supports_streaming", path),
        notes=tuple(_str_list(record, "notes", path)),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], data)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ValueError(f"{path}: missing required file")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(cast(dict[str, Any], data))
    return tuple(records)


def _required(record: Mapping[str, Any], field_name: str, path: Path) -> Any:
    if field_name not in record:
        raise ValueError(f"{path}: missing field {field_name}")
    return record[field_name]


def _dict_field(record: Mapping[str, Any], field_name: str, path: Path) -> dict[str, Any]:
    value = _required(record, field_name, path)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field_name} must be an object")
    return cast(dict[str, Any], value)


def _optional_str(record: Mapping[str, Any], field_name: str, path: Path) -> str | None:
    value = _required(record, field_name, path)
    if value is None:
        return None
    return validate_nonempty_str(field_name, str(value))


def _bool_field(record: Mapping[str, Any], field_name: str, path: Path) -> bool:
    value = _required(record, field_name, path)
    if not isinstance(value, bool):
        raise ValueError(f"{path}: {field_name} must be a bool")
    return value


def _str_list(record: Mapping[str, Any], field_name: str, path: Path) -> list[str]:
    value = _required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {field_name} must contain strings")
    return cast(list[str], value)


def _int_list(record: Mapping[str, Any], field_name: str, path: Path) -> list[int]:
    value = _required(record, field_name, path)
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a list")
    return [int(item) for item in value]


__all__ = ["validate_cache_directory"]
