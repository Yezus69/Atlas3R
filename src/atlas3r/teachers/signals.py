"""Stable teacher-signal cache contracts and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics, validate_transform
from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
)
from atlas3r.forge.clip_cache import (
    resolve_payload_path as resolve_clip_payload_path,
)

TEACHER_SIGNAL_FORMAT_NAME = "atlas3r_teacher_signal_cache"
TEACHER_SIGNAL_FORMAT_VERSION = 1
TEACHER_SIGNAL_MANIFEST_FILENAME = "atlas3r_teacher_signal_manifest.json"

_REQUIRED_ARRAYS = {
    "depth_m",
    "depth_sigma_m",
    "confidence",
    "valid_mask",
    "K",
    "T_world_camera",
    "frame_ids",
    "timestamps_s",
}
_TRUTH_BOOLEAN_FIELDS = {
    "diagnostic_only": True,
    "accuracy_report": False,
    "performance_report": False,
}


def teacher_signal_manifest_path_from_input(path: str | Path) -> Path:
    """Resolve a teacher-signal manifest from either a manifest file or cache folder."""

    candidate = Path(path)
    if candidate.is_dir():
        return candidate / TEACHER_SIGNAL_MANIFEST_FILENAME
    return candidate


def load_teacher_signal_manifest(
    path: str | Path,
    *,
    clip_cache: str | Path | None = None,
    validate_payloads: bool = False,
) -> dict[str, object]:
    """Load and validate a teacher-signal manifest."""

    manifest_path = teacher_signal_manifest_path_from_input(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{manifest_path}: failed to read teacher-signal manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON teacher-signal manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: teacher-signal manifest must be a JSON object")
    manifest = dict(payload)
    source_clip_manifest = _load_source_clip_manifest(
        manifest,
        manifest_path.parent,
        clip_cache=clip_cache,
    )
    validate_teacher_signal_manifest(
        manifest,
        cache_root=manifest_path.parent,
        validate_payloads=validate_payloads,
        source_clip_manifest=source_clip_manifest,
    )
    return manifest


def validate_teacher_signal_manifest(
    manifest: Mapping[str, object],
    *,
    cache_root: str | Path,
    validate_payloads: bool,
    source_clip_manifest: Mapping[str, object] | None = None,
) -> None:
    """Validate teacher-signal manifest metadata and optionally payload arrays."""

    root = Path(cache_root)
    if manifest.get("format_name") != TEACHER_SIGNAL_FORMAT_NAME:
        raise ValueError(f"manifest.format_name: expected {TEACHER_SIGNAL_FORMAT_NAME!r}")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version != TEACHER_SIGNAL_FORMAT_VERSION:
        raise ValueError("manifest.format_version: unsupported teacher-signal version")
    clip_length = _positive_int(manifest, "clip_length")
    width = _positive_int(manifest, "image_width")
    height = _positive_int(manifest, "image_height")
    _required_string(manifest, "source_clip_cache_manifest_path")
    _required_string(manifest, "source_dataset_name")
    _required_string(manifest, "source_sequence_name")
    _required_string(manifest, "split")
    _required_string(manifest, "teacher_name")
    _required_string(manifest, "teacher_version")
    _required_string(manifest, "teacher_source_type")
    _validate_truth_boundary(manifest)
    _validate_source_metadata(manifest)

    signals_value = manifest.get("signals")
    if not isinstance(signals_value, list):
        raise ValueError("manifest.signals: must be a list")
    signal_count = _non_negative_int(manifest, "signal_count")
    if signal_count != len(signals_value):
        raise ValueError("manifest.signal_count: must match len(signals)")
    for index, signal_value in enumerate(signals_value):
        if not isinstance(signal_value, Mapping):
            raise ValueError(f"manifest.signals[{index}]: must be a mapping")
        _validate_signal_entry(
            signal_value,
            index=index,
            clip_length=clip_length,
            cache_root=root,
        )
        if validate_payloads:
            arrays = read_teacher_signal_payload_from_entry(root, signal_value)
            validate_teacher_signal_payload(
                arrays,
                clip_length=clip_length,
                height=height,
                width=width,
            )
            validate_payload_matches_signal_entry(arrays, signal_value, index=index)
    if source_clip_manifest is not None:
        validate_teacher_signal_matches_clip_cache(manifest, source_clip_manifest)


def validate_teacher_signal_matches_clip_cache(
    teacher_manifest: Mapping[str, object],
    clip_manifest: Mapping[str, object],
) -> None:
    """Validate teacher-signal metadata against the source clip-cache metadata."""

    for teacher_key, clip_key in (
        ("source_dataset_name", "source_dataset_name"),
        ("source_sequence_name", "source_sequence_name"),
        ("split", "split"),
        ("clip_length", "clip_length"),
        ("image_width", "image_width"),
        ("image_height", "image_height"),
    ):
        if teacher_manifest.get(teacher_key) != clip_manifest.get(clip_key):
            raise ValueError(f"manifest.{teacher_key}: must match source clip cache")

    clip_entries = clip_manifest.get("clips")
    teacher_entries = teacher_manifest.get("signals")
    if not isinstance(clip_entries, list) or not isinstance(teacher_entries, list):
        raise ValueError("source clip metadata: clips/signals must be lists")
    for index, teacher_entry_value in enumerate(teacher_entries):
        if not isinstance(teacher_entry_value, Mapping):
            raise ValueError(f"manifest.signals[{index}]: must be a mapping")
        source_clip_id = _non_negative_int(teacher_entry_value, "source_clip_id")
        if source_clip_id >= len(clip_entries):
            raise ValueError(f"manifest.signals[{index}].source_clip_id: out of range")
        clip_entry_value = clip_entries[source_clip_id]
        if not isinstance(clip_entry_value, Mapping):
            raise ValueError(f"source clip {source_clip_id}: must be a mapping")
        _compare_entry_metadata(
            teacher_entry_value,
            clip_entry_value,
            index=index,
            clip_length=_positive_int(teacher_manifest, "clip_length"),
        )


def read_teacher_signal_payload(
    teacher_manifest_or_dir: str | Path,
    *,
    signal_index: int = 0,
) -> dict[str, Any]:
    """Load a teacher-signal payload by index from a manifest file or cache folder."""

    manifest_path = teacher_signal_manifest_path_from_input(teacher_manifest_or_dir)
    manifest = load_teacher_signal_manifest(manifest_path)
    signals = cast(list[object], manifest["signals"])
    if signal_index < 0 or signal_index >= len(signals):
        raise ValueError("signal_index: out of range")
    entry = signals[signal_index]
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest.signals[{signal_index}]: must be a mapping")
    arrays = read_teacher_signal_payload_from_entry(manifest_path.parent, entry)
    validate_teacher_signal_payload(
        arrays,
        clip_length=_positive_int(manifest, "clip_length"),
        height=_positive_int(manifest, "image_height"),
        width=_positive_int(manifest, "image_width"),
    )
    validate_payload_matches_signal_entry(arrays, entry, index=signal_index)
    return arrays


def read_teacher_signal_payload_from_entry(
    cache_root: str | Path,
    signal_entry: Mapping[str, object],
) -> dict[str, Any]:
    """Load the `.npz` payload referenced by one teacher-signal entry."""

    path = resolve_signal_payload_path(cache_root, signal_entry.get("payload_path"))
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except OSError as exc:
        raise ValueError(f"{path}: failed to read teacher-signal payload: {exc}") from exc


def resolve_signal_payload_path(cache_root: str | Path, relative_path: object) -> Path:
    """Resolve a relative teacher-signal payload path after rejecting traversal."""

    return resolve_clip_payload_path(cache_root, relative_path)


def write_teacher_signal_payload(
    path: str | Path,
    payload: Mapping[str, npt.NDArray[Any] | np.generic],
    *,
    clip_length: int,
    height: int,
    width: int,
) -> None:
    """Validate and write one compressed teacher-signal payload."""

    arrays = {key: np.asarray(value) for key, value in payload.items()}
    validate_teacher_signal_payload(arrays, clip_length=clip_length, height=height, width=width)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def write_teacher_signal_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    """Write deterministic teacher-signal JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_teacher_signal_payload(
    payload: Mapping[str, Any],
    *,
    clip_length: int,
    height: int,
    width: int,
) -> None:
    """Validate one loaded teacher-signal payload."""

    missing = sorted(_REQUIRED_ARRAYS.difference(payload))
    if missing:
        raise ValueError(f"teacher-signal payload: missing required keys {missing}")
    depth = _array(payload, "depth_m", (clip_length, height, width), np.float32)
    if np.any(depth < 0.0):
        raise ValueError("depth_m: must be non-negative")
    sigma = _array(payload, "depth_sigma_m", (clip_length, height, width), np.float32)
    if np.any(sigma < 0.0):
        raise ValueError("depth_sigma_m: must be non-negative")
    confidence = _array(payload, "confidence", (clip_length, height, width), np.float32)
    _validate_probability("confidence", confidence)
    valid_mask = _array(payload, "valid_mask", (clip_length, height, width), np.bool_)
    if np.any(sigma[valid_mask] <= 0.0):
        raise ValueError("depth_sigma_m: must be positive on valid pixels")
    K = _array(payload, "K", (clip_length, 3, 3), np.float32)
    transforms = _array(payload, "T_world_camera", (clip_length, 4, 4), np.float32)
    for index in range(clip_length):
        validate_intrinsics(f"K[{index}]", K[index])
        validate_transform(f"T_world_camera[{index}]", transforms[index])
    frame_ids = np.asarray(payload["frame_ids"])
    if frame_ids.shape != (clip_length,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("frame_ids: must have shape T and integer dtype")
    _array(payload, "timestamps_s", (clip_length,), np.float64)

    for key in ("pointmap_camera_m", "pointmap_world_m", "normal_camera"):
        if key in payload:
            _array(payload, key, (clip_length, height, width, 3), np.float32)
    if "object_mask_ids" in payload:
        _array(payload, "object_mask_ids", (clip_length, height, width), np.int32)
    for key in ("object_confidence", "dynamic_probability"):
        if key in payload:
            _validate_probability(
                key, _array(payload, key, (clip_length, height, width), np.float32)
            )


def validate_payload_matches_signal_entry(
    payload: Mapping[str, Any],
    entry: Mapping[str, object],
    *,
    index: int,
) -> None:
    """Validate per-entry frame metadata against the payload arrays."""

    frame_ids = tuple(int(value) for value in np.asarray(payload["frame_ids"]).tolist())
    timestamps = tuple(float(value) for value in np.asarray(payload["timestamps_s"]).tolist())
    if frame_ids != tuple(_int_sequence(entry, "frame_ids", len(frame_ids))):
        raise ValueError(f"manifest.signals[{index}].frame_ids: must match payload")
    entry_timestamps = tuple(_float_sequence(entry, "timestamps_s", len(timestamps)))
    if not np.allclose(np.asarray(timestamps), np.asarray(entry_timestamps), rtol=0.0, atol=1e-9):
        raise ValueError(f"manifest.signals[{index}].timestamps_s: must match payload")


def _load_source_clip_manifest(
    manifest: Mapping[str, object],
    cache_root: Path,
    *,
    clip_cache: str | Path | None,
) -> dict[str, object]:
    if clip_cache is not None:
        return load_clip_cache_manifest(clip_cache)
    raw_path = manifest.get("source_clip_cache_manifest_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest.source_clip_cache_manifest_path: must be a non-empty string")
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = cache_root / source_path
        if not source_path.exists():
            cwd_relative = Path(raw_path)
            if cwd_relative.exists():
                source_path = cwd_relative
    return load_clip_cache_manifest(manifest_path_from_input(source_path))


def _validate_signal_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    clip_length: int,
    cache_root: Path,
) -> None:
    signal_id = _non_negative_int(entry, "signal_id")
    if signal_id != index:
        raise ValueError(f"manifest.signals[{index}].signal_id: must equal list index")
    _non_negative_int(entry, "source_clip_id")
    resolve_signal_payload_path(cache_root, entry.get("payload_path"))
    frame_ids = _int_sequence(entry, "frame_ids", clip_length)
    _float_sequence(entry, "timestamps_s", clip_length)
    if len(set(frame_ids)) != clip_length:
        raise ValueError(f"manifest.signals[{index}].frame_ids: must be unique")


def _compare_entry_metadata(
    teacher_entry: Mapping[str, object],
    clip_entry: Mapping[str, object],
    *,
    index: int,
    clip_length: int,
) -> None:
    if tuple(_int_sequence(teacher_entry, "frame_ids", clip_length)) != tuple(
        _int_sequence(clip_entry, "frame_ids", clip_length)
    ):
        raise ValueError(f"manifest.signals[{index}].frame_ids: must match source clip")
    teacher_timestamps = np.asarray(_float_sequence(teacher_entry, "timestamps_s", clip_length))
    clip_timestamps = np.asarray(_float_sequence(clip_entry, "timestamps_s", clip_length))
    if not np.allclose(teacher_timestamps, clip_timestamps, rtol=0.0, atol=1e-9):
        raise ValueError(f"manifest.signals[{index}].timestamps_s: must match source clip")


def _validate_truth_boundary(manifest: Mapping[str, object]) -> None:
    truth = manifest.get("truth_boundary")
    if not isinstance(truth, Mapping):
        raise ValueError("manifest.truth_boundary: must be a mapping")
    for key, expected in _TRUTH_BOOLEAN_FIELDS.items():
        if truth.get(key) != expected:
            raise ValueError(f"truth_boundary.{key}: expected {expected!r}")
    teacher_source = truth.get("teacher_source")
    if not isinstance(teacher_source, str) or not teacher_source:
        raise ValueError("truth_boundary.teacher_source: must be a non-empty string")
    for key in ("measured_geometry", "pseudo_label"):
        if not isinstance(truth.get(key), bool):
            raise ValueError(f"truth_boundary.{key}: must be a bool")
    if bool(truth["measured_geometry"]) and bool(truth["pseudo_label"]):
        raise ValueError("truth_boundary: measured_geometry and pseudo_label cannot both be true")


def _validate_source_metadata(manifest: Mapping[str, object]) -> None:
    metadata = manifest.get("source_metadata")
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError("source_metadata: must be a mapping")
    pose_source = metadata.get("pose_source")
    if pose_source is not None and (not isinstance(pose_source, str) or not pose_source):
        raise ValueError("source_metadata.pose_source: must be a non-empty string when provided")
    pose_confidence = metadata.get("pose_confidence")
    if pose_confidence is not None:
        if not isinstance(pose_confidence, int | float) or isinstance(pose_confidence, bool):
            raise ValueError("source_metadata.pose_confidence: must be numeric when provided")
        if float(pose_confidence) < 0.0 or float(pose_confidence) > 1.0:
            raise ValueError("source_metadata.pose_confidence: must be in [0, 1]")


def _array(
    mapping: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> npt.NDArray[Any]:
    array = np.asarray(mapping[key])
    if array.shape != shape:
        raise ValueError(f"{key}: expected shape {shape}, got {array.shape}")
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{key}: expected dtype {np.dtype(dtype).name}, got {array.dtype}")
    if array.size > 0 and not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: must contain finite values")
    return array


def _validate_probability(key: str, array: npt.NDArray[Any]) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{key}: must contain values in [0, 1]")


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string")
    return value


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key}: must be a positive integer")
    return value


def _non_negative_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key}: must be a non-negative integer")
    return value


def _int_sequence(mapping: Mapping[str, object], key: str, length: int) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key}: must be a list of length {length}")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{key}: must contain integer values")
    return tuple(value)


def _float_sequence(mapping: Mapping[str, object], key: str, length: int) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key}: must be a list of length {length}")
    if not all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{key}: must contain numeric values")
    return tuple(float(item) for item in value)


__all__ = [
    "TEACHER_SIGNAL_FORMAT_NAME",
    "TEACHER_SIGNAL_FORMAT_VERSION",
    "TEACHER_SIGNAL_MANIFEST_FILENAME",
    "load_teacher_signal_manifest",
    "read_teacher_signal_payload",
    "validate_teacher_signal_manifest",
    "validate_teacher_signal_payload",
    "write_teacher_signal_payload",
]
