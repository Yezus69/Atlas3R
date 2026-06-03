"""Canonical Atlas3R multi-view clip-cache validation and IO."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics, validate_transform

CLIP_CACHE_FORMAT_NAME = "atlas3r_multiview_clip_cache"
CLIP_CACHE_FORMAT_VERSION = 1
CLIP_CACHE_MANIFEST_FILENAME = "atlas3r_clip_cache_manifest.json"

_REQUIRED_KEYS = {
    "images_rgb_u8",
    "depth_m",
    "valid_depth_mask",
    "K",
    "T_world_camera",
    "frame_ids",
    "timestamps_s",
    "center_index",
}
_TRUTH_BOUNDARY_FLAGS = {
    "diagnostic_only": True,
    "accuracy_report": False,
    "performance_report": False,
    "teacher_source": "tum_rgbd_sensor_depth_pose",
}


def manifest_path_from_input(path: str | Path) -> Path:
    """Resolve a clip-cache manifest path from either a manifest file or cache folder."""

    candidate = Path(path)
    if candidate.is_dir():
        return candidate / CLIP_CACHE_MANIFEST_FILENAME
    return candidate


def load_clip_cache_manifest(path: str | Path) -> dict[str, object]:
    """Load and validate a clip-cache manifest without loading every payload."""

    manifest_path = manifest_path_from_input(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{manifest_path}: failed to read clip-cache manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid JSON clip-cache manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest must be a JSON object")
    manifest = dict(payload)
    validate_clip_cache_manifest(manifest, cache_root=manifest_path.parent, validate_payloads=False)
    return manifest


def validate_clip_cache_manifest(
    manifest: Mapping[str, object],
    *,
    cache_root: str | Path,
    validate_payloads: bool,
) -> None:
    """Validate manifest metadata and optionally every referenced payload."""

    root = Path(cache_root)
    if manifest.get("format_name") != CLIP_CACHE_FORMAT_NAME:
        raise ValueError(f"manifest.format_name: expected {CLIP_CACHE_FORMAT_NAME!r}")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version != CLIP_CACHE_FORMAT_VERSION:
        raise ValueError("manifest.format_version: unsupported clip-cache version")
    clip_length = _positive_int(manifest, "clip_length")
    width = _positive_int(manifest, "image_width")
    height = _positive_int(manifest, "image_height")
    max_gap = _positive_float(manifest, "max_frame_gap_s")
    truth_boundary = _mapping_field(manifest, "truth_boundary")
    for key, expected in _TRUTH_BOUNDARY_FLAGS.items():
        if truth_boundary.get(key) != expected:
            raise ValueError(f"truth_boundary.{key}: expected {expected!r}")
    clips_value = manifest.get("clips")
    if not isinstance(clips_value, list):
        raise ValueError("manifest.clips: must be a list")
    clip_count = _non_negative_int(manifest, "clip_count")
    if clip_count != len(clips_value):
        raise ValueError("manifest.clip_count: must match len(clips)")
    for index, clip_value in enumerate(clips_value):
        if not isinstance(clip_value, Mapping):
            raise ValueError(f"manifest.clips[{index}]: must be a mapping")
        _validate_clip_entry(
            clip_value,
            index=index,
            clip_length=clip_length,
            max_frame_gap_s=max_gap,
            cache_root=root,
        )
        if validate_payloads:
            arrays = read_clip_payload_from_entry(root, clip_value)
            validate_clip_payload(arrays, clip_length=clip_length, height=height, width=width)


def resolve_payload_path(cache_root: str | Path, relative_path: object) -> Path:
    """Resolve a relative payload path after rejecting path traversal."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("payload_path: must be a non-empty relative string")
    payload_path = Path(relative_path)
    if payload_path.is_absolute() or ".." in payload_path.parts:
        raise ValueError(f"payload_path: unsafe path {relative_path!r}")
    root = Path(cache_root).resolve()
    resolved = (root / payload_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"payload_path: escapes cache root: {relative_path!r}")
    return resolved


def read_clip_payload(cache_manifest_or_dir: str | Path, *, clip_index: int = 0) -> dict[str, Any]:
    """Load a clip payload by index from a manifest file or cache directory."""

    manifest_path = manifest_path_from_input(cache_manifest_or_dir)
    manifest = load_clip_cache_manifest(manifest_path)
    clips = cast(list[object], manifest["clips"])
    if clip_index < 0 or clip_index >= len(clips):
        raise ValueError("clip_index: out of range")
    entry = clips[clip_index]
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest.clips[{clip_index}]: must be a mapping")
    arrays = read_clip_payload_from_entry(manifest_path.parent, entry)
    validate_clip_payload(
        arrays,
        clip_length=_positive_int(manifest, "clip_length"),
        height=_positive_int(manifest, "image_height"),
        width=_positive_int(manifest, "image_width"),
    )
    return arrays


def read_clip_payload_from_entry(
    cache_root: str | Path,
    clip_entry: Mapping[str, object],
) -> dict[str, Any]:
    """Load the `.npz` payload referenced by a manifest clip entry."""

    path = resolve_payload_path(cache_root, clip_entry.get("payload_path"))
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except OSError as exc:
        raise ValueError(f"{path}: failed to read clip payload: {exc}") from exc


def write_clip_payload(
    path: str | Path,
    payload: Mapping[str, npt.NDArray[Any] | np.generic],
    *,
    clip_length: int,
    height: int,
    width: int,
) -> None:
    """Validate and write one compressed clip payload."""

    arrays = {key: np.asarray(value) for key, value in payload.items()}
    validate_clip_payload(arrays, clip_length=clip_length, height=height, width=width)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def validate_clip_payload(
    payload: Mapping[str, Any],
    *,
    clip_length: int,
    height: int,
    width: int,
) -> None:
    """Validate one loaded clip payload against the stable Phase 5A schema."""

    missing = sorted(_REQUIRED_KEYS.difference(payload))
    if missing:
        raise ValueError(f"clip payload: missing required keys {missing}")
    _array(payload, "images_rgb_u8", (clip_length, height, width, 3), np.uint8)
    depth = _array(payload, "depth_m", (clip_length, height, width), np.float32)
    if np.any(depth < 0.0):
        raise ValueError("depth_m: must be non-negative")
    _array(payload, "valid_depth_mask", (clip_length, height, width), np.bool_)
    K = _array(payload, "K", (clip_length, 3, 3), np.float32)
    transforms = _array(payload, "T_world_camera", (clip_length, 4, 4), np.float32)
    for index in range(clip_length):
        validate_intrinsics(f"K[{index}]", K[index])
        validate_transform(f"T_world_camera[{index}]", transforms[index])
    frame_ids = np.asarray(payload["frame_ids"])
    if frame_ids.shape != (clip_length,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("frame_ids: must have shape T and integer dtype")
    _array(payload, "timestamps_s", (clip_length,), np.float64)
    center = np.asarray(payload["center_index"])
    if center.shape != () or not np.issubdtype(center.dtype, np.integer):
        raise ValueError("center_index: must be an integer scalar")
    center_value = int(center.item())
    if center_value < 0 or center_value >= clip_length:
        raise ValueError("center_index: must be inside the clip")
    for key in ("pointmap_camera_m", "pointmap_world_m", "normal_camera"):
        if key in payload:
            _array(payload, key, (clip_length, height, width, 3), np.float32)


def write_json_file(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write deterministic JSON with LF newlines."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_clip_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    clip_length: int,
    max_frame_gap_s: float,
    cache_root: Path,
) -> None:
    clip_id = _non_negative_int(entry, "clip_id")
    if clip_id != index:
        raise ValueError(f"manifest.clips[{index}].clip_id: must equal list index")
    frame_ids = _int_sequence(entry, "frame_ids", clip_length)
    timestamps = _float_sequence(entry, "timestamps_s", clip_length)
    center_index = _non_negative_int(entry, "center_index")
    if center_index >= clip_length:
        raise ValueError(f"manifest.clips[{index}].center_index: out of range")
    resolve_payload_path(cache_root, entry.get("payload_path"))
    for left, right in zip(timestamps[:-1], timestamps[1:], strict=False):
        if right - left > max_frame_gap_s:
            raise ValueError(f"manifest.clips[{index}]: timestamps exceed max_frame_gap_s")
    if len(set(frame_ids)) != clip_length:
        raise ValueError(f"manifest.clips[{index}].frame_ids: must be unique")


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


def _mapping_field(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}: must be a mapping")
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


def _positive_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0.0:
        raise ValueError(f"{key}: must be a positive number")
    return float(value)


def _int_sequence(
    mapping: Mapping[str, object],
    key: str,
    length: int,
) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key}: must be a list of length {length}")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{key}: must contain integer values")
    return tuple(value)


def _float_sequence(
    mapping: Mapping[str, object],
    key: str,
    length: int,
) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key}: must be a list of length {length}")
    if not all(isinstance(item, int | float) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{key}: must contain numeric values")
    return tuple(float(item) for item in value)


__all__ = [
    "CLIP_CACHE_FORMAT_NAME",
    "CLIP_CACHE_FORMAT_VERSION",
    "CLIP_CACHE_MANIFEST_FILENAME",
    "load_clip_cache_manifest",
    "manifest_path_from_input",
    "read_clip_payload",
    "read_clip_payload_from_entry",
    "resolve_payload_path",
    "validate_clip_cache_manifest",
    "validate_clip_payload",
    "write_clip_payload",
    "write_json_file",
]
