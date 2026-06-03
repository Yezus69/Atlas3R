"""VGGT local-output ingestion into Atlas3R teacher-signal caches."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.teachers.external.cache_writer import (
    ExternalSignalPayload,
    write_external_teacher_signal_cache,
)
from atlas3r.teachers.external.contracts import ExternalTeacherStatus

TEACHER_NAME = "vggt"
TEACHER_VERSION = "local-output-v1"
TEACHER_SOURCE_TYPE = "local_external_geometry_teacher"
INPUT_FORMAT = (
    "Folder of clip_<source_clip_id:06d>.npz or source_clip_<source_clip_id:06d>.npz "
    "files with depth_m and optional confidence, K, T_world_camera, pointmaps"
)
OUTPUT_FORMAT = "Atlas3R teacher-signal cache v1 with VGGT local pseudo-labels"
_RAW_NPZ_FILENAME_PATTERNS = (
    re.compile(r"^clip_(?P<source_clip_id>\d{6})\.npz$"),
    re.compile(r"^source_clip_(?P<source_clip_id>\d{6})\.npz$"),
)
_SIGMA_HEURISTIC = (
    "If local VGGT output does not provide depth_sigma_m, Atlas3R sets sigma to "
    "max(0.03 m, 0.03 * depth_m) / max(confidence, 0.25); invalid pixels use "
    "depth/sigma/confidence 0."
)


@dataclass(frozen=True)
class VGGTLocalIngestConfig:
    clip_cache: Path
    input: Path
    output: Path
    teacher_version: str = TEACHER_VERSION
    max_clips: int | None = None

    def __post_init__(self) -> None:
        if self.max_clips is not None and self.max_clips <= 0:
            raise ValueError("max_clips: must be positive when provided")


def get_vggt_local_status() -> ExternalTeacherStatus:
    return ExternalTeacherStatus(
        name=TEACHER_NAME,
        display_name="VGGT local-output ingest",
        availability="available",
        can_run_locally=True,
        install_hint=(
            "Run VGGT outside Atlas3R and provide local NPZ outputs; do not vendor VGGT "
            "code or weights into this repo."
        ),
        expected_input_format=INPUT_FORMAT,
        expected_output_format=OUTPUT_FORMAT,
        reason="Local-output ingestion has no optional model dependency.",
    )


def ingest_vggt_local_teacher_signal_cache(config: VGGTLocalIngestConfig) -> dict[str, object]:
    """Validate local VGGT-like arrays and write a teacher-signal cache."""

    if not config.input.is_dir():
        raise ValueError(f"{config.input}: VGGT local input must be a directory")
    if not config.teacher_version:
        raise ValueError("teacher_version: must be non-empty")
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clip_entries = _clip_entries(clip_manifest)
    clips_by_id = {_int_field(entry, "clip_id"): entry for entry in clip_entries}
    source_paths = _input_payload_paths(config.input)
    if config.max_clips is not None:
        source_paths = source_paths[: config.max_clips]
    if not source_paths:
        raise ValueError(f"{config.input}: no VGGT local NPZ payloads found")

    seen_source_clip_ids: dict[int, Path] = {}
    payloads: list[ExternalSignalPayload] = []
    for payload_path in source_paths:
        source_clip_id = _source_clip_id_from_raw_npz_filename(payload_path)
        if source_clip_id in seen_source_clip_ids:
            first_path = seen_source_clip_ids[source_clip_id]
            raise ValueError(
                f"{payload_path.name}: duplicate source clip id {source_clip_id} "
                f"already provided by {first_path.name}"
            )
        seen_source_clip_ids[source_clip_id] = payload_path
        clip_entry = clips_by_id.get(source_clip_id)
        if clip_entry is None:
            raise ValueError(
                f"{payload_path.name}: source clip id {source_clip_id} "
                "is outside the source clip-cache range"
            )
        clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
        validate_clip_payload(
            clip_payload,
            clip_length=_int_field(clip_manifest, "clip_length"),
            height=_int_field(clip_manifest, "image_height"),
            width=_int_field(clip_manifest, "image_width"),
        )
        with np.load(payload_path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        payloads.append(
            ExternalSignalPayload(
                source_clip_id=source_clip_id,
                arrays=_payload_from_vggt_arrays(raw, clip_payload=clip_payload),
            )
        )

    return write_external_teacher_signal_cache(
        clip_cache=clip_manifest_path,
        output=config.output,
        payloads=payloads,
        teacher_name=TEACHER_NAME,
        teacher_version=config.teacher_version,
        teacher_source_type=TEACHER_SOURCE_TYPE,
        source_metadata={
            "runner": "atlas3r.teachers.external.vggt_local",
            "ingest_input_path": str(config.input),
            "expected_arrays": [
                "depth_m",
                "confidence",
                "K",
                "T_world_camera",
                "pointmap_camera_m",
                "pointmap_world_m",
            ],
            "depth_sigma_m_heuristic": _SIGMA_HEURISTIC,
            "measured_geometry": False,
            "pseudo_label": True,
        },
    )


def _payload_from_vggt_arrays(
    raw: Mapping[str, npt.NDArray[Any]],
    *,
    clip_payload: Mapping[str, Any],
) -> dict[str, npt.NDArray[Any] | np.generic]:
    depth = _required_array(raw, "depth_m").astype(np.float32, copy=False)
    if depth.ndim != 3:
        raise ValueError("depth_m: expected shape T,H,W")
    confidence = _optional_array(raw, "confidence")
    if confidence is None:
        valid = np.isfinite(depth) & (depth > 0.0)
        confidence = np.where(valid, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    else:
        confidence = confidence.astype(np.float32, copy=False)
        if confidence.shape != depth.shape:
            raise ValueError(f"confidence: expected shape {depth.shape}, got {confidence.shape}")
        _validate_probability("confidence", confidence)
    valid_mask = np.isfinite(depth) & (depth > 0.0) & (confidence > 0.0)
    depth = np.where(valid_mask, depth, np.float32(0.0)).astype(np.float32, copy=False)
    sigma = _optional_array(raw, "depth_sigma_m")
    if sigma is None:
        sigma = _heuristic_sigma(depth, confidence, valid_mask)
    else:
        sigma = sigma.astype(np.float32, copy=False)
        if sigma.shape != depth.shape:
            raise ValueError(f"depth_sigma_m: expected shape {depth.shape}, got {sigma.shape}")
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("depth_sigma_m: must be finite and non-negative")
        if np.any(sigma[valid_mask] <= 0.0):
            raise ValueError("depth_sigma_m: must be positive on valid pixels")
        sigma = np.where(valid_mask, sigma, np.float32(0.0)).astype(np.float32, copy=False)

    payload: dict[str, npt.NDArray[Any] | np.generic] = {
        "depth_m": depth,
        "depth_sigma_m": sigma,
        "confidence": np.where(valid_mask, confidence, np.float32(0.0)).astype(np.float32),
        "valid_mask": valid_mask.astype(np.bool_, copy=False),
        "K": _array_or_clip(raw, clip_payload, "K", dtype=np.float32),
        "T_world_camera": _array_or_clip(raw, clip_payload, "T_world_camera", dtype=np.float32),
        "frame_ids": _array_or_clip(raw, clip_payload, "frame_ids", dtype=np.int32),
        "timestamps_s": _array_or_clip(raw, clip_payload, "timestamps_s", dtype=np.float64),
    }
    for key in ("pointmap_camera_m", "pointmap_world_m", "normal_camera"):
        value = _optional_array(raw, key)
        if value is not None:
            payload[key] = value.astype(np.float32, copy=False)
    return payload


def _heuristic_sigma(
    depth: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    sigma = np.zeros_like(depth, dtype=np.float32)
    base = np.maximum(np.float32(0.03), depth * np.float32(0.03))
    base = base / np.maximum(confidence, np.float32(0.25))
    sigma[valid_mask] = base[valid_mask]
    return sigma


def _input_payload_paths(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*.npz"))


def _source_clip_id_from_raw_npz_filename(path: Path) -> int:
    for pattern in _RAW_NPZ_FILENAME_PATTERNS:
        match = pattern.match(path.name)
        if match is not None:
            return int(match.group("source_clip_id"))
    raise ValueError(
        f"{path.name}: VGGT local NPZ filename must be "
        "clip_<source_clip_id:06d>.npz or source_clip_<source_clip_id:06d>.npz"
    )


def _required_array(
    mapping: Mapping[str, npt.NDArray[Any]],
    key: str,
) -> npt.NDArray[Any]:
    value = _optional_array(mapping, key)
    if value is None:
        raise ValueError(f"{key}: required local VGGT array is missing")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key}: must contain finite values")
    return value


def _optional_array(
    mapping: Mapping[str, npt.NDArray[Any]],
    key: str,
) -> npt.NDArray[Any] | None:
    if key not in mapping:
        return None
    return np.asarray(mapping[key])


def _array_or_clip(
    raw: Mapping[str, npt.NDArray[Any]],
    clip_payload: Mapping[str, Any],
    key: str,
    *,
    dtype: Any,
) -> npt.NDArray[Any]:
    value = raw[key] if key in raw else clip_payload[key]
    return np.asarray(value, dtype=dtype)


def _validate_probability(key: str, array: npt.NDArray[np.float32]) -> None:
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{key}: must contain finite values in [0, 1]")


def _clip_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    value = manifest.get("clips")
    if not isinstance(value, list):
        raise ValueError("clip manifest clips: must be a list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"clip manifest.clips[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], entry))
    return entries


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "TEACHER_NAME",
    "VGGTLocalIngestConfig",
    "get_vggt_local_status",
    "ingest_vggt_local_teacher_signal_cache",
]
