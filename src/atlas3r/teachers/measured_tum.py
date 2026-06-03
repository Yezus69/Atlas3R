"""Measured TUM RGB-D teacher-signal forge and local-folder ingestion."""

from __future__ import annotations

import re
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
from atlas3r.teachers.signals import (
    TEACHER_SIGNAL_FORMAT_NAME,
    TEACHER_SIGNAL_FORMAT_VERSION,
    TEACHER_SIGNAL_MANIFEST_FILENAME,
    _non_negative_int,
    _positive_int,
    load_teacher_signal_manifest,
    validate_payload_matches_signal_entry,
    validate_teacher_signal_manifest,
    validate_teacher_signal_payload,
    write_teacher_signal_manifest,
    write_teacher_signal_payload,
)

_RAW_NPZ_FILENAME_PATTERNS = (
    re.compile(r"^clip_(?P<source_clip_id>\d{6})\.npz$"),
    re.compile(r"^source_clip_(?P<source_clip_id>\d{6})\.npz$"),
)


@dataclass(frozen=True)
class MeasuredTumTeacherForgeConfig:
    clip_cache: Path
    output: Path
    sigma_m: float = 0.01
    max_clips: int | None = None


@dataclass(frozen=True)
class LocalTeacherIngestConfig:
    clip_cache: Path
    input: Path
    output: Path
    teacher_name: str
    teacher_version: str
    pseudo_label: bool


def forge_measured_tum_teacher_signal_cache(
    config: MeasuredTumTeacherForgeConfig,
) -> dict[str, object]:
    """Forge measured visible-surface teacher signals from a Phase 5A clip cache."""

    _validate_measured_config(config)
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clips = _clip_entries(clip_manifest)
    selected = clips if config.max_clips is None else clips[: config.max_clips]
    if not selected:
        raise ValueError(f"{clip_manifest_path}: no clips selected for teacher-signal forge")

    config.output.mkdir(parents=True, exist_ok=True)
    signals_dir = config.output / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_entries: list[dict[str, object]] = []
    for signal_id, clip_entry in enumerate(selected):
        clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
        validate_clip_payload(
            clip_payload,
            clip_length=_positive_int(clip_manifest, "clip_length"),
            height=_positive_int(clip_manifest, "image_height"),
            width=_positive_int(clip_manifest, "image_width"),
        )
        payload = _measured_payload_from_clip_payload(clip_payload, sigma_m=config.sigma_m)
        payload_path = signals_dir / f"clip_{signal_id:06d}.npz"
        write_teacher_signal_payload(
            payload_path,
            payload,
            clip_length=_positive_int(clip_manifest, "clip_length"),
            height=_positive_int(clip_manifest, "image_height"),
            width=_positive_int(clip_manifest, "image_width"),
        )
        signal_entries.append(_signal_entry(signal_id, clip_entry))

    manifest = _teacher_manifest(
        clip_manifest=clip_manifest,
        clip_manifest_path=clip_manifest_path,
        signal_entries=signal_entries,
        teacher_name="tum_rgbd_sensor_depth_pose",
        teacher_version="1",
        teacher_source_type="measured_rgbd_pose",
        measured_geometry=True,
        pseudo_label=False,
        source_metadata={
            "depth_source": "Phase 5A clip-cache TUM sensor depth",
            "pose_source": "Phase 5A clip-cache TUM pose",
            "valid_pixel_rule": "valid_depth_mask",
            "invalid_pixel_rule": "depth=0, sigma=0, confidence=0",
            "sigma_m": config.sigma_m,
        },
    )
    validate_teacher_signal_manifest(
        manifest,
        cache_root=config.output,
        validate_payloads=True,
        source_clip_manifest=clip_manifest,
    )
    manifest_path = config.output / TEACHER_SIGNAL_MANIFEST_FILENAME
    write_teacher_signal_manifest(manifest_path, manifest)
    return _result(config.output, manifest_path, manifest)


def ingest_local_teacher_signal_cache(config: LocalTeacherIngestConfig) -> dict[str, object]:
    """Validate and copy local `.npz` teacher payloads into the stable cache format."""

    _validate_ingest_config(config)
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    input_manifest_path = config.input / TEACHER_SIGNAL_MANIFEST_FILENAME
    if input_manifest_path.is_file():
        input_manifest = load_teacher_signal_manifest(
            input_manifest_path,
            clip_cache=clip_manifest_path,
            validate_payloads=True,
        )
        sources = _manifest_payload_sources(input_manifest, input_manifest_path.parent)
        input_format = "teacher_signal_manifest"
        preserved_metadata = {
            "input_manifest_path": str(input_manifest_path),
            "input_teacher_name": str(input_manifest["teacher_name"]),
            "input_teacher_version": str(input_manifest["teacher_version"]),
            "input_teacher_source_type": str(input_manifest["teacher_source_type"]),
        }
    else:
        sources = _raw_npz_payload_sources(config.input, clip_manifest)
        input_format = "raw_npz_payloads"
        preserved_metadata = {}
    if not sources:
        raise ValueError(f"{config.input}: no teacher-signal payloads found")

    config.output.mkdir(parents=True, exist_ok=True)
    signals_dir = config.output / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal_entries: list[dict[str, object]] = []
    for signal_id, source in enumerate(sources):
        arrays = source["arrays"]
        if not isinstance(arrays, dict):
            raise ValueError("local ingest: payload source arrays must be a mapping")
        validate_teacher_signal_payload(
            arrays,
            clip_length=_positive_int(clip_manifest, "clip_length"),
            height=_positive_int(clip_manifest, "image_height"),
            width=_positive_int(clip_manifest, "image_width"),
        )
        entry = cast(dict[str, object], source["entry"])
        validate_payload_matches_signal_entry(arrays, entry, index=signal_id)
        output_payload_path = signals_dir / f"clip_{signal_id:06d}.npz"
        write_teacher_signal_payload(
            output_payload_path,
            arrays,
            clip_length=_positive_int(clip_manifest, "clip_length"),
            height=_positive_int(clip_manifest, "image_height"),
            width=_positive_int(clip_manifest, "image_width"),
        )
        signal_entries.append(entry)

    manifest = _teacher_manifest(
        clip_manifest=clip_manifest,
        clip_manifest_path=clip_manifest_path,
        signal_entries=signal_entries,
        teacher_name=config.teacher_name,
        teacher_version=config.teacher_version,
        teacher_source_type="local_folder",
        measured_geometry=False,
        pseudo_label=config.pseudo_label,
        source_metadata={
            "ingest_input_path": str(config.input),
            "ingest_input_format": input_format,
            **preserved_metadata,
        },
    )
    validate_teacher_signal_manifest(
        manifest,
        cache_root=config.output,
        validate_payloads=True,
        source_clip_manifest=clip_manifest,
    )
    manifest_path = config.output / TEACHER_SIGNAL_MANIFEST_FILENAME
    write_teacher_signal_manifest(manifest_path, manifest)
    return _result(config.output, manifest_path, manifest)


def _measured_payload_from_clip_payload(
    clip_payload: dict[str, Any],
    *,
    sigma_m: float,
) -> dict[str, npt.NDArray[Any] | np.generic]:
    depth = np.asarray(clip_payload["depth_m"], dtype=np.float32)
    valid_mask = np.asarray(clip_payload["valid_depth_mask"], dtype=np.bool_)
    measured_depth = depth.copy()
    measured_depth[~valid_mask] = 0.0
    sigma = np.zeros_like(measured_depth, dtype=np.float32)
    sigma[valid_mask] = np.float32(sigma_m)
    confidence = np.zeros_like(measured_depth, dtype=np.float32)
    confidence[valid_mask] = 1.0
    payload: dict[str, npt.NDArray[Any] | np.generic] = {
        "depth_m": measured_depth,
        "depth_sigma_m": sigma,
        "confidence": confidence,
        "valid_mask": valid_mask.astype(np.bool_, copy=True),
        "K": np.asarray(clip_payload["K"], dtype=np.float32),
        "T_world_camera": np.asarray(clip_payload["T_world_camera"], dtype=np.float32),
        "frame_ids": np.asarray(clip_payload["frame_ids"], dtype=np.int32),
        "timestamps_s": np.asarray(clip_payload["timestamps_s"], dtype=np.float64),
    }
    for key in ("pointmap_camera_m", "pointmap_world_m", "normal_camera"):
        if key in clip_payload:
            payload[key] = np.asarray(clip_payload[key], dtype=np.float32)
    return payload


def _teacher_manifest(
    *,
    clip_manifest: dict[str, object],
    clip_manifest_path: Path,
    signal_entries: list[dict[str, object]],
    teacher_name: str,
    teacher_version: str,
    teacher_source_type: str,
    measured_geometry: bool,
    pseudo_label: bool,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "format_name": TEACHER_SIGNAL_FORMAT_NAME,
        "format_version": TEACHER_SIGNAL_FORMAT_VERSION,
        "source_clip_cache_manifest_path": str(clip_manifest_path),
        "source_dataset_name": str(clip_manifest["source_dataset_name"]),
        "source_sequence_name": str(clip_manifest["source_sequence_name"]),
        "split": str(clip_manifest["split"]),
        "clip_length": _positive_int(clip_manifest, "clip_length"),
        "image_width": _positive_int(clip_manifest, "image_width"),
        "image_height": _positive_int(clip_manifest, "image_height"),
        "teacher_name": teacher_name,
        "teacher_version": teacher_version,
        "teacher_source_type": teacher_source_type,
        "signal_count": len(signal_entries),
        "signals": signal_entries,
        "source_metadata": source_metadata,
        "truth_boundary": {
            "diagnostic_only": True,
            "accuracy_report": False,
            "performance_report": False,
            "teacher_source": teacher_name,
            "measured_geometry": measured_geometry,
            "pseudo_label": pseudo_label,
        },
    }


def _manifest_payload_sources(
    input_manifest: dict[str, object],
    input_root: Path,
) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    entries = input_manifest.get("signals")
    if not isinstance(entries, list):
        raise ValueError("input teacher-signal manifest signals: must be a list")
    for signal_id, entry_value in enumerate(entries):
        if not isinstance(entry_value, dict):
            raise ValueError(f"input manifest.signals[{signal_id}]: must be a mapping")
        payload_path = input_root / str(entry_value["payload_path"])
        with np.load(payload_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        sources.append({"entry": _entry_for_output(signal_id, entry_value), "arrays": arrays})
    return sources


def _raw_npz_payload_sources(
    input_root: Path,
    clip_manifest: dict[str, object],
) -> list[dict[str, object]]:
    paths = sorted(input_root.glob("*.npz"))
    clips = _clip_entries(clip_manifest)
    clips_by_id = {_non_negative_int(clip, "clip_id"): clip for clip in clips}
    seen_source_clip_ids: dict[int, Path] = {}
    sources: list[dict[str, object]] = []
    for payload_path in paths:
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
        with np.load(payload_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        sources.append({"entry": _signal_entry(len(sources), clip_entry), "arrays": arrays})
    return sources


def _signal_entry(signal_id: int, clip_entry: dict[str, object]) -> dict[str, object]:
    source_clip_id = _non_negative_int(clip_entry, "clip_id")
    return {
        "signal_id": signal_id,
        "source_clip_id": source_clip_id,
        "payload_path": f"signals/clip_{signal_id:06d}.npz",
        "frame_ids": list(_int_sequence(clip_entry, "frame_ids")),
        "timestamps_s": list(_float_sequence(clip_entry, "timestamps_s")),
        "center_index": _non_negative_int(clip_entry, "center_index")
        if "center_index" in clip_entry
        else 0,
    }


def _entry_for_output(signal_id: int, input_entry: dict[str, object]) -> dict[str, object]:
    entry = {
        "signal_id": signal_id,
        "source_clip_id": _non_negative_int(input_entry, "source_clip_id"),
        "payload_path": f"signals/clip_{signal_id:06d}.npz",
        "frame_ids": list(_int_sequence(input_entry, "frame_ids")),
        "timestamps_s": list(_float_sequence(input_entry, "timestamps_s")),
    }
    if "center_index" in input_entry:
        entry["center_index"] = _non_negative_int(input_entry, "center_index")
    return entry


def _clip_entries(manifest: dict[str, object]) -> list[dict[str, object]]:
    value = manifest.get("clips")
    if not isinstance(value, list):
        raise ValueError("clip manifest clips: must be a list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"clip manifest.clips[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], entry))
    return entries


def _source_clip_id_from_raw_npz_filename(path: Path) -> int:
    for pattern in _RAW_NPZ_FILENAME_PATTERNS:
        match = pattern.match(path.name)
        if match is not None:
            return int(match.group("source_clip_id"))
    raise ValueError(
        f"{path.name}: raw NPZ filename must be "
        "clip_<source_clip_id:06d>.npz or source_clip_<source_clip_id:06d>.npz"
    )


def _result(output: Path, manifest_path: Path, manifest: dict[str, object]) -> dict[str, object]:
    return {
        "format_name": "atlas3r_teacher_signal_cache_write_result",
        "output": str(output),
        "manifest_path": str(manifest_path),
        "signal_count": _non_negative_int(manifest, "signal_count"),
        "teacher_name": str(manifest["teacher_name"]),
        "teacher_source_type": str(manifest["teacher_source_type"]),
        "truth_boundary": dict(cast(dict[str, object], manifest["truth_boundary"])),
    }


def _validate_measured_config(config: MeasuredTumTeacherForgeConfig) -> None:
    if config.sigma_m <= 0.0:
        raise ValueError("sigma_m: must be positive")
    if config.max_clips is not None and config.max_clips <= 0:
        raise ValueError("max_clips: must be positive when provided")


def _validate_ingest_config(config: LocalTeacherIngestConfig) -> None:
    if not config.teacher_name:
        raise ValueError("teacher_name: must be non-empty")
    if not config.teacher_version:
        raise ValueError("teacher_version: must be non-empty")
    if not config.input.is_dir():
        raise ValueError(f"{config.input}: local teacher input must be a directory")


def _int_sequence(mapping: dict[str, object], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key}: must be a list of integers")
    return tuple(value)


def _float_sequence(mapping: dict[str, object], key: str) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key}: must be a list of numbers")
    return tuple(float(item) for item in value)
