"""Shared writer for dependency-isolated external teacher signal caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import load_clip_cache_manifest, manifest_path_from_input
from atlas3r.teachers.signals import (
    TEACHER_SIGNAL_FORMAT_NAME,
    TEACHER_SIGNAL_FORMAT_VERSION,
    TEACHER_SIGNAL_MANIFEST_FILENAME,
    validate_teacher_signal_manifest,
    write_teacher_signal_manifest,
    write_teacher_signal_payload,
)


@dataclass(frozen=True)
class ExternalSignalPayload:
    source_clip_id: int
    arrays: Mapping[str, npt.NDArray[Any] | np.generic]


def write_external_teacher_signal_cache(
    *,
    clip_cache: str | Path,
    output: str | Path,
    payloads: Sequence[ExternalSignalPayload],
    teacher_name: str,
    teacher_version: str,
    teacher_source_type: str,
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Write external teacher arrays into the stable Phase 5B signal format."""

    if not payloads:
        raise ValueError("payloads: at least one teacher signal payload is required")
    clip_manifest_path = manifest_path_from_input(clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clip_entries = _clip_entries(clip_manifest)
    clips_by_id = {_int_field(entry, "clip_id"): entry for entry in clip_entries}
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    signals_dir = output_path / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    signal_entries: list[dict[str, object]] = []
    for signal_id, payload in enumerate(payloads):
        clip_entry = clips_by_id.get(payload.source_clip_id)
        if clip_entry is None:
            raise ValueError(f"source_clip_id {payload.source_clip_id}: out of range")
        payload_path = signals_dir / f"clip_{signal_id:06d}.npz"
        write_teacher_signal_payload(
            payload_path,
            payload.arrays,
            clip_length=_int_field(clip_manifest, "clip_length"),
            height=_int_field(clip_manifest, "image_height"),
            width=_int_field(clip_manifest, "image_width"),
        )
        signal_entries.append(_signal_entry(signal_id, clip_entry))

    manifest = _teacher_manifest(
        clip_manifest=clip_manifest,
        clip_manifest_path=clip_manifest_path,
        signal_entries=signal_entries,
        teacher_name=teacher_name,
        teacher_version=teacher_version,
        teacher_source_type=teacher_source_type,
        source_metadata=dict(source_metadata),
    )
    validate_teacher_signal_manifest(
        manifest,
        cache_root=output_path,
        validate_payloads=True,
        source_clip_manifest=clip_manifest,
    )
    manifest_path = output_path / TEACHER_SIGNAL_MANIFEST_FILENAME
    write_teacher_signal_manifest(manifest_path, manifest)
    return {
        "format_name": "atlas3r_external_teacher_signal_cache_write_result",
        "output": str(output_path),
        "manifest_path": str(manifest_path),
        "signal_count": len(signal_entries),
        "teacher_name": teacher_name,
        "teacher_source_type": teacher_source_type,
        "truth_boundary": dict(cast(dict[str, object], manifest["truth_boundary"])),
    }


def _teacher_manifest(
    *,
    clip_manifest: dict[str, object],
    clip_manifest_path: Path,
    signal_entries: list[dict[str, object]],
    teacher_name: str,
    teacher_version: str,
    teacher_source_type: str,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "format_name": TEACHER_SIGNAL_FORMAT_NAME,
        "format_version": TEACHER_SIGNAL_FORMAT_VERSION,
        "source_clip_cache_manifest_path": str(clip_manifest_path),
        "source_dataset_name": str(clip_manifest["source_dataset_name"]),
        "source_sequence_name": str(clip_manifest["source_sequence_name"]),
        "split": str(clip_manifest["split"]),
        "clip_length": _int_field(clip_manifest, "clip_length"),
        "image_width": _int_field(clip_manifest, "image_width"),
        "image_height": _int_field(clip_manifest, "image_height"),
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
            "measured_geometry": False,
            "pseudo_label": True,
        },
    }


def _signal_entry(signal_id: int, clip_entry: dict[str, object]) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "source_clip_id": _int_field(clip_entry, "clip_id"),
        "payload_path": f"signals/clip_{signal_id:06d}.npz",
        "frame_ids": list(_int_sequence(clip_entry, "frame_ids")),
        "timestamps_s": list(_float_sequence(clip_entry, "timestamps_s")),
        "center_index": _int_field(clip_entry, "center_index")
        if "center_index" in clip_entry
        else 0,
    }


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


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


def _int_sequence(mapping: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key}: must be a list of integers")
    return tuple(value)


def _float_sequence(mapping: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key}: must be a list of numbers")
    return tuple(float(item) for item in value)


__all__ = [
    "ExternalSignalPayload",
    "write_external_teacher_signal_cache",
]
