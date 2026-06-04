"""Import existing Atlas3R/TUM data into the recording format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.data.tum_rgbd import load_tum_rgbd_manifest
from atlas3r.forge.clip_cache import load_clip_cache_manifest
from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    Atlas3RRecording,
    recording_truth_boundary,
    write_recording_files,
)
from atlas3r.recording.sensor_folder import (
    SENSOR_CAPTURE_FORMAT_NAME,
    SENSOR_CAPTURE_FORMAT_VERSION,
    SensorFolderRecordingImportConfig,
    recording_from_sensor_folder,
)
from atlas3r.runtime.student_stream import load_unique_frame_stream


@dataclass(frozen=True)
class TumRecordingImportConfig:
    manifest: Path
    output: Path
    split: str = "val"
    max_frames: int | None = None
    width: int = 160
    height: int = 120


@dataclass(frozen=True)
class ClipCacheRecordingImportConfig:
    clip_cache: Path
    output: Path
    dedupe_frame_id: bool = False


def recording_from_tum_manifest(config: TumRecordingImportConfig) -> dict[str, object]:
    """Write a recording that references an existing TUM RGB-D manifest's files."""

    _validate_tum_config(config)
    tum_manifest = load_tum_rgbd_manifest(config.manifest)
    records = _records_for_split(tum_manifest, config.split)
    if config.max_frames is not None:
        records = records[: config.max_frames]
    if not records:
        raise ValueError(f"{config.manifest}: no {config.split!r} frames selected")
    K = _scaled_intrinsics(tum_manifest, width=config.width, height=config.height)
    depth_scale = _depth_scale(tum_manifest)
    root_path = _string_field(tum_manifest, "root_path")
    frames = []
    for record in records:
        frame_id = _int_field(record, "frame_id")
        T_world_camera = _array_list(record["T_world_camera"], (4, 4))
        frames.append(
            {
                "K": K.tolist(),
                "T_world_camera": T_world_camera.tolist(),
                "camera_center_world_m": T_world_camera[:3, 3].tolist(),
                "depth_path": _string_field(record, "depth_path"),
                "depth_scale": depth_scale,
                "frame_id": frame_id,
                "rgb_path": _string_field(record, "rgb_path"),
                "source_metadata": {
                    "depth_timestamp_s": _float_field(record, "depth_timestamp_s"),
                    "pose_source": str(record.get("pose_source", "TUM RGB-D groundtruth.txt")),
                    "pose_timestamp_s": _float_field(record, "pose_timestamp_s"),
                    "rgb_timestamp_s": _float_field(record, "rgb_timestamp_s"),
                    "source_frame_id": frame_id,
                    "source_root_key": "tum_root",
                },
                "timestamp_s": _float_field(record, "rgb_timestamp_s"),
            }
        )
    manifest = _recording_manifest(
        frame_count=len(frames),
        width=config.width,
        height=config.height,
        source_dataset=_string_field(tum_manifest, "dataset_name"),
        source_sequence=_string_field(tum_manifest, "sequence_name"),
        depth_present=True,
        pose_present=True,
        capture_metadata={
            "importer": "tum_rgbd_manifest",
            "source_manifest_path": str(config.manifest),
            "split": config.split,
            "target_height": config.height,
            "target_width": config.width,
        },
        known_calibration_metadata={
            "K": K.tolist(),
            "coordinate_frame": RECORDING_COORDINATE_FRAME,
            "depth_scale": depth_scale,
            "depth_units": "meters = raw_png / depth_scale",
            "intrinsics_source": str(
                _mapping_field(tum_manifest, "intrinsics").get("source", "unknown")
            ),
            "original_image": dict(_mapping_field(tum_manifest, "image")),
        },
        external_roots={"tum_root": root_path},
    )
    recording = write_recording_files(config.output, manifest=manifest, frames=frames)
    return _import_result(
        "atlas3r_tum_recording_import_result",
        recording,
        source=str(config.manifest),
        split=config.split,
    )


def recording_from_clip_cache(config: ClipCacheRecordingImportConfig) -> dict[str, object]:
    """Write a recording stream from an existing Atlas3R clip cache."""

    manifest = load_clip_cache_manifest(config.clip_cache)
    stream = load_unique_frame_stream(config.clip_cache)
    if not stream:
        raise ValueError(f"{config.clip_cache}: no frames found in clip cache")
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "rgb").mkdir(exist_ok=True)
    (config.output / "depth").mkdir(exist_ok=True)
    frames = []
    for stream_frame in stream:
        rgb_path = f"rgb/frame_{stream_frame.frame_id:06d}.npz"
        depth_path = f"depth/frame_{stream_frame.frame_id:06d}.npz"
        np.savez_compressed(config.output / rgb_path, rgb_u8=stream_frame.rgb_u8)
        np.savez_compressed(
            config.output / depth_path,
            depth_m=stream_frame.depth_m,
            valid_depth_mask=stream_frame.valid_depth_mask,
        )
        frames.append(
            {
                "K": stream_frame.K.tolist(),
                "T_world_camera": stream_frame.T_world_camera.tolist(),
                "camera_center_world_m": stream_frame.T_world_camera[:3, 3].tolist(),
                "depth_path": depth_path,
                "frame_id": stream_frame.frame_id,
                "rgb_path": rgb_path,
                "source_metadata": {
                    **stream_frame.metadata(),
                    "dedupe_frame_id": config.dedupe_frame_id,
                    "source_format": "atlas3r_multiview_clip_cache",
                },
                "timestamp_s": stream_frame.timestamp_s,
            }
        )
    recording_manifest = _recording_manifest(
        frame_count=len(frames),
        width=_int_field(manifest, "image_width"),
        height=_int_field(manifest, "image_height"),
        source_dataset=_string_field(manifest, "source_dataset_name"),
        source_sequence=_string_field(manifest, "source_sequence_name"),
        depth_present=True,
        pose_present=True,
        capture_metadata={
            "dedupe_frame_id": config.dedupe_frame_id,
            "importer": "atlas3r_clip_cache",
            "source_clip_cache": str(config.clip_cache),
            "source_manifest_path": str(manifest.get("source_manifest_path", "")),
            "split": str(manifest.get("split", "")),
        },
        known_calibration_metadata={
            "coordinate_frame": RECORDING_COORDINATE_FRAME,
            "intrinsics_source": "per-frame clip-cache K",
            "source_teacher_metadata": dict(
                cast(dict[str, object], manifest.get("teacher_source_metadata", {}))
            ),
        },
        external_roots={},
    )
    recording = write_recording_files(config.output, manifest=recording_manifest, frames=frames)
    return _import_result(
        "atlas3r_clip_cache_recording_import_result",
        recording,
        source=str(config.clip_cache),
        split=str(manifest.get("split", "")),
    )


def _recording_manifest(
    *,
    frame_count: int,
    width: int,
    height: int,
    source_dataset: str,
    source_sequence: str,
    depth_present: bool,
    pose_present: bool,
    capture_metadata: dict[str, object],
    known_calibration_metadata: dict[str, object],
    external_roots: dict[str, str],
) -> dict[str, object]:
    return {
        "capture_metadata": capture_metadata,
        "coordinate_frame": RECORDING_COORDINATE_FRAME,
        "depth_present": depth_present,
        "external_roots": external_roots,
        "format_name": RECORDING_FORMAT_NAME,
        "format_version": RECORDING_FORMAT_VERSION,
        "frame_count": frame_count,
        "height": height,
        "known_calibration_metadata": known_calibration_metadata,
        "pose_present": pose_present,
        "source_dataset": source_dataset,
        "source_sequence": source_sequence,
        "truth_boundary": recording_truth_boundary(
            depth_present=depth_present,
            pose_present=pose_present,
        ),
        "width": width,
    }


def _import_result(
    format_name: str,
    recording: Atlas3RRecording,
    *,
    source: str,
    split: str,
) -> dict[str, object]:
    return {
        "depth_present": bool(recording.manifest["depth_present"]),
        "format_name": format_name,
        "format_version": 1,
        "frame_count": len(recording.frames),
        "manifest_path": str(recording.root / "atlas3r_recording.json"),
        "output": str(recording.root),
        "pose_present": bool(recording.manifest["pose_present"]),
        "source": source,
        "split": split,
        "truth_boundary": dict(cast(dict[str, object], recording.manifest["truth_boundary"])),
    }


def _records_for_split(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("manifest.frames: must be a list")
    records = [
        cast(dict[str, object], frame)
        for frame in frames
        if isinstance(frame, dict) and frame.get("split") == split
    ]
    records.sort(
        key=lambda record: (_float_field(record, "rgb_timestamp_s"), _int_field(record, "frame_id"))
    )
    return records


def _scaled_intrinsics(
    manifest: dict[str, object],
    *,
    width: int,
    height: int,
) -> npt.NDArray[np.float32]:
    image = _mapping_field(manifest, "image")
    original_width = _int_field(image, "width")
    original_height = _int_field(image, "height")
    K = _array_list(_mapping_field(manifest, "intrinsics")["K"], (3, 3))
    scaled = K.copy()
    scaled[0, 0] *= float(width) / float(original_width)
    scaled[0, 2] *= float(width) / float(original_width)
    scaled[1, 1] *= float(height) / float(original_height)
    scaled[1, 2] *= float(height) / float(original_height)
    return scaled.astype(np.float32, copy=False)


def _depth_scale(manifest: dict[str, object]) -> float:
    depth = _mapping_field(manifest, "depth")
    scale = _float_field(depth, "scale")
    if scale <= 0.0:
        raise ValueError("manifest.depth.scale: must be positive")
    return scale


def _validate_tum_config(config: TumRecordingImportConfig) -> None:
    if config.split not in {"train", "val"}:
        raise ValueError("split: must be train or val")
    for field_name in ("width", "height"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames: must be positive when provided")


def _mapping_field(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key}: must be a mapping")
    return value


def _string_field(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string")
    return value


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


def _float_field(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key}: must be numeric")
    return float(value)


def _array_list(value: object, shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"array: expected finite shape {shape}, got {array.shape}")
    return array


__all__ = [
    "ClipCacheRecordingImportConfig",
    "SENSOR_CAPTURE_FORMAT_NAME",
    "SENSOR_CAPTURE_FORMAT_VERSION",
    "SensorFolderRecordingImportConfig",
    "TumRecordingImportConfig",
    "recording_from_clip_cache",
    "recording_from_sensor_folder",
    "recording_from_tum_manifest",
]
