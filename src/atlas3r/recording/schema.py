"""Dependency-light Atlas3R recording format validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics, validate_transform, validate_vector

RECORDING_FORMAT_NAME = "atlas3r_recording"
RECORDING_FORMAT_VERSION = 1
RECORDING_COORDINATE_FRAME = "x_right_y_down_z_forward"
RECORDING_MANIFEST_FILENAME = "atlas3r_recording.json"
RECORDING_FRAMES_FILENAME = "frames.jsonl"


@dataclass(frozen=True)
class RecordingFrame:
    """One validated chronological Atlas3R recording frame record."""

    frame_id: int
    timestamp_s: float
    rgb_path: str
    depth_path: str | None
    depth_scale: float | None
    K: npt.NDArray[np.float32]
    T_world_camera: npt.NDArray[np.float32] | None
    camera_center_world_m: npt.NDArray[np.float32] | None
    source_metadata: dict[str, object]

    def to_json_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "K": self.K.astype(float).tolist(),
            "frame_id": self.frame_id,
            "rgb_path": self.rgb_path,
            "source_metadata": dict(self.source_metadata),
            "timestamp_s": self.timestamp_s,
        }
        if self.depth_path is not None:
            record["depth_path"] = self.depth_path
        if self.depth_scale is not None:
            record["depth_scale"] = self.depth_scale
        if self.T_world_camera is not None:
            record["T_world_camera"] = self.T_world_camera.astype(float).tolist()
        if self.camera_center_world_m is not None:
            record["camera_center_world_m"] = self.camera_center_world_m.astype(float).tolist()
        return record


@dataclass(frozen=True)
class Atlas3RRecording:
    """A loaded and validated Atlas3R recording folder."""

    root: Path
    manifest: dict[str, object]
    frames: tuple[RecordingFrame, ...]


def recording_truth_boundary(*, depth_present: bool, pose_present: bool) -> dict[str, object]:
    """Return the required non-claim truth boundary for recording artifacts."""

    return {
        "accuracy_report": False,
        "depth_present": depth_present,
        "diagnostic_only": True,
        "hidden_geometry_measured": False,
        "performance_report": False,
        "pose_present": pose_present,
    }


def write_recording_files(
    recording_dir: str | Path,
    *,
    manifest: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
) -> Atlas3RRecording:
    """Write deterministic recording files, validate them, and return the loaded recording."""

    root = Path(recording_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / RECORDING_MANIFEST_FILENAME, dict(manifest))
    with (root / RECORDING_FRAMES_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            json.dump(dict(frame), handle, sort_keys=True)
            handle.write("\n")
    return load_recording(root)


def load_recording(recording_dir: str | Path) -> Atlas3RRecording:
    """Load and validate a recording folder."""

    root = Path(recording_dir)
    manifest_path = root / RECORDING_MANIFEST_FILENAME
    frames_path = root / RECORDING_FRAMES_FILENAME
    manifest = _load_json_object(manifest_path)
    _validate_manifest(manifest)
    raw_frames = _load_frames_jsonl(frames_path)
    frames = tuple(
        _frame_from_mapping(raw, manifest=manifest, root=root, index=index)
        for index, raw in enumerate(raw_frames)
    )
    _validate_frame_sequence(manifest, frames)
    return Atlas3RRecording(root=root, manifest=manifest, frames=frames)


def validate_recording_folder(recording_dir: str | Path) -> dict[str, object]:
    """Validate a recording folder and return deterministic CLI JSON."""

    recording = load_recording(recording_dir)
    first = recording.frames[0] if recording.frames else None
    last = recording.frames[-1] if recording.frames else None
    truth_boundary = _mapping_field(recording.manifest, "truth_boundary")
    return {
        "coordinate_frame": str(recording.manifest["coordinate_frame"]),
        "depth_present": bool(recording.manifest["depth_present"]),
        "first_frame_id": None if first is None else first.frame_id,
        "first_timestamp_s": None if first is None else first.timestamp_s,
        "format_name": "atlas3r_recording_validation",
        "format_version": 1,
        "frame_count": len(recording.frames),
        "height": _int_field(recording.manifest, "height"),
        "input": str(recording.root),
        "last_frame_id": None if last is None else last.frame_id,
        "last_timestamp_s": None if last is None else last.timestamp_s,
        "pose_present": bool(recording.manifest["pose_present"]),
        "recording_format_name": str(recording.manifest["format_name"]),
        "recording_format_version": _int_field(recording.manifest, "format_version"),
        "source_dataset": str(recording.manifest["source_dataset"]),
        "source_sequence": str(recording.manifest["source_sequence"]),
        "truth_boundary": dict(truth_boundary),
        "width": _int_field(recording.manifest, "width"),
    }


def resolve_recording_path(
    recording: Atlas3RRecording | Mapping[str, object],
    recording_root: str | Path,
    relative_path: str,
) -> Path:
    """Resolve a safe frame path under the recording root or declared external roots."""

    _validate_safe_relative_path("path", relative_path)
    manifest = recording.manifest if isinstance(recording, Atlas3RRecording) else recording
    for root in _path_roots(Path(recording_root), manifest):
        resolved = (root / relative_path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"path: escapes declared root: {relative_path!r}")
        if resolved.is_file():
            return resolved
    roots = ", ".join(str(root) for root in _path_roots(Path(recording_root), manifest))
    raise ValueError(
        f"{relative_path}: file does not exist under recording/external roots: {roots}"
    )


def _validate_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("format_name") != RECORDING_FORMAT_NAME:
        raise ValueError(f"manifest.format_name: expected {RECORDING_FORMAT_NAME!r}")
    if _int_field(manifest, "format_version") != RECORDING_FORMAT_VERSION:
        raise ValueError("manifest.format_version: unsupported recording version")
    if _string_field(manifest, "coordinate_frame") != RECORDING_COORDINATE_FRAME:
        raise ValueError(f"manifest.coordinate_frame: expected {RECORDING_COORDINATE_FRAME!r}")
    _positive_int(manifest, "width")
    _positive_int(manifest, "height")
    _non_negative_int(manifest, "frame_count")
    _string_field(manifest, "source_dataset")
    _string_field(manifest, "source_sequence")
    _bool_field(manifest, "depth_present")
    _bool_field(manifest, "pose_present")
    for key in ("capture_metadata", "known_calibration_metadata"):
        _mapping_field(manifest, key)
    truth_boundary = _mapping_field(manifest, "truth_boundary")
    expected = recording_truth_boundary(
        depth_present=bool(manifest["depth_present"]),
        pose_present=bool(manifest["pose_present"]),
    )
    for key, value in expected.items():
        if truth_boundary.get(key) != value:
            raise ValueError(f"manifest.truth_boundary.{key}: expected {value!r}")
    if "external_roots" in manifest:
        roots = _mapping_field(manifest, "external_roots")
        for key, value in roots.items():
            if not isinstance(key, str) or not key:
                raise ValueError("manifest.external_roots: keys must be non-empty strings")
            if not isinstance(value, str) or not value:
                raise ValueError(f"manifest.external_roots.{key}: must be a non-empty path")


def _frame_from_mapping(
    raw: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    root: Path,
    index: int,
) -> RecordingFrame:
    frame_id = _non_negative_int(raw, "frame_id")
    timestamp_s = _finite_float(raw, "timestamp_s")
    rgb_path = _string_field(raw, "rgb_path")
    _validate_safe_relative_path("rgb_path", rgb_path)
    resolve_recording_path(manifest, root, rgb_path)
    depth_path = _optional_string(raw, "depth_path")
    depth_scale = _optional_positive_float(raw, "depth_scale")
    if bool(manifest["depth_present"]):
        if depth_path is None:
            raise ValueError(f"frames[{index}].depth_path: required when depth_present=true")
        _validate_safe_relative_path("depth_path", depth_path)
        resolve_recording_path(manifest, root, depth_path)
        if Path(depth_path).suffix.lower() == ".png" and depth_scale is None:
            raise ValueError(f"frames[{index}].depth_scale: required for PNG depth")
    elif depth_path is not None:
        raise ValueError(f"frames[{index}].depth_path: forbidden when depth_present=false")

    K = _float_array(raw.get("K"), (3, 3), field_name=f"frames[{index}].K")
    validate_intrinsics(f"frames[{index}].K", K)
    T_world_camera = None
    camera_center_world_m = None
    if "T_world_camera" in raw:
        T_world_camera = _float_array(
            raw.get("T_world_camera"),
            (4, 4),
            field_name=f"frames[{index}].T_world_camera",
        )
        validate_transform(f"frames[{index}].T_world_camera", T_world_camera)
    if bool(manifest["pose_present"]) and T_world_camera is None:
        raise ValueError(f"frames[{index}].T_world_camera: required when pose_present=true")
    if "camera_center_world_m" in raw:
        camera_center_world_m = _float_array(
            raw.get("camera_center_world_m"),
            (3,),
            field_name=f"frames[{index}].camera_center_world_m",
        )
        validate_vector(f"frames[{index}].camera_center_world_m", camera_center_world_m, 3)
        if T_world_camera is not None and not np.allclose(
            camera_center_world_m,
            T_world_camera[:3, 3],
            atol=1e-4,
        ):
            raise ValueError(
                f"frames[{index}].camera_center_world_m: must match T_world_camera[:3, 3]"
            )
    elif T_world_camera is not None:
        camera_center_world_m = T_world_camera[:3, 3].copy()
    source_metadata = _optional_mapping(raw, "source_metadata")
    return RecordingFrame(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        rgb_path=rgb_path,
        depth_path=depth_path,
        depth_scale=depth_scale,
        K=K.astype(np.float32, copy=False),
        T_world_camera=(
            None if T_world_camera is None else T_world_camera.astype(np.float32, copy=False)
        ),
        camera_center_world_m=(
            None
            if camera_center_world_m is None
            else camera_center_world_m.astype(np.float32, copy=False)
        ),
        source_metadata=source_metadata,
    )


def _validate_frame_sequence(
    manifest: Mapping[str, object],
    frames: tuple[RecordingFrame, ...],
) -> None:
    frame_count = _non_negative_int(manifest, "frame_count")
    if frame_count != len(frames):
        raise ValueError("manifest.frame_count: must match frames.jsonl record count")
    seen_ids: set[int] = set()
    previous_timestamp: float | None = None
    for index, frame in enumerate(frames):
        if frame.frame_id in seen_ids:
            raise ValueError(f"frames[{index}].frame_id: duplicate frame id {frame.frame_id}")
        seen_ids.add(frame.frame_id)
        if previous_timestamp is not None and frame.timestamp_s < previous_timestamp:
            raise ValueError("frames.jsonl: records must be chronological by timestamp_s")
        previous_timestamp = frame.timestamp_s


def _path_roots(recording_root: Path, manifest: Mapping[str, object]) -> tuple[Path, ...]:
    roots = [recording_root.resolve()]
    external = manifest.get("external_roots")
    if isinstance(external, Mapping):
        for value in external.values():
            if isinstance(value, str) and value:
                roots.append(Path(value).resolve())
    return tuple(roots)


def _validate_safe_relative_path(field_name: str, value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name}: unsafe path {value!r}")
    if not value or value.startswith(("/", "\\")):
        raise ValueError(f"{field_name}: must be a safe relative path")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: failed to read JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(dict[str, object], payload)


def _load_frames_jsonl(path: Path) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                frames.append(cast(dict[str, object], payload))
    except OSError as exc:
        raise ValueError(f"{path}: failed to read frames JSONL: {exc}") from exc
    return frames


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(record), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _float_array(
    value: object, shape: tuple[int, ...], *, field_name: str
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name}: expected finite shape {shape}, got {array.shape}")
    return array


def _string_field(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string")
    return value


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string when present")
    return value


def _mapping_field(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}: must be a mapping")
    return value


def _optional_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key}: must be a mapping when present")
    return dict(value)


def _bool_field(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key}: must be a bool")
    return value


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = _int_field(mapping, key)
    if value <= 0:
        raise ValueError(f"{key}: must be positive")
    return value


def _non_negative_int(mapping: Mapping[str, object], key: str) -> int:
    value = _int_field(mapping, key)
    if value < 0:
        raise ValueError(f"{key}: must be non-negative")
    return value


def _finite_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key}: must be numeric")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{key}: must be finite")
    return scalar


def _optional_positive_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key}: must be numeric when present")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{key}: must be positive when present")
    return scalar


__all__ = [
    "RECORDING_COORDINATE_FRAME",
    "RECORDING_FORMAT_NAME",
    "RECORDING_FORMAT_VERSION",
    "RECORDING_FRAMES_FILENAME",
    "RECORDING_MANIFEST_FILENAME",
    "Atlas3RRecording",
    "RecordingFrame",
    "load_recording",
    "recording_truth_boundary",
    "resolve_recording_path",
    "validate_recording_folder",
    "write_recording_files",
]
