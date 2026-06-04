"""Generic measured sensor-folder import boundary for Atlas3R recordings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics, validate_transform
from atlas3r.recording.schema import (
    RECORDING_COORDINATE_FRAME,
    RECORDING_FORMAT_NAME,
    RECORDING_FORMAT_VERSION,
    Atlas3RRecording,
    recording_truth_boundary,
    write_recording_files,
)

SENSOR_CAPTURE_FORMAT_NAME = "atlas3r_sensor_capture_input"
SENSOR_CAPTURE_FORMAT_VERSION = 1
SENSOR_CAPTURE_MANIFEST_FILENAME = "sensor_capture.json"
SENSOR_CAPTURE_FRAMES_FILENAME = "frames.jsonl"


@dataclass(frozen=True)
class SensorFolderRecordingImportConfig:
    input: Path
    output: Path


def recording_from_sensor_folder(config: SensorFolderRecordingImportConfig) -> dict[str, object]:
    """Validate a generic measured sensor-capture folder and write a recording."""

    input_root = config.input
    capture = _load_json_object(input_root / SENSOR_CAPTURE_MANIFEST_FILENAME)
    _validate_sensor_capture_manifest(capture)
    raw_frames = _load_jsonl_objects(input_root / SENSOR_CAPTURE_FRAMES_FILENAME)
    frames = [
        _sensor_frame_to_recording_frame(raw, capture=capture, input_root=input_root, index=index)
        for index, raw in enumerate(raw_frames)
    ]
    recording_manifest = _recording_manifest(
        frame_count=len(frames),
        width=_int_field(capture, "width"),
        height=_int_field(capture, "height"),
        source_dataset=_string_field(capture, "source_dataset"),
        source_sequence=_string_field(capture, "source_sequence"),
        depth_present=_bool_field(capture, "depth_present"),
        pose_present=_bool_field(capture, "pose_present"),
        capture_metadata={
            "importer": "sensor_folder",
            "source_capture_path": str(input_root),
            "source_format_name": SENSOR_CAPTURE_FORMAT_NAME,
            "source_format_version": SENSOR_CAPTURE_FORMAT_VERSION,
            "source_truth_boundary": dict(_mapping_field(capture, "truth_boundary")),
        },
        known_calibration_metadata={
            **dict(_mapping_field(capture, "calibration_metadata")),
            "coordinate_frame": RECORDING_COORDINATE_FRAME,
        },
        external_roots={"sensor_capture_root": str(input_root)},
    )
    recording = write_recording_files(config.output, manifest=recording_manifest, frames=frames)
    return _import_result(
        "atlas3r_sensor_folder_recording_import_result",
        recording,
        source=str(input_root),
        split="",
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


def _validate_sensor_capture_manifest(capture: Mapping[str, object]) -> None:
    if capture.get("format_name") != SENSOR_CAPTURE_FORMAT_NAME:
        raise ValueError(f"sensor_capture.format_name: expected {SENSOR_CAPTURE_FORMAT_NAME!r}")
    if _int_field(capture, "format_version") != SENSOR_CAPTURE_FORMAT_VERSION:
        raise ValueError("sensor_capture.format_version: unsupported version")
    if _string_field(capture, "coordinate_frame") != RECORDING_COORDINATE_FRAME:
        raise ValueError(
            "sensor_capture.coordinate_frame: input exporter must pre-convert to "
            f"{RECORDING_COORDINATE_FRAME!r}"
        )
    _positive_int(capture, "width")
    _positive_int(capture, "height")
    _string_field(capture, "source_dataset")
    _string_field(capture, "source_sequence")
    _bool_field(capture, "depth_present")
    _bool_field(capture, "pose_present")
    calibration = _mapping_field(capture, "calibration_metadata")
    for key in (
        "calibration_source",
        "intrinsics_source",
        "pose_source",
        "depth_source",
        "depth_units",
        "metric_scale_source",
    ):
        _string_field(calibration, key)
    truth_boundary = _mapping_field(capture, "truth_boundary")
    expected = {
        "accuracy_report": False,
        "diagnostic_only": True,
        "hidden_geometry_measured": False,
        "performance_report": False,
        "realtime_claim": False,
    }
    for key, value in expected.items():
        if truth_boundary.get(key) != value:
            raise ValueError(f"sensor_capture.truth_boundary.{key}: expected {value!r}")


def _sensor_frame_to_recording_frame(
    raw: Mapping[str, object],
    *,
    capture: Mapping[str, object],
    input_root: Path,
    index: int,
) -> dict[str, object]:
    width = _int_field(capture, "width")
    height = _int_field(capture, "height")
    frame_id = _non_negative_int(raw, "frame_id")
    timestamp_s = _finite_float_field(raw, "timestamp_s")
    rgb_path = _safe_relative_path(_string_field(raw, "rgb_path"), field_name="rgb_path")
    _validate_rgb_payload_dimensions(input_root / rgb_path, width=width, height=height)
    depth_present = _bool_field(capture, "depth_present")
    pose_present = _bool_field(capture, "pose_present")
    depth_path_value = _optional_string(raw, "depth_path")
    depth_path: Path | None = None
    depth_scale = _optional_positive_float(raw, "depth_scale")
    if depth_present:
        if depth_path_value is None:
            raise ValueError(f"frames[{index}].depth_path: required when depth_present=true")
        depth_path = _safe_relative_path(depth_path_value, field_name="depth_path")
        _validate_depth_payload_dimensions(
            input_root / depth_path,
            width=width,
            height=height,
            depth_scale=depth_scale,
            field_name=f"frames[{index}].depth_path",
        )
    elif depth_path_value is not None:
        raise ValueError(f"frames[{index}].depth_path: forbidden when depth_present=false")
    K = _array_list(raw.get("K"), (3, 3))
    validate_intrinsics(f"frames[{index}].K", K)
    T_world_camera = None
    camera_center_world_m = None
    if "T_world_camera" in raw:
        T_world_camera = _array_list(raw["T_world_camera"], (4, 4))
        validate_transform(f"frames[{index}].T_world_camera", T_world_camera)
    if pose_present and T_world_camera is None:
        raise ValueError(f"frames[{index}].T_world_camera: required when pose_present=true")
    if not pose_present and T_world_camera is not None:
        raise ValueError(f"frames[{index}].T_world_camera: forbidden when pose_present=false")
    if "camera_center_world_m" in raw:
        camera_center_world_m = _array_list(raw["camera_center_world_m"], (3,))
        if T_world_camera is None:
            raise ValueError(
                f"frames[{index}].camera_center_world_m: requires T_world_camera when present"
            )
        if not np.allclose(camera_center_world_m, T_world_camera[:3, 3], atol=1e-4):
            raise ValueError(
                f"frames[{index}].camera_center_world_m: must match T_world_camera[:3, 3]"
            )
    elif T_world_camera is not None:
        camera_center_world_m = T_world_camera[:3, 3].copy()
    source_metadata = _mapping_field(raw, "source_metadata")
    record: dict[str, object] = {
        "K": K.tolist(),
        "frame_id": frame_id,
        "rgb_path": str(rgb_path).replace("\\", "/"),
        "source_metadata": {
            **dict(source_metadata),
            "source_format": SENSOR_CAPTURE_FORMAT_NAME,
            "source_root_key": "sensor_capture_root",
        },
        "timestamp_s": timestamp_s,
    }
    if depth_path is not None:
        record["depth_path"] = str(depth_path).replace("\\", "/")
    if depth_scale is not None:
        record["depth_scale"] = depth_scale
    if T_world_camera is not None:
        record["T_world_camera"] = T_world_camera.tolist()
    if camera_center_world_m is not None:
        record["camera_center_world_m"] = camera_center_world_m.tolist()
    return record


def _validate_rgb_payload_dimensions(path: Path, *, width: int, height: int) -> None:
    _require_existing_file(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "rgb_u8" not in payload.files:
                raise ValueError(f"{path}: RGB NPZ must contain rgb_u8")
            rgb = np.asarray(payload["rgb_u8"])
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"{path}: rgb_u8 must be uint8 HxWx3 at capture dimensions")
    elif suffix == ".png":
        _validate_png_dimensions(path, width=width, height=height)
    elif suffix == ".ppm":
        _validate_ppm_dimensions(path, width=width, height=height)
    else:
        raise ValueError(f"{path}: unsupported RGB extension {suffix!r}")


def _validate_depth_payload_dimensions(
    path: Path,
    *,
    width: int,
    height: int,
    depth_scale: float | None,
    field_name: str,
) -> None:
    _require_existing_file(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "depth_m" not in payload.files:
                raise ValueError(f"{path}: depth NPZ must contain depth_m")
            depth = np.asarray(payload["depth_m"], dtype=np.float32)
            valid = (
                np.asarray(payload["valid_depth_mask"], dtype=np.bool_)
                if "valid_depth_mask" in payload.files
                else None
            )
            confidence = (
                np.asarray(payload["confidence"], dtype=np.float32)
                if "confidence" in payload.files
                else None
            )
        _validate_depth_arrays(
            path,
            depth=depth,
            valid=valid,
            confidence=confidence,
            height=height,
            width=width,
        )
    elif suffix == ".png":
        if depth_scale is None:
            raise ValueError(f"{field_name}: PNG depth requires positive depth_scale")
        _validate_png_dimensions(path, width=width, height=height)
    else:
        raise ValueError(f"{path}: unsupported depth extension {suffix!r}")


def _validate_depth_arrays(
    path: Path,
    *,
    depth: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_] | None,
    confidence: npt.NDArray[np.float32] | None,
    height: int,
    width: int,
) -> None:
    if depth.shape != (height, width) or not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError(f"{path}: depth_m must be finite non-negative HxW")
    if valid is not None and valid.shape != (height, width):
        raise ValueError(f"{path}: valid_depth_mask must be HxW")
    if confidence is not None:
        confidence_array = np.asarray(confidence, dtype=np.float32)
        if (
            confidence_array.shape != (height, width)
            or not np.all(np.isfinite(confidence_array))
            or np.any((confidence_array < 0.0) | (confidence_array > 1.0))
        ):
            raise ValueError(f"{path}: confidence must be finite HxW values in [0, 1]")


def _validate_png_dimensions(path: Path, *, width: int, height: int) -> None:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path}: expected a PNG image")
    png_width = int.from_bytes(header[16:20], byteorder="big")
    png_height = int.from_bytes(header[20:24], byteorder="big")
    if (png_width, png_height) != (width, height):
        raise ValueError(
            f"{path}: expected image dimensions {(width, height)}, got {(png_width, png_height)}"
        )


def _validate_ppm_dimensions(path: Path, *, width: int, height: int) -> None:
    tokens: list[bytes] = []
    with path.open("rb") as handle:
        while len(tokens) < 4:
            token = handle.readline()
            if token.startswith(b"#"):
                continue
            tokens.extend(token.split())
    if len(tokens) < 4 or tokens[0] != b"P6":
        raise ValueError(f"{path}: expected binary P6 PPM")
    try:
        ppm_width = int(tokens[1])
        ppm_height = int(tokens[2])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid PPM dimensions") from exc
    if (ppm_width, ppm_height) != (width, height):
        raise ValueError(
            f"{path}: expected image dimensions {(width, height)}, got {(ppm_width, ppm_height)}"
        )


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


def _require_existing_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"{path}: file does not exist")


def _safe_relative_path(value: str, *, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value or value.startswith(("/", "\\")):
        raise ValueError(f"{field_name}: unsafe path {value!r}")
    return path


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


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string when present")
    return value


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


def _finite_float_field(mapping: Mapping[str, object], key: str) -> float:
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


def _array_list(value: object, shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"array: expected finite shape {shape}, got {array.shape}")
    return array


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


def _load_jsonl_objects(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
                records.append(cast(dict[str, object], payload))
    except OSError as exc:
        raise ValueError(f"{path}: failed to read JSONL: {exc}") from exc
    return records


__all__ = [
    "SENSOR_CAPTURE_FORMAT_NAME",
    "SENSOR_CAPTURE_FORMAT_VERSION",
    "SensorFolderRecordingImportConfig",
    "recording_from_sensor_folder",
]
