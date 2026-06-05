"""Measured temporal cache generation for SMGT-v2 training."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.api.validation import validate_intrinsics, validate_transform
from atlas3r.recording.observations import observation_from_recording_frame
from atlas3r.recording.schema import load_recording
from atlas3r.runtime.rgb_student_preprocess import resize_nearest, scale_intrinsics

FORMAT_NAME = "atlas3r_measured_temporal_cache"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "atlas3r_measured_temporal_cache.json"
FORBIDDEN_CACHE_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}


@dataclass(frozen=True)
class MeasuredTemporalCacheBuildConfig:
    recording: Path
    output: Path
    clip_length: int = 8
    clip_stride: int = 4
    image_size: tuple[int, int] = (160, 224)
    depth_sigma_floor_m: float = 0.02


def measured_temporal_cache_truth_boundary() -> dict[str, object]:
    return {
        "diagnostic_only": True,
        "learned_inference": False,
        "student_rgb_only_used": False,
        "observed_only": True,
        "predicted_completion": False,
        "hidden_geometry_measured": False,
        "teacher_geometry_used": False,
        "measured_depth_used": False,
        "measured_pose_used": False,
        "measured_depth_used_for_training": True,
        "measured_pose_used_for_training": True,
        "measured_training_used": True,
        "pseudo_training_used": False,
        "metric_scale_source": "measured_recording_training_target",
        "rgb_only_mapping_ready": False,
        "realtime_claim": False,
        "final_smgt": False,
    }


def build_measured_temporal_cache(config: MeasuredTemporalCacheBuildConfig) -> dict[str, object]:
    _validate_build_config(config)
    recording = load_recording(config.recording)
    if not bool(recording.manifest["depth_present"]):
        raise ValueError(f"{config.recording}: measured depth is required")
    if not bool(recording.manifest["pose_present"]):
        raise ValueError(f"{config.recording}: measured T_world_camera is required")
    observations = [
        observation_from_recording_frame(
            recording,
            frame,
            depth_sigma_floor_m=config.depth_sigma_floor_m,
        )
        for frame in recording.frames
    ]
    if len(observations) < config.clip_length:
        raise ValueError("measured temporal cache: not enough frames for one clip")
    config.output.mkdir(parents=True, exist_ok=True)
    clips_dir = config.output / "clips"
    clips_dir.mkdir(exist_ok=True)
    height, width = config.image_size
    truth = measured_temporal_cache_truth_boundary()
    clip_entries: list[dict[str, object]] = []
    for clip_id, start in enumerate(
        range(0, len(observations) - config.clip_length + 1, config.clip_stride)
    ):
        clip = observations[start : start + config.clip_length]
        rel = f"clips/clip_{clip_id:06d}.npz"
        _write_clip_npz(
            config.output / rel,
            clip,
            clip_id=clip_id,
            target_size=config.image_size,
            truth_flags=truth,
        )
        frame_ids = [obs.frame_id for obs in clip]
        clip_entries.append(
            {
                "clip_id": clip_id,
                "path": rel,
                "frame_ids": frame_ids,
                "start_frame_id": frame_ids[0],
                "end_frame_id": frame_ids[-1],
            }
        )
    manifest = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_recording": str(config.recording),
        "source_dataset": str(recording.manifest["source_dataset"]),
        "source_sequence": str(recording.manifest["source_sequence"]),
        "frame_count": len(observations),
        "clip_count": len(clip_entries),
        "clip_length": config.clip_length,
        "clip_stride": config.clip_stride,
        "image_size": [height, width],
        "image_height": height,
        "image_width": width,
        "measured_depth_used_for_training": True,
        "measured_pose_used_for_training": True,
        "target_weights": {
            "depth_target_weight": 1.0,
            "pose_target_weight": 1.0,
            "confidence_target_weight": 1.0,
        },
        "truth_boundary": truth,
        "arrays_schema": _arrays_schema(
            height=height,
            width=width,
            clip_length=config.clip_length,
        ),
        "no_model_weights_in_cache": True,
        "clips": clip_entries,
        "inspect_report": "inspect_report.json",
        "inspect_report_markdown": "inspect_report.md",
    }
    _write_json(config.output / MANIFEST_FILENAME, manifest)
    report = inspect_measured_temporal_cache(config.output)
    _write_json(config.output / "inspect_report.json", report)
    _write_inspect_markdown(config.output / "inspect_report.md", report)
    return report


def inspect_measured_temporal_cache(cache_dir: str | Path) -> dict[str, object]:
    root = Path(cache_dir)
    manifest = _read_manifest(root)
    clips = _clips(manifest)
    clip_summaries = []
    for entry in clips:
        rel = _safe_relative_path(str(entry["path"]), "clips[].path")
        payload = _load_clip(root / rel)
        clip_summaries.append(_validate_clip_payload(payload, entry, manifest))
    _validate_no_forbidden_cache_files(root)
    weights = cast(dict[str, object], manifest["target_weights"])
    return {
        "format_name": "atlas3r_measured_temporal_cache_inspection",
        "format_version": 1,
        "cache": str(root),
        "cache_format_name": manifest["format_name"],
        "cache_format_version": manifest["format_version"],
        "source_recording": manifest["source_recording"],
        "clip_count": len(clip_summaries),
        "frame_count": manifest["frame_count"],
        "image_size": manifest["image_size"],
        "depth_target_weight": float(cast(float, weights["depth_target_weight"])),
        "pose_target_weight": float(cast(float, weights["pose_target_weight"])),
        "confidence_target_weight": float(cast(float, weights["confidence_target_weight"])),
        "truth_boundary": manifest["truth_boundary"],
        "no_model_weights_in_cache": manifest["no_model_weights_in_cache"],
        "clip_summaries": clip_summaries,
        "validation_passed": True,
    }


def format_measured_temporal_cache_inspection(cache_dir: str | Path) -> str:
    return json.dumps(inspect_measured_temporal_cache(cache_dir), indent=2, sort_keys=True) + "\n"


class MeasuredTemporalCacheDataset:
    """Array dataset for measured temporal cache clips."""

    def __init__(self, cache_dir: str | Path, *, indices: Sequence[int] | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.inspection = inspect_measured_temporal_cache(self.cache_dir)
        self.manifest = _read_manifest(self.cache_dir)
        all_entries = _clips(self.manifest)
        if indices is None:
            self.clip_entries = tuple(all_entries)
        else:
            self.clip_entries = tuple(all_entries[int(index)] for index in indices)
        if not self.clip_entries:
            raise ValueError("MeasuredTemporalCacheDataset: no clips selected")

    def __len__(self) -> int:
        return len(self.clip_entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= len(self.clip_entries):
            raise IndexError(index)
        entry = self.clip_entries[index]
        payload = _load_clip(self.cache_dir / str(entry["path"]))
        metadata = _clip_metadata(payload)
        weights = cast(dict[str, object], self.manifest["target_weights"])
        return {
            "rgb_u8": np.asarray(payload["rgb_u8"], dtype=np.uint8),
            "K": np.asarray(payload["K"], dtype=np.float32),
            "T_world_camera": np.asarray(payload["T_world_camera"], dtype=np.float32),
            "depth_m": np.asarray(payload["depth_m"], dtype=np.float32),
            "depth_sigma_m": np.asarray(payload["depth_sigma_m"], dtype=np.float32),
            "confidence": np.asarray(payload["confidence"], dtype=np.float32),
            "valid_mask": np.asarray(payload["valid_mask"], dtype=np.bool_),
            "frame_ids": np.asarray(payload["frame_ids"], dtype=np.int64),
            "timestamps_ns": np.asarray(payload["timestamps_ns"], dtype=np.int64),
            "target_source": "measured",
            "truth_flags": cast(dict[str, object], metadata["truth_flags"]),
            "depth_target_weight": float(cast(float, weights["depth_target_weight"])),
            "pose_target_weight": float(cast(float, weights["pose_target_weight"])),
            "confidence_target_weight": float(cast(float, weights["confidence_target_weight"])),
        }

    def cache_summary(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir),
            "selected_clip_count": len(self.clip_entries),
            "source_clip_count": int(cast(int, self.manifest["clip_count"])),
            "image_height": int(cast(int, self.manifest["image_height"])),
            "image_width": int(cast(int, self.manifest["image_width"])),
            "clip_length": int(cast(int, self.manifest["clip_length"])),
            "target_source": "measured",
            "truth_boundary": dict(cast(dict[str, object], self.manifest["truth_boundary"])),
        }


def _write_clip_npz(
    path: Path,
    observations: Sequence[Any],
    *,
    clip_id: int,
    target_size: tuple[int, int],
    truth_flags: Mapping[str, object],
) -> None:
    height, width = target_size
    rgbs: list[npt.NDArray[np.uint8]] = []
    depths: list[npt.NDArray[np.float32]] = []
    sigmas: list[npt.NDArray[np.float32]] = []
    confidences: list[npt.NDArray[np.float32]] = []
    masks: list[npt.NDArray[np.bool_]] = []
    intrinsics: list[npt.NDArray[np.float32]] = []
    transforms: list[npt.NDArray[np.float32]] = []
    for obs in observations:
        if obs.rgb_u8 is None:
            raise ValueError(f"frame {obs.frame_id}: measured cache requires RGB")
        original_size = (int(obs.camera.height), int(obs.camera.width))
        rgbs.append(resize_nearest(obs.rgb_u8, height=height, width=width))
        depths.append(_resize_hw_nearest(obs.depth_m, height=height, width=width))
        sigmas.append(_resize_hw_nearest(obs.depth_sigma_m, height=height, width=width))
        confidences.append(_resize_hw_nearest(obs.confidence, height=height, width=width))
        static_mask = np.asarray(obs.static_mask, dtype=np.bool_)
        masks.append(_resize_hw_nearest(static_mask, height=height, width=width).astype(np.bool_))
        intrinsics.append(scale_intrinsics(obs.camera.K, original_size, target_size))
        transforms.append(obs.pose.T_world_camera.astype(np.float32, copy=True))
    metadata = {
        "clip_id": clip_id,
        "truth_flags": dict(truth_flags),
        "target_source": "measured",
    }
    np.savez_compressed(
        path,
        frame_ids=np.asarray([obs.frame_id for obs in observations], dtype=np.int64),
        timestamps_ns=np.asarray([obs.pose.timestamp_ns for obs in observations], dtype=np.int64),
        rgb_u8=np.stack(rgbs, axis=0),
        K=np.stack(intrinsics, axis=0).astype(np.float32),
        T_world_camera=np.stack(transforms, axis=0).astype(np.float32),
        depth_m=np.stack(depths, axis=0).astype(np.float32),
        depth_sigma_m=np.stack(sigmas, axis=0).astype(np.float32),
        confidence=np.stack(confidences, axis=0).astype(np.float32),
        valid_mask=np.stack(masks, axis=0).astype(np.bool_),
        target_source=np.asarray("measured"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _validate_clip_payload(
    payload: Mapping[str, npt.NDArray[Any]],
    entry: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    clip_length = _int(manifest["clip_length"], "clip_length")
    height = _int(manifest["image_height"], "image_height")
    width = _int(manifest["image_width"], "image_width")
    frame_ids = _array(payload, "frame_ids", np.int64, (clip_length,))
    if not np.all(frame_ids[:-1] <= frame_ids[1:]):
        raise ValueError("frame_ids: must be sorted within clip")
    _array(payload, "timestamps_ns", np.int64, (clip_length,))
    _array(payload, "rgb_u8", np.uint8, (clip_length, height, width, 3))
    K = _finite_float(payload, "K", (clip_length, 3, 3))
    transforms = _finite_float(payload, "T_world_camera", (clip_length, 4, 4))
    for index in range(clip_length):
        validate_intrinsics(f"K[{index}]", K[index])
        validate_transform(f"T_world_camera[{index}]", transforms[index])
    depth = _finite_float(payload, "depth_m", (clip_length, height, width))
    sigma = _finite_float(payload, "depth_sigma_m", (clip_length, height, width))
    confidence = _finite_float(payload, "confidence", (clip_length, height, width))
    if np.any(depth < 0.0) or np.any(sigma < 0.0):
        raise ValueError("depth_m/depth_sigma_m: must be non-negative")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence: must be in [0,1]")
    valid = _array(payload, "valid_mask", np.bool_, (clip_length, height, width))
    target_source = str(np.asarray(payload.get("target_source", np.asarray(""))).item())
    if target_source != "measured":
        raise ValueError("target_source: expected measured")
    metadata = _clip_metadata(payload)
    truth = cast(dict[str, object], metadata["truth_flags"])
    if truth.get("measured_depth_used_for_training") is not True:
        raise ValueError("truth_flags.measured_depth_used_for_training must be true")
    if truth.get("measured_pose_used_for_training") is not True:
        raise ValueError("truth_flags.measured_pose_used_for_training must be true")
    if (
        truth.get("measured_depth_used") is not False
        or truth.get("measured_pose_used") is not False
    ):
        raise ValueError("truth_flags: measured mapping flags must be false")
    return {
        "clip_id": entry["clip_id"],
        "frame_ids": frame_ids.astype(int).tolist(),
        "valid_pixel_ratio": float(np.count_nonzero(valid) / max(valid.size, 1)),
    }


def _read_manifest(root: Path) -> dict[str, object]:
    try:
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{root}: invalid measured temporal cache manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest: must be a JSON object")
    if (
        manifest.get("format_name") != FORMAT_NAME
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("manifest: unexpected measured temporal cache format")
    if manifest.get("no_model_weights_in_cache") is not True:
        raise ValueError("manifest.no_model_weights_in_cache: must be true")
    for key in ("measured_depth_used_for_training", "measured_pose_used_for_training"):
        if manifest.get(key) is not True:
            raise ValueError(f"manifest.{key}: must be true")
    return cast(dict[str, object], manifest)


def _load_clip(path: Path) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"{path}: invalid measured temporal cache clip: {exc}") from exc


def _clip_metadata(payload: Mapping[str, npt.NDArray[Any]]) -> dict[str, object]:
    if "metadata_json" not in payload:
        raise ValueError("metadata_json: required in clip payload")
    metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json: must decode to an object")
    return cast(dict[str, object], metadata)


def _resize_hw_nearest(
    array: npt.NDArray[Any],
    *,
    height: int,
    width: int,
) -> npt.NDArray[Any]:
    if array.shape == (height, width):
        return array.copy()
    y = np.linspace(0, array.shape[0] - 1, height).round().astype(np.int64)
    x = np.linspace(0, array.shape[1] - 1, width).round().astype(np.int64)
    return cast(npt.NDArray[Any], array[y[:, None], x[None, :]].copy())


def _clips(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    value = manifest.get("clips")
    if not isinstance(value, list) or not value:
        raise ValueError("manifest.clips: must be a non-empty list")
    return [cast(dict[str, object], item) for item in value]


def _arrays_schema(*, height: int, width: int, clip_length: int) -> dict[str, object]:
    return {
        "rgb_u8": f"uint8[{clip_length},{height},{width},3]",
        "K": f"float32[{clip_length},3,3]",
        "T_world_camera": f"float32[{clip_length},4,4]",
        "depth_m": f"float32[{clip_length},{height},{width}]",
        "depth_sigma_m": f"float32[{clip_length},{height},{width}]",
        "confidence": f"float32[{clip_length},{height},{width}]",
        "valid_mask": f"bool[{clip_length},{height},{width}]",
        "target_source": "measured",
    }


def _array(
    payload: Mapping[str, npt.NDArray[Any]],
    key: str,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> npt.NDArray[Any]:
    if key not in payload:
        raise ValueError(f"{key}: missing from clip payload")
    array = np.asarray(payload[key])
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError(
            f"{key}: expected {np.dtype(dtype)} {shape}, got {array.dtype} {array.shape}"
        )
    return array


def _finite_float(
    payload: Mapping[str, npt.NDArray[Any]],
    key: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    if key not in payload:
        raise ValueError(f"{key}: missing from clip payload")
    array = np.asarray(payload[key], dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: expected finite float32 {shape}")
    return array


def _safe_relative_path(value: str, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"{field_name}: must be a safe relative path")
    return path


def _validate_no_forbidden_cache_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_CACHE_SUFFIXES:
            raise ValueError(f"{path}: model weights or generated checkpoints are not allowed")


def _validate_build_config(config: MeasuredTemporalCacheBuildConfig) -> None:
    if config.clip_length <= 0:
        raise ValueError("clip_length: must be positive")
    if config.clip_stride <= 0:
        raise ValueError("clip_stride: must be positive")
    if config.image_size[0] <= 0 or config.image_size[1] <= 0:
        raise ValueError("image_size: dimensions must be positive")
    if config.depth_sigma_floor_m <= 0.0:
        raise ValueError("depth_sigma_floor_m: must be positive")


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_inspect_markdown(path: Path, report: Mapping[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Measured Temporal Cache Inspection",
                "",
                f"- Validation passed: `{report['validation_passed']}`",
                f"- Clips: `{report['clip_count']}`",
                f"- Frames: `{report['frame_count']}`",
                f"- Depth target weight: `{report['depth_target_weight']}`",
                f"- Pose target weight: `{report['pose_target_weight']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name}: must be an integer")
    return value


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "MeasuredTemporalCacheBuildConfig",
    "MeasuredTemporalCacheDataset",
    "build_measured_temporal_cache",
    "format_measured_temporal_cache_inspection",
    "inspect_measured_temporal_cache",
    "measured_temporal_cache_truth_boundary",
]
