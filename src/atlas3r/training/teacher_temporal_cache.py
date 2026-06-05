"""Validated temporal cache for RGB teacher pseudo-label training data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.mapping.observations import DepthObservation
from atlas3r.runtime.rgb_teacher_outputs import rgb_teacher_truth_boundary

FORMAT_NAME = "atlas3r_teacher_temporal_cache"
FORMAT_VERSION = 1
FORMAT_ALIAS = "atlas3r_teacher_temporal_cache_v1"
MANIFEST_FILENAME = "atlas3r_teacher_temporal_cache.json"
FORBIDDEN_CACHE_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}


def write_teacher_temporal_cache(
    cache_dir: Path,
    observations: Sequence[DepthObservation],
    *,
    source_rgb_teacher_run: Path,
    teacher_metadata: Mapping[str, object],
    stitch_mode: str,
    metric_scale_source: str,
    clip_length: int,
    clip_stride: int,
    teacher_name: str = "vggt",
) -> dict[str, object]:
    if clip_length <= 0:
        raise ValueError("clip_length: must be positive")
    if clip_stride <= 0:
        raise ValueError("clip_stride: must be positive")
    ordered = tuple(sorted(observations, key=lambda item: item.frame_id))
    if len(ordered) < clip_length:
        raise ValueError("teacher temporal cache: not enough observations for one clip")
    cache_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = cache_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    truth = rgb_teacher_truth_boundary(metric_scale_source=metric_scale_source)
    clip_entries: list[dict[str, object]] = []
    frame_index: dict[int, dict[str, object]] = {}
    for clip_id, start in enumerate(range(0, len(ordered) - clip_length + 1, clip_stride)):
        clip_obs = ordered[start : start + clip_length]
        rel = f"clips/clip_{clip_id:06d}.npz"
        _write_clip_npz(
            cache_dir / rel,
            clip_obs,
            clip_id=clip_id,
            truth_flags=truth,
        )
        frame_ids = [obs.frame_id for obs in clip_obs]
        clip_entries.append(
            {
                "clip_id": clip_id,
                "path": rel,
                "frame_ids": frame_ids,
                "start_frame_id": frame_ids[0],
                "end_frame_id": frame_ids[-1],
            }
        )
        for obs in clip_obs:
            record = frame_index.setdefault(
                obs.frame_id,
                {
                    "frame_id": obs.frame_id,
                    "timestamp_ns": obs.pose.timestamp_ns,
                    "clip_ids": [],
                    "pseudo_submap_id": _pseudo_submap_id(obs),
                },
            )
            cast(list[int], record["clip_ids"]).append(clip_id)
    height = int(ordered[0].camera.height)
    width = int(ordered[0].camera.width)
    manifest = {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source_rgb_teacher_run": source_rgb_teacher_run.name,
        "teacher_name": teacher_name,
        "teacher_model_source": str(teacher_metadata.get("model_source", "unknown")),
        "teacher_checkpoint": teacher_metadata.get("checkpoint"),
        "device": str(teacher_metadata.get("resolved_device", "unknown")),
        "dtype": teacher_metadata.get("dtype"),
        "image_size": [height, width],
        "image_height": height,
        "image_width": width,
        "stitch_mode": stitch_mode,
        "metric_scale_source": metric_scale_source,
        "clip_length": clip_length,
        "clip_stride": clip_stride,
        "frame_count": len(ordered),
        "clip_count": len(clip_entries),
        "truth_boundary": truth,
        "target_weights": {
            "depth_target_weight": 0.25,
            "pose_target_weight": 0.25,
            "measured_depth_target_weight": 1.0,
            "measured_pose_target_weight": 1.0,
        },
        "arrays_schema": _arrays_schema(height=height, width=width, clip_length=clip_length),
        "source_paths_relative": [],
        "no_model_weights_in_cache": True,
        "clips": clip_entries,
        "frame_index": "frame_index.jsonl",
        "inspect_report": "inspect_report.json",
        "inspect_report_markdown": "inspect_report.md",
    }
    _write_json(cache_dir / MANIFEST_FILENAME, manifest)
    _write_jsonl(
        cache_dir / "frame_index.jsonl",
        sorted(frame_index.values(), key=lambda r: _int(r["frame_id"], "frame_id")),
    )
    report = inspect_teacher_temporal_cache(cache_dir)
    _write_json(cache_dir / "inspect_report.json", report)
    _write_inspect_markdown(cache_dir / "inspect_report.md", report)
    return report


def inspect_teacher_temporal_cache(cache_dir: str | Path) -> dict[str, object]:
    root = Path(cache_dir)
    manifest = _read_manifest(root)
    clips = _clips(manifest)
    clip_summaries: list[dict[str, object]] = []
    for entry in clips:
        rel = _safe_relative_path(str(entry["path"]), "clips[].path")
        payload = _load_clip(root / rel)
        clip_summaries.append(_validate_clip_payload(payload, entry, manifest))
    _validate_no_forbidden_cache_files(root)
    depth_weight = float(cast(Any, manifest["target_weights"])["depth_target_weight"])
    pose_weight = float(cast(Any, manifest["target_weights"])["pose_target_weight"])
    if depth_weight > 0.25 or pose_weight > 0.25:
        raise ValueError("target_weights: pseudo depth/pose weights must be <= 0.25")
    return {
        "format_name": "atlas3r_teacher_temporal_cache_inspection",
        "format_version": 1,
        "cache": str(root),
        "cache_format_name": manifest["format_name"],
        "cache_format_version": manifest["format_version"],
        "clip_count": len(clip_summaries),
        "frame_count": manifest["frame_count"],
        "stitch_mode": manifest["stitch_mode"],
        "metric_scale_source": manifest["metric_scale_source"],
        "depth_target_weight": depth_weight,
        "pose_target_weight": pose_weight,
        "truth_boundary": manifest["truth_boundary"],
        "no_model_weights_in_cache": manifest["no_model_weights_in_cache"],
        "clip_summaries": clip_summaries,
        "validation_passed": True,
    }


def format_teacher_temporal_cache_inspection(cache_dir: str | Path) -> str:
    return json.dumps(inspect_teacher_temporal_cache(cache_dir), indent=2, sort_keys=True) + "\n"


class TeacherTemporalCacheDataset:
    """Dependency-free array dataset for future SMGT temporal training."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.inspection = inspect_teacher_temporal_cache(self.cache_dir)
        self.manifest = _read_manifest(self.cache_dir)
        self.clip_entries = _clips(self.manifest)

    def __len__(self) -> int:
        return len(self.clip_entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= len(self.clip_entries):
            raise IndexError(index)
        entry = self.clip_entries[index]
        payload = _load_clip(self.cache_dir / str(entry["path"]))
        metadata = _clip_metadata(payload)
        target_weights = cast(dict[str, object], self.manifest["target_weights"])
        return {
            "rgb_u8": np.asarray(payload["rgb_u8"], dtype=np.uint8),
            "K": np.asarray(payload["K"], dtype=np.float32),
            "T_world_camera": np.asarray(payload["T_world_camera"], dtype=np.float32),
            "depth_m": np.asarray(payload["depth_m"], dtype=np.float32),
            "depth_sigma_m": np.asarray(payload["depth_sigma_m"], dtype=np.float32),
            "confidence": np.asarray(payload["confidence"], dtype=np.float32),
            "valid_mask": np.asarray(payload["valid_mask"], dtype=np.bool_),
            "teacher_valid": np.asarray(payload["teacher_valid"], dtype=np.bool_),
            "frame_ids": np.asarray(payload["frame_ids"], dtype=np.int64),
            "timestamps_ns": np.asarray(payload["timestamps_ns"], dtype=np.int64),
            "pseudo_submap_id": np.asarray(payload["pseudo_submap_id"], dtype=np.int32),
            "truth_flags": cast(dict[str, object], metadata["truth_flags"]),
            "depth_target_weight": float(cast(Any, target_weights["depth_target_weight"])),
            "pose_target_weight": float(cast(Any, target_weights["pose_target_weight"])),
        }


def _write_clip_npz(
    path: Path,
    observations: Sequence[DepthObservation],
    *,
    clip_id: int,
    truth_flags: Mapping[str, object],
) -> None:
    frame_ids = np.asarray([obs.frame_id for obs in observations], dtype=np.int64)
    timestamps_ns = np.asarray([obs.pose.timestamp_ns for obs in observations], dtype=np.int64)
    metadata = {
        "clip_id": clip_id,
        "truth_flags": dict(truth_flags),
        "diagnostic_only": True,
        "measured_depth_used": False,
        "measured_pose_used": False,
    }
    np.savez_compressed(
        path,
        frame_ids=frame_ids,
        timestamps_ns=timestamps_ns,
        rgb_u8=np.stack([_required_rgb(obs) for obs in observations], axis=0),
        K=np.stack([obs.camera.K.astype(np.float32, copy=False) for obs in observations], axis=0),
        T_world_camera=np.stack(
            [obs.pose.T_world_camera.astype(np.float32, copy=False) for obs in observations],
            axis=0,
        ),
        depth_m=np.stack(
            [obs.depth_m.astype(np.float32, copy=False) for obs in observations], axis=0
        ),
        depth_sigma_m=np.stack(
            [obs.depth_sigma_m.astype(np.float32, copy=False) for obs in observations],
            axis=0,
        ),
        confidence=np.stack(
            [obs.confidence.astype(np.float32, copy=False) for obs in observations],
            axis=0,
        ),
        valid_mask=np.stack(
            [np.asarray(obs.static_mask, dtype=np.bool_) for obs in observations], axis=0
        ),
        teacher_valid=np.ones((len(observations),), dtype=np.bool_),
        pseudo_submap_id=np.asarray(
            [_pseudo_submap_id(obs) for obs in observations], dtype=np.int32
        ),
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
    _finite_float(payload, "K", (clip_length, 3, 3))
    transforms = _finite_float(payload, "T_world_camera", (clip_length, 4, 4))
    _validate_transforms(transforms)
    depth = _finite_float(payload, "depth_m", (clip_length, height, width))
    sigma = _finite_float(payload, "depth_sigma_m", (clip_length, height, width))
    confidence = _finite_float(payload, "confidence", (clip_length, height, width))
    if np.any(depth < 0.0) or np.any(sigma < 0.0):
        raise ValueError("depth_m/depth_sigma_m: must be non-negative")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence: must be in [0,1]")
    _array(payload, "valid_mask", np.bool_, (clip_length, height, width))
    _array(payload, "teacher_valid", np.bool_, (clip_length,))
    _array(payload, "pseudo_submap_id", np.int32, (clip_length,))
    metadata = _clip_metadata(payload)
    truth = cast(dict[str, object], metadata["truth_flags"])
    if bool(truth.get("measured_depth_used")) or bool(truth.get("measured_pose_used")):
        raise ValueError("truth_flags: measured mapping flags must be false")
    return {
        "clip_id": entry["clip_id"],
        "frame_ids": frame_ids.astype(int).tolist(),
        "valid_pixel_ratio": float(
            np.count_nonzero(payload["valid_mask"]) / payload["valid_mask"].size
        ),
    }


def _read_manifest(root: Path) -> dict[str, object]:
    try:
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{root}: invalid teacher temporal cache manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest: must be a JSON object")
    if (
        manifest.get("format_name") != FORMAT_NAME
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("manifest: unexpected teacher temporal cache format")
    if manifest.get("no_model_weights_in_cache") is not True:
        raise ValueError("manifest.no_model_weights_in_cache: must be true")
    for rel in cast(list[object], manifest.get("source_paths_relative", [])):
        _safe_relative_path(str(rel), "source_paths_relative[]")
    return cast(dict[str, object], manifest)


def _load_clip(path: Path) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"{path}: invalid teacher temporal cache clip: {exc}") from exc


def _clip_metadata(payload: Mapping[str, npt.NDArray[Any]]) -> dict[str, object]:
    if "metadata_json" not in payload:
        raise ValueError("metadata_json: required in clip payload")
    metadata = json.loads(str(payload["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json: must decode to an object")
    return cast(dict[str, object], metadata)


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
    }


def _required_rgb(observation: DepthObservation) -> npt.NDArray[np.uint8]:
    if observation.rgb_u8 is None:
        raise ValueError(f"frame {observation.frame_id}: teacher cache requires RGB")
    return observation.rgb_u8.astype(np.uint8, copy=False)


def _pseudo_submap_id(observation: DepthObservation) -> int:
    stitching = observation.pose.diagnostics.get("stitching")
    if isinstance(stitching, Mapping):
        value = stitching.get("pseudo_submap_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


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


def _validate_transforms(transforms: npt.NDArray[np.float32]) -> None:
    expected = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(transforms[:, 3, :], expected[None, :], atol=1e-5):
        raise ValueError("T_world_camera: bottom row must be [0,0,0,1]")
    det = np.linalg.det(transforms[:, :3, :3].astype(np.float64))
    if np.any(np.abs(det) < 1e-6) or np.any(~np.isfinite(det)):
        raise ValueError("T_world_camera: rotation block is degenerate")


def _safe_relative_path(value: str, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"{field_name}: must be a safe relative path")
    return path


def _validate_no_forbidden_cache_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_CACHE_SUFFIXES:
            raise ValueError(f"{path}: model weights or generated checkpoints are not allowed")


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def _write_inspect_markdown(path: Path, report: Mapping[str, object]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Teacher Temporal Cache Inspection",
                "",
                f"- Validation passed: `{report['validation_passed']}`",
                f"- Clips: `{report['clip_count']}`",
                f"- Frames: `{report['frame_count']}`",
                f"- Stitch mode: `{report['stitch_mode']}`",
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
    "FORMAT_ALIAS",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_FILENAME",
    "TeacherTemporalCacheDataset",
    "format_teacher_temporal_cache_inspection",
    "inspect_teacher_temporal_cache",
    "write_teacher_temporal_cache",
]
