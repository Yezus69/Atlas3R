"""Forge Atlas3R multi-view clip caches from TUM RGB-D manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.data.image_runtime import require_pillow_image
from atlas3r.data.tum_rgbd import load_tum_rgbd_manifest
from atlas3r.forge.clip_cache import (
    CLIP_CACHE_FORMAT_NAME,
    CLIP_CACHE_FORMAT_VERSION,
    CLIP_CACHE_MANIFEST_FILENAME,
    validate_clip_cache_manifest,
    write_clip_payload,
    write_json_file,
)


@dataclass(frozen=True)
class TumRgbdClipForgeConfig:
    manifest: Path
    output: Path
    split: str = "train"
    clip_length: int = 5
    stride: int = 1
    width: int = 160
    height: int = 120
    max_clips: int | None = None
    max_frame_gap_s: float = 0.12
    write_pointmaps: bool = False
    write_normals: bool = False


def forge_tum_rgbd_clip_cache(config: TumRgbdClipForgeConfig) -> dict[str, object]:
    """Write a TUM RGB-D multi-view clip cache and return deterministic CLI JSON."""

    _validate_config(config)
    manifest = load_tum_rgbd_manifest(config.manifest)
    records = _records_for_split(manifest, config.split)
    windows = _contiguous_windows(
        records,
        clip_length=config.clip_length,
        stride=config.stride,
        max_frame_gap_s=config.max_frame_gap_s,
    )
    if config.max_clips is not None:
        windows = windows[: config.max_clips]
    if not windows:
        raise ValueError(f"{config.manifest}: no contiguous {config.split!r} clips were produced")

    config.output.mkdir(parents=True, exist_ok=True)
    clips_dir = config.output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    cache_clips: list[dict[str, object]] = []
    for clip_id, window in enumerate(windows):
        payload_path = clips_dir / f"clip_{clip_id:06d}.npz"
        payload = _payload_from_window(
            manifest,
            window,
            width=config.width,
            height=config.height,
            write_pointmaps=config.write_pointmaps,
            write_normals=config.write_normals,
        )
        write_clip_payload(
            payload_path,
            payload,
            clip_length=config.clip_length,
            height=config.height,
            width=config.width,
        )
        frame_ids = [_int_field(record, "frame_id") for record in window]
        timestamps = [_float_field(record, "rgb_timestamp_s") for record in window]
        cache_clips.append(
            {
                "clip_id": clip_id,
                "payload_path": f"clips/clip_{clip_id:06d}.npz",
                "frame_ids": frame_ids,
                "timestamps_s": timestamps,
                "center_index": config.clip_length // 2,
                "center_frame_id": frame_ids[config.clip_length // 2],
                "source_paths": {
                    "rgb": [str(record["rgb_path"]) for record in window],
                    "depth": [str(record["depth_path"]) for record in window],
                },
            }
        )

    truth_boundary = _truth_boundary()
    cache_manifest: dict[str, object] = {
        "format_name": CLIP_CACHE_FORMAT_NAME,
        "format_version": CLIP_CACHE_FORMAT_VERSION,
        "source_dataset_name": str(manifest["dataset_name"]),
        "source_sequence_name": str(manifest["sequence_name"]),
        "source_manifest_path": str(config.manifest),
        "split": config.split,
        "clip_length": config.clip_length,
        "stride": config.stride,
        "image_width": config.width,
        "image_height": config.height,
        "max_frame_gap_s": config.max_frame_gap_s,
        "clip_count": len(cache_clips),
        "clips": cache_clips,
        "teacher_source_metadata": {
            "depth_source": "TUM RGB-D sensor depth, depth_raw / scale",
            "pose_source": "TUM RGB-D groundtruth pose",
            "coordinate_frame": str(manifest.get("coordinate_frame", "")),
            "computed_pointmaps": config.write_pointmaps,
            "computed_normals": config.write_normals,
        },
        "truth_boundary": truth_boundary,
    }
    validate_clip_cache_manifest(
        cache_manifest,
        cache_root=config.output,
        validate_payloads=True,
    )
    manifest_path = config.output / CLIP_CACHE_MANIFEST_FILENAME
    write_json_file(manifest_path, cache_manifest)
    return {
        "format_name": "atlas3r_tum_rgbd_clip_forge_result",
        "manifest_path": str(manifest_path),
        "clip_count": len(cache_clips),
        "split": config.split,
        "first_clip_id": 0,
        "last_clip_id": len(cache_clips) - 1,
        "truth_boundary": truth_boundary,
    }


def _payload_from_window(
    manifest: dict[str, object],
    records: list[dict[str, object]],
    *,
    width: int,
    height: int,
    write_pointmaps: bool,
    write_normals: bool,
) -> dict[str, npt.NDArray[Any] | np.generic]:
    root = Path(_string_field(manifest, "root_path"))
    K = _scaled_intrinsics(manifest, width=width, height=height)
    depth_scale = _depth_scale(manifest)
    rgb_frames: list[npt.NDArray[np.uint8]] = []
    depth_frames: list[npt.NDArray[np.float32]] = []
    mask_frames: list[npt.NDArray[np.bool_]] = []
    transforms: list[npt.NDArray[np.float32]] = []
    for record in records:
        rgb_frames.append(_load_rgb(root / _string_field(record, "rgb_path"), width, height))
        depth_m, valid_mask = _load_depth(
            root / _string_field(record, "depth_path"),
            width,
            height,
            depth_scale,
        )
        depth_frames.append(depth_m)
        mask_frames.append(valid_mask)
        transforms.append(_float_array(record["T_world_camera"], (4, 4)))
    depth_stack = np.stack(depth_frames).astype(np.float32, copy=False)
    mask_stack = np.stack(mask_frames).astype(np.bool_, copy=False)
    transform_stack = np.stack(transforms).astype(np.float32, copy=False)
    K_stack = np.repeat(K[np.newaxis, :, :], len(records), axis=0).astype(np.float32, copy=False)
    payload: dict[str, npt.NDArray[Any] | np.generic] = {
        "images_rgb_u8": np.stack(rgb_frames).astype(np.uint8, copy=False),
        "depth_m": depth_stack,
        "valid_depth_mask": mask_stack,
        "K": K_stack,
        "T_world_camera": transform_stack,
        "frame_ids": np.asarray(
            [_int_field(record, "frame_id") for record in records], dtype=np.int32
        ),
        "timestamps_s": np.asarray(
            [_float_field(record, "rgb_timestamp_s") for record in records],
            dtype=np.float64,
        ),
        "center_index": np.asarray(len(records) // 2, dtype=np.int32),
    }
    if write_pointmaps or write_normals:
        pointmap_camera = np.stack(
            [
                _pointmap_camera(depth, K, mask)
                for depth, mask in zip(depth_stack, mask_stack, strict=False)
            ]
        ).astype(np.float32, copy=False)
        if write_pointmaps:
            payload["pointmap_camera_m"] = pointmap_camera
            payload["pointmap_world_m"] = np.stack(
                [
                    _pointmap_world(points, transform, mask)
                    for points, transform, mask in zip(
                        pointmap_camera,
                        transform_stack,
                        mask_stack,
                        strict=False,
                    )
                ]
            ).astype(np.float32, copy=False)
        if write_normals:
            payload["normal_camera"] = np.stack(
                [
                    _normal_camera(points, mask)
                    for points, mask in zip(pointmap_camera, mask_stack, strict=False)
                ]
            ).astype(np.float32, copy=False)
    return payload


def _load_rgb(path: Path, width: int, height: int) -> npt.NDArray[np.uint8]:
    image_module = require_pillow_image()
    resampling = getattr(image_module, "Resampling", image_module)
    try:
        with image_module.open(path) as image:
            resized = image.convert("RGB").resize((width, height), resample=resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"{path}: failed to load RGB image: {exc}") from exc
    if array.shape != (height, width, 3):
        raise ValueError(f"{path}: RGB image did not resize to H,W,3")
    return array


def _load_depth(
    path: Path,
    width: int,
    height: int,
    depth_scale: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    image_module = require_pillow_image()
    resampling = getattr(image_module, "Resampling", image_module)
    try:
        with image_module.open(path) as image:
            raw = np.asarray(image.resize((width, height), resample=resampling.NEAREST))
    except OSError as exc:
        raise ValueError(f"{path}: failed to load depth image: {exc}") from exc
    if raw.shape != (height, width):
        raise ValueError(f"{path}: depth image did not resize to H,W")
    raw_float = raw.astype(np.float32, copy=False)
    valid_mask = raw_float > 0.0
    depth_m = raw_float / np.float32(depth_scale)
    depth_m[~valid_mask] = 0.0
    return (
        cast(npt.NDArray[np.float32], depth_m.astype(np.float32, copy=False)),
        cast(npt.NDArray[np.bool_], valid_mask.astype(np.bool_, copy=False)),
    )


def _pointmap_camera(
    depth_m: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    height, width = depth_m.shape
    u = np.arange(width, dtype=np.float32)[np.newaxis, :]
    v = np.arange(height, dtype=np.float32)[:, np.newaxis]
    x = (u - K[0, 2]) / max(float(K[0, 0]), 1e-6) * depth_m
    y = (v - K[1, 2]) / max(float(K[1, 1]), 1e-6) * depth_m
    points = np.stack([x, y, depth_m], axis=-1).astype(np.float32, copy=False)
    points[~valid_mask] = 0.0
    return cast(npt.NDArray[np.float32], points)


def _pointmap_world(
    pointmap_camera: npt.NDArray[np.float32],
    T_world_camera: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    flat = pointmap_camera.reshape(-1, 3).astype(np.float64, copy=False)
    transform = T_world_camera.astype(np.float64, copy=False)
    world = (transform[:3, :3] @ flat.T).T + transform[:3, 3]
    world = world.reshape(pointmap_camera.shape).astype(np.float32, copy=False)
    world[~valid_mask] = 0.0
    return cast(npt.NDArray[np.float32], world)


def _normal_camera(
    pointmap_camera: npt.NDArray[np.float32],
    valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float32]:
    normals = np.zeros_like(pointmap_camera, dtype=np.float32)
    if pointmap_camera.shape[0] < 3 or pointmap_camera.shape[1] < 3:
        return normals
    dx = pointmap_camera[1:-1, 2:, :] - pointmap_camera[1:-1, :-2, :]
    dy = pointmap_camera[2:, 1:-1, :] - pointmap_camera[:-2, 1:-1, :]
    raw_normals = np.cross(dx, dy).astype(np.float32, copy=False)
    norms = np.linalg.norm(raw_normals, axis=-1)
    valid = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[1:-1, 2:]
        & valid_mask[1:-1, :-2]
        & valid_mask[2:, 1:-1]
        & valid_mask[:-2, 1:-1]
        & (norms > 1e-8)
    )
    raw_normals[valid] /= norms[valid, np.newaxis]
    raw_normals[~valid] = 0.0
    normals[1:-1, 1:-1, :] = raw_normals
    return normals


def _contiguous_windows(
    records: list[dict[str, object]],
    *,
    clip_length: int,
    stride: int,
    max_frame_gap_s: float,
) -> list[list[dict[str, object]]]:
    windows: list[list[dict[str, object]]] = []
    for start in range(0, len(records) - clip_length + 1, stride):
        window = records[start : start + clip_length]
        timestamps = [_float_field(record, "rgb_timestamp_s") for record in window]
        gaps_ok = all(
            right - left <= max_frame_gap_s
            for left, right in zip(timestamps[:-1], timestamps[1:], strict=False)
        )
        if gaps_ok:
            windows.append(window)
    return windows


def _records_for_split(manifest: dict[str, object], split: str) -> list[dict[str, object]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError("manifest.frames: must be a list")
    records = [
        cast(dict[str, object], frame)
        for frame in frames
        if isinstance(frame, dict) and frame.get("split") == split
    ]
    records.sort(key=lambda record: _int_field(record, "frame_id"))
    if not records:
        raise ValueError(f"manifest.frames: no records found for split {split!r}")
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
    K = _float_array(_mapping_field(manifest, "intrinsics")["K"], (3, 3))
    scaled = K.copy()
    scaled[0, 0] *= float(width) / float(original_width)
    scaled[0, 2] *= float(width) / float(original_width)
    scaled[1, 1] *= float(height) / float(original_height)
    scaled[1, 2] *= float(height) / float(original_height)
    return scaled.astype(np.float32, copy=False)


def _float_array(value: object, shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"array: expected finite shape {shape}, got {array.shape}")
    return array


def _depth_scale(manifest: dict[str, object]) -> float:
    depth = _mapping_field(manifest, "depth")
    scale = _float_field(depth, "scale")
    if scale <= 0.0:
        raise ValueError("manifest.depth.scale: must be positive")
    return scale


def _string_field(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}: must be a non-empty string")
    return value


def _mapping_field(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key}: must be a mapping")
    return value


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


def _float_field(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key}: must be numeric")
    return float(value)


def _truth_boundary() -> dict[str, object]:
    return {
        "diagnostic_only": True,
        "accuracy_report": False,
        "performance_report": False,
        "teacher_source": "tum_rgbd_sensor_depth_pose",
        "rotation_learned": False,
        "hidden_geometry_measured": False,
    }


def _validate_config(config: TumRgbdClipForgeConfig) -> None:
    if config.split not in {"train", "val"}:
        raise ValueError("split: must be train or val")
    for field_name in ("clip_length", "stride", "width", "height"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if config.max_clips is not None and config.max_clips <= 0:
        raise ValueError("max_clips: must be positive when provided")
    if config.max_frame_gap_s <= 0.0:
        raise ValueError("max_frame_gap_s: must be positive")


__all__ = [
    "TumRgbdClipForgeConfig",
    "forge_tum_rgbd_clip_cache",
]
