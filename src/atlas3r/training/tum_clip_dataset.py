"""Lazy Torch dataset for Atlas3R multi-view clip caches."""

from __future__ import annotations

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
from atlas3r.pose.transforms import compose_transforms, invert_transform
from atlas3r.training.torch_runtime import require_torch


class TumRgbdClipCacheDataset:
    """Load forged multi-view TUM RGB-D clips lazily from `.npz` payloads."""

    def __init__(self, clip_cache: str | Path) -> None:
        self.manifest_path = manifest_path_from_input(clip_cache)
        self.manifest = load_clip_cache_manifest(self.manifest_path)
        clips = self.manifest.get("clips")
        if not isinstance(clips, list) or not clips:
            raise ValueError(f"{self.manifest_path}: clips must be a non-empty list")
        self.clips = tuple(cast(dict[str, object], clip) for clip in clips)
        self.cache_root = self.manifest_path.parent
        self.clip_length = _int_field(self.manifest, "clip_length")
        self.height = _int_field(self.manifest, "image_height")
        self.width = _int_field(self.manifest, "image_width")
        require_torch()

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.clips):
            raise IndexError(index)
        clip = self.clips[index]
        arrays = read_clip_payload_from_entry(self.cache_root, clip)
        validate_clip_payload(
            arrays,
            clip_length=self.clip_length,
            height=self.height,
            width=self.width,
        )
        torch = require_torch()
        images_rgb = arrays["images_rgb_u8"].astype(np.float32) / np.float32(255.0)
        images_chw = np.transpose(images_rgb, (0, 3, 1, 2)).copy()
        depth_m = arrays["depth_m"].astype(np.float32, copy=False)
        valid_mask = arrays["valid_depth_mask"].astype(np.bool_, copy=False)
        K = arrays["K"].astype(np.float32, copy=False)
        T_world_camera = arrays["T_world_camera"].astype(np.float32, copy=False)
        center_index = int(np.asarray(arrays["center_index"]).item())
        relative = _relative_T_center_camera(T_world_camera, center_index)
        frame_ids = arrays["frame_ids"].astype(np.int32, copy=False)
        timestamps = arrays["timestamps_s"].astype(np.float64, copy=False)
        center_valid = valid_mask[center_index][np.newaxis, :, :]
        return {
            "images_rgb": torch.from_numpy(images_chw.astype(np.float32, copy=False)),
            "intrinsics": torch.from_numpy(K.copy()),
            "T_world_camera": torch.from_numpy(T_world_camera.copy()),
            "target": {
                "center_depth_m": torch.from_numpy(
                    depth_m[center_index][np.newaxis, :, :].astype(np.float32, copy=False)
                ),
                "center_valid_depth_mask": torch.from_numpy(
                    center_valid.astype(np.bool_, copy=False)
                ),
                "center_confidence": torch.from_numpy(center_valid.astype(np.float32, copy=False)),
                "relative_T_center_camera": torch.from_numpy(relative),
            },
            "metadata": {
                "clip_id": _int_field(clip, "clip_id"),
                "frame_ids": [int(frame_id) for frame_id in frame_ids.tolist()],
                "center_frame_id": int(frame_ids[center_index]),
                "center_index": center_index,
                "timestamps_s": [float(timestamp) for timestamp in timestamps.tolist()],
                "payload_path": str(clip["payload_path"]),
                "manifest_path": str(self.manifest_path),
            },
        }


def _relative_T_center_camera(
    T_world_camera: npt.NDArray[np.float32],
    center_index: int,
) -> npt.NDArray[np.float32]:
    T_center_world = invert_transform(T_world_camera[center_index])
    relative = [
        compose_transforms(T_center_world, T_world_camera[index])
        for index in range(T_world_camera.shape[0])
    ]
    return np.stack(relative).astype(np.float32, copy=False)


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "TumRgbdClipCacheDataset",
]
