"""SMGT-tiny dataset adapter for Phase 6H teacher temporal caches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.training.teacher_temporal_cache import TeacherTemporalCacheDataset
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()


class SMGTTinyTeacherCacheDataset:
    """Torch dataset exposing Phase 6H pseudo labels in SMGT training shape."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        indices: Sequence[int] | None = None,
        debug_subset_clips: int | None = None,
        pseudo_weight_override: float | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.source = TeacherTemporalCacheDataset(self.cache_dir)
        all_indices = (
            tuple(range(len(self.source))) if indices is None else tuple(int(i) for i in indices)
        )
        if debug_subset_clips is not None:
            if debug_subset_clips <= 0:
                raise ValueError("debug_subset_clips: must be positive when provided")
            all_indices = all_indices[:debug_subset_clips]
        if not all_indices:
            raise ValueError("SMGTTinyTeacherCacheDataset: no clips selected")
        if pseudo_weight_override is not None and (
            pseudo_weight_override <= 0.0 or pseudo_weight_override > 1.0
        ):
            raise ValueError("pseudo_weight_override: must be in (0, 1]")
        self.indices = all_indices
        self.pseudo_weight_override = pseudo_weight_override
        self.manifest = self.source.manifest
        self.inspection = self.source.inspection

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.source[self.indices[index]]
        _validate_raw_sample(sample)
        rgb_u8 = cast(npt.NDArray[np.uint8], sample["rgb_u8"])
        rgb = np.transpose(rgb_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
        depth_weight = float(cast(float, sample["depth_target_weight"]))
        pose_weight = float(cast(float, sample["pose_target_weight"]))
        if self.pseudo_weight_override is not None:
            depth_weight = self.pseudo_weight_override
            pose_weight = self.pseudo_weight_override
        return {
            "images_rgb": _TORCH.from_numpy(rgb.astype(np.float32, copy=False)),
            "K": _TORCH.from_numpy(cast(npt.NDArray[np.float32], sample["K"]).copy()),
            "target": {
                "depth_m": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], sample["depth_m"]).copy()
                ),
                "depth_sigma_m": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], sample["depth_sigma_m"]).copy()
                ),
                "confidence": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], sample["confidence"]).copy()
                ),
                "valid_mask": _TORCH.from_numpy(
                    cast(npt.NDArray[np.bool_], sample["valid_mask"]).copy()
                ),
                "T_world_camera": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], sample["T_world_camera"]).copy()
                ),
            },
            "frame_ids": _TORCH.from_numpy(cast(npt.NDArray[np.int64], sample["frame_ids"]).copy()),
            "depth_target_weight": _TORCH.tensor(depth_weight, dtype=_TORCH.float32),
            "pose_target_weight": _TORCH.tensor(pose_weight, dtype=_TORCH.float32),
            "truth_flags": _truth_tensor_record(cast(dict[str, object], sample["truth_flags"])),
            "metadata": {
                "cache_dir": str(self.cache_dir),
                "source_sequence_name": str(
                    self.manifest.get("source_rgb_teacher_run", "teacher_cache")
                ),
                "metric_scale_source": str(self.manifest["metric_scale_source"]),
            },
        }

    def cache_summary(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir),
            "selected_clip_count": len(self.indices),
            "source_clip_count": len(self.source),
            "image_height": int(cast(int, self.manifest["image_height"])),
            "image_width": int(cast(int, self.manifest["image_width"])),
            "clip_length": int(cast(int, self.manifest["clip_length"])),
            "depth_target_weight": float(cast(float, self.inspection["depth_target_weight"])),
            "pose_target_weight": float(cast(float, self.inspection["pose_target_weight"])),
            "pseudo_weight_override": self.pseudo_weight_override,
            "truth_boundary": dict(cast(dict[str, object], self.manifest["truth_boundary"])),
        }


def smgt_tiny_train_val_indices(
    clip_count: int,
    *,
    val_split: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if clip_count <= 0:
        raise ValueError("clip_count: must be positive")
    if val_split < 0.0 or val_split >= 1.0:
        raise ValueError("val_split: must be in [0, 1)")
    if clip_count == 1 or val_split == 0.0:
        return tuple(range(clip_count)), tuple(range(clip_count))
    val_count = max(1, int(round(clip_count * val_split)))
    val_count = min(val_count, clip_count - 1)
    train_count = clip_count - val_count
    return tuple(range(train_count)), tuple(range(train_count, clip_count))


def _validate_raw_sample(sample: dict[str, object]) -> None:
    rgb = np.asarray(sample["rgb_u8"])
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb_u8: expected uint8 T,H,W,3")
    frame_count, height, width = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
    _array(sample, "K", np.float32, (frame_count, 3, 3))
    _array(sample, "T_world_camera", np.float32, (frame_count, 4, 4))
    for key in ("depth_m", "depth_sigma_m", "confidence"):
        array = _array(sample, key, np.float32, (frame_count, height, width))
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key}: must be finite")
    if np.any(cast(npt.NDArray[np.float32], sample["depth_m"]) < 0.0):
        raise ValueError("depth_m: must be non-negative")
    if np.any(cast(npt.NDArray[np.float32], sample["depth_sigma_m"]) < 0.0):
        raise ValueError("depth_sigma_m: must be non-negative")
    confidence = cast(npt.NDArray[np.float32], sample["confidence"])
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence: must be in [0,1]")
    _array(sample, "valid_mask", np.bool_, (frame_count, height, width))
    if (
        float(cast(float, sample["depth_target_weight"])) > 0.25
        or float(cast(float, sample["pose_target_weight"])) > 0.25
    ):
        raise ValueError("default pseudo target weights must be <= 0.25")


def _array(
    sample: dict[str, object],
    key: str,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> npt.NDArray[Any]:
    array = np.asarray(sample[key])
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError(
            f"{key}: expected {np.dtype(dtype)} {shape}, got {array.dtype} {array.shape}"
        )
    return array


def _truth_tensor_record(truth_flags: dict[str, object]) -> dict[str, Any]:
    return {
        "diagnostic_only": bool(truth_flags.get("diagnostic_only", True)),
        "measured_depth_used": bool(truth_flags.get("measured_depth_used", False)),
        "measured_pose_used": bool(truth_flags.get("measured_pose_used", False)),
        "teacher_geometry_used": bool(truth_flags.get("teacher_geometry_used", True)),
    }


__all__ = [
    "SMGTTinyTeacherCacheDataset",
    "smgt_tiny_train_val_indices",
]
