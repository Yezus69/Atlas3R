"""Mixed measured/pseudo temporal datasets for SMGT-small-v2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.runtime.rgb_student_preprocess import resize_nearest, scale_intrinsics
from atlas3r.training.measured_temporal_cache import MeasuredTemporalCacheDataset
from atlas3r.training.teacher_temporal_cache import TeacherTemporalCacheDataset
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()


class SMGTV2MixedTemporalDataset:
    """Torch dataset exposing measured and pseudo temporal cache clips."""

    def __init__(
        self,
        *,
        measured_caches: Sequence[str | Path],
        pseudo_caches: Sequence[str | Path] = (),
        measured_indices: Sequence[Sequence[int] | None] | None = None,
        pseudo_weight: float = 0.25,
    ) -> None:
        if not measured_caches:
            raise ValueError("measured_caches: at least one measured cache is required")
        if pseudo_weight <= 0.0 or pseudo_weight > 0.25:
            raise ValueError("pseudo_weight: must be in (0, 0.25]")
        measured_indices = measured_indices or tuple(None for _ in measured_caches)
        if len(measured_indices) != len(measured_caches):
            raise ValueError("measured_indices: must match measured_caches length")
        self.sources: list[tuple[str, int, Any]] = []
        self.measured = [
            MeasuredTemporalCacheDataset(cache, indices=indices)
            for cache, indices in zip(measured_caches, measured_indices, strict=True)
        ]
        self.pseudo = [TeacherTemporalCacheDataset(cache) for cache in pseudo_caches]
        self.pseudo_weight = float(pseudo_weight)
        for source_index, measured_dataset in enumerate(self.measured):
            for clip_index in range(len(measured_dataset)):
                self.sources.append(("measured", source_index, clip_index))
        for source_index, pseudo_dataset in enumerate(self.pseudo):
            for clip_index in range(len(pseudo_dataset)):
                self.sources.append(("pseudo", source_index, clip_index))
        if not self.sources:
            raise ValueError("SMGTV2MixedTemporalDataset: no clips selected")
        self.image_height = int(cast(int, self.measured[0].manifest["image_height"]))
        self.image_width = int(cast(int, self.measured[0].manifest["image_width"]))
        self.clip_length = int(cast(int, self.measured[0].manifest["clip_length"]))
        self._validate_cache_shapes()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_kind, source_index, clip_index = self.sources[index]
        if source_kind == "measured":
            raw = self.measured[source_index][clip_index]
            target_source_id = 1.0
            depth_weight = float(cast(float, raw["depth_target_weight"]))
            pose_weight = float(cast(float, raw["pose_target_weight"]))
            confidence_weight = float(cast(float, raw["confidence_target_weight"]))
        else:
            raw = self.pseudo[source_index][clip_index]
            target_source_id = 0.0
            depth_weight = min(float(cast(float, raw["depth_target_weight"])), self.pseudo_weight)
            pose_weight = min(float(cast(float, raw["pose_target_weight"])), self.pseudo_weight)
            confidence_weight = self.pseudo_weight
        raw = _normalize_raw_sample_shape(
            raw,
            target_height=self.image_height,
            target_width=self.image_width,
        )
        _validate_raw_sample(raw, source_kind=source_kind)
        rgb_u8 = cast(npt.NDArray[np.uint8], raw["rgb_u8"])
        rgb = np.transpose(rgb_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))
        target_T_world_camera = _clip_relative_transforms(
            cast(npt.NDArray[np.float32], raw["T_world_camera"])
        )
        return {
            "images_rgb": _TORCH.from_numpy(rgb.astype(np.float32, copy=False)),
            "K": _TORCH.from_numpy(cast(npt.NDArray[np.float32], raw["K"]).copy()),
            "target": {
                "depth_m": _TORCH.from_numpy(cast(npt.NDArray[np.float32], raw["depth_m"]).copy()),
                "depth_sigma_m": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], raw["depth_sigma_m"]).copy()
                ),
                "confidence": _TORCH.from_numpy(
                    cast(npt.NDArray[np.float32], raw["confidence"]).copy()
                ),
                "valid_mask": _TORCH.from_numpy(
                    cast(npt.NDArray[np.bool_], raw["valid_mask"]).copy()
                ),
                "T_world_camera": _TORCH.from_numpy(target_T_world_camera.copy()),
            },
            "frame_ids": _TORCH.from_numpy(cast(npt.NDArray[np.int64], raw["frame_ids"]).copy()),
            "target_source": source_kind,
            "target_source_id": _TORCH.tensor(target_source_id, dtype=_TORCH.float32),
            "depth_target_weight": _TORCH.tensor(depth_weight, dtype=_TORCH.float32),
            "pose_target_weight": _TORCH.tensor(pose_weight, dtype=_TORCH.float32),
            "confidence_target_weight": _TORCH.tensor(confidence_weight, dtype=_TORCH.float32),
            "metadata": {
                "source_kind": source_kind,
                "source_index": source_index,
                "clip_index": clip_index,
            },
        }

    def cache_summary(self) -> dict[str, object]:
        return {
            "measured_caches": [dataset.cache_summary() for dataset in self.measured],
            "pseudo_caches": [
                {
                    "cache_dir": str(dataset.cache_dir),
                    "clip_count": len(dataset),
                    "target_source": "pseudo",
                    "pseudo_weight": self.pseudo_weight,
                }
                for dataset in self.pseudo
            ],
            "total_clip_count": len(self.sources),
            "image_height": self.image_height,
            "image_width": self.image_width,
            "clip_length": self.clip_length,
        }

    def _validate_cache_shapes(self) -> None:
        for measured_dataset in self.measured:
            manifest = measured_dataset.manifest
            if int(cast(int, manifest["image_height"])) != self.image_height:
                raise ValueError("all v2 caches must share image_height")
            if int(cast(int, manifest["image_width"])) != self.image_width:
                raise ValueError("all v2 caches must share image_width")
            if int(cast(int, manifest["clip_length"])) != self.clip_length:
                raise ValueError("all v2 caches must share clip_length")
        for pseudo_dataset in self.pseudo:
            if int(cast(int, pseudo_dataset.manifest["clip_length"])) != self.clip_length:
                raise ValueError("all v2 caches must share clip_length")


def smgt_v2_measured_train_val_indices(
    clip_count: int,
    *,
    val_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if clip_count <= 0:
        raise ValueError("clip_count: must be positive")
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction: must be in [0, 1)")
    if clip_count == 1 or val_fraction == 0.0:
        all_indices = tuple(range(clip_count))
        return all_indices, all_indices
    val_count = max(1, int(round(clip_count * val_fraction)))
    val_count = min(val_count, clip_count - 1)
    train_count = clip_count - val_count
    return tuple(range(train_count)), tuple(range(train_count, clip_count))


def _validate_raw_sample(sample: dict[str, object], *, source_kind: str) -> None:
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
    if source_kind == "pseudo":
        if float(cast(float, sample["depth_target_weight"])) > 0.25:
            raise ValueError("pseudo depth target weight must be <= 0.25")
        if float(cast(float, sample["pose_target_weight"])) > 0.25:
            raise ValueError("pseudo pose target weight must be <= 0.25")


def _normalize_raw_sample_shape(
    sample: dict[str, object],
    *,
    target_height: int,
    target_width: int,
) -> dict[str, object]:
    rgb = np.asarray(sample["rgb_u8"])
    if rgb.shape[1:3] == (target_height, target_width):
        return sample
    original_size = (int(rgb.shape[1]), int(rgb.shape[2]))
    normalized = dict(sample)
    normalized["rgb_u8"] = np.stack(
        [
            resize_nearest(frame, height=target_height, width=target_width)
            for frame in cast(npt.NDArray[np.uint8], rgb)
        ],
        axis=0,
    )
    for key in ("depth_m", "depth_sigma_m", "confidence"):
        normalized[key] = np.stack(
            [
                _resize_hw_nearest(frame, height=target_height, width=target_width)
                for frame in np.asarray(sample[key])
            ],
            axis=0,
        ).astype(np.float32)
    normalized["valid_mask"] = np.stack(
        [
            _resize_hw_nearest(frame, height=target_height, width=target_width).astype(np.bool_)
            for frame in np.asarray(sample["valid_mask"])
        ],
        axis=0,
    )
    normalized["K"] = np.stack(
        [
            scale_intrinsics(K, original_size, (target_height, target_width))
            for K in cast(npt.NDArray[np.float32], np.asarray(sample["K"], dtype=np.float32))
        ],
        axis=0,
    ).astype(np.float32)
    return normalized


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


def _clip_relative_transforms(
    T_world_camera: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    transforms = np.asarray(T_world_camera, dtype=np.float32)
    anchor_inv = np.linalg.inv(transforms[0].astype(np.float64))
    relative = np.einsum("ij,tjk->tik", anchor_inv, transforms.astype(np.float64))
    return cast(npt.NDArray[np.float32], relative.astype(np.float32))


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


__all__ = [
    "SMGTV2MixedTemporalDataset",
    "smgt_v2_measured_train_val_indices",
]
