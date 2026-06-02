"""Lazy disk-backed PyTorch dataset for TUM RGB-D manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.data.image_runtime import require_pillow_image
from atlas3r.data.tum_rgbd import load_tum_rgbd_manifest
from atlas3r.training.torch_runtime import require_torch


class TumRgbdDepthDataset:
    """Load TUM RGB-D RGB/depth/pose records lazily from an Atlas3R manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int,
        height: int,
        split: str = "train",
        min_valid_depth_pixels: int = 1,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if min_valid_depth_pixels < 0:
            raise ValueError("min_valid_depth_pixels: must be non-negative")
        manifest = load_tum_rgbd_manifest(manifest_path)
        records = manifest.get("frames")
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path}: frames must be a list")
        selected = [record for record in records if _record_split(record) == split]
        if not selected:
            raise ValueError(f"{manifest_path}: no frames found for split {split!r}")
        self.manifest_path = Path(manifest_path)
        self.manifest = manifest
        self.root_path = Path(_string_field(manifest, "root_path"))
        self.width = width
        self.height = height
        self.split = split
        self.min_valid_depth_pixels = min_valid_depth_pixels
        self.records = tuple(cast(dict[str, object], record) for record in selected)
        require_torch()
        require_pillow_image()
        self._K = scaled_intrinsics_from_manifest(manifest, width=width, height=height)
        self._depth_scale = _depth_scale(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        rgb_u8 = self._load_rgb(_record_path(self.root_path, record, "rgb_path"))
        depth_m, valid_mask = self._load_depth(_record_path(self.root_path, record, "depth_path"))
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < self.min_valid_depth_pixels:
            frame_id = _int_field(record, "frame_id")
            raise ValueError(
                f"{self.manifest_path}: frame_id {frame_id} has {valid_count} valid depth "
                f"pixels, below min_valid_depth_pixels={self.min_valid_depth_pixels}"
            )
        torch = require_torch()
        image_float = rgb_u8.astype(np.float32) / np.float32(255.0)
        image_chw = np.transpose(image_float, (2, 0, 1)).copy()
        valid_chw = valid_mask[np.newaxis, :, :]
        depth_chw = depth_m[np.newaxis, :, :]
        confidence = valid_chw.astype(np.float32)
        T_world_camera = _array_field(record, "T_world_camera", shape=(4, 4))
        camera_center = _array_field(record, "camera_center_world_m", shape=(3,))
        return {
            "images_rgb": torch.from_numpy(image_chw.astype(np.float32, copy=False)),
            "intrinsics": torch.from_numpy(self._K.copy()),
            "target": {
                "depth_m": torch.from_numpy(depth_chw.astype(np.float32, copy=False)),
                "valid_depth_mask": torch.from_numpy(valid_chw.astype(np.bool_, copy=False)),
                "confidence": torch.from_numpy(confidence),
                "camera_center_world_m": torch.from_numpy(camera_center),
                "T_world_camera": torch.from_numpy(T_world_camera),
            },
            "metadata": {
                "dataset": self.manifest["dataset_name"],
                "sequence": self.manifest["sequence_name"],
                "frame_id": _int_field(record, "frame_id"),
                "split": str(record["split"]),
                "timestamp_s": _float_field(record, "rgb_timestamp_s"),
                "rgb_path": str(record["rgb_path"]),
                "depth_path": str(record["depth_path"]),
                "manifest_path": str(self.manifest_path),
            },
        }

    def _load_rgb(self, path: Path) -> npt.NDArray[np.uint8]:
        image_module = require_pillow_image()
        resampling = getattr(image_module, "Resampling", image_module)
        try:
            with image_module.open(path) as image:
                resized = image.convert("RGB").resize(
                    (self.width, self.height),
                    resample=resampling.BILINEAR,
                )
                array = np.asarray(resized, dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"{path}: failed to load TUM RGB image: {exc}") from exc
        if array.shape != (self.height, self.width, 3):
            raise ValueError(f"{path}: RGB image did not resize to H,W,3")
        return array

    def _load_depth(self, path: Path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        image_module = require_pillow_image()
        resampling = getattr(image_module, "Resampling", image_module)
        try:
            with image_module.open(path) as image:
                resized = image.resize(
                    (self.width, self.height),
                    resample=resampling.NEAREST,
                )
                raw = np.asarray(resized)
        except OSError as exc:
            raise ValueError(f"{path}: failed to load TUM depth image: {exc}") from exc
        if raw.shape != (self.height, self.width):
            raise ValueError(f"{path}: depth image did not resize to H,W")
        raw_float = raw.astype(np.float32, copy=False)
        valid_mask = raw_float > 0.0
        depth_m = raw_float / np.float32(self._depth_scale)
        depth_m[~valid_mask] = 0.0
        return (
            cast(npt.NDArray[np.float32], depth_m.astype(np.float32, copy=False)),
            cast(npt.NDArray[np.bool_], valid_mask.astype(np.bool_, copy=False)),
        )


def scaled_intrinsics_from_manifest(
    manifest: dict[str, object],
    *,
    width: int,
    height: int,
) -> npt.NDArray[np.float32]:
    """Scale manifest intrinsics from original TUM image size to a resize target."""

    image = manifest.get("image")
    if not isinstance(image, dict):
        raise ValueError("manifest.image: must be a dictionary")
    original_width = int(image.get("width", 0))
    original_height = int(image.get("height", 0))
    if original_width <= 0 or original_height <= 0:
        raise ValueError("manifest.image width/height must be positive")
    K = _manifest_intrinsics(manifest)
    sx = float(width) / float(original_width)
    sy = float(height) / float(original_height)
    scaled = K.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled.astype(np.float32, copy=False)


def _manifest_intrinsics(manifest: dict[str, object]) -> npt.NDArray[np.float32]:
    intrinsics = manifest.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("manifest.intrinsics: must be a dictionary")
    return _as_array(intrinsics.get("K"), field_name="manifest.intrinsics.K", shape=(3, 3))


def _depth_scale(manifest: dict[str, object]) -> float:
    depth = manifest.get("depth")
    if not isinstance(depth, dict):
        raise ValueError("manifest.depth: must be a dictionary")
    scale = float(depth.get("scale", 0.0))
    if scale <= 0.0:
        raise ValueError("manifest.depth.scale: must be positive")
    return scale


def _record_path(root: Path, record: dict[str, object], field_name: str) -> Path:
    path_value = _string_field(record, field_name)
    path = root / path_value
    if not path.is_file():
        raise ValueError(f"{path}: manifest record file does not exist")
    return path


def _record_split(record: object) -> str:
    if not isinstance(record, dict):
        raise ValueError("manifest frames must be dictionaries")
    value = record.get("split")
    if not isinstance(value, str):
        raise ValueError("manifest frame split must be a string")
    return value


def _string_field(mapping: dict[str, object], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name}: must be a non-empty string")
    return value


def _int_field(mapping: dict[str, object], field_name: str) -> int:
    value = mapping.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"{field_name}: must be an integer")
    return value


def _float_field(mapping: dict[str, object], field_name: str) -> float:
    value = mapping.get(field_name)
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name}: must be numeric")
    return float(value)


def _array_field(
    mapping: dict[str, object],
    field_name: str,
    *,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    return _as_array(mapping.get(field_name), field_name=field_name, shape=shape)


def _as_array(
    value: object,
    *,
    field_name: str,
    shape: tuple[int, ...],
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{field_name}: expected shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name}: must contain finite values")
    return cast(npt.NDArray[np.float32], array.astype(np.float32, copy=False))


__all__ = [
    "TumRgbdDepthDataset",
    "scaled_intrinsics_from_manifest",
]
