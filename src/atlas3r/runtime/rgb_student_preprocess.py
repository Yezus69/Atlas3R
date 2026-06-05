"""Preprocessing helpers for RGB student runtime inference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def resize_nearest(
    rgb: npt.NDArray[np.uint8],
    *,
    height: int,
    width: int,
) -> npt.NDArray[np.uint8]:
    if rgb.shape[:2] == (height, width):
        return rgb.astype(np.uint8, copy=True)
    y = np.linspace(0, rgb.shape[0] - 1, height).round().astype(np.int64)
    x = np.linspace(0, rgb.shape[1] - 1, width).round().astype(np.int64)
    return rgb[y[:, None], x[None, :], :].astype(np.uint8, copy=True)


def scale_intrinsics(
    K: npt.NDArray[np.float32],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> npt.NDArray[np.float32]:
    in_h, in_w = original_size
    out_h, out_w = target_size
    scaled = K.astype(np.float32, copy=True)
    scaled[0, 0] *= out_w / max(float(in_w), 1.0)
    scaled[0, 2] *= out_w / max(float(in_w), 1.0)
    scaled[1, 1] *= out_h / max(float(in_h), 1.0)
    scaled[1, 2] *= out_h / max(float(in_h), 1.0)
    return scaled


__all__ = ["resize_nearest", "scale_intrinsics"]
