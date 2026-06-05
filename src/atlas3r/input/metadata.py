"""Small camera metadata helpers."""

from __future__ import annotations

import numpy as np

from atlas3r.contracts.frames import CameraModel


def camera_from_focal(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float | None = None,
    cy: float | None = None,
    source: str = "metadata",
    confidence: float = 1.0,
) -> CameraModel:
    K = np.array(
        [
            [fx, 0.0, float(width - 1) / 2.0 if cx is None else cx],
            [0.0, fy, float(height - 1) / 2.0 if cy is None else cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CameraModel(width=width, height=height, K=K, confidence=confidence, source=source)


def scale_camera_model(camera: CameraModel, *, width: int, height: int) -> CameraModel:
    sx = float(width) / float(camera.width)
    sy = float(height) / float(camera.height)
    K = camera.K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return CameraModel(
        width=width,
        height=height,
        K=K,
        distortion_model=camera.distortion_model,
        distortion_params=camera.distortion_params,
        rolling_shutter_row_time_s=camera.rolling_shutter_row_time_s,
        confidence=camera.confidence,
        source=camera.source,
    )
