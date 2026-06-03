"""Shared Depth Pro runner type aliases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DepthProFramePrediction:
    depth_m: npt.NDArray[Any]
    confidence: npt.NDArray[Any] | None = None
    depth_sigma_m: npt.NDArray[Any] | None = None


DepthProFramePredictor = Callable[
    [npt.NDArray[np.uint8], npt.NDArray[np.float32]],
    DepthProFramePrediction,
]


__all__ = [
    "DepthProFramePrediction",
    "DepthProFramePredictor",
]
