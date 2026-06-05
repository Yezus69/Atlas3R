from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_ppm(path: Path, rgb: NDArray[np.uint8]) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("expected RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


def write_ppm_sequence(root: Path, *, count: int = 4, width: int = 4, height: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    for index in range(count):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = (x_grid * 30 + index * 20).astype(np.uint8)
        rgb[:, :, 1] = (y_grid * 40 + index * 10).astype(np.uint8)
        rgb[:, :, 2] = np.uint8(50 + index * 20)
        write_ppm(root / f"{index:03d}.ppm", rgb)
