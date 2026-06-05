"""Small NumPy validation helpers for public contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


def as_float32_array(value: object, shape: tuple[int | None, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    _require_shape(array, shape, name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def as_uint8_array(value: object, shape: tuple[int | None, ...], name: str) -> UInt8Array:
    array = np.asarray(value, dtype=np.uint8)
    _require_shape(array, shape, name)
    return array


def as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def as_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float, not bool")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"{name} must be a float")


def require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def array_to_list(value: NDArray[Any] | None) -> object:
    if value is None:
        return None
    return value.tolist()


def _require_shape(array: NDArray[Any], shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {array.ndim}")
    for index, expected in enumerate(shape):
        if expected is not None and array.shape[index] != expected:
            actual = array.shape[index]
            raise ValueError(
                f"{name} shape mismatch at dim {index}: expected {expected}, got {actual}"
            )
