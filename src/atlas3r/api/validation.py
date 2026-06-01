"""Validation helpers for Atlas3R public contracts."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from enum import Enum
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]

FLOAT_ATOL = 1e-4


def _fail(field_name: str, message: str) -> NoReturn:
    raise ValueError(f"{field_name}: {message}")


def validate_nonempty_str(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(field_name, "must be a non-empty string")
    return value


def validate_non_negative_int(field_name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(field_name, "must be a non-negative integer")
    return value


def validate_positive_int(field_name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(field_name, "must be a positive integer")
    return value


def validate_finite_scalar(field_name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        _fail(field_name, "must be finite")
    return scalar


def validate_non_negative_scalar(field_name: str, value: float) -> float:
    scalar = validate_finite_scalar(field_name, value)
    if scalar < 0.0:
        _fail(field_name, "must be non-negative")
    return scalar


def validate_positive_scalar(field_name: str, value: float) -> float:
    scalar = validate_finite_scalar(field_name, value)
    if scalar <= 0.0:
        _fail(field_name, "must be positive")
    return scalar


def validate_confidence(field_name: str, value: float) -> float:
    scalar = validate_finite_scalar(field_name, value)
    if scalar < 0.0 or scalar > 1.0:
        _fail(field_name, "must be in [0, 1]")
    return scalar


def validate_choice(field_name: str, value: object, allowed: Collection[str]) -> str:
    if isinstance(value, Enum):
        candidate: object = value.value
    else:
        candidate = value
    if not isinstance(candidate, str) or candidate not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        _fail(field_name, f"must be one of: {allowed_values}")
    return candidate


def validate_mapping(field_name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(field_name, "must be a mapping")
    return value


def validate_str_sequence(field_name: str, value: Sequence[str]) -> Sequence[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        _fail(field_name, "must be a sequence of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _fail(field_name, f"item {index} must be a string")
    return value


def validate_int_sequence(
    field_name: str, value: Sequence[int], *, non_empty: bool
) -> Sequence[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(field_name, "must be a sequence of integers")
    if non_empty and len(value) == 0:
        _fail(field_name, "must be non-empty")
    for index, item in enumerate(value):
        validate_non_negative_int(f"{field_name}[{index}]", item)
    return value


def validate_finite_numeric_array(field_name: str, value: Array) -> Array:
    array = np.asarray(value)
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        _fail(field_name, "must be a numeric array")
    if array.size > 0 and not np.all(np.isfinite(array)):
        _fail(field_name, "must contain only finite values")
    return array


def validate_array_shape(field_name: str, value: Array, shape: tuple[int, ...]) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.shape != shape:
        expected = "x".join(str(dim) for dim in shape)
        actual = "x".join(str(dim) for dim in array.shape)
        _fail(field_name, f"must have shape {expected}, got {actual}")
    return array


def validate_vector(field_name: str, value: Array, length: int) -> Array:
    return validate_array_shape(field_name, value, (length,))


def validate_matrix_nx3(field_name: str, value: Array) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.ndim != 2 or array.shape[1] != 3:
        _fail(field_name, "must have shape Nx3")
    return array


def validate_matrix_nx2(field_name: str, value: Array) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.ndim != 2 or array.shape[1] != 2:
        _fail(field_name, "must have shape Nx2")
    return array


def validate_image_hw(field_name: str, value: Array) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.ndim != 2:
        _fail(field_name, "must have shape HxW")
    return array


def validate_confidence_array(field_name: str, value: Array) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.size > 0 and (np.any(array < 0.0) or np.any(array > 1.0)):
        _fail(field_name, "must contain values in [0, 1]")
    return array


def validate_rotation_matrix(field_name: str, value: Array, *, atol: float = FLOAT_ATOL) -> Array:
    matrix = validate_array_shape(field_name, value, (3, 3)).astype(np.float64, copy=False)
    should_be_identity = matrix.T @ matrix
    if not np.allclose(should_be_identity, np.eye(3), atol=atol):
        _fail(field_name, "rotation block must be approximately orthonormal")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=1e-3):
        _fail(field_name, "rotation block determinant must be positive and near 1")
    return np.asarray(value)


def validate_transform(field_name: str, value: Array, *, atol: float = FLOAT_ATOL) -> Array:
    transform = validate_array_shape(field_name, value, (4, 4)).astype(np.float64, copy=False)
    if not np.allclose(transform[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        _fail(field_name, "bottom row must be [0, 0, 0, 1]")
    validate_rotation_matrix(f"{field_name}[:3, :3]", transform[:3, :3], atol=atol)
    return np.asarray(value)


def validate_intrinsics(field_name: str, value: Array) -> Array:
    K = validate_array_shape(field_name, value, (3, 3)).astype(np.float64, copy=False)
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        _fail(field_name, "focal lengths must be positive")
    if not np.isfinite(K[0, 2]) or not np.isfinite(K[1, 2]):
        _fail(field_name, "principal point must be finite")
    return np.asarray(value)


def validate_covariance_6x6(
    field_name: str, value: Array | None, *, atol: float = 1e-6
) -> Array | None:
    if value is None:
        return None
    covariance = validate_array_shape(field_name, value, (6, 6)).astype(np.float64, copy=False)
    if not np.allclose(covariance, covariance.T, atol=atol):
        _fail(field_name, "must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.min(eigenvalues) < -atol:
        _fail(field_name, "must be positive semi-definite")
    return np.asarray(value)


def validate_faces(field_name: str, value: Array, vertex_count: int) -> Array:
    faces = np.asarray(value)
    if faces.ndim != 2 or faces.shape[1] != 3:
        _fail(field_name, "must have shape Mx3")
    if not np.issubdtype(faces.dtype, np.integer):
        _fail(field_name, "must be an integer array")
    if faces.size > 0 and (np.any(faces < 0) or np.any(faces >= vertex_count)):
        _fail(field_name, "contains vertex indices outside vertices_m")
    return faces


def validate_same_shape(field_name: str, value: Array, expected_shape: tuple[int, ...]) -> Array:
    array = validate_finite_numeric_array(field_name, value)
    if array.shape != expected_shape:
        _fail(field_name, f"must have shape {expected_shape}, got {array.shape}")
    return array
