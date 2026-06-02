"""Dependency-safe helpers for optional image loading."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


class PillowDependencyError(RuntimeError):
    """Raised when optional Pillow image support is requested but unavailable."""


def pillow_available() -> bool:
    """Return whether the optional `Pillow` package can be imported."""

    return importlib.util.find_spec("PIL") is not None


def require_pillow_image() -> Any:
    """Import `PIL.Image` or raise a concise optional-dependency error."""

    if not pillow_available():
        raise PillowDependencyError(
            "Pillow is required for Atlas3R RGB-D data loading. Install the optional train "
            "extra: python -m pip install -e .[train]"
        )
    return importlib.import_module("PIL.Image")


__all__ = [
    "PillowDependencyError",
    "pillow_available",
    "require_pillow_image",
]
