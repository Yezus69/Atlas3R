"""Dependency-safe PyTorch runtime helpers for optional training commands."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


class TorchDependencyError(RuntimeError):
    """Raised when optional PyTorch training support is requested but unavailable."""


def torch_available() -> bool:
    """Return whether the optional `torch` package can be imported."""

    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    """Import torch or raise a concise optional-dependency error."""

    if not torch_available():
        raise TorchDependencyError(
            "PyTorch is required for Atlas3R training. Install the optional train extra: "
            "python -m pip install -e .[train]"
        )
    return importlib.import_module("torch")


def select_device(device: str) -> str:
    """Resolve a requested training device against the available PyTorch backends."""

    normalized = device.lower()
    explicit_cuda_index: int | None = None
    if normalized.startswith("cuda:"):
        try:
            explicit_cuda_index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("device: cuda index must be an integer") from exc
        if explicit_cuda_index < 0:
            raise ValueError("device: cuda index must be non-negative")
    elif normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device: must be one of auto, cuda, cuda:N, mps, or cpu")

    torch = require_torch()
    if normalized == "cpu":
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if normalized == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if normalized == "cuda" and not cuda_available:
        raise ValueError("device: cuda was requested but torch.cuda is not available")
    if explicit_cuda_index is not None:
        if not cuda_available:
            raise ValueError("device: cuda was requested but torch.cuda is not available")
        if explicit_cuda_index >= int(torch.cuda.device_count()):
            raise ValueError("device: requested cuda index is not available")
        return f"cuda:{explicit_cuda_index}"
    if normalized == "mps" and not mps_available:
        raise ValueError("device: mps was requested but torch.backends.mps is not available")
    return normalized


__all__ = [
    "TorchDependencyError",
    "require_torch",
    "select_device",
    "torch_available",
]
