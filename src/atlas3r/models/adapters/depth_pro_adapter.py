"""Dependency-safe adapter for Apple's external Depth Pro runtime."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class DepthProRuntimeError(RuntimeError):
    """Raised when the external Depth Pro runtime cannot produce a prediction."""


@dataclass(frozen=True)
class DepthProAdapterConfig:
    repo_path: str | None = None
    checkpoint: str | None = None
    device: str = "cuda:0"
    image_size: int | None = None


@dataclass(frozen=True)
class DepthProFramePrediction:
    frame_id: int
    depth_m: NDArray[np.float32]
    focal_px: float | None
    runtime_ms: float
    output_shapes: dict[str, object]


@dataclass(frozen=True)
class DepthProBatchPrediction:
    frame_predictions: tuple[DepthProFramePrediction, ...]
    model_source: str
    checkpoint: str | None
    device: str
    image_size: int | None
    runtime_ms: float


def probe_depth_pro_runtime(repo_path: str | None = None) -> tuple[bool, str]:
    with _temporary_sys_path(repo_path):
        torch_available = importlib.util.find_spec("torch") is not None
        depth_pro_available = importlib.util.find_spec("depth_pro") is not None
    if torch_available and depth_pro_available:
        return True, "torch and depth_pro import specs are available"
    missing = []
    if not torch_available:
        missing.append("torch")
    if not depth_pro_available:
        missing.append("depth_pro")
    return False, f"missing optional runtime dependencies: {', '.join(missing)}"


def run_depth_pro_frames(
    image_paths: Sequence[str | Path],
    frame_ids: Sequence[int],
    config: DepthProAdapterConfig,
) -> DepthProBatchPrediction:
    if len(image_paths) != len(frame_ids):
        raise ValueError("image_paths and frame_ids must have the same length")
    if not image_paths:
        raise ValueError("Depth Pro batch cannot be empty")
    if config.image_size is not None and config.image_size <= 0:
        raise ValueError("depth_pro_image_size must be positive when provided")
    with _temporary_sys_path(config.repo_path):
        torch = _import_module("torch")
        depth_pro = _import_module("depth_pro")
        device = _resolve_device(torch, config.device)
        precision = torch.half if str(device).startswith("cuda") else torch.float32
        model, transform = _create_model(depth_pro, torch, config, device, precision)
        model.eval()
        predictions: list[DepthProFramePrediction] = []
        total_runtime_ms = 0.0
        for image_path, frame_id in zip(image_paths, frame_ids, strict=True):
            started = time.perf_counter()
            image, f_px = _load_image(depth_pro, image_path, config.image_size)
            image_tensor = transform(image)
            with torch.no_grad():
                prediction = model.infer(image_tensor, f_px=f_px)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            total_runtime_ms += runtime_ms
            depth = _tensor_to_numpy(prediction["depth"]).astype(np.float32)
            if depth.ndim != 2:
                raise DepthProRuntimeError(f"Depth Pro depth must be HxW, got {depth.shape}")
            focal_px = _optional_float(prediction.get("focallength_px"))
            predictions.append(
                DepthProFramePrediction(
                    frame_id=int(frame_id),
                    depth_m=depth,
                    focal_px=focal_px,
                    runtime_ms=runtime_ms,
                    output_shapes={
                        "depth": list(depth.shape),
                        "focallength_px": [] if focal_px is not None else None,
                    },
                )
            )
    return DepthProBatchPrediction(
        frame_predictions=tuple(predictions),
        model_source=config.repo_path or "python_package:depth_pro",
        checkpoint=config.checkpoint,
        device=device,
        image_size=config.image_size,
        runtime_ms=total_runtime_ms,
    )


def _create_model(
    depth_pro: ModuleType,
    torch: ModuleType,
    config: DepthProAdapterConfig,
    device: str,
    precision: Any,
) -> tuple[Any, Any]:
    if not hasattr(depth_pro, "create_model_and_transforms"):
        raise DepthProRuntimeError("depth_pro.create_model_and_transforms is unavailable")
    model_config = getattr(depth_pro, "DEFAULT_MONODEPTH_CONFIG_DICT", None)
    if model_config is None:
        depth_module = _import_module("depth_pro.depth_pro")
        model_config = getattr(depth_module, "DEFAULT_MONODEPTH_CONFIG_DICT", None)
    if config.checkpoint and model_config is not None:
        from dataclasses import replace

        model_config = replace(model_config, checkpoint_uri=config.checkpoint)
    kwargs: dict[str, object] = {
        "device": torch.device(device),
        "precision": precision,
    }
    if model_config is not None:
        kwargs["config"] = model_config
    try:
        created = depth_pro.create_model_and_transforms(**kwargs)
        return cast(tuple[Any, Any], created)
    except TypeError as exc:
        if config.checkpoint:
            raise DepthProRuntimeError(
                "Depth Pro runtime does not expose checkpoint override support"
            ) from exc
        created = depth_pro.create_model_and_transforms()
        return cast(tuple[Any, Any], created)


def _load_image(
    depth_pro: ModuleType, image_path: str | Path, image_size: int | None
) -> tuple[Any, Any]:
    if not hasattr(depth_pro, "load_rgb"):
        raise DepthProRuntimeError("depth_pro.load_rgb is unavailable")
    image, _metadata, f_px = depth_pro.load_rgb(str(image_path))
    if image_size is not None:
        original_width, _original_height = image.size
        scale = float(image_size) / float(original_width)
        image = image.resize((image_size, image_size))
        if f_px is not None:
            f_px = float(f_px) * scale
    return image, f_px


def _resolve_device(torch: ModuleType, requested: str) -> str:
    if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise DepthProRuntimeError(f"requested device {requested} but CUDA is unavailable")
    return requested


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exercised by unavailable path
        raise DepthProRuntimeError(f"failed to import optional module {name}: {exc}") from exc


def _tensor_to_numpy(value: Any) -> NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    array = _tensor_to_numpy(value).reshape((-1,))
    if array.size == 0:
        return None
    scalar = float(array[0])
    return scalar if np.isfinite(scalar) and scalar > 0.0 else None


@contextmanager
def _temporary_sys_path(repo_path: str | None) -> Iterator[None]:
    if repo_path is None:
        yield
        return
    root = str(Path(repo_path).resolve())
    src = str((Path(repo_path) / "src").resolve())
    added: list[str] = []
    for candidate in (src, root):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
            added.append(candidate)
    try:
        yield
    finally:
        for candidate in added:
            if candidate in sys.path:
                sys.path.remove(candidate)
