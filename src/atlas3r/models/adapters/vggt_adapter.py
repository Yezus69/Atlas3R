"""Dependency-safe VGGT runtime adapter."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class VggtRuntimeError(RuntimeError):
    """Raised when VGGT cannot run in the external runtime environment."""


@dataclass(frozen=True)
class VggtAdapterConfig:
    repo_path: str | None
    checkpoint: str | None
    device: str
    image_size: int


@dataclass(frozen=True)
class VggtWindowPrediction:
    frame_ids: tuple[int, ...]
    intrinsics: NDArray[np.float32]
    extrinsics_camera_from_world: NDArray[np.float32]
    depth_m: NDArray[np.float32]
    depth_confidence: NDArray[np.float32]
    runtime_ms: float
    dtype: str
    model_source: str
    checkpoint: str
    output_shapes: dict[str, tuple[int, ...]]


def probe_vggt_runtime(repo_path: str | None = None) -> tuple[bool, str]:
    with _temporary_sys_path(repo_path):
        torch_available = importlib.util.find_spec("torch") is not None
        vggt_available = importlib.util.find_spec("vggt") is not None
    if torch_available and vggt_available:
        return True, "torch and vggt import specs are available"
    missing = []
    if not torch_available:
        missing.append("torch")
    if not vggt_available:
        missing.append("vggt")
    return False, f"missing optional runtime dependencies: {', '.join(missing)}"


def run_vggt_window(
    image_paths: Sequence[str | Path],
    frame_ids: Sequence[int],
    config: VggtAdapterConfig,
) -> VggtWindowPrediction:
    if len(image_paths) != len(frame_ids):
        raise ValueError("image_paths and frame_ids must have the same length")
    if not image_paths:
        raise ValueError("VGGT window cannot be empty")
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    with _temporary_sys_path(config.repo_path):
        started = time.perf_counter()
        torch = _import_module("torch")
        vggt_module = _import_module("vggt.models.vggt")
        load_fn_module = _import_module("vggt.utils.load_fn")
        pose_module = _import_module("vggt.utils.pose_enc")
        device = _resolve_device(torch, config.device)
        dtype = _select_dtype(torch, device)
        model = _load_model(torch, vggt_module, config)
        model = model.to(device)
        model.eval()
        image_loader = cast(Any, load_fn_module).load_and_preprocess_images
        images = image_loader([str(path) for path in image_paths]).to(device)
        if images.ndim == 4:
            images = images[None]
        if hasattr(torch, "no_grad"):
            grad_context = torch.no_grad()
        else:
            grad_context = nullcontext()
        with grad_context:
            with _autocast(torch, device, dtype):
                tokens, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(tokens)[-1]
                pose_to_mats = cast(Any, pose_module).pose_encoding_to_extri_intri
                extrinsic, intrinsic = pose_to_mats(pose_enc, images.shape[-2:])
                depth_map, depth_conf = model.depth_head(tokens, images, ps_idx)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        intrinsic_np = _tensor_to_numpy(intrinsic)
        extrinsic_np = _tensor_to_numpy(extrinsic)
        depth_np = _tensor_to_numpy(depth_map)
        depth_conf_np = _tensor_to_numpy(depth_conf)
    intrinsics = _normalize_intrinsics(intrinsic_np, len(frame_ids))
    extrinsics = _normalize_extrinsics(extrinsic_np, len(frame_ids))
    depth = _normalize_depth(depth_np, len(frame_ids))
    confidence = _normalize_depth(depth_conf_np, len(frame_ids))
    return VggtWindowPrediction(
        frame_ids=tuple(int(item) for item in frame_ids),
        intrinsics=intrinsics,
        extrinsics_camera_from_world=extrinsics,
        depth_m=depth,
        depth_confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32),
        runtime_ms=float(runtime_ms),
        dtype=str(dtype).replace("torch.", ""),
        model_source=_model_source(config),
        checkpoint=config.checkpoint or "facebook/VGGT-1B",
        output_shapes={
            "intrinsics": tuple(int(item) for item in intrinsics.shape),
            "extrinsics_camera_from_world": tuple(int(item) for item in extrinsics.shape),
            "depth_m": tuple(int(item) for item in depth.shape),
            "depth_confidence": tuple(int(item) for item in confidence.shape),
        },
    )


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise VggtRuntimeError(f"could not import optional module {name}: {exc}") from exc


def _load_model(torch: ModuleType, vggt_module: ModuleType, config: VggtAdapterConfig) -> Any:
    model_class = cast(Any, vggt_module).VGGT
    checkpoint = config.checkpoint or "facebook/VGGT-1B"
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        try:
            model = model_class(img_size=config.image_size)
        except TypeError:
            model = model_class()
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model
    try:
        return model_class.from_pretrained(checkpoint)
    except Exception as exc:
        raise VggtRuntimeError(f"could not load VGGT checkpoint {checkpoint}: {exc}") from exc


def _resolve_device(torch: ModuleType, requested: str) -> str:
    if requested.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise VggtRuntimeError(f"requested {requested}, but CUDA is not available")
    return requested


def _select_dtype(torch: ModuleType, device: str) -> Any:
    if not device.startswith("cuda"):
        return torch.float32
    try:
        major = int(torch.cuda.get_device_capability()[0])
        return torch.bfloat16 if major >= 8 else torch.float16
    except Exception:
        return torch.float16


@contextmanager
def _autocast(torch: ModuleType, device: str, dtype: Any) -> Iterator[None]:
    if not device.startswith("cuda") or str(dtype).endswith("float32"):
        yield
        return
    autocast = cast(Any, torch).cuda.amp.autocast
    with autocast(dtype=dtype):
        yield


@contextmanager
def _temporary_sys_path(repo_path: str | None) -> Iterator[None]:
    if not repo_path:
        yield
        return
    resolved = str(Path(repo_path).resolve())
    inserted = False
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(resolved)
            except ValueError:
                pass


def _tensor_to_numpy(value: Any) -> NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        return cast(NDArray[np.float32], np.asarray(value.numpy(), dtype=np.float32))
    return cast(NDArray[np.float32], np.asarray(value, dtype=np.float32))


def _normalize_intrinsics(value: NDArray[np.float32], frame_count: int) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.shape == (3, 3):
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[-2:] != (3, 3):
        raise VggtRuntimeError(f"unexpected VGGT intrinsic shape: {array.shape}")
    if array.shape[0] != frame_count:
        raise VggtRuntimeError(f"expected {frame_count} intrinsics, got {array.shape[0]}")
    return array.astype(np.float32)


def _normalize_extrinsics(value: NDArray[np.float32], frame_count: int) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.shape == (3, 4):
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[-2:] not in {(3, 4), (4, 4)}:
        raise VggtRuntimeError(f"unexpected VGGT extrinsic shape: {array.shape}")
    if array.shape[0] != frame_count:
        raise VggtRuntimeError(f"expected {frame_count} extrinsics, got {array.shape[0]}")
    if array.shape[-2:] == (4, 4):
        return array.astype(np.float32)
    padded = np.tile(np.eye(4, dtype=np.float32), (array.shape[0], 1, 1))
    padded[:, :3, :4] = array
    return padded.astype(np.float32)


def _normalize_depth(value: NDArray[np.float32], frame_count: int) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise VggtRuntimeError(f"unexpected VGGT depth shape: {array.shape}")
    if array.shape[0] != frame_count:
        raise VggtRuntimeError(f"expected {frame_count} depth maps, got {array.shape[0]}")
    return array.astype(np.float32)


def _model_source(config: VggtAdapterConfig) -> str:
    if config.repo_path:
        return str(Path(config.repo_path).resolve())
    return "python_package:vggt"
