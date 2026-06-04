"""Optional real VGGT model loading for the Phase 5H teacher runner."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from atlas3r.teachers.external.contracts import (
    ExternalTeacherDependencyError,
    ExternalTeacherError,
)

INSTALL_HINT = (
    "Install VGGT in the active environment or set ATLAS3R_VGGT_REPO to a local "
    "VGGT checkout; keep VGGT code and weights outside Atlas3R."
)
DEFAULT_MODEL_ID = "facebook/VGGT-1B"
DEFAULT_REMOTE_CHECKPOINT = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
MODEL_INPUT_RESOLUTION = 518

VGGTRawClipPredictor = Callable[[Mapping[str, Any]], Mapping[str, object]]


@dataclass(frozen=True)
class LoadedVGGTPredictor:
    predictor: VGGTRawClipPredictor
    resolved_device: str
    model_source: str
    checkpoint: str | None
    input_resolution: int


def load_vggt_predictor(
    *,
    vggt_repo: Path | None,
    checkpoint: str | None,
    device: str,
) -> LoadedVGGTPredictor:
    """Load VGGT and return a per-clip prediction callable."""

    if vggt_repo is not None:
        _add_repo_to_path(vggt_repo)
    try:
        torch = import_module("torch")
        functional = import_module("torch.nn.functional")
        model_module = import_module("vggt.models.vggt")
        pose_module = import_module("vggt.utils.pose_enc")
    except ImportError as exc:
        raise ExternalTeacherDependencyError("VGGT", str(exc), INSTALL_HINT) from exc

    resolved_device = _resolve_device(torch, device)
    model_class = getattr(model_module, "VGGT", None)
    if model_class is None:
        raise ExternalTeacherDependencyError(
            "VGGT",
            "vggt.models.vggt.VGGT is unavailable",
            INSTALL_HINT,
        )
    model, model_source = _construct_model(torch, model_class, checkpoint)
    model = model.to(resolved_device) if hasattr(model, "to") else model
    if hasattr(model, "eval"):
        model.eval()
    pose_decode = getattr(pose_module, "pose_encoding_to_extri_intri", None)
    if pose_decode is None:
        raise ExternalTeacherDependencyError(
            "VGGT",
            "vggt.utils.pose_enc.pose_encoding_to_extri_intri is unavailable",
            INSTALL_HINT,
        )

    def predict(clip_payload: Mapping[str, Any]) -> Mapping[str, object]:
        rgb = np.asarray(clip_payload["images_rgb_u8"], dtype=np.uint8)
        images = torch.from_numpy(np.transpose(rgb, (0, 3, 1, 2))).float()
        images = images.to(resolved_device) / 255.0
        images = functional.interpolate(
            images,
            size=(MODEL_INPUT_RESOLUTION, MODEL_INPUT_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )
        dtype = _autocast_dtype(torch, resolved_device)
        context = _autocast_context(torch, resolved_device, dtype)
        with torch.no_grad():
            with context:
                predictions = model(images)
        if not isinstance(predictions, Mapping):
            raise ExternalTeacherError("VGGT prediction must be a mapping")
        output = dict(predictions)
        if "pose_enc" in output and "extrinsic" not in output:
            with torch.no_grad():
                extrinsic, intrinsic = pose_decode(output["pose_enc"], images.shape[-2:])
            output["extrinsic"] = extrinsic
            output["intrinsic"] = intrinsic
        output["model_input_resolution"] = MODEL_INPUT_RESOLUTION
        return output

    return LoadedVGGTPredictor(
        predictor=predict,
        resolved_device=resolved_device,
        model_source=model_source,
        checkpoint=checkpoint,
        input_resolution=MODEL_INPUT_RESOLUTION,
    )


def _add_repo_to_path(vggt_repo: Path) -> None:
    repo = vggt_repo.expanduser().resolve()
    if not repo.is_dir():
        raise ExternalTeacherDependencyError(
            "VGGT",
            f"{repo}: path is not a directory",
            INSTALL_HINT,
        )
    if not (repo / "vggt").is_dir():
        raise ExternalTeacherDependencyError(
            "VGGT",
            f"{repo}: expected a VGGT checkout containing a vggt package directory",
            INSTALL_HINT,
        )
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def _resolve_device(torch: Any, device: str) -> str:
    normalized = device.lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device: must be one of auto, cuda, mps, or cpu")
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
    if normalized == "mps" and not mps_available:
        raise ValueError("device: mps was requested but torch.backends.mps is not available")
    return normalized


def _construct_model(torch: Any, model_class: Any, checkpoint: str | None) -> tuple[Any, str]:
    if checkpoint:
        model = model_class()
        if checkpoint.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
        else:
            state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model, "explicit_checkpoint"
    if hasattr(model_class, "from_pretrained"):
        return model_class.from_pretrained(DEFAULT_MODEL_ID), DEFAULT_MODEL_ID
    model = model_class()
    state = torch.hub.load_state_dict_from_url(DEFAULT_REMOTE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    return model, DEFAULT_REMOTE_CHECKPOINT


def _autocast_dtype(torch: Any, device: str) -> Any:
    if device != "cuda":
        return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(torch: Any, device: str, dtype: Any) -> Any:
    if device == "cuda" and dtype is not None:
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


__all__ = [
    "INSTALL_HINT",
    "LoadedVGGTPredictor",
    "VGGTRawClipPredictor",
    "load_vggt_predictor",
]
