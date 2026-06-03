"""Real Depth Pro model loading and tensor-device helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.teachers.external.contracts import ExternalTeacherDependencyError
from atlas3r.teachers.external.depth_pro_types import (
    DepthProFramePrediction,
    DepthProFramePredictor,
)


def load_depth_pro_predictor(
    *,
    checkpoint_uri: str,
    device: str,
    install_hint: str,
) -> DepthProFramePredictor:
    try:
        depth_pro = import_module("depth_pro")
    except ImportError as exc:
        raise ExternalTeacherDependencyError("Depth Pro", str(exc), install_hint) from exc
    resolved_device = _resolve_depth_pro_device(device, install_hint=install_hint)
    try:
        model_config = _depth_pro_config_with_checkpoint(
            depth_pro,
            checkpoint_uri,
            install_hint=install_hint,
        )
        model, transform = _create_depth_pro_model(
            depth_pro,
            config=model_config,
            device=resolved_device,
        )
    except AttributeError as exc:
        raise ExternalTeacherDependencyError(
            "Depth Pro",
            "depth_pro.create_model_and_transforms is unavailable",
            install_hint,
        ) from exc
    model = _move_model_to_device(model, resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    def predict(
        rgb_u8: npt.NDArray[np.uint8], K: npt.NDArray[np.float32]
    ) -> DepthProFramePrediction:
        image = _pil_image_from_rgb(rgb_u8, install_hint=install_hint)
        model_input = transform(image) if transform is not None else image
        model_input = _move_tensors_to_device(model_input, resolved_device)
        prediction = _infer_depth_pro(model, model_input, K)
        depth = _prediction_array(prediction, ("depth_m", "depth"))
        confidence = _optional_prediction_array(
            prediction,
            ("confidence", "confidence_map", "valid_confidence"),
        )
        sigma = _optional_prediction_array(
            prediction,
            ("depth_sigma_m", "depth_uncertainty_m", "uncertainty"),
        )
        return DepthProFramePrediction(depth_m=depth, confidence=confidence, depth_sigma_m=sigma)

    return predict


def _resolve_depth_pro_device(device: str, *, install_hint: str) -> str:
    from atlas3r.training.torch_runtime import TorchDependencyError, select_device

    try:
        return select_device(device)
    except TorchDependencyError as exc:
        if device in {"auto", "cpu"}:
            return "cpu"
        raise ExternalTeacherDependencyError("Depth Pro", str(exc), install_hint) from exc


def _create_depth_pro_model(depth_pro: Any, *, config: Any, device: str) -> tuple[Any, Any]:
    create_model = depth_pro.create_model_and_transforms
    parameters = signature(create_model).parameters
    kwargs: dict[str, Any] = {}
    if "config" in parameters:
        kwargs["config"] = config
    if "device" in parameters:
        torch = _optional_import("torch")
        kwargs["device"] = torch.device(device) if torch is not None else device
    return cast(tuple[Any, Any], create_model(**kwargs))


def _move_model_to_device(model: Any, device: str) -> Any:
    if not hasattr(model, "to"):
        return model
    try:
        return model.to(device)
    except TypeError:
        return model


def _move_tensors_to_device(value: Any, device: str) -> Any:
    torch = _optional_import("torch")
    if torch is not None and torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    return value


def _depth_pro_config_with_checkpoint(
    depth_pro: Any,
    checkpoint_uri: str,
    *,
    install_hint: str,
) -> Any:
    config_class = getattr(depth_pro, "DepthProConfig", None)
    if config_class is not None:
        try:
            return config_class(checkpoint_uri=checkpoint_uri)
        except TypeError:
            pass
    create_model = getattr(depth_pro, "create_model_and_transforms", None)
    if create_model is not None:
        config_parameter = signature(create_model).parameters.get("config")
        if config_parameter is not None and config_parameter.default is not Parameter.empty:
            try:
                return replace(config_parameter.default, checkpoint_uri=checkpoint_uri)
            except (TypeError, ValueError):
                pass
    raise ExternalTeacherDependencyError(
        "Depth Pro",
        "DepthProConfig cannot be constructed with an external checkpoint_uri",
        install_hint,
    )


def _infer_depth_pro(model: Any, model_input: Any, K: npt.NDArray[np.float32]) -> Any:
    f_px = float((float(K[0, 0]) + float(K[1, 1])) * 0.5)
    torch = _optional_import("torch")
    context = torch.no_grad() if torch is not None else nullcontext()
    f_px_argument = _focal_length_argument(f_px, model_input, torch)
    with context:
        try:
            return model.infer(model_input, f_px=f_px_argument)
        except TypeError:
            return model.infer(model_input)


def _focal_length_argument(f_px: float, model_input: Any, torch: Any | None) -> Any:
    if torch is None:
        return f_px
    reference = _first_tensor(model_input, torch)
    device = None if reference is None else reference.device
    return torch.tensor(f_px, dtype=torch.float32, device=device)


def _first_tensor(value: Any, torch: Any) -> Any | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    if isinstance(value, tuple | list):
        for item in value:
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    return None


def _pil_image_from_rgb(rgb_u8: npt.NDArray[np.uint8], *, install_hint: str) -> Any:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise ExternalTeacherDependencyError("Depth Pro", str(exc), install_hint) from exc
    return image_module.fromarray(np.asarray(rgb_u8, dtype=np.uint8), mode="RGB")


def _prediction_array(prediction: Any, keys: tuple[str, ...]) -> npt.NDArray[Any]:
    value = _prediction_value(prediction, keys)
    if value is None:
        raise ValueError(f"Depth Pro prediction: missing one of {keys}")
    return _to_numpy(value)


def _optional_prediction_array(prediction: Any, keys: tuple[str, ...]) -> npt.NDArray[Any] | None:
    value = _prediction_value(prediction, keys)
    return None if value is None else _to_numpy(value)


def _prediction_value(prediction: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(prediction, Mapping):
        for key in keys:
            if key in prediction:
                return prediction[key]
        return None
    for key in keys:
        if hasattr(prediction, key):
            return getattr(prediction, key)
    return None


def _to_numpy(value: Any) -> npt.NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _optional_import(module_name: str) -> Any | None:
    try:
        return import_module(module_name)
    except ImportError:
        return None


__all__ = [
    "load_depth_pro_predictor",
]
