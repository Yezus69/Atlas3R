"""Depth Pro external teacher runner for Atlas3R teacher-signal caches."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib import import_module
from inspect import Parameter, signature
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.models.adapters._dependency import missing_optional_modules
from atlas3r.teachers.external.cache_writer import (
    ExternalSignalPayload,
    write_external_teacher_signal_cache,
)
from atlas3r.teachers.external.contracts import (
    ExternalTeacherDependencyError,
    ExternalTeacherRunConfig,
    ExternalTeacherStatus,
)
from atlas3r.teachers.map_eval import TeacherSignalInspectConfig, inspect_teacher_signals

TEACHER_NAME = "depth_pro"
TEACHER_VERSION = "external-depth-pro-v1"
TEACHER_SOURCE_TYPE = "single_frame_metric_depth_teacher"
REQUIRED_MODULES = ("depth_pro",)
INSTALL_HINT = (
    "Install Depth Pro in the active environment and keep its repository and checkpoints "
    "outside Atlas3R; do not vendor third-party code or weights into this repo."
)
INPUT_FORMAT = "Atlas3R clip cache with images_rgb_u8, K, T_world_camera, frame IDs, timestamps"
OUTPUT_FORMAT = "Atlas3R teacher-signal cache v1 with pseudo-label metric depth"
CHECKPOINT_ENV_VAR = "ATLAS3R_DEPTH_PRO_CHECKPOINT"
_SIGMA_HEURISTIC = (
    "If Depth Pro does not provide depth_sigma_m, Atlas3R sets sigma to "
    "max(0.05 m, 0.05 * depth_m), divided by max(confidence, 0.25) when confidence is "
    "provided; invalid pixels use depth/sigma/confidence 0."
)


@dataclass(frozen=True)
class DepthProRunConfig(ExternalTeacherRunConfig):
    checkpoint_uri: str | None = None


@dataclass(frozen=True)
class DepthProFramePrediction:
    depth_m: npt.NDArray[Any]
    confidence: npt.NDArray[Any] | None = None
    depth_sigma_m: npt.NDArray[Any] | None = None


DepthProFramePredictor = Callable[
    [npt.NDArray[np.uint8], npt.NDArray[np.float32]],
    DepthProFramePrediction,
]


class DepthProExternalTeacherRunner:
    """Run Depth Pro frame-by-frame into Atlas3R teacher-signal payloads."""

    def __init__(self, predictor: DepthProFramePredictor | None = None) -> None:
        self._predictor = predictor

    def status(self) -> ExternalTeacherStatus:
        return get_depth_pro_status()

    def run(self, config: ExternalTeacherRunConfig) -> dict[str, object]:
        predictor = self._predictor
        if predictor is None:
            status = self.status()
            checkpoint_uri = _checkpoint_uri_from_config(config)
            if status.availability != "available":
                raise ExternalTeacherDependencyError(
                    status.display_name,
                    status.reason or "missing optional dependencies",
                    status.install_hint,
                )
            if checkpoint_uri is None:
                raise ExternalTeacherDependencyError(
                    status.display_name,
                    (
                        "no external checkpoint URI configured; pass --checkpoint-uri "
                        f"or set {CHECKPOINT_ENV_VAR}"
                    ),
                    status.install_hint,
                )
            predictor = _load_depth_pro_predictor(checkpoint_uri=checkpoint_uri)
        return run_depth_pro_teacher_signal_cache(config, predictor=predictor)


def get_depth_pro_status() -> ExternalTeacherStatus:
    missing = missing_optional_modules(REQUIRED_MODULES)
    if missing:
        return ExternalTeacherStatus(
            name=TEACHER_NAME,
            display_name="Depth Pro",
            availability="unavailable",
            can_run_locally=False,
            install_hint=INSTALL_HINT,
            expected_input_format=INPUT_FORMAT,
            expected_output_format=OUTPUT_FORMAT,
            reason=f"missing optional dependency: {', '.join(missing)}",
        )
    checkpoint_uri = os.environ.get(CHECKPOINT_ENV_VAR)
    can_run_locally = bool(checkpoint_uri)
    reason = (
        f"external checkpoint configured by {CHECKPOINT_ENV_VAR}"
        if checkpoint_uri
        else (
            "Depth Pro import target is present, but no external checkpoint URI is configured; "
            f"pass --checkpoint-uri or set {CHECKPOINT_ENV_VAR}."
        )
    )
    return ExternalTeacherStatus(
        name=TEACHER_NAME,
        display_name="Depth Pro",
        availability="available",
        can_run_locally=can_run_locally,
        install_hint=INSTALL_HINT,
        expected_input_format=INPUT_FORMAT,
        expected_output_format=OUTPUT_FORMAT,
        reason=reason,
    )


def run_depth_pro_teacher_signal_cache(
    config: ExternalTeacherRunConfig,
    *,
    predictor: DepthProFramePredictor | None = None,
) -> dict[str, object]:
    """Run Depth Pro, write a validated signal cache, and optionally inspect it."""

    if predictor is None:
        checkpoint_uri = _checkpoint_uri_from_config(config)
        if checkpoint_uri is None:
            raise ExternalTeacherDependencyError(
                "Depth Pro",
                (
                    "no external checkpoint URI configured; pass --checkpoint-uri "
                    f"or set {CHECKPOINT_ENV_VAR}"
                ),
                INSTALL_HINT,
            )
        predictor = _load_depth_pro_predictor(checkpoint_uri=checkpoint_uri)
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clips = _clip_entries(clip_manifest)
    selected = clips if config.max_clips is None else clips[: config.max_clips]
    if not selected:
        raise ValueError(f"{clip_manifest_path}: no clips selected for Depth Pro")

    payloads: list[ExternalSignalPayload] = []
    for clip_entry in selected:
        clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
        validate_clip_payload(
            clip_payload,
            clip_length=_int_field(clip_manifest, "clip_length"),
            height=_int_field(clip_manifest, "image_height"),
            width=_int_field(clip_manifest, "image_width"),
        )
        payloads.append(
            ExternalSignalPayload(
                source_clip_id=_int_field(clip_entry, "clip_id"),
                arrays=_payload_from_clip_payload(clip_payload, predictor=predictor),
            )
        )

    result = write_external_teacher_signal_cache(
        clip_cache=clip_manifest_path,
        output=config.output,
        payloads=payloads,
        teacher_name=TEACHER_NAME,
        teacher_version=TEACHER_VERSION,
        teacher_source_type=TEACHER_SOURCE_TYPE,
        source_metadata={
            "runner": "atlas3r.teachers.external.depth_pro",
            "depth_source": "Depth Pro single-frame RGB metric depth",
            "pose_source": "source Atlas3R clip-cache T_world_camera",
            "valid_pixel_rule": "finite positive predicted depth and positive confidence",
            "measured_geometry": False,
            "pseudo_label": True,
            "depth_sigma_m_heuristic": _SIGMA_HEURISTIC,
        },
    )
    if config.run_inspect:
        inspect_output = config.inspect_output or config.output.with_name(
            f"{config.output.name}_inspect"
        )
        inspection = inspect_teacher_signals(
            TeacherSignalInspectConfig(
                clip_cache=clip_manifest_path,
                teacher_cache=config.output,
                output=inspect_output,
                max_clips=config.max_clips or 64,
            )
        )
        result["inspection"] = {
            "summary_path": inspection["summary_path"],
            "per_clip_metrics_path": inspection["per_clip_metrics_path"],
        }
    return result


def _payload_from_clip_payload(
    clip_payload: Mapping[str, Any],
    *,
    predictor: DepthProFramePredictor,
) -> dict[str, npt.NDArray[Any] | np.generic]:
    rgb = np.asarray(clip_payload["images_rgb_u8"], dtype=np.uint8)
    clip_length, height, width, _channels = rgb.shape
    K = np.asarray(clip_payload["K"], dtype=np.float32)
    depth = np.zeros((clip_length, height, width), dtype=np.float32)
    sigma = np.zeros_like(depth, dtype=np.float32)
    confidence = np.zeros_like(depth, dtype=np.float32)
    valid_mask = np.zeros((clip_length, height, width), dtype=np.bool_)
    for frame_offset in range(clip_length):
        frame_prediction = predictor(rgb[frame_offset], K[frame_offset])
        frame_arrays = _normalise_frame_prediction(
            frame_prediction,
            height=height,
            width=width,
        )
        depth[frame_offset] = frame_arrays["depth_m"]
        sigma[frame_offset] = frame_arrays["depth_sigma_m"]
        confidence[frame_offset] = frame_arrays["confidence"]
        valid_mask[frame_offset] = frame_arrays["valid_mask"]
    return {
        "depth_m": depth,
        "depth_sigma_m": sigma,
        "confidence": confidence,
        "valid_mask": valid_mask,
        "K": K.astype(np.float32, copy=False),
        "T_world_camera": np.asarray(clip_payload["T_world_camera"], dtype=np.float32),
        "frame_ids": np.asarray(clip_payload["frame_ids"], dtype=np.int32),
        "timestamps_s": np.asarray(clip_payload["timestamps_s"], dtype=np.float64),
    }


def _normalise_frame_prediction(
    prediction: DepthProFramePrediction,
    *,
    height: int,
    width: int,
) -> dict[str, npt.NDArray[Any]]:
    depth = _image_float32(prediction.depth_m, "depth_m", height, width)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth = np.where(valid, depth, np.float32(0.0)).astype(np.float32, copy=False)
    if prediction.confidence is None:
        confidence = np.where(valid, np.float32(0.5), np.float32(0.0)).astype(np.float32)
    else:
        confidence = _image_float32(prediction.confidence, "confidence", height, width)
        _validate_probability("confidence", confidence)
        confidence = np.where(valid, confidence, np.float32(0.0)).astype(np.float32, copy=False)
    valid = valid & (confidence > 0.0)
    if prediction.depth_sigma_m is None:
        sigma = _heuristic_sigma(
            depth, confidence, valid, confidence_provided=prediction.confidence is not None
        )
    else:
        sigma = _image_float32(prediction.depth_sigma_m, "depth_sigma_m", height, width)
        if np.any(sigma < 0.0):
            raise ValueError("depth_sigma_m: must be non-negative")
        if np.any(sigma[valid] <= 0.0):
            raise ValueError("depth_sigma_m: must be positive on valid pixels")
        sigma = np.where(valid, sigma, np.float32(0.0)).astype(np.float32, copy=False)
    return {
        "depth_m": np.where(valid, depth, np.float32(0.0)).astype(np.float32, copy=False),
        "depth_sigma_m": sigma,
        "confidence": np.where(valid, confidence, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        ),
        "valid_mask": valid.astype(np.bool_, copy=False),
    }


def _heuristic_sigma(
    depth: npt.NDArray[np.float32],
    confidence: npt.NDArray[np.float32],
    valid: npt.NDArray[np.bool_],
    *,
    confidence_provided: bool,
) -> npt.NDArray[np.float32]:
    sigma = np.zeros_like(depth, dtype=np.float32)
    base = np.maximum(np.float32(0.05), depth * np.float32(0.05))
    if confidence_provided:
        base = base / np.maximum(confidence, np.float32(0.25))
    sigma[valid] = base[valid]
    return sigma


def _checkpoint_uri_from_config(config: ExternalTeacherRunConfig) -> str | None:
    value = getattr(config, "checkpoint_uri", None)
    if isinstance(value, str) and value:
        return value
    env_value = os.environ.get(CHECKPOINT_ENV_VAR)
    return env_value if env_value else None


def _load_depth_pro_predictor(*, checkpoint_uri: str) -> DepthProFramePredictor:
    try:
        depth_pro = import_module("depth_pro")
    except ImportError as exc:
        raise ExternalTeacherDependencyError("Depth Pro", str(exc), INSTALL_HINT) from exc
    try:
        model_config = _depth_pro_config_with_checkpoint(depth_pro, checkpoint_uri)
        model, transform = depth_pro.create_model_and_transforms(config=model_config)
    except AttributeError as exc:
        raise ExternalTeacherDependencyError(
            "Depth Pro",
            "depth_pro.create_model_and_transforms is unavailable",
            INSTALL_HINT,
        ) from exc
    if hasattr(model, "eval"):
        model.eval()

    def predict(
        rgb_u8: npt.NDArray[np.uint8], K: npt.NDArray[np.float32]
    ) -> DepthProFramePrediction:
        image = _pil_image_from_rgb(rgb_u8)
        model_input = transform(image) if transform is not None else image
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


def _depth_pro_config_with_checkpoint(depth_pro: Any, checkpoint_uri: str) -> Any:
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
        INSTALL_HINT,
    )


def _infer_depth_pro(model: Any, model_input: Any, K: npt.NDArray[np.float32]) -> Any:
    f_px = float((float(K[0, 0]) + float(K[1, 1])) * 0.5)
    torch = _optional_import("torch")
    context = torch.no_grad() if torch is not None else nullcontext()
    with context:
        try:
            return model.infer(model_input, f_px=f_px)
        except TypeError:
            return model.infer(model_input)


def _pil_image_from_rgb(rgb_u8: npt.NDArray[np.uint8]) -> Any:
    try:
        image_module = import_module("PIL.Image")
    except ImportError as exc:
        raise ExternalTeacherDependencyError("Depth Pro", str(exc), INSTALL_HINT) from exc
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


def _image_float32(
    value: npt.NDArray[Any],
    key: str,
    height: int,
    width: int,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (height, width):
        raise ValueError(f"{key}: expected shape {(height, width)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key}: must contain finite values")
    return array


def _validate_probability(key: str, array: npt.NDArray[np.float32]) -> None:
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{key}: must contain values in [0, 1]")


def _optional_import(module_name: str) -> Any | None:
    try:
        return import_module(module_name)
    except ImportError:
        return None


def _clip_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    value = manifest.get("clips")
    if not isinstance(value, list):
        raise ValueError("clip manifest clips: must be a list")
    entries: list[dict[str, object]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"clip manifest.clips[{index}]: must be a mapping")
        entries.append(cast(dict[str, object], entry))
    return entries


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "DepthProExternalTeacherRunner",
    "DepthProFramePrediction",
    "DepthProFramePredictor",
    "DepthProRunConfig",
    "get_depth_pro_status",
    "run_depth_pro_teacher_signal_cache",
]
