"""Depth Pro external teacher runner for Atlas3R teacher-signal caches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
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
from atlas3r.teachers.external.depth_pro_arrays import (
    DepthProRunStats,
    normalise_frame_prediction_arrays,
)
from atlas3r.teachers.external.depth_pro_model import load_depth_pro_predictor
from atlas3r.teachers.external.depth_pro_types import (
    DepthProFramePrediction,
    DepthProFramePredictor,
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
    device: str = "auto"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device: must be one of auto, cuda, mps, or cpu")


@dataclass(frozen=True)
class _CachedFramePrediction:
    rgb_u8: npt.NDArray[np.uint8]
    K: npt.NDArray[np.float32]
    arrays: dict[str, npt.NDArray[Any]]


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
            predictor = load_depth_pro_predictor(
                checkpoint_uri=checkpoint_uri,
                device=_device_from_config(config),
                install_hint=INSTALL_HINT,
            )
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
        predictor = load_depth_pro_predictor(
            checkpoint_uri=checkpoint_uri,
            device=_device_from_config(config),
            install_hint=INSTALL_HINT,
        )
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clips = _clip_entries(clip_manifest)
    selected = clips if config.max_clips is None else clips[: config.max_clips]
    if not selected:
        raise ValueError(f"{clip_manifest_path}: no clips selected for Depth Pro")

    payloads: list[ExternalSignalPayload] = []
    stats = DepthProRunStats()
    prediction_cache: dict[int, _CachedFramePrediction] = {}
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
                arrays=_payload_from_clip_payload(
                    clip_payload,
                    predictor=predictor,
                    prediction_cache=prediction_cache,
                    stats=stats,
                ),
            )
        )

    run_metadata = stats.source_metadata()
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
            **run_metadata,
        },
    )
    result.update(stats.result_fields())
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
    prediction_cache: dict[int, _CachedFramePrediction],
    stats: DepthProRunStats,
) -> dict[str, npt.NDArray[Any] | np.generic]:
    rgb = np.asarray(clip_payload["images_rgb_u8"], dtype=np.uint8)
    clip_length, height, width, _channels = rgb.shape
    K = np.asarray(clip_payload["K"], dtype=np.float32)
    frame_ids = np.asarray(clip_payload["frame_ids"], dtype=np.int64)
    stats.record_clip(clip_length)
    depth = np.zeros((clip_length, height, width), dtype=np.float32)
    sigma = np.zeros_like(depth, dtype=np.float32)
    confidence = np.zeros_like(depth, dtype=np.float32)
    valid_mask = np.zeros((clip_length, height, width), dtype=np.bool_)
    for frame_offset in range(clip_length):
        frame_id = int(frame_ids[frame_offset])
        frame_arrays = _cached_or_predict_frame(
            frame_id=frame_id,
            rgb_u8=rgb[frame_offset],
            K=K[frame_offset],
            predictor=predictor,
            prediction_cache=prediction_cache,
            height=height,
            width=width,
            stats=stats,
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
        "frame_ids": frame_ids.astype(np.int32, copy=False),
        "timestamps_s": np.asarray(clip_payload["timestamps_s"], dtype=np.float64),
    }


def _cached_or_predict_frame(
    *,
    frame_id: int,
    rgb_u8: npt.NDArray[np.uint8],
    K: npt.NDArray[np.float32],
    predictor: DepthProFramePredictor,
    prediction_cache: dict[int, _CachedFramePrediction],
    height: int,
    width: int,
    stats: DepthProRunStats,
) -> dict[str, npt.NDArray[Any]]:
    cached = prediction_cache.get(frame_id)
    if cached is not None:
        _validate_duplicate_frame(frame_id, rgb_u8=rgb_u8, K=K, cached=cached)
        stats.record_duplicate_reuse()
        return cached.arrays
    frame_prediction = predictor(rgb_u8, K)
    stats.record_unique_prediction()
    arrays = normalise_frame_prediction_arrays(
        depth_m=frame_prediction.depth_m,
        confidence=frame_prediction.confidence,
        depth_sigma_m=frame_prediction.depth_sigma_m,
        height=height,
        width=width,
        stats=stats,
    )
    prediction_cache[frame_id] = _CachedFramePrediction(
        rgb_u8=rgb_u8.copy(),
        K=K.astype(np.float32, copy=True),
        arrays={key: np.asarray(value).copy() for key, value in arrays.items()},
    )
    return arrays


def _validate_duplicate_frame(
    frame_id: int,
    *,
    rgb_u8: npt.NDArray[np.uint8],
    K: npt.NDArray[np.float32],
    cached: _CachedFramePrediction,
) -> None:
    if not np.array_equal(rgb_u8, cached.rgb_u8):
        raise ValueError(f"frame_id {frame_id}: duplicate RGB payload does not match")
    if not np.allclose(K, cached.K, rtol=0.0, atol=0.0):
        raise ValueError(f"frame_id {frame_id}: duplicate K metadata does not match")


def _checkpoint_uri_from_config(config: ExternalTeacherRunConfig) -> str | None:
    value = getattr(config, "checkpoint_uri", None)
    if isinstance(value, str) and value:
        return value
    env_value = os.environ.get(CHECKPOINT_ENV_VAR)
    return env_value if env_value else None


def _device_from_config(config: ExternalTeacherRunConfig) -> str:
    value = getattr(config, "device", "auto")
    if not isinstance(value, str):
        raise ValueError("device: must be one of auto, cuda, mps, or cpu")
    return value


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
