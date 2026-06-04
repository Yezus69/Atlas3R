"""Export Phase 5D temporal student checkpoints as teacher-signal caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from atlas3r.forge.clip_cache import (
    load_clip_cache_manifest,
    manifest_path_from_input,
    read_clip_payload_from_entry,
    validate_clip_payload,
)
from atlas3r.teachers.external.cache_writer import (
    ExternalSignalPayload,
    write_external_teacher_signal_cache,
)
from atlas3r.training.teacher_signal_temporal_artifacts import (
    load_teacher_signal_temporal_checkpoint,
)
from atlas3r.training.torch_runtime import require_torch


@dataclass(frozen=True)
class StudentTemporalTeacherRunConfig:
    checkpoint: Path
    clip_cache: Path
    output: Path
    device: str = "auto"
    max_clips: int | None = None


def run_student_temporal_teacher_signal_cache(
    config: StudentTemporalTeacherRunConfig,
) -> dict[str, object]:
    """Run a temporal-v1 checkpoint and write a pseudo-label teacher-signal cache."""

    _validate_config(config)
    torch = require_torch()
    loaded = load_teacher_signal_temporal_checkpoint(config.checkpoint, device=config.device)
    model = loaded["model"]
    resolved_device = str(loaded["device"])
    checkpoint = loaded["checkpoint"]
    clip_manifest_path = manifest_path_from_input(config.clip_cache)
    clip_manifest = load_clip_cache_manifest(clip_manifest_path)
    clip_entries = _entries(clip_manifest, "clips")
    selected = clip_entries[: config.max_clips or len(clip_entries)]
    if not selected:
        raise ValueError(f"{clip_manifest_path}: no clips selected for student export")
    payloads: list[ExternalSignalPayload] = []
    clip_length = _int_field(clip_manifest, "clip_length")
    height = _int_field(clip_manifest, "image_height")
    width = _int_field(clip_manifest, "image_width")
    model.eval()
    predictions_by_frame_id: dict[
        int,
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ] = {}
    with torch.no_grad():
        for clip_entry in selected:
            clip_payload = read_clip_payload_from_entry(clip_manifest_path.parent, clip_entry)
            validate_clip_payload(
                clip_payload,
                clip_length=clip_length,
                height=height,
                width=width,
            )
            prediction = _predict_clip(model, clip_payload, resolved_device)
            _reuse_duplicate_frame_predictions(prediction, clip_payload, predictions_by_frame_id)
            confidence = prediction["confidence"]
            valid_mask = (
                np.isfinite(prediction["depth_m"])
                & np.isfinite(prediction["depth_sigma_m"])
                & np.isfinite(confidence)
                & (prediction["depth_m"] > 0.0)
                & (prediction["depth_sigma_m"] > 0.0)
                & (confidence > 0.0)
            )
            payloads.append(
                ExternalSignalPayload(
                    source_clip_id=_int_field(clip_entry, "clip_id"),
                    arrays={
                        "depth_m": prediction["depth_m"],
                        "depth_sigma_m": prediction["depth_sigma_m"],
                        "confidence": confidence,
                        "valid_mask": valid_mask.astype(np.bool_),
                        "K": np.asarray(clip_payload["K"], dtype=np.float32),
                        "T_world_camera": np.asarray(
                            clip_payload["T_world_camera"],
                            dtype=np.float32,
                        ),
                        "frame_ids": np.asarray(clip_payload["frame_ids"], dtype=np.int32),
                        "timestamps_s": np.asarray(clip_payload["timestamps_s"], dtype=np.float64),
                    },
                )
            )
    result = write_external_teacher_signal_cache(
        clip_cache=clip_manifest_path,
        output=config.output,
        payloads=payloads,
        teacher_name="atlas3r_temporal_v1_student",
        teacher_version=f"checkpoint-step-{int(checkpoint['step'])}",
        teacher_source_type="atlas3r_student_checkpoint",
        source_metadata={
            "checkpoint": str(config.checkpoint),
            "checkpoint_format_name": str(checkpoint["format_name"]),
            "checkpoint_step": int(checkpoint["step"]),
            "model_config": dict(checkpoint["model_config"]),
            "loss_config": dict(checkpoint["loss_config"]),
            "truth_boundary": dict(checkpoint["truth_boundary"]),
            "pose_source": "source_clip_cache_T_world_camera",
            "pose_source_type": "source_clip_cache_pose_not_external_teacher",
            "pose_confidence": 1.0,
            "pose_note": (
                "Predicted relative SE(3) is diagnostic only for runtime; exported "
                "teacher-signal T_world_camera comes from the source clip cache."
            ),
        },
    )
    return {
        "format_name": "atlas3r_student_temporal_teacher_signal_run",
        "checkpoint": str(config.checkpoint),
        "clip_cache": str(clip_manifest_path),
        "output": str(config.output),
        "device": resolved_device,
        "max_clips": config.max_clips,
        "signal_count": result["signal_count"],
        "teacher_name": result["teacher_name"],
        "truth_boundary": result["truth_boundary"],
    }


def _predict_clip(
    model: Any,
    clip_payload: dict[str, Any],
    device: str,
) -> dict[str, npt.NDArray[np.float32]]:
    torch = require_torch()
    images = np.asarray(clip_payload["images_rgb_u8"], dtype=np.float32) / np.float32(255.0)
    images = np.transpose(images, (0, 3, 1, 2))[np.newaxis, ...].copy()
    intrinsics = np.asarray(clip_payload["K"], dtype=np.float32)[np.newaxis, ...].copy()
    batch_images = torch.from_numpy(images).to(device)
    batch_intrinsics = torch.from_numpy(intrinsics).to(device)
    prediction = model(batch_images, batch_intrinsics)
    return {
        "depth_m": prediction["depth_m"][0, :, 0].detach().cpu().numpy().astype(np.float32),
        "depth_sigma_m": prediction["depth_sigma_m"][0, :, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "confidence": prediction["confidence"][0, :, 0].detach().cpu().numpy().astype(np.float32),
    }


def _reuse_duplicate_frame_predictions(
    prediction: dict[str, npt.NDArray[np.float32]],
    clip_payload: dict[str, Any],
    predictions_by_frame_id: dict[
        int,
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ],
) -> None:
    frame_ids = np.asarray(clip_payload["frame_ids"], dtype=np.int64)
    for frame_offset, frame_id_value in enumerate(frame_ids.tolist()):
        frame_id = int(frame_id_value)
        cached = predictions_by_frame_id.get(frame_id)
        if cached is None:
            predictions_by_frame_id[frame_id] = (
                prediction["depth_m"][frame_offset].copy(),
                prediction["depth_sigma_m"][frame_offset].copy(),
                prediction["confidence"][frame_offset].copy(),
            )
            continue
        prediction["depth_m"][frame_offset] = cached[0]
        prediction["depth_sigma_m"][frame_offset] = cached[1]
        prediction["confidence"][frame_offset] = cached[2]


def _validate_config(config: StudentTemporalTeacherRunConfig) -> None:
    if config.max_clips is not None and config.max_clips <= 0:
        raise ValueError("max_clips: must be positive when provided")


def _entries(manifest: dict[str, object], key: str) -> list[dict[str, object]]:
    value = manifest.get(key)
    if not isinstance(value, list):
        raise ValueError(f"manifest.{key}: must be a list")
    entries: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"manifest.{key}[{index}]: must be a mapping")
        entries.append(dict(item))
    return entries


def _int_field(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key}: must be an integer")
    return value


__all__ = [
    "StudentTemporalTeacherRunConfig",
    "run_student_temporal_teacher_signal_cache",
]
