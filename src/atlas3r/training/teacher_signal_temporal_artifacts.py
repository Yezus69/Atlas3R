"""Artifacts for Phase 5D teacher-signal temporal training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.teacher_signal_dataset import TeacherSignalTemporalDataset
from atlas3r.training.teacher_signal_losses import TeacherSignalLossConfig
from atlas3r.training.torch_runtime import require_torch, select_device


def load_teacher_signal_temporal_checkpoint(path: str | Path, *, device: str = "auto") -> Any:
    """Load a Phase 5D temporal checkpoint and instantiate its model."""

    torch = require_torch()
    resolved_device = select_device(device)
    payload = torch.load(path, map_location=resolved_device)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint must be a mapping")
    if payload.get("format_name") != "atlas3r_teacher_signal_temporal_checkpoint":
        raise ValueError(f"{path}: unsupported temporal teacher-signal checkpoint")
    model_config = payload.get("model_config")
    if (
        not isinstance(model_config, dict)
        or model_config.get("model_name") != "TemporalMetricNetV1"
    ):
        raise ValueError(f"{path}: checkpoint model must be TemporalMetricNetV1")
    truth_boundary = payload.get("truth_boundary")
    if not isinstance(truth_boundary, dict) or truth_boundary.get("mapping") is not False:
        raise ValueError(f"{path}: checkpoint truth boundary is not a Phase 5D debug model")
    from atlas3r.training.tiny_temporal_geometry_model import TemporalMetricNetV1

    model = TemporalMetricNetV1(
        hidden_channels=int(model_config["hidden_channels"]),
        bottleneck_channels=int(model_config["bottleneck_channels"]),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "checkpoint": payload,
        "device": resolved_device,
    }


def write_teacher_signal_prediction_sample_and_preview(
    output: Path,
    *,
    model: Any,
    dataset: TeacherSignalTemporalDataset,
    device: str,
    metrics: dict[str, float],
    truth_boundary: dict[str, object],
) -> tuple[Path, Path]:
    """Write a compact center-frame NPZ plus HTML/SVG preview."""

    torch = require_torch()
    sample = dataset[0]
    batch = _move_batch_to_device(
        {
            "images_rgb": sample["images_rgb"].unsqueeze(0),
            "intrinsics": sample["intrinsics"].unsqueeze(0),
            "target": {
                key: value.unsqueeze(0)
                for key, value in sample["target"].items()
                if torch.is_tensor(value)
            },
        },
        device,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(batch["images_rgb"], batch["intrinsics"])
    center_index = int(sample["images_rgb"].shape[0] // 2)
    rgb_u8 = (
        (sample["images_rgb"][center_index].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    target_depth = sample["target"]["depth_m"][center_index, 0].detach().cpu().numpy()
    valid_mask = sample["target"]["valid_mask"][center_index, 0].detach().cpu().numpy()
    predicted_depth = prediction["depth_m"][0, center_index, 0].detach().cpu().numpy()
    abs_error = np.abs(predicted_depth - target_depth).astype(np.float32)
    np.savez(
        output / "prediction_sample.npz",
        rgb_u8=rgb_u8,
        target_depth_m=target_depth.astype(np.float32),
        valid_depth_mask=valid_mask.astype(np.bool_),
        predicted_depth_m=predicted_depth.astype(np.float32),
        abs_depth_error_m=abs_error,
        predicted_depth_sigma_m=prediction["depth_sigma_m"][0, center_index, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        predicted_confidence=prediction["confidence"][0, center_index, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        frame_ids=sample["frame_ids"].detach().cpu().numpy().astype(np.int64),
    )
    return write_prediction_preview(
        output_dir=output,
        rgb_u8=rgb_u8,
        target_depth_m=target_depth.astype(np.float32),
        predicted_depth_m=predicted_depth.astype(np.float32),
        abs_error_m=abs_error,
        valid_depth_mask=valid_mask.astype(np.bool_),
        metrics=metrics,
        truth_boundary=truth_boundary,
        title="Atlas3R Teacher-Signal Temporal Preview",
    )


def write_teacher_signal_temporal_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    config_record: dict[str, object],
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    truth_boundary: dict[str, object],
    loss_config: TeacherSignalLossConfig,
) -> None:
    """Write a validated Phase 5D temporal training checkpoint payload."""

    torch = require_torch()
    torch.save(
        {
            "format_name": "atlas3r_teacher_signal_temporal_checkpoint",
            "format_version": 1,
            "step": step,
            "model_name": "temporal-v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_record,
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "model_config": model.model_config(),
            "loss_config": loss_config.to_json(),
            "teacher_caches": list(cast(list[str], config_record["teacher_caches"])),
            "val_teacher_caches": list(cast(list[str], config_record["val_teacher_caches"])),
            "truth_boundary": truth_boundary,
        },
        path,
    )


def teacher_signal_temporal_config_record(
    config: Any,
    *,
    train_dataset: TeacherSignalTemporalDataset,
    val_dataset: TeacherSignalTemporalDataset,
    split_mode: str,
    resolved_device: str,
    amp_enabled: bool,
) -> dict[str, object]:
    """Return deterministic JSON config metadata for a Phase 5D run."""

    return {
        "teacher_caches": [str(path) for path in config.teacher_caches],
        "val_teacher_caches": [str(path) for path in config.val_teacher_caches],
        "output": str(config.output),
        "model": config.model,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "num_workers": config.num_workers,
        "learning_rate": config.learning_rate,
        "log_every": config.log_every,
        "val_every": config.val_every,
        "checkpoint_every": config.checkpoint_every,
        "preview_every": config.preview_every,
        "seed": config.seed,
        "amp_requested": config.amp,
        "amp_enabled": amp_enabled,
        "max_runtime_minutes": config.max_runtime_minutes,
        "hidden_channels": config.hidden_channels,
        "bottleneck_channels": config.bottleneck_channels,
        "loss_config": config.loss_config.to_json(),
        "validation_split_mode": split_mode,
        "train_dataset": train_dataset.cache_summary(),
        "validation_dataset": val_dataset.cache_summary(),
    }


def _move_batch_to_device(value: Any, device: str) -> Any:
    torch = require_torch()
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    return value


__all__ = [
    "load_teacher_signal_temporal_checkpoint",
    "teacher_signal_temporal_config_record",
    "write_teacher_signal_prediction_sample_and_preview",
    "write_teacher_signal_temporal_checkpoint",
]
