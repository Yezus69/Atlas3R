"""Artifact writers for TUM RGB-D debug training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.torch_runtime import require_torch
from atlas3r.training.tum_rgbd_dataset import TumRgbdDepthDataset


def write_tum_rgbd_prediction_sample_and_preview(
    output: Path,
    *,
    model: Any,
    dataset: TumRgbdDepthDataset,
    device: str,
    metrics: dict[str, float],
    truth_boundary: dict[str, object],
) -> tuple[Path, Path]:
    torch = require_torch()
    sample = dataset[0]
    batch = {
        "images_rgb": sample["images_rgb"].unsqueeze(0).to(device),
        "intrinsics": sample["intrinsics"].unsqueeze(0).to(device),
        "target": {
            key: value.unsqueeze(0).to(device)
            for key, value in cast(dict[str, Any], sample["target"]).items()
            if torch.is_tensor(value)
        },
    }
    model.eval()
    with torch.no_grad():
        prediction = model(batch["images_rgb"], batch["intrinsics"])
    rgb_u8 = (
        (sample["images_rgb"].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    target_depth = sample["target"]["depth_m"][0].detach().cpu().numpy().astype(np.float32)
    valid_mask = sample["target"]["valid_depth_mask"][0].detach().cpu().numpy().astype(np.bool_)
    predicted_depth = prediction["depth_m"][0, 0].detach().cpu().numpy().astype(np.float32)
    predicted_sigma = prediction["depth_sigma_m"][0, 0].detach().cpu().numpy().astype(np.float32)
    predicted_confidence = prediction["confidence"][0, 0].detach().cpu().numpy().astype(np.float32)
    abs_error = np.abs(predicted_depth - target_depth).astype(np.float32)
    np.savez(
        output / "prediction_sample.npz",
        rgb_u8=rgb_u8,
        target_depth_m=target_depth,
        valid_depth_mask=valid_mask,
        predicted_depth_m=predicted_depth,
        abs_depth_error_m=abs_error,
        predicted_depth_sigma_m=predicted_sigma,
        predicted_confidence=predicted_confidence,
        K=sample["intrinsics"].detach().cpu().numpy().astype(np.float32),
        T_world_camera=sample["target"]["T_world_camera"].detach().cpu().numpy().astype(np.float32),
    )
    return write_prediction_preview(
        output_dir=output,
        rgb_u8=rgb_u8,
        target_depth_m=target_depth,
        predicted_depth_m=predicted_depth,
        abs_error_m=abs_error,
        valid_depth_mask=valid_mask,
        metrics=metrics,
        truth_boundary=truth_boundary,
        title="Atlas3R TUM RGB-D Training Preview",
    )


def write_tum_rgbd_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    step: int,
    config_record: dict[str, object],
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    truth_boundary: dict[str, object],
) -> None:
    torch = require_torch()
    model_config = model.model_config()
    checkpoint = {
        "format_name": "atlas3r_tiny_depth_pose_checkpoint",
        "format_version": 1,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_record,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "model_config": {
            **model_config,
            "model_role": "tum_rgbd_real_capture_debug_training_mvp",
            "truth_boundary": truth_boundary,
        },
        "truth_boundary": truth_boundary,
    }
    torch.save(checkpoint, path)


def write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


__all__ = [
    "append_jsonl",
    "write_json",
    "write_tum_rgbd_checkpoint",
    "write_tum_rgbd_prediction_sample_and_preview",
]
