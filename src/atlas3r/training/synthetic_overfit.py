"""Synthetic-overfit training runner for the Phase 4A MVP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.synthetic_depth_dataset import (
    SyntheticDepthSample,
    generate_synthetic_depth_samples,
)
from atlas3r.training.torch_runtime import require_torch, select_device


@dataclass(frozen=True)
class SyntheticOverfitConfig:
    output: Path
    steps: int = 200
    batch_size: int = 8
    num_samples: int = 64
    width: int = 64
    height: int = 48
    seed: int = 0
    device: str = "auto"
    learning_rate: float = 0.001
    log_every: int = 10

    def to_json_dict(self, *, resolved_device: str) -> dict[str, object]:
        return {
            "output": str(self.output),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "num_samples": self.num_samples,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "requested_device": self.device,
            "resolved_device": resolved_device,
            "learning_rate": self.learning_rate,
            "log_every": self.log_every,
        }


def training_mvp_truth_boundary() -> dict[str, object]:
    return {
        "training_mvp": True,
        "synthetic_only": True,
        "real_capture_model": False,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "learned_inference": True,
        "generalizes_to_real_world": False,
    }


def run_synthetic_overfit(config: SyntheticOverfitConfig) -> dict[str, object]:
    """Run the synthetic training MVP and write all Phase 4A artifacts."""

    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    from atlas3r.training.losses import synthetic_depth_pose_loss
    from atlas3r.training.tiny_depth_pose_model import TinyDepthPoseNet, model_config

    config.output.mkdir(parents=True, exist_ok=True)
    config_record = config.to_json_dict(resolved_device=resolved_device)
    truth_boundary = training_mvp_truth_boundary()
    _write_json(config.output / "config.json", config_record)
    _seed_torch(torch, config.seed)
    samples = generate_synthetic_depth_samples(
        count=config.num_samples,
        width=config.width,
        height=config.height,
        seed=config.seed,
    )
    model = TinyDepthPoseNet().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics_path = config.output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    rng = np.random.default_rng(config.seed)
    best_depth_mae = float("inf")
    final_metrics: dict[str, float] = {}
    for step in range(1, config.steps + 1):
        indices = rng.integers(0, len(samples), size=config.batch_size)
        batch = _samples_to_tensors(samples, indices.tolist(), torch, resolved_device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["images_rgb"], batch["intrinsics"])
        loss, metrics = synthetic_depth_pose_loss(prediction, batch["target"])
        loss.backward()
        optimizer.step()
        final_metrics = metrics
        best_depth_mae = min(best_depth_mae, metrics["depth_mae_m"])
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            _append_jsonl(
                metrics_path,
                {
                    "format_name": "atlas3r_synthetic_overfit_metric",
                    "step": step,
                    "device": resolved_device,
                    **metrics,
                },
            )

    sample_arrays = _predict_one_sample(samples[0], model, torch, resolved_device)
    sample_npz = config.output / "prediction_sample.npz"
    np.savez(sample_npz, **sample_arrays)
    abs_error = np.abs(sample_arrays["predicted_depth_m"] - sample_arrays["target_depth_m"])
    html_path, svg_path = write_prediction_preview(
        output_dir=config.output,
        rgb_u8=samples[0].rgb_u8,
        target_depth_m=sample_arrays["target_depth_m"],
        predicted_depth_m=sample_arrays["predicted_depth_m"],
        abs_error_m=abs_error.astype(np.float32),
        metrics=final_metrics,
        truth_boundary=truth_boundary,
    )
    checkpoint_path = config.output / "checkpoint_last.pt"
    checkpoint = {
        "format_name": "atlas3r_tiny_depth_pose_checkpoint",
        "format_version": 1,
        "step": config.steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_record,
        "model_config": model_config(),
        "metrics": final_metrics,
        "truth_boundary": truth_boundary,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "format_name": "atlas3r_synthetic_overfit_train_run_summary",
        "format_version": 1,
        "config": config_record,
        "final_metrics": final_metrics,
        "best_depth_mae_m": best_depth_mae,
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "metrics": metrics_path.name,
            "prediction_sample": sample_npz.name,
            "preview_html": html_path.name,
            "preview_svg": svg_path.name,
        },
        "truth_boundary": truth_boundary,
    }
    _write_json(config.output / "summary.json", summary)
    return {
        "format_name": "atlas3r_synthetic_overfit_train_run",
        "output": str(config.output),
        "device": resolved_device,
        "steps": config.steps,
        "final_loss": final_metrics.get("loss_total", float("nan")),
        "best_depth_mae_m": best_depth_mae,
        "checkpoint": checkpoint_path.name,
        "preview": html_path.name,
        "truth_boundary": truth_boundary,
    }


def _samples_to_tensors(
    samples: tuple[SyntheticDepthSample, ...],
    indices: list[int],
    torch: Any,
    device: str,
) -> dict[str, Any]:
    selected = [samples[index] for index in indices]
    images = np.stack([sample.rgb_model for sample in selected], axis=0).astype(np.float32)
    depth = np.stack([sample.depth_m for sample in selected], axis=0)[:, np.newaxis, :, :]
    sigma = np.stack([sample.depth_sigma_m for sample in selected], axis=0)[:, np.newaxis, :, :]
    confidence = np.stack([sample.confidence for sample in selected], axis=0)[:, np.newaxis, :, :]
    centers = np.stack([sample.camera_center_world_m for sample in selected], axis=0)
    intrinsics = np.stack([sample.K for sample in selected], axis=0)
    return {
        "images_rgb": torch.from_numpy(images).to(device),
        "intrinsics": torch.from_numpy(intrinsics.astype(np.float32)).to(device),
        "target": {
            "depth_m": torch.from_numpy(depth.astype(np.float32)).to(device),
            "depth_sigma_m": torch.from_numpy(sigma.astype(np.float32)).to(device),
            "confidence": torch.from_numpy(confidence.astype(np.float32)).to(device),
            "camera_center_world_m": torch.from_numpy(centers.astype(np.float32)).to(device),
        },
    }


def _predict_one_sample(
    sample: SyntheticDepthSample, model: Any, torch: Any, device: str
) -> dict[str, Any]:
    batch = _samples_to_tensors((sample,), [0], torch, device)
    model.eval()
    with torch.no_grad():
        prediction = model(batch["images_rgb"], batch["intrinsics"])
    predicted_depth = prediction["depth_m"][0, 0].detach().cpu().numpy().astype(np.float32)
    predicted_sigma = prediction["depth_sigma_m"][0, 0].detach().cpu().numpy().astype(np.float32)
    predicted_confidence = prediction["confidence"][0, 0].detach().cpu().numpy().astype(np.float32)
    predicted_center = (
        prediction["camera_center_world_m"][0].detach().cpu().numpy().astype(np.float32)
    )
    return {
        "rgb_u8": sample.rgb_u8,
        "target_depth_m": sample.depth_m,
        "predicted_depth_m": predicted_depth,
        "abs_depth_error_m": np.abs(predicted_depth - sample.depth_m).astype(np.float32),
        "target_depth_sigma_m": sample.depth_sigma_m,
        "predicted_depth_sigma_m": predicted_sigma,
        "target_confidence": sample.confidence,
        "predicted_confidence": predicted_confidence,
        "object_mask": sample.object_mask,
        "K": sample.K,
        "T_world_camera": sample.T_world_camera,
        "target_camera_center_world_m": sample.camera_center_world_m,
        "predicted_camera_center_world_m": predicted_center,
    }


def _validate_config(config: SyntheticOverfitConfig) -> None:
    for field_name in ("steps", "batch_size", "num_samples", "width", "height", "log_every"):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate: must be positive")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


__all__ = [
    "SyntheticOverfitConfig",
    "run_synthetic_overfit",
    "training_mvp_truth_boundary",
]
