"""Real TUM RGB-D debug training runner for Phase 4C."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_rgbd_artifacts import (
    append_jsonl,
    write_json,
    write_tum_rgbd_checkpoint,
    write_tum_rgbd_prediction_sample_and_preview,
)
from atlas3r.training.tum_rgbd_dataset import TumRgbdDepthDataset


@dataclass(frozen=True)
class TumRgbdTrainConfig:
    manifest: Path
    output: Path
    steps: int = 20000
    batch_size: int = 16
    width: int = 160
    height: int = 120
    device: str = "cuda"
    num_workers: int = 4
    learning_rate: float = 0.0005
    log_every: int = 50
    val_every: int = 500
    checkpoint_every: int = 1000
    preview_every: int = 1000
    seed: int = 0
    amp: bool = False
    max_runtime_minutes: float | None = None
    model: str = "tiny-v1"
    depth_loss: str = "metric_l1"
    hidden_channels: int = 32
    min_valid_depth_pixels: int = 1

    def to_json_dict(
        self,
        *,
        resolved_device: str,
        amp_enabled: bool,
        cuda_device_name: str | None,
    ) -> dict[str, object]:
        return {
            "manifest": str(self.manifest),
            "output": str(self.output),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "width": self.width,
            "height": self.height,
            "requested_device": self.device,
            "resolved_device": resolved_device,
            "num_workers": self.num_workers,
            "learning_rate": self.learning_rate,
            "log_every": self.log_every,
            "val_every": self.val_every,
            "checkpoint_every": self.checkpoint_every,
            "preview_every": self.preview_every,
            "seed": self.seed,
            "amp_requested": self.amp,
            "amp_enabled": amp_enabled,
            "cuda_device_name": cuda_device_name,
            "max_runtime_minutes": self.max_runtime_minutes,
            "model": self.model,
            "depth_loss": self.depth_loss,
            "hidden_channels": self.hidden_channels,
            "min_valid_depth_pixels": self.min_valid_depth_pixels,
        }


def real_rgbd_truth_boundary() -> dict[str, object]:
    return {
        "training_mvp": True,
        "trained_on_real_rgbd": True,
        "dataset": "TUM RGB-D freiburg1_xyz",
        "synthetic_only": False,
        "real_capture_debug_model": True,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "learned_inference": True,
        "generalizes_to_real_world": False,
    }


def run_tum_rgbd_depth_pose_training(config: TumRgbdTrainConfig) -> dict[str, object]:
    """Train the tiny debug model on a TUM RGB-D manifest and write run artifacts."""

    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    amp_enabled = bool(config.amp and resolved_device == "cuda")
    cuda_device_name = _cuda_device_name(torch, resolved_device)
    config.output.mkdir(parents=True, exist_ok=True)
    config_record = config.to_json_dict(
        resolved_device=resolved_device,
        amp_enabled=amp_enabled,
        cuda_device_name=cuda_device_name,
    )
    truth_boundary = real_rgbd_truth_boundary()
    write_json(config.output / "config.json", config_record)
    _seed_torch(torch, config.seed)

    train_dataset = TumRgbdDepthDataset(
        config.manifest,
        width=config.width,
        height=config.height,
        split="train",
        min_valid_depth_pixels=config.min_valid_depth_pixels,
    )
    val_dataset = TumRgbdDepthDataset(
        config.manifest,
        width=config.width,
        height=config.height,
        split="val",
        min_valid_depth_pixels=config.min_valid_depth_pixels,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    train_iterator = iter(train_loader)

    from atlas3r.training.losses import masked_rgbd_depth_pose_loss
    from atlas3r.training.tiny_depth_pose_model import build_tiny_depth_pose_model

    model = build_tiny_depth_pose_model(config.model, hidden_channels=config.hidden_channels).to(
        resolved_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    metrics_path = config.output / "metrics.jsonl"
    validation_path = config.output / "validation_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    validation_path.write_text("", encoding="utf-8")
    final_train_metrics: dict[str, float] = {}
    best_val_metrics: dict[str, float] | None = None
    best_val_rmse = float("inf")
    stopped_reason = "completed_steps"
    start_time = time.monotonic()
    completed_steps = 0

    for step in range(1, config.steps + 1):
        completed_steps = step
        model.train()
        batch, train_iterator = _next_training_batch(train_loader, train_iterator)
        batch = _move_batch_to_device(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            prediction = model(batch["images_rgb"], batch["intrinsics"])
            loss, metrics = masked_rgbd_depth_pose_loss(
                prediction,
                batch["target"],
                depth_loss=config.depth_loss,
            )
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        final_train_metrics = metrics
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            append_jsonl(
                metrics_path,
                {
                    "format_name": "atlas3r_tum_rgbd_train_metric",
                    "step": step,
                    "device": resolved_device,
                    **metrics,
                },
            )
        if step == 1 or step % config.val_every == 0 or step == config.steps:
            val_metrics = _validate(
                model, val_loader, resolved_device, depth_loss=config.depth_loss
            )
            append_jsonl(
                validation_path,
                {
                    "format_name": "atlas3r_tum_rgbd_validation_metric",
                    "step": step,
                    "device": resolved_device,
                    **val_metrics,
                },
            )
            if val_metrics["depth_rmse_m"] < best_val_rmse:
                best_val_rmse = val_metrics["depth_rmse_m"]
                best_val_metrics = val_metrics
                write_tum_rgbd_checkpoint(
                    config.output / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config_record=config_record,
                    metrics=metrics,
                    validation_metrics=val_metrics,
                    truth_boundary=truth_boundary,
                )
        if step % config.checkpoint_every == 0 or step == config.steps:
            write_tum_rgbd_checkpoint(
                config.output / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config_record=config_record,
                metrics=metrics,
                validation_metrics=best_val_metrics or {},
                truth_boundary=truth_boundary,
            )
        if step % config.preview_every == 0:
            write_tum_rgbd_prediction_sample_and_preview(
                config.output,
                model=model,
                dataset=val_dataset,
                device=resolved_device,
                metrics=best_val_metrics or final_train_metrics,
                truth_boundary=truth_boundary,
            )
        if _runtime_expired(config, start_time):
            stopped_reason = "max_runtime_minutes"
            break

    if best_val_metrics is None:
        best_val_metrics = _validate(
            model,
            val_loader,
            resolved_device,
            depth_loss=config.depth_loss,
        )
        best_val_rmse = best_val_metrics["depth_rmse_m"]
        append_jsonl(
            validation_path,
            {
                "format_name": "atlas3r_tum_rgbd_validation_metric",
                "step": completed_steps,
                "device": resolved_device,
                **best_val_metrics,
            },
        )
    write_tum_rgbd_checkpoint(
        config.output / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        step=completed_steps,
        config_record=config_record,
        metrics=final_train_metrics,
        validation_metrics=best_val_metrics,
        truth_boundary=truth_boundary,
    )
    if not (config.output / "checkpoint_best.pt").is_file():
        write_tum_rgbd_checkpoint(
            config.output / "checkpoint_best.pt",
            model=model,
            optimizer=optimizer,
            step=completed_steps,
            config_record=config_record,
            metrics=final_train_metrics,
            validation_metrics=best_val_metrics,
            truth_boundary=truth_boundary,
        )
    html_path, svg_path = write_tum_rgbd_prediction_sample_and_preview(
        config.output,
        model=model,
        dataset=val_dataset,
        device=resolved_device,
        metrics=best_val_metrics,
        truth_boundary=truth_boundary,
    )
    summary = {
        "format_name": "atlas3r_tum_rgbd_train_run_summary",
        "format_version": 1,
        "dataset_name": val_dataset.manifest["dataset_name"],
        "sequence_name": val_dataset.manifest["sequence_name"],
        "manifest_path": str(config.manifest),
        "train_frame_count": len(train_dataset),
        "val_frame_count": len(val_dataset),
        "best_validation": {
            "depth_rmse_m": best_val_metrics["depth_rmse_m"],
            "depth_mae_m": best_val_metrics["depth_mae_m"],
            "depth_absrel": best_val_metrics["depth_absrel"],
        },
        "final_train_losses": final_train_metrics,
        "artifacts": {
            "config": "config.json",
            "metrics": "metrics.jsonl",
            "validation_metrics": "validation_metrics.jsonl",
            "checkpoint_last": "checkpoint_last.pt",
            "checkpoint_best": "checkpoint_best.pt",
            "prediction_sample": "prediction_sample.npz",
            "preview_html": html_path.name,
            "preview_svg": svg_path.name,
        },
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "cuda_device_name": cuda_device_name,
        "stopped_reason": stopped_reason,
        "steps_completed": completed_steps,
        "truth_boundary": truth_boundary,
    }
    write_json(config.output / "summary.json", summary)
    return {
        "format_name": "atlas3r_tum_rgbd_train_run",
        "output": str(config.output),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "train_frame_count": len(train_dataset),
        "val_frame_count": len(val_dataset),
        "steps_completed": completed_steps,
        "best_depth_rmse_m": best_val_metrics["depth_rmse_m"],
        "best_depth_mae_m": best_val_metrics["depth_mae_m"],
        "best_depth_absrel": best_val_metrics["depth_absrel"],
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_best": "checkpoint_best.pt",
        "truth_boundary": truth_boundary,
    }


def _validate(model: Any, val_loader: Any, device: str, *, depth_loss: str) -> dict[str, float]:
    torch = require_torch()
    from atlas3r.training.losses import masked_rgbd_depth_pose_loss

    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["intrinsics"])
            _loss, metrics = masked_rgbd_depth_pose_loss(
                prediction,
                batch["target"],
                depth_loss=depth_loss,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    if batches <= 0:
        raise ValueError("validation: no batches were produced")
    return {key: value / float(batches) for key, value in totals.items()}


def _next_training_batch(loader: Any, iterator: Any) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _move_batch_to_device(value: Any, device: str) -> Any:
    torch = require_torch()
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    return value


def _runtime_expired(config: TumRgbdTrainConfig, start_time: float) -> bool:
    if config.max_runtime_minutes is None:
        return False
    return (time.monotonic() - start_time) >= config.max_runtime_minutes * 60.0


def _cuda_device_name(torch: Any, device: str) -> str | None:
    if device != "cuda" or not bool(torch.cuda.is_available()):
        return None
    return str(torch.cuda.get_device_name(0))


def _validate_config(config: TumRgbdTrainConfig) -> None:
    for field_name in (
        "steps",
        "batch_size",
        "width",
        "height",
        "log_every",
        "val_every",
        "checkpoint_every",
        "preview_every",
        "hidden_channels",
    ):
        value = getattr(config, field_name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name}: must be a positive integer")
    if config.num_workers < 0:
        raise ValueError("num_workers: must be non-negative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate: must be positive")
    if config.max_runtime_minutes is not None and config.max_runtime_minutes <= 0.0:
        raise ValueError("max_runtime_minutes: must be positive when provided")
    if config.model not in {"tiny-v1", "tiny-v2"}:
        raise ValueError("model: must be 'tiny-v1' or 'tiny-v2'")
    if config.depth_loss not in {"metric_l1", "log_l1"}:
        raise ValueError("depth_loss: must be 'metric_l1' or 'log_l1'")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "TumRgbdTrainConfig",
    "real_rgbd_truth_boundary",
    "run_tum_rgbd_depth_pose_training",
]
