"""Tiny temporal TUM RGB-D training runner for Phase 5A."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.training.preview import write_prediction_preview
from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_clip_dataset import TumRgbdClipCacheDataset
from atlas3r.training.tum_rgbd_artifacts import append_jsonl, write_json


@dataclass(frozen=True)
class TumRgbdTemporalTrainConfig:
    clip_cache: Path
    output: Path
    val_clip_cache: Path | None = None
    steps: int = 12000
    batch_size: int = 8
    device: str = "cuda"
    num_workers: int = 4
    learning_rate: float = 0.0003
    log_every: int = 50
    val_every: int = 500
    checkpoint_every: int = 1000
    preview_every: int = 1000
    seed: int = 0
    amp: bool = False
    max_runtime_minutes: float | None = 330.0
    hidden_channels: int = 32
    depth_loss: str = "metric_l1"


def run_tum_rgbd_temporal_training(config: TumRgbdTemporalTrainConfig) -> dict[str, object]:
    """Train `TinyTemporalMetricNetV0` from a forged clip cache."""

    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    amp_enabled = bool(config.amp and resolved_device == "cuda")
    config.output.mkdir(parents=True, exist_ok=True)
    _seed_torch(torch, config.seed)
    train_dataset = TumRgbdClipCacheDataset(config.clip_cache)
    val_dataset = TumRgbdClipCacheDataset(config.val_clip_cache or config.clip_cache)
    config_record = _config_record(config, resolved_device=resolved_device, amp_enabled=amp_enabled)
    write_json(config.output / "config.json", config_record)

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
    from atlas3r.training.temporal_losses import temporal_geometry_loss
    from atlas3r.training.tiny_temporal_geometry_model import (
        TinyTemporalMetricNetV0,
        temporal_v0_truth_boundary,
    )

    model = TinyTemporalMetricNetV0(hidden_channels=config.hidden_channels).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    truth_boundary = temporal_v0_truth_boundary()
    metrics_path = config.output / "metrics.jsonl"
    validation_path = config.output / "validation_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    validation_path.write_text("", encoding="utf-8")
    best_val_metrics: dict[str, float] | None = None
    best_val_rmse = float("inf")
    final_train_metrics: dict[str, float] = {}
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
            loss, metrics = temporal_geometry_loss(
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
                {"format_name": "atlas3r_temporal_train_metric", "step": step, **metrics},
            )
        if step == 1 or step % config.val_every == 0 or step == config.steps:
            val_metrics = _validate(
                model,
                val_loader,
                resolved_device,
                depth_loss=config.depth_loss,
            )
            append_jsonl(
                validation_path,
                {
                    "format_name": "atlas3r_temporal_validation_metric",
                    "step": step,
                    **val_metrics,
                },
            )
            if val_metrics["center_depth_rmse_m"] < best_val_rmse:
                best_val_rmse = val_metrics["center_depth_rmse_m"]
                best_val_metrics = val_metrics
                _write_checkpoint(
                    config.output / "checkpoint_best.pt",
                    model,
                    optimizer,
                    step,
                    config_record,
                    metrics,
                    val_metrics,
                    truth_boundary,
                )
        if step % config.checkpoint_every == 0 or step == config.steps:
            _write_checkpoint(
                config.output / "checkpoint_last.pt",
                model,
                optimizer,
                step,
                config_record,
                metrics,
                best_val_metrics or {},
                truth_boundary,
            )
        if step % config.preview_every == 0:
            _write_prediction_sample_and_preview(
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
    _write_checkpoint(
        config.output / "checkpoint_last.pt",
        model,
        optimizer,
        completed_steps,
        config_record,
        final_train_metrics,
        best_val_metrics,
        truth_boundary,
    )
    if not (config.output / "checkpoint_best.pt").is_file():
        _write_checkpoint(
            config.output / "checkpoint_best.pt",
            model,
            optimizer,
            completed_steps,
            config_record,
            final_train_metrics,
            best_val_metrics,
            truth_boundary,
        )
    html_path, svg_path = _write_prediction_sample_and_preview(
        config.output,
        model=model,
        dataset=val_dataset,
        device=resolved_device,
        metrics=best_val_metrics,
        truth_boundary=truth_boundary,
    )
    summary = {
        "format_name": "atlas3r_tum_rgbd_temporal_train_run_summary",
        "format_version": 1,
        "clip_cache": str(config.clip_cache),
        "val_clip_cache": str(config.val_clip_cache or config.clip_cache),
        "train_clip_count": len(train_dataset),
        "val_clip_count": len(val_dataset),
        "best_validation": best_val_metrics,
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
        "stopped_reason": stopped_reason,
        "steps_completed": completed_steps,
        "truth_boundary": truth_boundary,
    }
    write_json(config.output / "summary.json", summary)
    return {
        "format_name": "atlas3r_tum_rgbd_temporal_train_run",
        "output": str(config.output),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "train_clip_count": len(train_dataset),
        "val_clip_count": len(val_dataset),
        "steps_completed": completed_steps,
        "best_center_depth_rmse_m": best_val_metrics["center_depth_rmse_m"],
        "best_center_depth_mae_m": best_val_metrics["center_depth_mae_m"],
        "best_center_depth_absrel": best_val_metrics["center_depth_absrel"],
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_best": "checkpoint_best.pt",
        "truth_boundary": truth_boundary,
    }


def _validate(model: Any, val_loader: Any, device: str, *, depth_loss: str) -> dict[str, float]:
    torch = require_torch()
    from atlas3r.training.temporal_losses import temporal_geometry_loss

    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["intrinsics"])
            _loss, metrics = temporal_geometry_loss(
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


def _write_prediction_sample_and_preview(
    output: Path,
    *,
    model: Any,
    dataset: TumRgbdClipCacheDataset,
    device: str,
    metrics: dict[str, float],
    truth_boundary: dict[str, object],
) -> tuple[Path, Path]:
    torch = require_torch()
    sample = dataset[0]
    batch = _move_batch_to_device(
        {
            "images_rgb": sample["images_rgb"].unsqueeze(0),
            "intrinsics": sample["intrinsics"].unsqueeze(0),
            "target": {
                key: value.unsqueeze(0)
                for key, value in cast(dict[str, Any], sample["target"]).items()
                if torch.is_tensor(value)
            },
        },
        device,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(batch["images_rgb"], batch["intrinsics"])
    metadata = cast(dict[str, Any], sample["metadata"])
    center_index = int(metadata["center_index"])
    rgb_u8 = (
        (sample["images_rgb"][center_index].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    target_depth = sample["target"]["center_depth_m"][0].detach().cpu().numpy().astype(np.float32)
    valid_mask = (
        sample["target"]["center_valid_depth_mask"][0].detach().cpu().numpy().astype(np.bool_)
    )
    predicted_depth = prediction["center_depth_m"][0, 0].detach().cpu().numpy().astype(np.float32)
    abs_error = np.abs(predicted_depth - target_depth).astype(np.float32)
    np.savez(
        output / "prediction_sample.npz",
        rgb_u8=rgb_u8,
        target_center_depth_m=target_depth,
        valid_depth_mask=valid_mask,
        predicted_center_depth_m=predicted_depth,
        abs_depth_error_m=abs_error,
        predicted_center_depth_sigma_m=prediction["center_depth_sigma_m"][0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        predicted_center_confidence=prediction["center_confidence"][0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        target_relative_translation=batch["target"]["relative_T_center_camera"][0, :, :3, 3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        predicted_relative_translation=prediction["relative_translation_center_from_camera"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
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
        title="Atlas3R TUM RGB-D Temporal Preview",
    )


def _write_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    step: int,
    config_record: dict[str, object],
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    truth_boundary: dict[str, object],
) -> None:
    torch = require_torch()
    torch.save(
        {
            "format_name": "atlas3r_tiny_temporal_geometry_checkpoint",
            "format_version": 1,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_record,
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "model_config": model.model_config(),
            "truth_boundary": truth_boundary,
        },
        path,
    )


def _config_record(
    config: TumRgbdTemporalTrainConfig,
    *,
    resolved_device: str,
    amp_enabled: bool,
) -> dict[str, object]:
    return {
        "clip_cache": str(config.clip_cache),
        "val_clip_cache": str(config.val_clip_cache) if config.val_clip_cache else None,
        "output": str(config.output),
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
        "depth_loss": config.depth_loss,
        "model": "TinyTemporalMetricNetV0",
    }


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


def _runtime_expired(config: TumRgbdTemporalTrainConfig, start_time: float) -> bool:
    if config.max_runtime_minutes is None:
        return False
    return (time.monotonic() - start_time) >= config.max_runtime_minutes * 60.0


def _validate_config(config: TumRgbdTemporalTrainConfig) -> None:
    for field_name in (
        "steps",
        "batch_size",
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
    if config.depth_loss not in {"metric_l1", "log_l1"}:
        raise ValueError("depth_loss: must be 'metric_l1' or 'log_l1'")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "TumRgbdTemporalTrainConfig",
    "run_tum_rgbd_temporal_training",
]
