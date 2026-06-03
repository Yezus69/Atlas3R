"""Teacher-signal temporal training runner for Phase 5D."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas3r.training.teacher_signal_dataset import (
    TeacherSignalTemporalDataset,
    teacher_signal_train_val_indices,
)
from atlas3r.training.teacher_signal_losses import (
    TeacherSignalLossConfig,
    teacher_signal_temporal_loss,
)
from atlas3r.training.teacher_signal_temporal_artifacts import (
    teacher_signal_temporal_config_record,
    write_teacher_signal_prediction_sample_and_preview,
    write_teacher_signal_temporal_checkpoint,
)
from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_rgbd_artifacts import append_jsonl, write_json


@dataclass(frozen=True)
class TeacherSignalTemporalTrainConfig:
    teacher_caches: tuple[Path, ...]
    output: Path
    val_teacher_caches: tuple[Path, ...] = ()
    model: str = "temporal-v1"
    steps: int = 12000
    batch_size: int = 8
    device: str = "cuda"
    num_workers: int = 0
    learning_rate: float = 0.0003
    log_every: int = 50
    val_every: int = 500
    checkpoint_every: int = 1000
    preview_every: int = 1000
    seed: int = 0
    amp: bool = False
    max_runtime_minutes: float | None = 330.0
    hidden_channels: int = 24
    bottleneck_channels: int = 32
    loss_config: TeacherSignalLossConfig = TeacherSignalLossConfig()


def run_teacher_signal_temporal_training(
    config: TeacherSignalTemporalTrainConfig,
) -> dict[str, object]:
    """Train `TemporalMetricNetV1` from measured and pseudo teacher-signal caches."""

    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    amp_enabled = bool(config.amp and resolved_device == "cuda")
    config.output.mkdir(parents=True, exist_ok=True)
    _seed_torch(torch, config.seed)

    train_dataset, val_dataset, split_mode = _datasets(config)
    config_record = teacher_signal_temporal_config_record(
        config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        split_mode=split_mode,
        resolved_device=resolved_device,
        amp_enabled=amp_enabled,
    )
    write_json(config.output / "config.json", config_record)
    metrics_path = config.output / "metrics.jsonl"
    validation_path = config.output / "validation_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    validation_path.write_text("", encoding="utf-8")

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
    from atlas3r.training.tiny_temporal_geometry_model import (
        TemporalMetricNetV1,
        temporal_v1_truth_boundary,
    )

    model = TemporalMetricNetV1(
        hidden_channels=config.hidden_channels,
        bottleneck_channels=config.bottleneck_channels,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    truth_boundary = temporal_v1_truth_boundary()
    best_val_metrics: dict[str, float] | None = None
    best_val_rmse = float("inf")
    final_train_metrics: dict[str, float] = {}
    stopped_reason = "completed_steps"
    success = True
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
        try:
            with context:
                prediction = model(batch["images_rgb"], batch["intrinsics"])
                loss, metrics = teacher_signal_temporal_loss(
                    prediction,
                    _loss_target(batch),
                    config=config.loss_config,
                )
        except ValueError as exc:
            if not _is_nonfinite_error(exc):
                raise
            stopped_reason = "nonfinite_loss"
            success = False
            break
        if not bool(torch.isfinite(loss).all()):
            stopped_reason = "nonfinite_loss"
            success = False
            break
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
                {"format_name": "atlas3r_teacher_signal_train_metric", "step": step, **metrics},
            )
        if step == 1 or step % config.val_every == 0 or step == config.steps:
            val_metrics = _validate(
                model,
                val_loader,
                resolved_device,
                loss_config=config.loss_config,
            )
            append_jsonl(
                validation_path,
                {
                    "format_name": "atlas3r_teacher_signal_validation_metric",
                    "step": step,
                    **val_metrics,
                },
            )
            if val_metrics["depth_rmse_m"] < best_val_rmse:
                best_val_rmse = val_metrics["depth_rmse_m"]
                best_val_metrics = val_metrics
                write_teacher_signal_temporal_checkpoint(
                    config.output / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config_record=config_record,
                    metrics=metrics,
                    validation_metrics=val_metrics,
                    truth_boundary=truth_boundary,
                    loss_config=config.loss_config,
                )
        if step % config.checkpoint_every == 0 or step == config.steps:
            write_teacher_signal_temporal_checkpoint(
                config.output / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config_record=config_record,
                metrics=metrics,
                validation_metrics=best_val_metrics or {},
                truth_boundary=truth_boundary,
                loss_config=config.loss_config,
            )
        if step % config.preview_every == 0:
            write_teacher_signal_prediction_sample_and_preview(
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

    if success:
        if best_val_metrics is None:
            best_val_metrics = _validate(
                model,
                val_loader,
                resolved_device,
                loss_config=config.loss_config,
            )
        write_teacher_signal_temporal_checkpoint(
            config.output / "checkpoint_last.pt",
            model=model,
            optimizer=optimizer,
            step=completed_steps,
            config_record=config_record,
            metrics=final_train_metrics,
            validation_metrics=best_val_metrics,
            truth_boundary=truth_boundary,
            loss_config=config.loss_config,
        )
        if not (config.output / "checkpoint_best.pt").is_file():
            write_teacher_signal_temporal_checkpoint(
                config.output / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                step=completed_steps,
                config_record=config_record,
                metrics=final_train_metrics,
                validation_metrics=best_val_metrics,
                truth_boundary=truth_boundary,
                loss_config=config.loss_config,
            )
        write_teacher_signal_prediction_sample_and_preview(
            config.output,
            model=model,
            dataset=val_dataset,
            device=resolved_device,
            metrics=best_val_metrics,
            truth_boundary=truth_boundary,
        )

    summary = {
        "format_name": "atlas3r_teacher_signal_temporal_train_run_summary",
        "format_version": 1,
        "success": success,
        "train_summary": train_dataset.cache_summary(),
        "validation_summary": val_dataset.cache_summary(),
        "best_validation": best_val_metrics,
        "final_train_losses": final_train_metrics,
        "artifacts": {
            "config": "config.json",
            "metrics": "metrics.jsonl",
            "validation_metrics": "validation_metrics.jsonl",
            "checkpoint_last": "checkpoint_last.pt" if success else None,
            "checkpoint_best": "checkpoint_best.pt" if success else None,
            "prediction_sample": "prediction_sample.npz" if success else None,
            "preview_html": "prediction_preview.html" if success else None,
            "preview_svg": "prediction_preview.svg" if success else None,
        },
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "stopped_reason": stopped_reason,
        "steps_completed": completed_steps,
        "truth_boundary": truth_boundary,
    }
    write_json(config.output / "summary.json", summary)
    return {
        "format_name": "atlas3r_teacher_signal_temporal_train_run",
        "output": str(config.output),
        "success": success,
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "steps_completed": completed_steps,
        "best_depth_rmse_m": None if best_val_metrics is None else best_val_metrics["depth_rmse_m"],
        "best_depth_mae_m": None if best_val_metrics is None else best_val_metrics["depth_mae_m"],
        "checkpoint_last": "checkpoint_last.pt" if success else None,
        "checkpoint_best": "checkpoint_best.pt" if success else None,
        "truth_boundary": truth_boundary,
    }


def _datasets(
    config: TeacherSignalTemporalTrainConfig,
) -> tuple[TeacherSignalTemporalDataset, TeacherSignalTemporalDataset, str]:
    if config.val_teacher_caches:
        return (
            TeacherSignalTemporalDataset(config.teacher_caches),
            TeacherSignalTemporalDataset(config.val_teacher_caches),
            "explicit_val_teacher_cache",
        )
    all_dataset = TeacherSignalTemporalDataset(config.teacher_caches)
    train_indices, val_indices = teacher_signal_train_val_indices(len(all_dataset))
    return (
        TeacherSignalTemporalDataset(config.teacher_caches, record_indices=train_indices),
        TeacherSignalTemporalDataset(config.teacher_caches, record_indices=val_indices),
        "deterministic_tail_block",
    )


def _validate(
    model: Any,
    val_loader: Any,
    device: str,
    *,
    loss_config: TeacherSignalLossConfig,
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["intrinsics"])
            _loss, metrics = teacher_signal_temporal_loss(
                prediction,
                _loss_target(batch),
                config=loss_config,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    if batches <= 0:
        raise ValueError("validation: no batches were produced")
    return {key: value / float(batches) for key, value in totals.items()}


def _loss_target(batch: dict[str, Any]) -> dict[str, Any]:
    target = dict(batch["target"])
    target["intrinsics"] = batch["intrinsics"]
    return target


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


def _runtime_expired(config: TeacherSignalTemporalTrainConfig, start_time: float) -> bool:
    if config.max_runtime_minutes is None:
        return False
    return (time.monotonic() - start_time) >= config.max_runtime_minutes * 60.0


def _is_nonfinite_error(exc: ValueError) -> bool:
    text = str(exc).lower()
    return "nan" in text or "inf" in text or "not finite" in text


def _validate_config(config: TeacherSignalTemporalTrainConfig) -> None:
    if not config.teacher_caches:
        raise ValueError("teacher_cache: at least one --teacher-cache is required")
    if config.model != "temporal-v1":
        raise ValueError("model: only temporal-v1 is supported")
    for field_name in (
        "steps",
        "batch_size",
        "log_every",
        "val_every",
        "checkpoint_every",
        "preview_every",
        "hidden_channels",
        "bottleneck_channels",
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


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "TeacherSignalTemporalTrainConfig",
    "run_teacher_signal_temporal_training",
]
