"""Training runner for SMGT-small-v2 measured/pseudo supervision."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas3r.models.smgt import (
    SMGTSmallV2,
    SMGTSmallV2Config,
    save_smgt_small_v2_checkpoint,
    smgt_small_v2_truth_boundary,
)
from atlas3r.training.smgt_tiny_eval import move_batch_to_device
from atlas3r.training.smgt_v2_dataset import (
    SMGTV2MixedTemporalDataset,
    smgt_v2_measured_train_val_indices,
)
from atlas3r.training.smgt_v2_eval import evaluate_smgt_v2
from atlas3r.training.smgt_v2_losses import (
    SMGTV2LossConfig,
    smgt_v2_candidate_can_be_best,
    smgt_v2_checkpoint_selection_score,
    smgt_v2_loss,
)
from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_rgbd_artifacts import append_jsonl, write_json


@dataclass(frozen=True)
class SMGTV2TrainConfig:
    measured_caches: Sequence[Path]
    output: Path
    pseudo_caches: Sequence[Path] = ()
    val_source: Path | None = None
    steps: int = 10000
    batch_size: int = 4
    device: str = "cuda"
    amp: bool = False
    learning_rate: float = 1e-4
    num_workers: int = 2
    save_every: int = 1000
    seed: int = 0
    val_fraction: float = 0.2
    pseudo_weight: float = 0.25
    loss_config: SMGTV2LossConfig = field(default_factory=SMGTV2LossConfig)


def run_smgt_v2_training(config: SMGTV2TrainConfig) -> dict[str, object]:
    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    amp_enabled = bool(config.amp and resolved_device.startswith("cuda"))
    _seed_torch(torch, config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset = _build_datasets(config)
    model_config = SMGTSmallV2Config(
        image_height=train_dataset.image_height,
        image_width=train_dataset.image_width,
        clip_length=train_dataset.clip_length,
    )
    truth = smgt_small_v2_truth_boundary(
        measured_training_used=True,
        pseudo_training_used=bool(config.pseudo_caches),
    )
    config_record = _config_record(
        config,
        resolved_device=resolved_device,
        amp_enabled=amp_enabled,
        model_config=model_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    write_json(config.output / "config.json", config_record)
    write_json(config.output / "truth_boundary.json", truth)
    train_metrics_path = config.output / "train_metrics.jsonl"
    val_metrics_path = config.output / "val_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    val_metrics_path.write_text("", encoding="utf-8")
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
    model = SMGTSmallV2(model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_iterator = iter(train_loader)
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")
    best_step: int | None = None
    final_metrics: dict[str, float] = {}
    start_time = time.monotonic()
    for step in range(1, config.steps + 1):
        model.train()
        batch, train_iterator = _next_batch(train_loader, train_iterator)
        batch = move_batch_to_device(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            prediction = model(batch["images_rgb"], batch["K"])
            loss, metrics = smgt_v2_loss(prediction, batch, config=config.loss_config)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        final_metrics = metrics
        append_jsonl(
            train_metrics_path,
            {"format_name": "atlas3r_smgt_v2_train_metric", "step": step, **metrics},
        )
        if step == 1 or step % config.save_every == 0 or step == config.steps:
            val_metrics = evaluate_smgt_v2(
                model,
                val_loader,
                device=resolved_device,
                loss_config=config.loss_config,
            )
            val_metrics["mesh_success"] = float(
                bool(val_metrics["val_mapped_pixel_ratio_at_selected_threshold"] > 0.0)
            )
            val_metrics["mesh_chunk_count"] = float(1.0 if val_metrics["mesh_success"] else 0.0)
            score = smgt_v2_checkpoint_selection_score(val_metrics)
            val_record = {
                "format_name": "atlas3r_smgt_v2_val_metric",
                "step": step,
                "selection_score": score,
                "candidate_can_be_best": float(smgt_v2_candidate_can_be_best(val_metrics)),
                **val_metrics,
            }
            append_jsonl(val_metrics_path, val_record)
            if score < best_score and smgt_v2_candidate_can_be_best(val_metrics):
                best_score = score
                best_metrics = dict(val_metrics)
                best_step = step
                save_smgt_small_v2_checkpoint(
                    config.output / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config_record=config_record,
                    metrics=metrics,
                    validation_metrics=val_metrics,
                    truth_boundary=truth,
                    loss_config=config.loss_config.to_json(),
                    selection_score=score,
                )
            save_smgt_small_v2_checkpoint(
                config.output / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config_record=config_record,
                metrics=metrics,
                validation_metrics=val_metrics,
                truth_boundary=truth,
                loss_config=config.loss_config.to_json(),
                selection_score=score,
            )
    summary = {
        "format_name": "atlas3r_smgt_v2_train_summary",
        "format_version": 1,
        "output": str(config.output),
        "measured_caches": [str(path) for path in config.measured_caches],
        "pseudo_caches": [str(path) for path in config.pseudo_caches],
        "val_source": None if config.val_source is None else str(config.val_source),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "steps_completed": config.steps,
        "runtime_seconds": time.monotonic() - start_time,
        "train_dataset": train_dataset.cache_summary(),
        "validation_dataset": val_dataset.cache_summary(),
        "final_train_metrics": final_metrics,
        "best_validation_metrics": best_metrics,
        "best_validation_step": best_step,
        "best_selection_score": None if best_metrics is None else best_score,
        "checkpoint_best_written": best_metrics is not None,
        "artifacts": {
            "config": "config.json",
            "train_metrics": "train_metrics.jsonl",
            "val_metrics": "val_metrics.jsonl",
            "checkpoint_last": "checkpoint_last.pt",
            "checkpoint_best": "checkpoint_best.pt" if best_metrics is not None else None,
            "training_report": "training_report.md",
            "truth_boundary": "truth_boundary.json",
        },
        "truth_boundary": truth,
    }
    write_json(config.output / "summary.json", summary)
    _write_training_report(config.output / "training_report.md", summary)
    return {
        "format_name": "atlas3r_smgt_v2_train_run",
        "output": str(config.output),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "steps_completed": config.steps,
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_best": "checkpoint_best.pt" if best_metrics is not None else None,
        "best_validation_step": best_step,
        "best_selection_score": None if best_metrics is None else best_score,
        "truth_boundary": truth,
    }


def _build_datasets(
    config: SMGTV2TrainConfig,
) -> tuple[SMGTV2MixedTemporalDataset, SMGTV2MixedTemporalDataset]:
    if config.val_source is not None:
        train_dataset = SMGTV2MixedTemporalDataset(
            measured_caches=config.measured_caches,
            pseudo_caches=config.pseudo_caches,
            pseudo_weight=config.pseudo_weight,
        )
        val_dataset = SMGTV2MixedTemporalDataset(measured_caches=[config.val_source])
        return train_dataset, val_dataset
    first_cache_probe = SMGTV2MixedTemporalDataset(measured_caches=[config.measured_caches[0]])
    train_indices, val_indices = smgt_v2_measured_train_val_indices(
        len(first_cache_probe.measured[0]),
        val_fraction=config.val_fraction,
    )
    measured_train_indices = [train_indices, *[None for _ in config.measured_caches[1:]]]
    train_dataset = SMGTV2MixedTemporalDataset(
        measured_caches=config.measured_caches,
        pseudo_caches=config.pseudo_caches,
        measured_indices=measured_train_indices,
        pseudo_weight=config.pseudo_weight,
    )
    val_dataset = SMGTV2MixedTemporalDataset(
        measured_caches=[config.measured_caches[0]],
        measured_indices=[val_indices],
    )
    return train_dataset, val_dataset


def _config_record(
    config: SMGTV2TrainConfig,
    *,
    resolved_device: str,
    amp_enabled: bool,
    model_config: SMGTSmallV2Config,
    train_dataset: SMGTV2MixedTemporalDataset,
    val_dataset: SMGTV2MixedTemporalDataset,
) -> dict[str, object]:
    return {
        "measured_caches": [str(path) for path in config.measured_caches],
        "pseudo_caches": [str(path) for path in config.pseudo_caches],
        "output": str(config.output),
        "val_source": None if config.val_source is None else str(config.val_source),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "amp_requested": config.amp,
        "amp_enabled": amp_enabled,
        "learning_rate": config.learning_rate,
        "num_workers": config.num_workers,
        "save_every": config.save_every,
        "seed": config.seed,
        "val_fraction": config.val_fraction,
        "pseudo_weight": config.pseudo_weight,
        "model_config": model_config.to_json(),
        "loss_config": config.loss_config.to_json(),
        "train_dataset": train_dataset.cache_summary(),
        "validation_dataset": val_dataset.cache_summary(),
    }


def _write_training_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# SMGT Small V2 Training Report",
        "",
        f"- Steps completed: `{summary['steps_completed']}`",
        f"- Device: `{summary['device']}`",
        f"- AMP enabled: `{summary['amp_enabled']}`",
        f"- Checkpoint best written: `{summary['checkpoint_best_written']}`",
        f"- Best validation step: `{summary['best_validation_step']}`",
        f"- Best selection score: `{summary['best_selection_score']}`",
        "",
        "This is diagnostic measured/pseudo training. It is not final SMGT, "
        "not realtime evidence, and not RGB-only production readiness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _next_batch(loader: Any, iterator: Any) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _validate_config(config: SMGTV2TrainConfig) -> None:
    if not config.measured_caches:
        raise ValueError("measured_caches: at least one measured cache is required")
    if config.steps <= 0:
        raise ValueError("steps: must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size: must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers: must be non-negative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate: must be positive")
    if config.save_every <= 0:
        raise ValueError("save_every: must be positive")
    if config.val_fraction < 0.0 or config.val_fraction >= 1.0:
        raise ValueError("val_fraction: must be in [0, 1)")
    if config.pseudo_weight <= 0.0 or config.pseudo_weight > 0.25:
        raise ValueError("pseudo_weight: must be in (0, 0.25]")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


__all__ = ["SMGTV2TrainConfig", "run_smgt_v2_training"]
