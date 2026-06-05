"""Training runner for the first diagnostic SMGT-tiny student."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from atlas3r.models.smgt import (
    SMGTTiny,
    SMGTTinyConfig,
    save_smgt_tiny_checkpoint,
    smgt_tiny_truth_boundary,
)
from atlas3r.training.smgt_tiny_dataset import (
    SMGTTinyTeacherCacheDataset,
    smgt_tiny_train_val_indices,
)
from atlas3r.training.smgt_tiny_eval import evaluate_smgt_tiny, move_batch_to_device
from atlas3r.training.smgt_tiny_losses import SMGTTinyLossConfig, smgt_tiny_loss
from atlas3r.training.smgt_tiny_metric_reporting import summarize_smgt_tiny_training_metrics
from atlas3r.training.smgt_tiny_split import (
    SPLIT_MANIFEST_FILENAME,
    SPLIT_REPORT_FILENAME,
    format_smgt_tiny_split_report,
    load_smgt_tiny_split_manifest,
    split_indices,
    write_smgt_tiny_split_manifest,
)
from atlas3r.training.torch_runtime import require_torch, select_device
from atlas3r.training.tum_rgbd_artifacts import append_jsonl, write_json


@dataclass(frozen=True)
class SMGTTinyTrainConfig:
    teacher_cache: Path
    output: Path
    steps: int = 2000
    batch_size: int = 4
    device: str = "cuda"
    amp: bool = False
    learning_rate: float = 1e-4
    val_split: float = 0.2
    heldout_split: float = 0.0
    split_manifest: Path | None = None
    num_workers: int = 2
    save_every: int = 500
    seed: int = 0
    debug_subset_clips: int | None = None
    debug_allow_pseudo_weight: float | None = None
    hidden_dim: int = 64
    feature_dim: int = 96
    memory_dim: int = 128
    loss_config: SMGTTinyLossConfig = field(default_factory=SMGTTinyLossConfig)


def run_smgt_tiny_training(config: SMGTTinyTrainConfig) -> dict[str, object]:
    """Train SMGT-tiny from a Phase 6H teacher temporal cache."""

    _validate_config(config)
    torch = require_torch()
    resolved_device = select_device(config.device)
    amp_enabled = bool(config.amp and resolved_device.startswith("cuda"))
    _seed_torch(torch, config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    full_dataset = SMGTTinyTeacherCacheDataset(
        config.teacher_cache,
        debug_subset_clips=config.debug_subset_clips,
        pseudo_weight_override=config.debug_allow_pseudo_weight,
    )
    split_manifest: dict[str, object] | None = None
    if config.split_manifest is not None or config.heldout_split > 0.0:
        split_manifest = _prepare_split_manifest(config)
        source_indices = split_indices(split_manifest, "train")
        source_val_indices = split_indices(split_manifest, "val")
        source_heldout_indices = split_indices(split_manifest, "heldout")
    else:
        train_indices, val_indices = smgt_tiny_train_val_indices(
            len(full_dataset),
            val_split=config.val_split,
        )
        source_indices = tuple(full_dataset.indices[index] for index in train_indices)
        source_val_indices = tuple(full_dataset.indices[index] for index in val_indices)
        source_heldout_indices = ()
    train_dataset = SMGTTinyTeacherCacheDataset(
        config.teacher_cache,
        indices=source_indices,
        pseudo_weight_override=config.debug_allow_pseudo_weight,
    )
    val_dataset = SMGTTinyTeacherCacheDataset(
        config.teacher_cache,
        indices=source_val_indices,
        pseudo_weight_override=config.debug_allow_pseudo_weight,
    )
    heldout_dataset = (
        SMGTTinyTeacherCacheDataset(
            config.teacher_cache,
            indices=source_heldout_indices,
            pseudo_weight_override=config.debug_allow_pseudo_weight,
        )
        if source_heldout_indices
        else None
    )
    model_config = SMGTTinyConfig(
        image_height=int(cast(int, full_dataset.manifest["image_height"])),
        image_width=int(cast(int, full_dataset.manifest["image_width"])),
        clip_length=int(cast(int, full_dataset.manifest["clip_length"])),
        hidden_dim=config.hidden_dim,
        feature_dim=config.feature_dim,
        memory_dim=config.memory_dim,
    )
    truth_boundary = smgt_tiny_truth_boundary(pseudo_training_used=True)
    config_record = _config_record(
        config,
        resolved_device=resolved_device,
        amp_enabled=amp_enabled,
        model_config=model_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        heldout_dataset=heldout_dataset,
        split_manifest=split_manifest,
    )
    write_json(config.output / "config.json", config_record)
    write_json(config.output / "truth_boundary.json", truth_boundary)
    train_metrics_path = config.output / "train_metrics.jsonl"
    val_metrics_path = config.output / "val_metrics.jsonl"
    heldout_metrics_path = config.output / "heldout_metrics.jsonl"
    train_metrics_path.write_text("", encoding="utf-8")
    val_metrics_path.write_text("", encoding="utf-8")
    if heldout_dataset is not None:
        heldout_metrics_path.write_text("", encoding="utf-8")
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
    heldout_loader = (
        torch.utils.data.DataLoader(
            heldout_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if heldout_dataset is not None
        else None
    )
    model = SMGTTiny(model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_iterator = iter(train_loader)
    best_val_metrics: dict[str, float] | None = None
    best_heldout_metrics: dict[str, float] | None = None
    best_val_loss = float("inf")
    best_heldout_loss = float("inf")
    final_metrics: dict[str, float] = {}
    train_history: list[dict[str, float]] = []
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
            loss, metrics = smgt_tiny_loss(prediction, batch, config=config.loss_config)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        final_metrics = metrics
        train_history.append(metrics)
        append_jsonl(
            train_metrics_path,
            {"format_name": "atlas3r_smgt_tiny_train_metric", "step": step, **metrics},
        )
        if step == 1 or step % config.save_every == 0 or step == config.steps:
            val_metrics = evaluate_smgt_tiny(
                model,
                val_loader,
                device=resolved_device,
                loss_config=config.loss_config,
            )
            append_jsonl(
                val_metrics_path,
                {"format_name": "atlas3r_smgt_tiny_val_metric", "step": step, **val_metrics},
            )
            heldout_metrics = None
            if heldout_loader is not None:
                heldout_metrics = evaluate_smgt_tiny(
                    model,
                    heldout_loader,
                    device=resolved_device,
                    loss_config=config.loss_config,
                )
                append_jsonl(
                    heldout_metrics_path,
                    {
                        "format_name": "atlas3r_smgt_tiny_heldout_metric",
                        "step": step,
                        **heldout_metrics,
                    },
                )
                if heldout_metrics["loss_total"] < best_heldout_loss:
                    best_heldout_loss = heldout_metrics["loss_total"]
                    best_heldout_metrics = heldout_metrics
                    save_smgt_tiny_checkpoint(
                        config.output / "checkpoint_heldout_best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        config_record=config_record,
                        metrics=metrics,
                        validation_metrics=val_metrics,
                        truth_boundary=truth_boundary,
                        loss_config=config.loss_config.to_json(),
                    )
            if val_metrics["loss_total"] < best_val_loss:
                best_val_loss = val_metrics["loss_total"]
                best_val_metrics = val_metrics
                save_smgt_tiny_checkpoint(
                    config.output / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config_record=config_record,
                    metrics=metrics,
                    validation_metrics=val_metrics,
                    truth_boundary=truth_boundary,
                    loss_config=config.loss_config.to_json(),
                )
            save_smgt_tiny_checkpoint(
                config.output / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                config_record=config_record,
                metrics=metrics,
                validation_metrics=val_metrics,
                truth_boundary=truth_boundary,
                loss_config=config.loss_config.to_json(),
            )
    if best_val_metrics is None:
        best_val_metrics = evaluate_smgt_tiny(
            model,
            val_loader,
            device=resolved_device,
            loss_config=config.loss_config,
        )
    if heldout_loader is not None and best_heldout_metrics is None:
        best_heldout_metrics = evaluate_smgt_tiny(
            model,
            heldout_loader,
            device=resolved_device,
            loss_config=config.loss_config,
        )
    _write_prediction_preview_npz(
        config.output / "prediction_preview.npz", model, val_dataset, resolved_device
    )
    metric_windows = summarize_smgt_tiny_training_metrics(train_history)
    summary = {
        "format_name": "atlas3r_smgt_tiny_train_summary",
        "format_version": 1,
        "output": str(config.output),
        "teacher_cache": str(config.teacher_cache),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "steps_completed": config.steps,
        "runtime_seconds": time.monotonic() - start_time,
        "train_dataset": train_dataset.cache_summary(),
        "validation_dataset": val_dataset.cache_summary(),
        "heldout_dataset": None if heldout_dataset is None else heldout_dataset.cache_summary(),
        "split_manifest": split_manifest,
        "final_train_metrics": final_metrics,
        "best_validation_metrics": best_val_metrics,
        "best_heldout_metrics": best_heldout_metrics,
        "training_metric_windows": metric_windows,
        "training_quality_pass": metric_windows["training_quality_pass"],
        "loss_decrease": metric_windows,
        "artifacts": {
            "config": "config.json",
            "train_metrics": "train_metrics.jsonl",
            "val_metrics": "val_metrics.jsonl",
            "heldout_metrics": "heldout_metrics.jsonl" if heldout_dataset is not None else None,
            "checkpoint_last": "checkpoint_last.pt",
            "checkpoint_best": "checkpoint_best.pt",
            "checkpoint_heldout_best": (
                "checkpoint_heldout_best.pt" if heldout_dataset is not None else None
            ),
            "split_manifest": SPLIT_MANIFEST_FILENAME if split_manifest is not None else None,
            "split_report": SPLIT_REPORT_FILENAME if split_manifest is not None else None,
            "training_report": "training_report.md",
            "prediction_preview": "prediction_preview.npz",
            "truth_boundary": "truth_boundary.json",
        },
        "truth_boundary": truth_boundary,
    }
    write_json(config.output / "summary.json", summary)
    _write_training_report(config.output / "training_report.md", summary)
    return {
        "format_name": "atlas3r_smgt_tiny_train_run",
        "output": str(config.output),
        "device": resolved_device,
        "amp_enabled": amp_enabled,
        "steps_completed": config.steps,
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_best": "checkpoint_best.pt",
        "checkpoint_heldout_best": (
            "checkpoint_heldout_best.pt" if heldout_dataset is not None else None
        ),
        "loss_decrease": metric_windows,
        "best_validation_loss_total": best_val_metrics["loss_total"],
        "best_heldout_loss_total": (
            None if best_heldout_metrics is None else best_heldout_metrics["loss_total"]
        ),
        "truth_boundary": truth_boundary,
    }


def _config_record(
    config: SMGTTinyTrainConfig,
    *,
    resolved_device: str,
    amp_enabled: bool,
    model_config: SMGTTinyConfig,
    train_dataset: SMGTTinyTeacherCacheDataset,
    val_dataset: SMGTTinyTeacherCacheDataset,
    heldout_dataset: SMGTTinyTeacherCacheDataset | None,
    split_manifest: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "teacher_cache": str(config.teacher_cache),
        "output": str(config.output),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "amp_requested": config.amp,
        "amp_enabled": amp_enabled,
        "learning_rate": config.learning_rate,
        "val_split": config.val_split,
        "heldout_split": config.heldout_split,
        "split_manifest_input": None
        if config.split_manifest is None
        else str(config.split_manifest),
        "num_workers": config.num_workers,
        "save_every": config.save_every,
        "seed": config.seed,
        "debug_subset_clips": config.debug_subset_clips,
        "debug_allow_pseudo_weight": config.debug_allow_pseudo_weight,
        "model_config": model_config.to_json(),
        "loss_config": config.loss_config.to_json(),
        "train_dataset": train_dataset.cache_summary(),
        "validation_dataset": val_dataset.cache_summary(),
        "heldout_dataset": None if heldout_dataset is None else heldout_dataset.cache_summary(),
        "split_manifest": split_manifest,
    }


def _prepare_split_manifest(config: SMGTTinyTrainConfig) -> dict[str, object]:
    if config.split_manifest is not None:
        manifest = load_smgt_tiny_split_manifest(config.split_manifest)
        write_json(config.output / SPLIT_MANIFEST_FILENAME, manifest)
        (config.output / SPLIT_REPORT_FILENAME).write_text(
            format_smgt_tiny_split_report(manifest),
            encoding="utf-8",
        )
        return manifest
    return write_smgt_tiny_split_manifest(
        config.output,
        config.teacher_cache,
        val_split=config.val_split,
        heldout_split=config.heldout_split,
    )


def _write_prediction_preview_npz(path: Path, model: Any, dataset: Any, device: str) -> None:
    torch = require_torch()
    sample = dataset[0]
    batch = move_batch_to_device(
        {
            "images_rgb": sample["images_rgb"].unsqueeze(0),
            "K": sample["K"].unsqueeze(0),
        },
        device,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(batch["images_rgb"], batch["K"])
    np.savez_compressed(
        path,
        rgb_u8=(sample["images_rgb"][0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(
            np.uint8
        ),
        frame_ids=sample["frame_ids"].detach().cpu().numpy().astype(np.int64),
        target_depth_m=sample["target"]["depth_m"].detach().cpu().numpy().astype(np.float32),
        target_valid_mask=sample["target"]["valid_mask"].detach().cpu().numpy().astype(np.bool_),
        predicted_depth_m=prediction["depth_m"][0].detach().cpu().numpy().astype(np.float32),
        predicted_depth_sigma_m=prediction["depth_sigma_m"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        predicted_confidence=prediction["confidence"][0].detach().cpu().numpy().astype(np.float32),
        predicted_T_world_camera=prediction["T_world_camera"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        target_T_world_camera=sample["target"]["T_world_camera"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    )


def _write_training_report(path: Path, summary: dict[str, object]) -> None:
    decrease = cast(dict[str, object], summary["loss_decrease"])
    first = cast(dict[str, object], decrease.get("first_100", {}))
    final = cast(dict[str, object], decrease.get("final_100", {}))
    lines = [
        "# SMGT Tiny Training Report",
        "",
        f"- Steps completed: `{summary['steps_completed']}`",
        f"- Device: `{summary['device']}`",
        f"- AMP enabled: `{summary['amp_enabled']}`",
        f"- Window size: `{decrease['window_size']}`",
        f"- First-window loss total: `{first.get('loss_total')}`",
        f"- Final-window loss total: `{final.get('loss_total')}`",
        f"- Loss total delta: `{decrease['loss_total_delta']}`",
        f"- Loss total reduction absolute: `{decrease['loss_total_reduction_absolute']}`",
        f"- Loss crossed zero: `{decrease['loss_total_crossed_zero']}`",
        f"- Loss decrease percent meaningful: `{decrease['loss_total_percent_meaningful']}`",
        f"- Loss decrease percent: `{decrease['loss_total_decrease_percent']}`",
        f"- First/final depth AbsRel: "
        f"`{first.get('depth_absrel')}` / `{final.get('depth_absrel')}`",
        f"- First/final depth RMSE m: "
        f"`{first.get('depth_rmse_m')}` / `{final.get('depth_rmse_m')}`",
        f"- First/final pose center mean m: `{first.get('pose_center_mean_m')}` / "
        f"`{final.get('pose_center_mean_m')}`",
        f"- First/final pose relative translation mean m: "
        f"`{first.get('pose_relative_translation_mean_m')}` / "
        f"`{final.get('pose_relative_translation_mean_m')}`",
        f"- First/final pose rotation mean deg: `{first.get('pose_rotation_mean_deg')}` / "
        f"`{final.get('pose_rotation_mean_deg')}`",
        f"- First/final confidence Brier: `{first.get('confidence_brier')}` / "
        f"`{final.get('confidence_brier')}`",
        f"- Training quality pass: `{summary['training_quality_pass']}`",
        "",
        "This is diagnostic pseudo-label training from the Phase 6H teacher temporal cache. "
        "It is not final SMGT, not realtime evidence, not an accuracy report, and not "
        "RGB-only production mapping readiness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _next_batch(loader: Any, iterator: Any) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _validate_config(config: SMGTTinyTrainConfig) -> None:
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
    if config.val_split < 0.0 or config.val_split >= 1.0:
        raise ValueError("val_split: must be in [0, 1)")
    if config.heldout_split < 0.0 or config.val_split + config.heldout_split >= 1.0:
        raise ValueError("heldout_split: must be non-negative and val_split + heldout_split < 1")
    if (config.heldout_split > 0.0 or config.split_manifest is not None) and (
        config.debug_subset_clips is not None
    ):
        raise ValueError("debug_subset_clips cannot be combined with heldout split manifests")
    if config.debug_allow_pseudo_weight is not None and (
        config.debug_allow_pseudo_weight <= 0.0 or config.debug_allow_pseudo_weight > 1.0
    ):
        raise ValueError("debug_allow_pseudo_weight: must be in (0, 1]")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


__all__ = ["SMGTTinyTrainConfig", "run_smgt_tiny_training"]
