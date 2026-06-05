"""Evaluation and confidence/sigma calibration helpers for SMGT-small-v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from atlas3r.models.smgt.small_v2_checkpoint import load_smgt_small_v2_checkpoint
from atlas3r.training.smgt_tiny_eval import move_batch_to_device
from atlas3r.training.smgt_v2_dataset import SMGTV2MixedTemporalDataset
from atlas3r.training.smgt_v2_losses import SMGTV2LossConfig, smgt_v2_loss
from atlas3r.training.torch_runtime import require_torch
from atlas3r.training.tum_rgbd_artifacts import write_json


@dataclass(frozen=True)
class SMGTV2CalibrationConfig:
    checkpoint: Path
    cache: Path
    output: Path
    device: str = "cuda"
    batch_size: int = 2
    target_min_mapped_ratio: float = 0.10
    target_max_mapped_ratio: float = 0.70


def evaluate_smgt_v2(
    model: Any,
    loader: Any,
    *,
    device: str,
    loss_config: SMGTV2LossConfig,
) -> dict[str, float]:
    torch = require_torch()
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["K"])
            _loss, metrics = smgt_v2_loss(prediction, batch, config=loss_config)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    if batches <= 0:
        raise ValueError("validation: no batches were produced")
    return {key: value / float(batches) for key, value in totals.items()}


def run_smgt_v2_gate_calibration(config: SMGTV2CalibrationConfig) -> dict[str, object]:
    _validate_calibration_config(config)
    torch = require_torch()
    loaded = load_smgt_small_v2_checkpoint(config.checkpoint, device=config.device)
    dataset = SMGTV2MixedTemporalDataset(measured_caches=[config.cache])
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    records = _collect_calibration_arrays(loaded.model, loader, device=loaded.device)
    calibration = select_smgt_v2_gate_thresholds(
        records["confidence"],
        records["sigma"],
        records["depth_pred"],
        records["depth_target"],
        records["valid"],
        target_min_mapped_ratio=config.target_min_mapped_ratio,
        target_max_mapped_ratio=config.target_max_mapped_ratio,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    write_json(config.output / "calibration.json", calibration)
    _write_calibration_report(config.output / "calibration_report.md", calibration)
    return {
        "format_name": "atlas3r_smgt_v2_gate_calibration_run",
        "checkpoint": str(config.checkpoint),
        "cache": str(config.cache),
        "output": str(config.output),
        "calibration": calibration,
    }


def select_smgt_v2_gate_thresholds(
    confidence: npt.NDArray[Any],
    sigma: npt.NDArray[Any],
    depth_pred: npt.NDArray[Any],
    depth_target: npt.NDArray[Any],
    valid: npt.NDArray[Any],
    *,
    target_min_mapped_ratio: float = 0.10,
    target_max_mapped_ratio: float = 0.70,
) -> dict[str, object]:
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    sig = np.asarray(sigma, dtype=np.float32).reshape(-1)
    pred = np.asarray(depth_pred, dtype=np.float32).reshape(-1)
    target = np.asarray(depth_target, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(valid, dtype=np.bool_).reshape(-1)
    finite = (
        valid_mask
        & np.isfinite(conf)
        & np.isfinite(sig)
        & np.isfinite(pred)
        & np.isfinite(target)
        & (target > 0.0)
        & (pred > 0.0)
        & (sig >= 0.0)
    )
    if not np.any(finite):
        raise ValueError("calibration: no finite valid pixels")
    abs_error = np.abs(pred - target)
    absrel = abs_error / np.maximum(target, 1e-6)
    candidates: list[dict[str, object]] = []
    conf_values = np.quantile(conf[finite], np.linspace(0.05, 0.95, 10))
    sigma_values = np.quantile(sig[finite], np.linspace(0.10, 0.95, 10))
    for conf_threshold in conf_values:
        for sigma_threshold in sigma_values:
            mapped = finite & (conf >= conf_threshold) & (sig <= sigma_threshold)
            ratio = float(np.count_nonzero(mapped) / np.count_nonzero(finite))
            if ratio <= 0.0:
                continue
            mapped_error = float(np.mean(absrel[mapped]))
            rejected = finite & ~mapped
            rejected_error = float(np.mean(absrel[rejected])) if np.any(rejected) else mapped_error
            in_target = target_min_mapped_ratio <= ratio <= target_max_mapped_ratio
            score = mapped_error + 0.25 * abs(ratio - 0.40)
            if not in_target:
                score += 1.0 + abs(
                    ratio - np.clip(ratio, target_min_mapped_ratio, target_max_mapped_ratio)
                )
            candidates.append(
                {
                    "confidence_threshold": float(conf_threshold),
                    "max_sigma_m": float(sigma_threshold),
                    "mapped_pixel_ratio": ratio,
                    "mapped_absrel": mapped_error,
                    "rejected_absrel": rejected_error,
                    "mapped_pixels_lower_error_than_rejected": bool(mapped_error <= rejected_error),
                    "score": float(score),
                }
            )
    if not candidates:
        raise ValueError("calibration: no valid threshold candidates")
    best = min(candidates, key=lambda item: float(cast(Any, item["score"])))
    brier = float(np.mean((conf[finite] - (absrel[finite] < 0.10).astype(np.float32)) ** 2))
    high_error = finite & (absrel >= np.quantile(absrel[finite], 0.75))
    selected = (
        finite
        & (conf >= float(cast(Any, best["confidence_threshold"])))
        & (sig <= float(cast(Any, best["max_sigma_m"])))
    )
    random_reject_rate = float(np.count_nonzero(high_error) / np.count_nonzero(finite))
    selected_high_error_rate = float(
        np.count_nonzero(high_error & selected) / max(np.count_nonzero(selected), 1)
    )
    return {
        "format_name": "atlas3r_smgt_v2_gate_calibration",
        "format_version": 1,
        "confidence_threshold": best["confidence_threshold"],
        "max_sigma_m": best["max_sigma_m"],
        "mapped_pixel_ratio": best["mapped_pixel_ratio"],
        "target_mapped_pixel_ratio_min": target_min_mapped_ratio,
        "target_mapped_pixel_ratio_max": target_max_mapped_ratio,
        "mapped_absrel": best["mapped_absrel"],
        "rejected_absrel": best["rejected_absrel"],
        "mapped_pixels_lower_error_than_rejected": best["mapped_pixels_lower_error_than_rejected"],
        "confidence_brier": brier,
        "selected_high_error_rate": selected_high_error_rate,
        "random_high_error_rate": random_reject_rate,
        "rejects_high_error_better_than_random": bool(
            selected_high_error_rate < random_reject_rate
        ),
        "candidate_count": len(candidates),
    }


def _collect_calibration_arrays(
    model: Any, loader: Any, *, device: str
) -> dict[str, npt.NDArray[np.float32]]:
    torch = require_torch()
    arrays: dict[str, list[npt.NDArray[np.float32]]] = {
        "confidence": [],
        "sigma": [],
        "depth_pred": [],
        "depth_target": [],
        "valid": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch["images_rgb"], batch["K"])
            target = batch["target"]
            arrays["confidence"].append(_numpy(prediction["confidence"]))
            arrays["sigma"].append(_numpy(prediction["depth_sigma_m"]))
            arrays["depth_pred"].append(_numpy(prediction["depth_m"]))
            arrays["depth_target"].append(_numpy(target["depth_m"]))
            arrays["valid"].append(_numpy(target["valid_mask"]).astype(np.float32))
    return {key: np.concatenate(value, axis=0) for key, value in arrays.items()}


def _write_calibration_report(path: Path, calibration: Mapping[str, object]) -> None:
    lines = [
        "# SMGT V2 Gate Calibration",
        "",
        f"- Confidence threshold: `{calibration['confidence_threshold']}`",
        f"- Max sigma m: `{calibration['max_sigma_m']}`",
        f"- Mapped pixel ratio: `{calibration['mapped_pixel_ratio']}`",
        f"- Mapped AbsRel: `{calibration['mapped_absrel']}`",
        f"- Rejected AbsRel: `{calibration['rejected_absrel']}`",
        f"- Confidence Brier: `{calibration['confidence_brier']}`",
        "- Rejects high-error better than random: "
        f"`{calibration['rejects_high_error_better_than_random']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_calibration_config(config: SMGTV2CalibrationConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size: must be positive")
    if not 0.0 < config.target_min_mapped_ratio < config.target_max_mapped_ratio <= 1.0:
        raise ValueError("target mapped ratio bounds must satisfy 0 < min < max <= 1")


def _numpy(value: Any) -> npt.NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "SMGTV2CalibrationConfig",
    "evaluate_smgt_v2",
    "run_smgt_v2_gate_calibration",
    "select_smgt_v2_gate_thresholds",
]
