"""Stable SMGT-tiny training metric summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

STABLE_WINDOW_METRICS = (
    "depth_absrel",
    "depth_rmse_m",
    "pose_center_mean_m",
    "pose_relative_translation_mean_m",
    "pose_rotation_mean_deg",
    "confidence_brier",
    "constant_depth_baseline_absrel",
    "depth_absrel_vs_constant_baseline_ratio",
    "no_motion_pose_baseline_center_mean_m",
    "pose_center_vs_no_motion_baseline_ratio",
    "loss_total",
)


def summarize_smgt_tiny_training_metrics(history: Sequence[dict[str, float]]) -> dict[str, object]:
    """Summarize training using sign-stable first/final metric windows."""

    if not history:
        return {
            "window_size": 0,
            "first_100": {},
            "final_100": {},
            "loss_total_delta": None,
            "loss_total_reduction_absolute": None,
            "loss_total_decrease_percent": None,
            "loss_total_percent_meaningful": False,
            "loss_total_crossed_zero": False,
            "training_quality_pass": False,
            "training_quality_basis": "no history",
            "first_window_loss_total_mean": None,
            "final_window_loss_total_mean": None,
        }
    window = min(100, len(history))
    first = _window_mean(history[:window])
    final = _window_mean(history[-window:])
    first_loss = first.get("loss_total")
    final_loss = final.get("loss_total")
    loss_delta = None if first_loss is None or final_loss is None else final_loss - first_loss
    loss_reduction = None if loss_delta is None else -loss_delta
    crossed_zero = _loss_crossed_zero(history)
    percent = None
    percent_meaningful = False
    if first_loss is not None and final_loss is not None and first_loss > 0.0 and final_loss >= 0.0:
        percent = float((first_loss - final_loss) / first_loss * 100.0)
        percent_meaningful = True
    quality_pass = _training_quality_pass(first, final)
    return {
        "window_size": window,
        "first_100": first,
        "final_100": final,
        "loss_total_delta": loss_delta,
        "loss_total_reduction_absolute": loss_reduction,
        "loss_total_decrease_percent": percent,
        "loss_total_percent_meaningful": percent_meaningful,
        "loss_total_crossed_zero": crossed_zero,
        "training_quality_pass": quality_pass,
        "training_quality_basis": (
            "final depth_absrel/depth_rmse_m/pose_center_mean_m/"
            "pose_relative_translation_mean_m are no worse than first window"
        ),
        "first_window_loss_total_mean": first_loss,
        "final_window_loss_total_mean": final_loss,
        "depth_first_window_mean": first.get("depth_absrel"),
        "depth_final_window_mean": final.get("depth_absrel"),
        "pose_first_window_mean": first.get("pose_relative_translation_mean_m"),
        "pose_final_window_mean": final.get("pose_relative_translation_mean_m"),
    }


def _window_mean(window: Sequence[dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in STABLE_WINDOW_METRICS:
        values = [float(item[key]) for item in window if key in item and np.isfinite(item[key])]
        if values:
            output[key] = float(np.mean(values))
    return output


def _loss_crossed_zero(history: Sequence[dict[str, float]]) -> bool:
    values = [float(item["loss_total"]) for item in history if "loss_total" in item]
    return bool(values and min(values) < 0.0 < max(values))


def _training_quality_pass(first: dict[str, float], final: dict[str, float]) -> bool:
    required = (
        "depth_absrel",
        "depth_rmse_m",
        "pose_center_mean_m",
        "pose_relative_translation_mean_m",
    )
    return all(key in first and key in final and final[key] <= first[key] for key in required)


__all__ = ["STABLE_WINDOW_METRICS", "summarize_smgt_tiny_training_metrics"]
