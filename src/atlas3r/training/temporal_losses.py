"""Temporal real-RGBD losses and diagnostic metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


def temporal_geometry_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    depth_loss: str = "metric_l1",
    depth_weight: float = 1.0,
    sigma_nll_weight: float = 0.05,
    confidence_weight: float = 0.01,
    relative_translation_weight: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Compute masked center-depth and relative-translation losses for temporal-v0."""

    pred_depth = _required_tensor(prediction, "center_depth_m")
    pred_sigma = _required_tensor(prediction, "center_depth_sigma_m").clamp(min=1e-4, max=10.0)
    pred_confidence = _required_tensor(prediction, "center_confidence")
    pred_translation = _required_tensor(prediction, "relative_translation_center_from_camera")
    target_depth = _required_tensor(target, "center_depth_m")
    valid_mask = _required_tensor(target, "center_valid_depth_mask").to(dtype=_TORCH.bool)
    target_confidence = _required_tensor(target, "center_confidence")
    relative_T = _required_tensor(target, "relative_T_center_camera")
    target_translation = relative_T[:, :, :3, 3]
    _validate_shapes(
        pred_depth=pred_depth,
        pred_sigma=pred_sigma,
        pred_confidence=pred_confidence,
        target_depth=target_depth,
        valid_mask=valid_mask,
        target_confidence=target_confidence,
        pred_translation=pred_translation,
        target_translation=target_translation,
    )
    valid_count = int(valid_mask.sum().detach().cpu().item())
    if valid_count <= 0:
        raise ValueError("center_valid_depth_mask: batch has no valid depth pixels")
    valid_pred_depth = pred_depth[valid_mask]
    valid_target_depth = target_depth[valid_mask]
    abs_depth_error = (valid_pred_depth - valid_target_depth).abs()
    valid_sigma = pred_sigma[valid_mask]
    if depth_loss == "metric_l1":
        loss_depth = _F.smooth_l1_loss(valid_pred_depth, valid_target_depth)
    elif depth_loss == "log_l1":
        loss_depth = _F.smooth_l1_loss(
            _TORCH.log(valid_pred_depth.clamp(min=1e-4)),
            _TORCH.log(valid_target_depth.clamp(min=1e-4)),
        )
    else:
        raise ValueError("depth_loss: must be 'metric_l1' or 'log_l1'")
    loss_sigma_nll = (abs_depth_error / valid_sigma + valid_sigma.log()).mean()
    loss_confidence = _F.mse_loss(pred_confidence, target_confidence)
    loss_translation = _F.smooth_l1_loss(pred_translation, target_translation)
    loss_total = (
        depth_weight * loss_depth
        + sigma_nll_weight * loss_sigma_nll
        + confidence_weight * loss_confidence
        + relative_translation_weight * loss_translation
    )
    _raise_if_not_finite("loss_total", loss_total)
    translation_error = _TORCH.linalg.norm(pred_translation - target_translation, dim=-1)
    metrics = {
        "loss_total": _float_item(loss_total),
        "loss_depth": _float_item(loss_depth),
        "loss_sigma_nll": _float_item(loss_sigma_nll),
        "loss_confidence": _float_item(loss_confidence),
        "loss_relative_translation": _float_item(loss_translation),
        "center_depth_mae_m": _float_item(abs_depth_error.mean()),
        "center_depth_rmse_m": _float_item(_TORCH.sqrt((abs_depth_error.square()).mean())),
        "center_depth_absrel": _float_item(
            (abs_depth_error / valid_target_depth.clamp(min=1e-6)).mean()
        ),
        "center_depth_within_1mm_percent": _within_percent(abs_depth_error, 0.001),
        "center_depth_within_5mm_percent": _within_percent(abs_depth_error, 0.005),
        "center_depth_within_1cm_percent": _within_percent(abs_depth_error, 0.01),
        "center_depth_within_5cm_percent": _within_percent(abs_depth_error, 0.05),
        "center_depth_within_10cm_percent": _within_percent(abs_depth_error, 0.10),
        "relative_translation_mean_m": _float_item(translation_error.mean()),
        "relative_translation_median_m": _float_item(translation_error.median()),
        "relative_translation_p95_m": _float_item(_TORCH.quantile(translation_error, 0.95)),
        "relative_rotation_mean_deg": 0.0,
        "valid_depth_pixels": float(valid_count),
    }
    for key, value in metrics.items():
        if not _TORCH.isfinite(_TORCH.tensor(value)):
            raise ValueError(f"{key}: metric is not finite")
    return loss_total, metrics


def _validate_shapes(**tensors: Any) -> None:
    pred_depth = tensors["pred_depth"]
    target_depth = tensors["target_depth"]
    if pred_depth.shape != target_depth.shape:
        raise ValueError("center_depth_m: prediction and target shapes must match")
    for key in ("pred_sigma", "pred_confidence", "valid_mask", "target_confidence"):
        if tensors[key].shape != target_depth.shape:
            raise ValueError(f"{key}: shape must match center_depth_m")
    if tensors["pred_translation"].shape != tensors["target_translation"].shape:
        raise ValueError("relative translation prediction and target shapes must match")


def _required_tensor(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{key}: required tensor is missing")
    tensor = mapping[key]
    if not _TORCH.is_tensor(tensor):
        raise ValueError(f"{key}: must be a torch.Tensor")
    if not bool(_TORCH.isfinite(tensor).all()):
        raise ValueError(f"{key}: tensor contains NaN or Inf")
    return tensor


def _within_percent(abs_error: Any, threshold_m: float) -> float:
    return _float_item((abs_error <= threshold_m).to(dtype=abs_error.dtype).mean() * 100.0)


def _raise_if_not_finite(field_name: str, tensor: Any) -> None:
    if not bool(_TORCH.isfinite(tensor).all()):
        raise ValueError(f"{field_name}: tensor contains NaN or Inf")


def _float_item(tensor: Any) -> float:
    return float(tensor.detach().cpu().item())


__all__ = [
    "temporal_geometry_loss",
]
