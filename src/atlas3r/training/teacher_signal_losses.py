"""Confidence and uncertainty weighted temporal teacher-signal losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


@dataclass(frozen=True)
class TeacherSignalLossConfig:
    min_sigma_m: float = 0.001
    max_sigma_m: float = 1.0
    max_pixel_weight: float = 100.0
    measured_teacher_weight: float = 1.0
    pseudo_teacher_weight: float = 0.25
    log_depth_weight: float = 1.0
    sigma_nll_weight: float = 0.05
    confidence_weight: float = 0.02
    pointmap_weight: float = 0.05
    relative_translation_weight: float = 0.1

    def to_json(self) -> dict[str, object]:
        return {
            "min_sigma_m": self.min_sigma_m,
            "max_sigma_m": self.max_sigma_m,
            "max_pixel_weight": self.max_pixel_weight,
            "measured_teacher_weight": self.measured_teacher_weight,
            "pseudo_teacher_weight": self.pseudo_teacher_weight,
            "log_depth_weight": self.log_depth_weight,
            "sigma_nll_weight": self.sigma_nll_weight,
            "confidence_weight": self.confidence_weight,
            "pointmap_weight": self.pointmap_weight,
            "relative_translation_weight": self.relative_translation_weight,
        }


def teacher_signal_temporal_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    config: TeacherSignalLossConfig | None = None,
) -> tuple[Any, dict[str, float]]:
    """Compute weighted temporal depth, uncertainty, confidence, and pose losses."""

    cfg = config or TeacherSignalLossConfig()
    _validate_config(cfg)
    pred_depth = _required_tensor(prediction, "depth_m").clamp(min=1e-4)
    pred_sigma = _required_tensor(prediction, "depth_sigma_m").clamp(
        min=cfg.min_sigma_m,
        max=cfg.max_sigma_m,
    )
    pred_confidence = _required_tensor(prediction, "confidence").clamp(0.0, 1.0)
    target_depth = _required_tensor(target, "depth_m").clamp(min=1e-4)
    target_sigma = _required_tensor(target, "depth_sigma_m")
    target_confidence = _required_tensor(target, "confidence").clamp(0.0, 1.0)
    valid_mask = _required_tensor(target, "valid_mask").to(dtype=_TORCH.bool)
    teacher_is_measured = _required_tensor(target, "teacher_is_measured").to(dtype=_TORCH.bool)
    _validate_pixel_shapes(
        pred_depth,
        pred_sigma,
        pred_confidence,
        target_depth,
        target_sigma,
        target_confidence,
        valid_mask,
    )
    valid_count = int(valid_mask.sum().detach().cpu().item())
    if valid_count <= 0:
        raise ValueError("valid_mask: batch has no valid teacher pixels")
    weights = _normalized_pixel_weights(
        target_sigma=target_sigma,
        target_confidence=target_confidence,
        valid_mask=valid_mask,
        teacher_is_measured=teacher_is_measured,
        config=cfg,
    )
    if float(weights.sum().detach().cpu().item()) <= 0.0:
        raise ValueError("pixel_weight: batch has no positive teacher weights")

    log_depth_loss = _weighted_mean(
        _F.smooth_l1_loss(
            _TORCH.log(pred_depth),
            _TORCH.log(target_depth),
            reduction="none",
        ),
        weights,
    )
    residual = pred_depth - target_depth
    combined_var = (
        pred_sigma.square()
        + target_sigma.clamp(
            min=cfg.min_sigma_m,
            max=cfg.max_sigma_m,
        ).square()
    )
    sigma_nll = _weighted_mean(
        0.5 * residual.square() / combined_var + 0.5 * combined_var.log(), weights
    )
    confidence_target = _TORCH.where(
        valid_mask, target_confidence, _TORCH.zeros_like(target_confidence)
    )
    confidence_loss = _F.smooth_l1_loss(pred_confidence, confidence_target)
    metric_depth_loss = _weighted_mean(
        _F.smooth_l1_loss(pred_depth, target_depth, reduction="none"),
        weights,
    )

    pointmap_loss = pred_depth.new_tensor(0.0)
    if "pointmap_camera_m" in target:
        pointmap_loss = _pointmap_loss(pred_depth, target, weights)

    translation_loss = pred_depth.new_tensor(0.0)
    translation_error = pred_depth.new_zeros((teacher_is_measured.shape[0],))
    if "relative_translation_center_from_camera" in prediction and (
        "relative_translation_center_from_camera" in target
    ):
        pred_translation = _required_tensor(prediction, "relative_translation_center_from_camera")
        target_translation = _required_tensor(target, "relative_translation_center_from_camera")
        if pred_translation.shape != target_translation.shape:
            raise ValueError("relative_translation_center_from_camera: shapes must match")
        translation_loss = _F.smooth_l1_loss(pred_translation, target_translation)
        translation_error = _TORCH.linalg.norm(pred_translation - target_translation, dim=-1)

    loss_total = (
        cfg.log_depth_weight * log_depth_loss
        + cfg.sigma_nll_weight * sigma_nll
        + cfg.confidence_weight * confidence_loss
        + cfg.pointmap_weight * pointmap_loss
        + cfg.relative_translation_weight * translation_loss
    )
    _raise_if_not_finite("loss_total", loss_total)
    abs_error = (pred_depth - target_depth).abs()[valid_mask]
    target_valid_depth = target_depth[valid_mask]
    metrics = {
        "loss_total": _float_item(loss_total),
        "loss_log_depth": _float_item(log_depth_loss),
        "loss_metric_depth": _float_item(metric_depth_loss),
        "loss_sigma_nll": _float_item(sigma_nll),
        "loss_confidence": _float_item(confidence_loss),
        "loss_pointmap_camera": _float_item(pointmap_loss),
        "loss_relative_translation": _float_item(translation_loss),
        "depth_rmse_m": _float_item(_TORCH.sqrt(abs_error.square().mean())),
        "depth_mae_m": _float_item(abs_error.mean()),
        "depth_absrel": _float_item((abs_error / target_valid_depth.clamp(min=1e-6)).mean()),
        "within_1mm_percent": _within_percent(abs_error, 0.001),
        "within_5mm_percent": _within_percent(abs_error, 0.005),
        "within_1cm_percent": _within_percent(abs_error, 0.01),
        "within_5cm_percent": _within_percent(abs_error, 0.05),
        "within_10cm_percent": _within_percent(abs_error, 0.10),
        "valid_depth_pixels": float(valid_count),
        "positive_weight_pixels": float(int((weights > 0.0).sum().detach().cpu().item())),
        "mean_normalized_pixel_weight": _float_item(weights[weights > 0.0].mean()),
        "measured_batch_count": float(int(teacher_is_measured.sum().detach().cpu().item())),
        "pseudo_batch_count": float(int((~teacher_is_measured).sum().detach().cpu().item())),
        "relative_translation_mean_m": _float_item(translation_error.mean()),
    }
    for key, value in metrics.items():
        if not _TORCH.isfinite(_TORCH.tensor(value)):
            raise ValueError(f"{key}: metric is not finite")
    return loss_total, metrics


def _normalized_pixel_weights(
    *,
    target_sigma: Any,
    target_confidence: Any,
    valid_mask: Any,
    teacher_is_measured: Any,
    config: TeacherSignalLossConfig,
) -> Any:
    sigma = target_sigma.clamp(min=config.min_sigma_m, max=config.max_sigma_m)
    weights = target_confidence / sigma.square()
    weights = _TORCH.where(valid_mask, weights, _TORCH.zeros_like(weights))
    weights = weights.clamp(max=config.max_pixel_weight)
    teacher_weight = _TORCH.where(
        teacher_is_measured,
        _TORCH.full_like(teacher_is_measured, config.measured_teacher_weight, dtype=weights.dtype),
        _TORCH.full_like(teacher_is_measured, config.pseudo_teacher_weight, dtype=weights.dtype),
    )
    while teacher_weight.ndim < weights.ndim:
        teacher_weight = teacher_weight.unsqueeze(-1)
    weights = weights * teacher_weight
    positive = weights > 0.0
    if bool(positive.any()):
        weights = weights / weights[positive].mean().clamp(min=1e-12)
    return weights


def _pointmap_loss(pred_depth: Any, target: Mapping[str, Any], weights: Any) -> Any:
    target_pointmap = _required_tensor(target, "pointmap_camera_m")
    intrinsics = _required_tensor(target, "intrinsics")
    if target_pointmap.shape[:3] != (pred_depth.shape[0], pred_depth.shape[1], 3):
        raise ValueError("pointmap_camera_m: must have shape B,T,3,H,W")
    predicted_pointmap = _unproject_depth_camera(pred_depth, intrinsics)
    point_error = _F.smooth_l1_loss(predicted_pointmap, target_pointmap, reduction="none")
    point_weights = weights.expand_as(pred_depth).expand_as(point_error)
    return _weighted_mean(point_error, point_weights)


def _unproject_depth_camera(depth: Any, intrinsics: Any) -> Any:
    batch_size, clip_length, _channels, height, width = depth.shape
    flat_depth = depth.reshape(batch_size * clip_length, 1, height, width)
    flat_intrinsics = intrinsics.reshape(batch_size * clip_length, 3, 3)
    dtype = depth.dtype
    device = depth.device
    u = _TORCH.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)
    v = _TORCH.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
    fx = flat_intrinsics[:, 0, 0].view(-1, 1, 1, 1).to(dtype=dtype).clamp(min=1e-6)
    fy = flat_intrinsics[:, 1, 1].view(-1, 1, 1, 1).to(dtype=dtype).clamp(min=1e-6)
    cx = flat_intrinsics[:, 0, 2].view(-1, 1, 1, 1).to(dtype=dtype)
    cy = flat_intrinsics[:, 1, 2].view(-1, 1, 1, 1).to(dtype=dtype)
    x = (u - cx) / fx * flat_depth
    y = (v - cy) / fy * flat_depth
    pointmap = _TORCH.cat([x, y, flat_depth], dim=1)
    return pointmap.reshape(batch_size, clip_length, 3, height, width)


def _weighted_mean(values: Any, weights: Any) -> Any:
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


def _validate_config(config: TeacherSignalLossConfig) -> None:
    if config.min_sigma_m <= 0.0 or config.max_sigma_m <= config.min_sigma_m:
        raise ValueError("sigma clamps: require 0 < min_sigma_m < max_sigma_m")
    if config.max_pixel_weight <= 0.0:
        raise ValueError("max_pixel_weight: must be positive")
    if config.measured_teacher_weight <= 0.0 or config.pseudo_teacher_weight <= 0.0:
        raise ValueError("teacher weights: must be positive")


def _validate_pixel_shapes(*tensors: Any) -> None:
    expected = tensors[0].shape
    if len(expected) != 5:
        raise ValueError("depth tensors: must have shape B,T,1,H,W")
    for tensor in tensors[1:]:
        if tensor.shape != expected:
            raise ValueError("teacher-signal pixel tensors must share shape B,T,1,H,W")


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
    "TeacherSignalLossConfig",
    "teacher_signal_temporal_loss",
]
