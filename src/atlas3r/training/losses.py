"""Supervised losses for the synthetic training MVP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


def synthetic_depth_pose_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    depth_weight: float = 1.0,
    sigma_nll_weight: float = 0.05,
    confidence_weight: float = 0.01,
    pose_center_weight: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Compute finite supervised synthetic depth, uncertainty, confidence, and pose losses."""

    pred_depth = _required_tensor(prediction, "depth_m")
    pred_sigma = _required_tensor(prediction, "depth_sigma_m").clamp(min=1e-4, max=10.0)
    pred_confidence = _required_tensor(prediction, "confidence")
    pred_center = _required_tensor(prediction, "camera_center_world_m")
    target_depth = _required_tensor(target, "depth_m")
    target_confidence = _required_tensor(target, "confidence")
    target_center = _required_tensor(target, "camera_center_world_m")

    abs_depth_error = (pred_depth - target_depth).abs()
    loss_depth = _F.smooth_l1_loss(pred_depth, target_depth)
    loss_sigma_nll = (abs_depth_error / pred_sigma + pred_sigma.log()).mean()
    loss_confidence = _F.mse_loss(pred_confidence, target_confidence)
    loss_pose_center = _F.l1_loss(pred_center, target_center)
    loss_total = (
        depth_weight * loss_depth
        + sigma_nll_weight * loss_sigma_nll
        + confidence_weight * loss_confidence
        + pose_center_weight * loss_pose_center
    )
    _raise_if_not_finite("loss_total", loss_total)
    depth_rmse = _TORCH.sqrt(_TORCH.mean((pred_depth - target_depth) ** 2))
    metrics = {
        "loss_total": _float_item(loss_total),
        "loss_depth": _float_item(loss_depth),
        "loss_sigma_nll": _float_item(loss_sigma_nll),
        "loss_confidence": _float_item(loss_confidence),
        "loss_pose_center": _float_item(loss_pose_center),
        "depth_mae_m": _float_item(abs_depth_error.mean()),
        "depth_rmse_m": _float_item(depth_rmse),
        "pose_center_mae_m": _float_item((pred_center - target_center).abs().mean()),
    }
    for key, value in metrics.items():
        if not _TORCH.isfinite(_TORCH.tensor(value)):
            raise ValueError(f"{key}: metric is not finite")
    return loss_total, metrics


def masked_rgbd_depth_pose_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    depth_weight: float = 1.0,
    sigma_nll_weight: float = 0.05,
    confidence_weight: float = 0.01,
    pose_center_weight: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Compute masked real RGB-D depth, uncertainty, confidence, and pose losses."""

    pred_depth = _required_tensor(prediction, "depth_m")
    pred_sigma = _required_tensor(prediction, "depth_sigma_m").clamp(min=1e-4, max=10.0)
    pred_confidence = _required_tensor(prediction, "confidence")
    pred_center = _required_tensor(prediction, "camera_center_world_m")
    target_depth = _required_tensor(target, "depth_m")
    valid_mask = _required_tensor(target, "valid_depth_mask").to(dtype=_TORCH.bool)
    target_center = target.get("camera_center_world_m")
    if not _TORCH.is_tensor(target_center):
        target_center = None

    if pred_depth.shape != target_depth.shape:
        raise ValueError("depth_m: prediction and target shapes must match")
    if pred_sigma.shape != target_depth.shape:
        raise ValueError("depth_sigma_m: prediction and target depth shapes must match")
    if pred_confidence.shape != target_depth.shape:
        raise ValueError("confidence: prediction and target depth shapes must match")
    if valid_mask.shape != target_depth.shape:
        raise ValueError("valid_depth_mask: shape must match target depth_m")
    valid_count = int(valid_mask.sum().detach().cpu().item())
    if valid_count <= 0:
        raise ValueError("valid_depth_mask: batch has no valid depth pixels")

    valid_pred_depth = pred_depth[valid_mask]
    valid_target_depth = target_depth[valid_mask]
    valid_abs_error = (valid_pred_depth - valid_target_depth).abs()
    valid_sigma = pred_sigma[valid_mask]
    loss_depth = _F.smooth_l1_loss(valid_pred_depth, valid_target_depth)
    loss_sigma_nll = (valid_abs_error / valid_sigma + valid_sigma.log()).mean()
    confidence_target = valid_mask.to(dtype=pred_confidence.dtype)
    loss_confidence = _F.mse_loss(pred_confidence, confidence_target)
    if target_center is None:
        loss_pose_center = pred_center.new_tensor(0.0)
        pose_center_mae = pred_center.new_tensor(0.0)
    else:
        target_center = _required_tensor(target, "camera_center_world_m")
        loss_pose_center = _F.l1_loss(pred_center, target_center)
        pose_center_mae = (pred_center - target_center).abs().mean()
    loss_total = (
        depth_weight * loss_depth
        + sigma_nll_weight * loss_sigma_nll
        + confidence_weight * loss_confidence
        + pose_center_weight * loss_pose_center
    )
    _raise_if_not_finite("loss_total", loss_total)
    depth_rmse = _TORCH.sqrt(_TORCH.mean((valid_pred_depth - valid_target_depth) ** 2))
    depth_absrel = (valid_abs_error / valid_target_depth.clamp(min=1e-6)).mean()
    metrics = {
        "loss_total": _float_item(loss_total),
        "loss_depth": _float_item(loss_depth),
        "loss_sigma_nll": _float_item(loss_sigma_nll),
        "loss_confidence": _float_item(loss_confidence),
        "loss_pose_center": _float_item(loss_pose_center),
        "depth_mae_m": _float_item(valid_abs_error.mean()),
        "depth_rmse_m": _float_item(depth_rmse),
        "depth_absrel": _float_item(depth_absrel),
        "pose_center_mae_m": _float_item(pose_center_mae),
        "valid_depth_pixels": float(valid_count),
    }
    for key, value in metrics.items():
        if not _TORCH.isfinite(_TORCH.tensor(value)):
            raise ValueError(f"{key}: metric is not finite")
    return loss_total, metrics


def _required_tensor(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{key}: required tensor is missing")
    tensor = mapping[key]
    if not _TORCH.is_tensor(tensor):
        raise ValueError(f"{key}: must be a torch.Tensor")
    if not bool(_TORCH.isfinite(tensor).all()):
        raise ValueError(f"{key}: tensor contains NaN or Inf")
    return tensor


def _raise_if_not_finite(field_name: str, tensor: Any) -> None:
    if not bool(_TORCH.isfinite(tensor).all()):
        raise ValueError(f"{field_name}: tensor contains NaN or Inf")


def _float_item(tensor: Any) -> float:
    return float(tensor.detach().cpu().item())


__all__ = [
    "masked_rgbd_depth_pose_loss",
    "synthetic_depth_pose_loss",
]
