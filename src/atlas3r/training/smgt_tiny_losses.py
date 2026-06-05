"""Losses for training the diagnostic SMGT-tiny student."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from atlas3r.models.smgt.geometry import unproject_depth_camera
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


@dataclass(frozen=True)
class SMGTTinyLossConfig:
    depth_log_weight: float = 1.0
    depth_si_weight: float = 0.1
    uncertainty_weight: float = 0.05
    confidence_weight: float = 0.05
    pose_relative_weight: float = 0.25
    pose_center_weight: float = 0.1
    pointmap_weight: float = 0.05
    smoothness_weight: float = 0.005
    dynamic_weight: float = 0.002
    min_depth_m: float = 1e-4
    min_sigma_m: float = 1e-3
    max_sigma_m: float = 5.0

    def to_json(self) -> dict[str, object]:
        return {
            "depth_log_weight": self.depth_log_weight,
            "depth_si_weight": self.depth_si_weight,
            "uncertainty_weight": self.uncertainty_weight,
            "confidence_weight": self.confidence_weight,
            "pose_relative_weight": self.pose_relative_weight,
            "pose_center_weight": self.pose_center_weight,
            "pointmap_weight": self.pointmap_weight,
            "smoothness_weight": self.smoothness_weight,
            "dynamic_weight": self.dynamic_weight,
            "min_depth_m": self.min_depth_m,
            "min_sigma_m": self.min_sigma_m,
            "max_sigma_m": self.max_sigma_m,
        }


def smgt_tiny_loss(
    prediction: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    config: SMGTTinyLossConfig | None = None,
) -> tuple[Any, dict[str, float]]:
    cfg = config or SMGTTinyLossConfig()
    target = batch["target"]
    if not isinstance(target, Mapping):
        raise ValueError("batch.target: must be a mapping")
    pred_depth = _tensor(prediction, "depth_m").float().clamp(min=cfg.min_depth_m)
    pred_sigma = (
        _tensor(prediction, "depth_sigma_m").float().clamp(cfg.min_sigma_m, cfg.max_sigma_m)
    )
    pred_conf = _tensor(prediction, "confidence").float().clamp(0.0, 1.0)
    pred_dynamic = _tensor(prediction, "dynamic_probability").float().clamp(0.0, 1.0)
    target_depth = _tensor(target, "depth_m").float().clamp(min=cfg.min_depth_m)
    target_conf = _tensor(target, "confidence").float().clamp(0.0, 1.0)
    valid = _tensor(target, "valid_mask").to(dtype=_TORCH.bool)
    if pred_depth.shape != target_depth.shape or valid.shape != target_depth.shape:
        raise ValueError("depth, target depth, and valid mask must share B,T,H,W shape")
    pixel_weight = _pixel_weights(batch, valid, target_conf)
    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count <= 0:
        raise ValueError("valid_mask: batch has no valid target pixels")
    log_error = _TORCH.log(pred_depth) - _TORCH.log(target_depth)
    depth_log = _weighted_mean(
        _F.smooth_l1_loss(log_error, _TORCH.zeros_like(log_error), reduction="none"), pixel_weight
    )
    depth_si = _scale_invariant_log_loss(log_error, pixel_weight)
    depth_residual = pred_depth - target_depth
    uncertainty = _weighted_mean(depth_residual.abs() / pred_sigma + pred_sigma.log(), pixel_weight)
    confidence_target = _TORCH.where(valid, target_conf, _TORCH.zeros_like(target_conf))
    confidence = _bce_probability(pred_conf, confidence_target)
    confidence_brier = (pred_conf - confidence_target).square().mean()
    pointmap = _pointmap_loss(prediction, batch, target_depth, pixel_weight)
    smoothness = _edge_aware_smoothness(pred_depth, _tensor(batch, "images_rgb").float())
    dynamic = _bce_probability(pred_dynamic, _TORCH.zeros_like(pred_dynamic))
    pose_relative, pose_center, pose_metrics = _pose_losses(prediction, target, batch)
    total = (
        cfg.depth_log_weight * depth_log
        + cfg.depth_si_weight * depth_si
        + cfg.uncertainty_weight * uncertainty
        + cfg.confidence_weight * confidence
        + cfg.pose_relative_weight * pose_relative
        + cfg.pose_center_weight * pose_center
        + cfg.pointmap_weight * pointmap
        + cfg.smoothness_weight * smoothness
        + cfg.dynamic_weight * dynamic
    )
    _raise_if_nonfinite(total)
    abs_error = depth_residual.abs()[valid]
    depth_absrel_metric = (abs_error / target_depth[valid].clamp(min=1e-6)).mean()
    depth_rmse_metric = _TORCH.sqrt(abs_error.square().mean())
    constant_depth = target_depth[valid].median()
    constant_error = (constant_depth - target_depth).abs()[valid]
    constant_absrel = (constant_error / target_depth[valid].clamp(min=1e-6)).mean()
    constant_rmse = _TORCH.sqrt(constant_error.square().mean())
    metrics = {
        "loss_total": _float(total),
        "loss_depth_log": _float(depth_log),
        "loss_depth_si": _float(depth_si),
        "loss_uncertainty": _float(uncertainty),
        "loss_confidence": _float(confidence),
        "confidence_brier": _float(confidence_brier),
        "loss_pose_relative": _float(pose_relative),
        "loss_pose_center": _float(pose_center),
        "loss_pointmap_consistency": _float(pointmap),
        "loss_smoothness": _float(smoothness),
        "loss_dynamic": _float(dynamic),
        "depth_rmse_m": _float(depth_rmse_metric),
        "depth_absrel": _float(depth_absrel_metric),
        "constant_depth_baseline_absrel": _float(constant_absrel),
        "constant_depth_baseline_rmse_m": _float(constant_rmse),
        "depth_absrel_vs_constant_baseline_ratio": _float(
            depth_absrel_metric / constant_absrel.clamp(min=1e-12)
        ),
        "student_depth_beats_constant_baseline_25pct": float(
            bool(depth_absrel_metric <= 0.75 * constant_absrel)
        ),
        "valid_pixel_ratio": float(valid_count / valid.numel()),
        **pose_metrics,
    }
    return total, metrics


def _pixel_weights(batch: Mapping[str, Any], valid: Any, target_conf: Any) -> Any:
    depth_weight = _tensor(batch, "depth_target_weight").float()
    while depth_weight.ndim < target_conf.ndim:
        depth_weight = depth_weight.unsqueeze(-1)
    weights = _TORCH.where(valid, target_conf * depth_weight, _TORCH.zeros_like(target_conf))
    positive = weights > 0.0
    if bool(positive.any()):
        weights = weights / weights[positive].mean().clamp(min=1e-12)
    return weights


def _scale_invariant_log_loss(log_error: Any, weights: Any) -> Any:
    mean_error = _weighted_mean(log_error, weights)
    mean_square = _weighted_mean(log_error.square(), weights)
    return (mean_square - 0.85 * mean_error.square()).clamp(min=0.0)


def _pointmap_loss(
    prediction: Mapping[str, Any],
    batch: Mapping[str, Any],
    target_depth: Any,
    weights: Any,
) -> Any:
    K = _tensor(batch, "K").float()
    target_pointmap = unproject_depth_camera(target_depth, K)
    pred_pointmap = _tensor(prediction, "pointmap_camera_m").float()
    point_weights = weights.unsqueeze(2).expand_as(pred_pointmap)
    return _weighted_mean(
        _F.smooth_l1_loss(pred_pointmap, target_pointmap, reduction="none"), point_weights
    )


def _pose_losses(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, float]]:
    pred_T = _tensor(prediction, "T_world_camera").float()
    target_T = _tensor(target, "T_world_camera").float()
    if pred_T.shape != target_T.shape:
        raise ValueError("T_world_camera prediction and target shapes must match")
    pose_weight = _tensor(batch, "pose_target_weight").float()
    if int(pred_T.shape[1]) <= 1:
        zero = pred_T.new_tensor(0.0)
        return (
            zero,
            zero,
            {
                "pose_relative_translation_mean_m": 0.0,
                "pose_rotation_mean_deg": 0.0,
                "pose_center_mean_m": 0.0,
                "no_motion_pose_baseline_center_mean_m": 0.0,
                "pose_center_vs_no_motion_baseline_ratio": 0.0,
                "student_pose_beats_no_motion_baseline_15pct": 1.0,
            },
        )
    pred_rel = _relative_transforms(pred_T)
    target_rel = _relative_transforms(target_T)
    weight = pose_weight.view(-1, 1)
    trans_error = _TORCH.linalg.norm(pred_rel[..., :3, 3] - target_rel[..., :3, 3], dim=-1)
    rot_error = _geodesic_rotation_error(pred_rel[..., :3, :3], target_rel[..., :3, :3])
    relative = ((trans_error + rot_error) * weight).sum() / weight.sum().clamp(min=1e-12)
    centers = _TORCH.linalg.norm(pred_T[..., :3, 3] - target_T[..., :3, 3], dim=-1)
    center = (centers * pose_weight.view(-1, 1)).sum() / pose_weight.sum().clamp(min=1e-12)
    target_centers = target_T[..., :3, 3]
    no_motion_centers = target_centers[:, :1, :].expand_as(target_centers)
    no_motion_error = _TORCH.linalg.norm(no_motion_centers - target_centers, dim=-1)
    no_motion_baseline = no_motion_error.mean()
    center_mean = centers.mean()
    return (
        relative,
        center,
        {
            "pose_relative_translation_mean_m": _float(trans_error.mean()),
            "pose_rotation_mean_deg": _float(rot_error.mean() * (180.0 / 3.141592653589793)),
            "pose_center_mean_m": _float(center_mean),
            "no_motion_pose_baseline_center_mean_m": _float(no_motion_baseline),
            "pose_center_vs_no_motion_baseline_ratio": _float(
                center_mean / no_motion_baseline.clamp(min=1e-12)
            ),
            "student_pose_beats_no_motion_baseline_15pct": float(
                bool(center_mean <= 0.85 * no_motion_baseline)
            ),
        },
    )


def _edge_aware_smoothness(depth: Any, images_rgb: Any) -> Any:
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]
    gray = images_rgb.mean(dim=2)
    wx = _TORCH.exp(-(gray[..., :, 1:] - gray[..., :, :-1]).abs())
    wy = _TORCH.exp(-(gray[..., 1:, :] - gray[..., :-1, :]).abs())
    return (dx.abs() * wx).mean() + (dy.abs() * wy).mean()


def _relative_transforms(T_world_camera: Any) -> Any:
    left = T_world_camera[:, :-1]
    right = T_world_camera[:, 1:]
    return _TORCH.matmul(_TORCH.linalg.inv(left), right)


def _geodesic_rotation_error(pred_rotation: Any, target_rotation: Any) -> Any:
    relative = _TORCH.matmul(pred_rotation.transpose(-1, -2), target_rotation)
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vector = _TORCH.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * _TORCH.linalg.norm(skew_vector, dim=-1)
    sin_theta = _TORCH.sqrt(sin_theta.square() + 1e-6 * 1e-6)
    return _TORCH.atan2(sin_theta, cos_theta)


def _weighted_mean(values: Any, weights: Any) -> Any:
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


def _bce_probability(prediction: Any, target: Any) -> Any:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log()).mean()


def _tensor(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping[key]
    if not _TORCH.is_tensor(value):
        raise ValueError(f"{key}: must be a torch.Tensor")
    if not bool(_TORCH.isfinite(value.float()).all()):
        raise ValueError(f"{key}: tensor contains NaN or Inf")
    return value


def _raise_if_nonfinite(value: Any) -> None:
    if not bool(_TORCH.isfinite(value).all()):
        raise ValueError("loss_total: tensor contains NaN or Inf")


def _float(value: Any) -> float:
    return float(value.detach().cpu().item())


__all__ = ["SMGTTinyLossConfig", "smgt_tiny_loss"]
