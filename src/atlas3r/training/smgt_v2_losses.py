"""Losses and checkpoint scoring for SMGT-small-v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from atlas3r.models.smgt.geometry import unproject_depth_camera
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_F: Any = _TORCH.nn.functional


@dataclass(frozen=True)
class SMGTV2LossConfig:
    depth_log_weight: float = 1.0
    depth_silog_weight: float = 0.2
    depth_gradient_weight: float = 0.05
    depth_nll_weight: float = 0.05
    confidence_weight: float = 0.15
    pose_relative_weight: float = 0.35
    pose_center_weight: float = 0.2
    rotation_weight: float = 0.1
    pointmap_weight: float = 0.08
    temporal_depth_weight: float = 0.03
    dynamic_prior_weight: float = 0.002
    min_depth_m: float = 1e-4
    min_sigma_m: float = 1e-3
    max_sigma_m: float = 5.0

    def to_json(self) -> dict[str, object]:
        return {
            "depth_log_weight": self.depth_log_weight,
            "depth_silog_weight": self.depth_silog_weight,
            "depth_gradient_weight": self.depth_gradient_weight,
            "depth_nll_weight": self.depth_nll_weight,
            "confidence_weight": self.confidence_weight,
            "pose_relative_weight": self.pose_relative_weight,
            "pose_center_weight": self.pose_center_weight,
            "rotation_weight": self.rotation_weight,
            "pointmap_weight": self.pointmap_weight,
            "temporal_depth_weight": self.temporal_depth_weight,
            "dynamic_prior_weight": self.dynamic_prior_weight,
            "min_depth_m": self.min_depth_m,
            "min_sigma_m": self.min_sigma_m,
            "max_sigma_m": self.max_sigma_m,
        }


def smgt_v2_loss(
    prediction: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    config: SMGTV2LossConfig | None = None,
) -> tuple[Any, dict[str, float]]:
    cfg = config or SMGTV2LossConfig()
    target = batch["target"]
    if not isinstance(target, Mapping):
        raise ValueError("batch.target: must be a mapping")
    pred_depth = _tensor(prediction, "depth_m").float().clamp(min=cfg.min_depth_m)
    pred_sigma = (
        _tensor(prediction, "depth_sigma_m").float().clamp(cfg.min_sigma_m, cfg.max_sigma_m)
    )
    pred_conf = _tensor(prediction, "confidence").float().clamp(1e-6, 1.0 - 1e-6)
    pred_dynamic = _tensor(prediction, "dynamic_probability").float().clamp(1e-6, 1.0 - 1e-6)
    target_depth = _tensor(target, "depth_m").float().clamp(min=cfg.min_depth_m)
    target_conf = _tensor(target, "confidence").float().clamp(0.0, 1.0)
    target_sigma = _tensor(target, "depth_sigma_m").float().clamp(cfg.min_sigma_m, cfg.max_sigma_m)
    valid = _tensor(target, "valid_mask").to(dtype=_TORCH.bool)
    if pred_depth.shape != target_depth.shape or valid.shape != target_depth.shape:
        raise ValueError("depth, target depth, and valid mask must share B,T,H,W shape")
    depth_weights = _pixel_weights(batch, valid, target_conf, "depth_target_weight")
    confidence_weights = _pixel_weights(
        batch, valid, _TORCH.ones_like(target_conf), "confidence_target_weight"
    )
    log_error = _TORCH.log(pred_depth) - _TORCH.log(target_depth)
    depth_log = _weighted_mean(
        _F.smooth_l1_loss(log_error, _TORCH.zeros_like(log_error), reduction="none"),
        depth_weights,
    )
    depth_silog = _scale_invariant_log_loss(log_error, depth_weights)
    depth_gradient = _depth_gradient_loss(pred_depth, target_depth, depth_weights)
    depth_residual = pred_depth - target_depth
    depth_nll = _weighted_mean(
        depth_residual.abs() / pred_sigma
        + pred_sigma.log()
        + 0.1 * (pred_sigma - target_sigma).abs(),
        depth_weights,
    )
    confidence_target = good_pixel_confidence_target(
        pred_depth.detach(),
        target_depth,
        valid,
        target_conf,
        _tensor(batch, "target_source_id").float(),
    )
    confidence = _focal_bce_probability(pred_conf, confidence_target, confidence_weights)
    confidence_brier = _weighted_mean((pred_conf - confidence_target).square(), confidence_weights)
    pointmap = _pointmap_loss(prediction, batch, target_depth, depth_weights)
    temporal_depth = _temporal_depth_consistency(pred_depth, target_depth, depth_weights)
    dynamic = _focal_bce_probability(
        pred_dynamic, _TORCH.zeros_like(pred_dynamic), _TORCH.ones_like(pred_dynamic)
    )
    pose_relative, pose_center, pose_rotation, pose_metrics = _pose_losses(
        prediction, target, batch
    )
    total = (
        cfg.depth_log_weight * depth_log
        + cfg.depth_silog_weight * depth_silog
        + cfg.depth_gradient_weight * depth_gradient
        + cfg.depth_nll_weight * depth_nll
        + cfg.confidence_weight * confidence
        + cfg.pose_relative_weight * pose_relative
        + cfg.pose_center_weight * pose_center
        + cfg.rotation_weight * pose_rotation
        + cfg.pointmap_weight * pointmap
        + cfg.temporal_depth_weight * temporal_depth
        + cfg.dynamic_prior_weight * dynamic
    )
    _raise_if_nonfinite(total)
    valid_count = int(valid.sum().detach().cpu().item())
    if valid_count <= 0:
        raise ValueError("valid_mask: batch has no valid target pixels")
    abs_error = depth_residual.abs()[valid]
    depth_absrel_metric = (abs_error / target_depth[valid].clamp(min=1e-6)).mean()
    depth_rmse_metric = _TORCH.sqrt(abs_error.square().mean())
    constant_depth = target_depth[valid].median()
    constant_error = (constant_depth - target_depth).abs()[valid]
    constant_absrel = (constant_error / target_depth[valid].clamp(min=1e-6)).mean()
    constant_rmse = _TORCH.sqrt(constant_error.square().mean())
    mapped = (pred_conf >= 0.5) & (pred_sigma <= 1.0) & valid
    mapped_ratio = mapped.float().mean()
    metrics = {
        "loss_total": _float(total),
        "loss_depth_log": _float(depth_log),
        "loss_depth_silog": _float(depth_silog),
        "loss_depth_gradient": _float(depth_gradient),
        "loss_depth_nll": _float(depth_nll),
        "loss_confidence": _float(confidence),
        "confidence_brier": _float(confidence_brier),
        "loss_pose_relative": _float(pose_relative),
        "loss_pose_center": _float(pose_center),
        "loss_pose_rotation": _float(pose_rotation),
        "loss_pointmap_consistency": _float(pointmap),
        "loss_temporal_depth_consistency": _float(temporal_depth),
        "loss_dynamic_prior": _float(dynamic),
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
        "val_mapped_pixel_ratio_at_selected_threshold": _float(mapped_ratio),
        "mesh_success": float(bool(_float(mapped_ratio) > 0.0)),
        **pose_metrics,
    }
    return total, metrics


def good_pixel_confidence_target(
    pred_depth: Any,
    target_depth: Any,
    valid: Any,
    teacher_confidence: Any,
    target_source_id: Any,
) -> Any:
    abs_error = (pred_depth - target_depth).abs()
    rel_error = abs_error / target_depth.clamp(min=1e-6)
    good_measured = valid & ((rel_error < 0.10) | (abs_error < 0.10))
    measured = target_source_id.view(-1, *([1] * (target_depth.ndim - 1))) >= 0.5
    pseudo_target = _TORCH.where(
        valid, teacher_confidence.clamp(0.0, 1.0), _TORCH.zeros_like(teacher_confidence)
    )
    measured_target = good_measured.float()
    return _TORCH.where(measured, measured_target, pseudo_target)


def smgt_v2_checkpoint_selection_score(metrics: Mapping[str, float]) -> float:
    depth = float(metrics.get("depth_absrel", 1e6))
    pose_ratio = float(metrics.get("pose_center_vs_no_motion_baseline_ratio", 1e6))
    confidence = float(metrics.get("confidence_brier", 1.0))
    mapped = float(metrics.get("val_mapped_pixel_ratio_at_selected_threshold", 0.0))
    mapped_penalty = abs(mapped - 0.40)
    zero_mesh_penalty = 1e6 if not smgt_v2_candidate_can_be_best(metrics) else 0.0
    return depth + 0.5 * pose_ratio + 0.25 * confidence + 0.25 * mapped_penalty + zero_mesh_penalty


def smgt_v2_candidate_can_be_best(metrics: Mapping[str, float]) -> bool:
    mesh_success = bool(metrics.get("mesh_success", 0.0))
    chunks = int(metrics.get("mesh_chunk_count", 1.0 if mesh_success else 0.0))
    mapped = float(metrics.get("val_mapped_pixel_ratio_at_selected_threshold", 0.0))
    return mesh_success and chunks > 0 and mapped > 0.0


def _pixel_weights(batch: Mapping[str, Any], valid: Any, target_conf: Any, key: str) -> Any:
    weight = _tensor(batch, key).float()
    while weight.ndim < target_conf.ndim:
        weight = weight.unsqueeze(-1)
    weights = _TORCH.where(valid, target_conf * weight, _TORCH.zeros_like(target_conf))
    positive = weights > 0.0
    if bool(positive.any()):
        weights = weights / weights[positive].mean().clamp(min=1e-12)
    return weights


def _scale_invariant_log_loss(log_error: Any, weights: Any) -> Any:
    mean_error = _weighted_mean(log_error, weights)
    mean_square = _weighted_mean(log_error.square(), weights)
    return (mean_square - 0.85 * mean_error.square()).clamp(min=0.0)


def _depth_gradient_loss(pred_depth: Any, target_depth: Any, weights: Any) -> Any:
    pred_dx = pred_depth[..., :, 1:] - pred_depth[..., :, :-1]
    target_dx = target_depth[..., :, 1:] - target_depth[..., :, :-1]
    pred_dy = pred_depth[..., 1:, :] - pred_depth[..., :-1, :]
    target_dy = target_depth[..., 1:, :] - target_depth[..., :-1, :]
    return _weighted_mean((pred_dx - target_dx).abs(), weights[..., :, 1:]) + _weighted_mean(
        (pred_dy - target_dy).abs(),
        weights[..., 1:, :],
    )


def _pointmap_loss(
    prediction: Mapping[str, Any], batch: Mapping[str, Any], target_depth: Any, weights: Any
) -> Any:
    K = _tensor(batch, "K").float()
    target_pointmap = unproject_depth_camera(target_depth, K)
    pred_pointmap = _tensor(prediction, "pointmap_camera_m").float()
    point_weights = weights.unsqueeze(2).expand_as(pred_pointmap)
    return _weighted_mean(
        _F.smooth_l1_loss(pred_pointmap, target_pointmap, reduction="none"),
        point_weights,
    )


def _temporal_depth_consistency(pred_depth: Any, target_depth: Any, weights: Any) -> Any:
    if int(pred_depth.shape[1]) <= 1:
        return pred_depth.new_tensor(0.0)
    pred_delta = pred_depth[:, 1:] - pred_depth[:, :-1]
    target_delta = target_depth[:, 1:] - target_depth[:, :-1]
    pair_weight = 0.5 * (weights[:, 1:] + weights[:, :-1])
    return _weighted_mean((pred_delta - target_delta).abs(), pair_weight)


def _pose_losses(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, float]]:
    pred_T = _tensor(prediction, "T_world_camera").float()
    target_T = _tensor(target, "T_world_camera").float()
    pose_weight = _tensor(batch, "pose_target_weight").float()
    if int(pred_T.shape[1]) <= 1:
        zero = pred_T.new_tensor(0.0)
        return zero, zero, zero, _zero_pose_metrics()
    pred_rel = _relative_transforms(pred_T)
    target_rel = _relative_transforms(target_T)
    weight = pose_weight.view(-1, 1)
    trans_error = _TORCH.linalg.norm(pred_rel[..., :3, 3] - target_rel[..., :3, 3], dim=-1)
    rot_error = _geodesic_rotation_error(pred_rel[..., :3, :3], target_rel[..., :3, :3])
    relative = (trans_error * weight).sum() / weight.sum().clamp(min=1e-12)
    rotation = (rot_error * weight).sum() / weight.sum().clamp(min=1e-12)
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
        rotation,
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


def _zero_pose_metrics() -> dict[str, float]:
    return {
        "pose_relative_translation_mean_m": 0.0,
        "pose_rotation_mean_deg": 0.0,
        "pose_center_mean_m": 0.0,
        "no_motion_pose_baseline_center_mean_m": 0.0,
        "pose_center_vs_no_motion_baseline_ratio": 0.0,
        "student_pose_beats_no_motion_baseline_15pct": 1.0,
    }


def _relative_transforms(T_world_camera: Any) -> Any:
    return _TORCH.matmul(_TORCH.linalg.inv(T_world_camera[:, :-1]), T_world_camera[:, 1:])


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
    sin_theta = _TORCH.sqrt(sin_theta.square() + 1e-12)
    return _TORCH.atan2(sin_theta, cos_theta)


def _focal_bce_probability(prediction: Any, target: Any, weights: Any) -> Any:
    bce = -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())
    focal = (prediction - target).abs().square()
    return _weighted_mean(bce * (1.0 + focal), weights)


def _weighted_mean(values: Any, weights: Any) -> Any:
    return (values * weights).sum() / weights.sum().clamp(min=1e-12)


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


__all__ = [
    "SMGTV2LossConfig",
    "good_pixel_confidence_target",
    "smgt_v2_candidate_can_be_best",
    "smgt_v2_checkpoint_selection_score",
    "smgt_v2_loss",
]
