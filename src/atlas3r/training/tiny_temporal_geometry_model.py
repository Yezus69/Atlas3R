"""Tiny temporal geometry model for Phase 5A data/loss plumbing."""

from __future__ import annotations

from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_NN: Any = _TORCH.nn


def temporal_v0_truth_boundary() -> dict[str, object]:
    """Truth flags for tiny temporal debug checkpoints."""

    return {
        "training_mvp": True,
        "temporal_v0": True,
        "trained_on_real_rgbd": True,
        "synthetic_only": False,
        "real_capture_debug_model": True,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "learned_inference": True,
        "generalizes_to_real_world": False,
        "relative_rotation_learned": False,
    }


def temporal_v1_truth_boundary() -> dict[str, object]:
    """Truth flags for teacher-signal temporal-v1 checkpoints."""

    return {
        "training_mvp": True,
        "temporal_v1": True,
        "teacher_signal_training": True,
        "learned_inference": True,
        "realtime": False,
        "mapping": False,
        "accuracy": False,
        "performance": False,
        "final_smgt": False,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "generalizes_to_real_world": False,
        "relative_rotation_learned": True,
        "outputs_are_pseudo_labels": True,
    }


class TinyTemporalMetricNetV0(_NN.Module):  # type: ignore[misc]
    """Small RGB+ray temporal model with mean feature aggregation over clips."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.frame_encoder = _NN.Sequential(
            _NN.Conv2d(6, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.depth_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.translation_head = _NN.Sequential(
            _NN.Linear(hidden_channels, hidden_channels),
            _NN.ReLU(inplace=True),
            _NN.Linear(hidden_channels, 3),
        )

    def forward(self, images_rgb: Any, intrinsics: Any) -> dict[str, Any]:
        """Run a temporal forward pass.

        Args:
            images_rgb: `B,T,3,H,W` float RGB in `[0,1]`.
            intrinsics: `B,T,3,3` scaled camera intrinsics.
        """

        if images_rgb.ndim != 5 or int(images_rgb.shape[2]) != 3:
            raise ValueError("images_rgb: must have shape B,T,3,H,W")
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[2:]) != (3, 3):
            raise ValueError("intrinsics: must have shape B,T,3,3")
        batch_size, clip_length, _channels, height, width = images_rgb.shape
        flat_images = images_rgb.reshape(batch_size * clip_length, 3, height, width)
        flat_intrinsics = intrinsics.reshape(batch_size * clip_length, 3, 3)
        ray_channels = _camera_ray_channels(
            batch_size=batch_size * clip_length,
            height=height,
            width=width,
            intrinsics=flat_intrinsics,
            dtype=images_rgb.dtype,
            device=images_rgb.device,
        )
        encoded = self.frame_encoder(_TORCH.cat([flat_images, ray_channels], dim=1))
        features = encoded.reshape(batch_size, clip_length, self.hidden_channels, height, width)
        aggregated = features.mean(dim=1)
        pooled = features.mean(dim=(-2, -1))
        return {
            "center_depth_m": _TORCH.nn.functional.softplus(self.depth_head(aggregated)) + 1e-4,
            "center_depth_sigma_m": (
                _TORCH.nn.functional.softplus(self.sigma_head(aggregated)) + 1e-4
            ),
            "center_confidence": _TORCH.sigmoid(self.confidence_head(aggregated)),
            "relative_translation_center_from_camera": self.translation_head(pooled),
        }

    def model_config(self) -> dict[str, object]:
        return {
            "model_name": "TinyTemporalMetricNetV0",
            "model_role": "tum_rgbd_temporal_debug_training_mvp",
            "final_smgt": False,
            "input_shape": "B,T,3,H,W RGB plus B,T,3,3 intrinsics-derived ray channels",
            "outputs": [
                "center_depth_m",
                "center_depth_sigma_m",
                "center_confidence",
                "relative_translation_center_from_camera",
            ],
            "relative_rotation_output": False,
            "hidden_channels": self.hidden_channels,
            "truth_boundary": temporal_v0_truth_boundary(),
        }


class TemporalMetricNetV1(_NN.Module):  # type: ignore[misc]
    """Small per-frame RGB+ray temporal model with a ConvGRU bottleneck."""

    def __init__(self, hidden_channels: int = 24, bottleneck_channels: int = 32) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.bottleneck_channels = bottleneck_channels
        self.frame_encoder = _NN.Sequential(
            _NN.Conv2d(6, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.gru_z = _NN.Conv2d(
            bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1
        )
        self.gru_r = _NN.Conv2d(
            bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1
        )
        self.gru_n = _NN.Conv2d(
            bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1
        )
        self.decoder = _NN.Sequential(
            _NN.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(bottleneck_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.depth_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.translation_head = _NN.Sequential(
            _NN.Linear(bottleneck_channels, hidden_channels),
            _NN.ReLU(inplace=True),
            _NN.Linear(hidden_channels, 3),
        )
        self.rotation_head = _NN.Sequential(
            _NN.Linear(bottleneck_channels, hidden_channels),
            _NN.ReLU(inplace=True),
            _NN.Linear(hidden_channels, 6),
        )
        self._initialize_rotation_head()

    def forward(self, images_rgb: Any, intrinsics: Any) -> dict[str, Any]:
        """Run a temporal-v1 forward pass.

        Args:
            images_rgb: `B,T,3,H,W` float RGB in `[0,1]`.
            intrinsics: `B,T,3,3` scaled camera intrinsics.
        """

        if images_rgb.ndim != 5 or int(images_rgb.shape[2]) != 3:
            raise ValueError("images_rgb: must have shape B,T,3,H,W")
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[2:]) != (3, 3):
            raise ValueError("intrinsics: must have shape B,T,3,3")
        batch_size, clip_length, _channels, height, width = images_rgb.shape
        flat_images = images_rgb.reshape(batch_size * clip_length, 3, height, width)
        flat_intrinsics = intrinsics.reshape(batch_size * clip_length, 3, 3)
        ray_channels = _camera_ray_channels(
            batch_size=batch_size * clip_length,
            height=height,
            width=width,
            intrinsics=flat_intrinsics,
            dtype=images_rgb.dtype,
            device=images_rgb.device,
        )
        encoded = self.frame_encoder(_TORCH.cat([flat_images, ray_channels], dim=1))
        _, channels, low_height, low_width = encoded.shape
        features = encoded.reshape(batch_size, clip_length, channels, low_height, low_width)
        fused = self._conv_gru(features)
        flat_fused = fused.reshape(batch_size * clip_length, channels, low_height, low_width)
        decoded = self.decoder(flat_fused)
        decoded = _TORCH.nn.functional.interpolate(
            decoded,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        depth = _TORCH.nn.functional.softplus(self.depth_head(decoded)) + 1e-4
        sigma = _TORCH.nn.functional.softplus(self.sigma_head(decoded)) + 1e-4
        confidence = _TORCH.sigmoid(self.confidence_head(decoded))
        pooled = fused.mean(dim=(-2, -1))
        return {
            "depth_m": depth.reshape(batch_size, clip_length, 1, height, width),
            "depth_sigma_m": sigma.reshape(batch_size, clip_length, 1, height, width),
            "confidence": confidence.reshape(batch_size, clip_length, 1, height, width),
            "relative_translation_center_from_camera": self.translation_head(pooled),
            "relative_rotation_6d_center_from_camera": self.rotation_head(pooled),
        }

    def _initialize_rotation_head(self) -> None:
        final_layer = self.rotation_head[-1]
        with _TORCH.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.copy_(
                _TORCH.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=final_layer.bias.dtype)
            )

    def _conv_gru(self, features: Any) -> Any:
        batch_size, clip_length, channels, height, width = features.shape
        hidden = features.new_zeros((batch_size, channels, height, width))
        outputs = []
        for index in range(clip_length):
            current = features[:, index]
            stacked = _TORCH.cat([current, hidden], dim=1)
            update_gate = _TORCH.sigmoid(self.gru_z(stacked))
            reset_gate = _TORCH.sigmoid(self.gru_r(stacked))
            candidate = _TORCH.tanh(self.gru_n(_TORCH.cat([current, reset_gate * hidden], dim=1)))
            hidden = (1.0 - update_gate) * hidden + update_gate * candidate
            outputs.append(hidden)
        return _TORCH.stack(outputs, dim=1)

    def model_config(self) -> dict[str, object]:
        return {
            "model_name": "TemporalMetricNetV1",
            "model_role": "teacher_signal_temporal_training_mvp",
            "final_smgt": False,
            "realtime": False,
            "mapping": False,
            "accuracy": False,
            "performance": False,
            "input_shape": "B,T,3,H,W RGB plus B,T,3,3 intrinsics-derived ray channels",
            "outputs": [
                "depth_m B,T,1,H,W",
                "depth_sigma_m B,T,1,H,W",
                "confidence B,T,1,H,W",
                "relative_translation_center_from_camera B,T,3",
                "relative_rotation_6d_center_from_camera B,T,6",
            ],
            "relative_rotation_output": True,
            "relative_rotation_representation": "6d_first_two_rotation_columns_center_from_camera",
            "hidden_channels": self.hidden_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "temporal_fusion": "single-pass ConvGRU bottleneck",
            "truth_boundary": temporal_v1_truth_boundary(),
        }


def _camera_ray_channels(
    *,
    batch_size: int,
    height: int,
    width: int,
    intrinsics: Any,
    dtype: Any,
    device: Any,
) -> Any:
    u = _TORCH.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)
    v = _TORCH.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
    fx = intrinsics[:, 0, 0].view(batch_size, 1, 1, 1).to(dtype=dtype)
    fy = intrinsics[:, 1, 1].view(batch_size, 1, 1, 1).to(dtype=dtype)
    cx = intrinsics[:, 0, 2].view(batch_size, 1, 1, 1).to(dtype=dtype)
    cy = intrinsics[:, 1, 2].view(batch_size, 1, 1, 1).to(dtype=dtype)
    ray_x = (u - cx) / fx.clamp(min=1e-6)
    ray_y = (v - cy) / fy.clamp(min=1e-6)
    ray_x = ray_x.expand(batch_size, 1, height, width)
    ray_y = ray_y.expand(batch_size, 1, height, width)
    radius = _TORCH.sqrt(ray_x.square() + ray_y.square())
    return _TORCH.cat([ray_x, ray_y, radius], dim=1)


__all__ = [
    "TemporalMetricNetV1",
    "TinyTemporalMetricNetV0",
    "temporal_v0_truth_boundary",
    "temporal_v1_truth_boundary",
]
