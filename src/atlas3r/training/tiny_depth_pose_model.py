"""Tiny PyTorch depth/pose model for the synthetic training MVP."""

from __future__ import annotations

from typing import Any

from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_NN: Any = _TORCH.nn


def training_mvp_truth_boundary() -> dict[str, object]:
    """Truth-boundary flags required for Phase 4A artifacts."""

    return {
        "training_mvp": True,
        "synthetic_only": True,
        "real_capture_model": False,
        "usable_for_realtime_mapping": False,
        "usable_for_mapping": False,
        "accuracy_report": False,
        "performance_report": False,
        "learned_inference": True,
        "generalizes_to_real_world": False,
    }


def model_config() -> dict[str, object]:
    """Return dependency-light metadata for the tiny synthetic MVP model."""

    return {
        "model_name": "TinyDepthPoseNet",
        "model_role": "synthetic_overfit_training_mvp",
        "final_smgt": False,
        "input_shape": "B,3,H,W float32 RGB in [0,1]",
        "outputs": [
            "depth_m",
            "depth_sigma_m",
            "confidence",
            "camera_center_world_m",
        ],
        "truth_boundary": training_mvp_truth_boundary(),
    }


def v2_model_config() -> dict[str, object]:
    """Return metadata for the single targeted TUM RGB-D v2 debug model."""

    return {
        "model_name": "TinyMetricDepthNetV2",
        "model_role": "tum_rgbd_real_capture_debug_training_mvp",
        "final_smgt": False,
        "input_shape": "B,3,H,W float32 RGB in [0,1] plus intrinsics-derived ray channels",
        "ray_channels": [
            "(u - cx) / fx",
            "(v - cy) / fy",
            "sqrt(ray_x^2 + ray_y^2)",
        ],
        "outputs": [
            "depth_m",
            "depth_sigma_m",
            "confidence",
            "camera_center_world_m",
        ],
        "truth_boundary": training_mvp_truth_boundary(),
    }


class TinyDepthPoseNet(_NN.Module):  # type: ignore[misc]
    """Small ConvNet that predicts depth, uncertainty, confidence, and camera center."""

    def __init__(self, hidden_channels: int = 24) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = _NN.Sequential(
            _NN.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.depth_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.center_head = _NN.Sequential(
            _NN.AdaptiveAvgPool2d((1, 1)),
            _NN.Flatten(),
            _NN.Linear(hidden_channels, hidden_channels),
            _NN.ReLU(inplace=True),
            _NN.Linear(hidden_channels, 3),
        )

    def forward(self, images_rgb: Any, intrinsics: Any | None = None) -> dict[str, Any]:
        """Run a forward pass.

        `intrinsics` is accepted to match the future student boundary, but this
        small MVP model only uses RGB.
        """

        del intrinsics
        if images_rgb.ndim != 4 or int(images_rgb.shape[1]) != 3:
            raise ValueError("images_rgb: must have shape B,3,H,W")
        features = self.encoder(images_rgb)
        depth_m = _TORCH.nn.functional.softplus(self.depth_head(features)) + 1e-4
        depth_sigma_m = _TORCH.nn.functional.softplus(self.sigma_head(features)) + 1e-4
        confidence = _TORCH.sigmoid(self.confidence_head(features))
        camera_center_world_m = self.center_head(features)
        return {
            "depth_m": depth_m,
            "depth_sigma_m": depth_sigma_m,
            "confidence": confidence,
            "camera_center_world_m": camera_center_world_m,
        }

    def model_config(self) -> dict[str, object]:
        return {**model_config(), "hidden_channels": self.hidden_channels}


class TinyMetricDepthNetV2(_NN.Module):  # type: ignore[misc]
    """Small RGB+ray depth model with a shallow skip decoder."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.stem = _NN.Sequential(
            _NN.Conv2d(6, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.down = _NN.Sequential(
            _NN.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.mid = _NN.Sequential(
            _NN.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.decoder = _NN.Sequential(
            _NN.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
            _NN.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _NN.ReLU(inplace=True),
        )
        self.depth_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(hidden_channels, 1, kernel_size=1)
        self.center_head = _NN.Sequential(
            _NN.AdaptiveAvgPool2d((1, 1)),
            _NN.Flatten(),
            _NN.Linear(hidden_channels * 2, hidden_channels),
            _NN.ReLU(inplace=True),
            _NN.Linear(hidden_channels, 3),
        )

    def forward(self, images_rgb: Any, intrinsics: Any | None = None) -> dict[str, Any]:
        """Run a forward pass using RGB and normalized camera ray channels."""

        if images_rgb.ndim != 4 or int(images_rgb.shape[1]) != 3:
            raise ValueError("images_rgb: must have shape B,3,H,W")
        if intrinsics is None or intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
            raise ValueError("intrinsics: TinyMetricDepthNetV2 requires shape B,3,3")
        ray_channels = _camera_ray_channels(
            batch_size=int(images_rgb.shape[0]),
            height=int(images_rgb.shape[2]),
            width=int(images_rgb.shape[3]),
            intrinsics=intrinsics,
            dtype=images_rgb.dtype,
            device=images_rgb.device,
        )
        encoded = self.stem(_TORCH.cat([images_rgb, ray_channels], dim=1))
        coarse = self.mid(self.down(encoded))
        upsampled = _TORCH.nn.functional.interpolate(
            coarse,
            size=tuple(encoded.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder(_TORCH.cat([encoded, upsampled], dim=1))
        depth_m = _TORCH.nn.functional.softplus(self.depth_head(decoded)) + 1e-4
        depth_sigma_m = _TORCH.nn.functional.softplus(self.sigma_head(decoded)) + 1e-4
        confidence = _TORCH.sigmoid(self.confidence_head(decoded))
        camera_center_world_m = self.center_head(coarse)
        return {
            "depth_m": depth_m,
            "depth_sigma_m": depth_sigma_m,
            "confidence": confidence,
            "camera_center_world_m": camera_center_world_m,
        }

    def model_config(self) -> dict[str, object]:
        return {**v2_model_config(), "hidden_channels": self.hidden_channels}


def build_tiny_depth_pose_model(model_name: str, *, hidden_channels: int) -> Any:
    """Instantiate a supported tiny checkpoint model by stable checkpoint name."""

    if model_name in {"tiny-v1", "TinyDepthPoseNet"}:
        return TinyDepthPoseNet(hidden_channels=hidden_channels)
    if model_name in {"tiny-v2", "TinyMetricDepthNetV2"}:
        return TinyMetricDepthNetV2(hidden_channels=hidden_channels)
    raise ValueError(f"model_name: unsupported tiny depth model {model_name!r}")


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
    "TinyMetricDepthNetV2",
    "TinyDepthPoseNet",
    "build_tiny_depth_pose_model",
    "model_config",
    "training_mvp_truth_boundary",
    "v2_model_config",
]
