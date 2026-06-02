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


__all__ = [
    "TinyDepthPoseNet",
    "model_config",
    "training_mvp_truth_boundary",
]
