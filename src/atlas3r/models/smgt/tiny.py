"""First learned diagnostic SMGT-tiny student model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from atlas3r.models.smgt.config import SMGTTinyConfig
from atlas3r.models.smgt.geometry import (
    camera_ray_channels,
    normals_from_depth,
    transforms_from_relative_deltas,
    unproject_depth_camera,
)
from atlas3r.models.smgt.memory import ConvGRUCell, SMGTTinyMemoryState
from atlas3r.models.student.contracts import StudentForwardOutput
from atlas3r.training.torch_runtime import require_torch

_TORCH: Any = require_torch()
_NN: Any = _TORCH.nn
_F: Any = _TORCH.nn.functional


class SMGTTiny(_NN.Module):  # type: ignore[misc]
    """Tiny multi-scale RGB+ray ConvGRU student for depth, pose, and uncertainty."""

    def __init__(self, config: SMGTTinyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        f = config.feature_dim
        m = config.memory_dim
        self.scale1 = _conv_block(6, h, stride=1)
        self.scale2 = _conv_block(h, f, stride=2)
        self.scale3 = _conv_block(f, f, stride=2)
        self.scale4 = _conv_block(f, m, stride=2)
        self.memory = ConvGRUCell(m, m)
        self.up3 = _conv_block(m + f, f, stride=1)
        self.up2 = _conv_block(f + f, f, stride=1)
        self.up1 = _conv_block(f + h, h, stride=1)
        self.depth_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.sigma_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.confidence_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.dynamic_head = _NN.Conv2d(h, 1, kernel_size=1)
        self.pose_head = _NN.Sequential(
            _NN.Linear(m, h),
            _NN.SiLU(inplace=True),
            _NN.Linear(h, 9),
        )
        self._initialize_pose_head()

    def forward(
        self,
        images_rgb: Any,
        K: Any,
        T_world_camera_prior: Any | None = None,
        memory_state: SMGTTinyMemoryState | None = None,
        return_memory_state: bool = False,
    ) -> dict[str, Any]:
        del T_world_camera_prior
        _validate_inputs(images_rgb, K)
        batch_size, frame_count, _channels, height, width = images_rgb.shape
        if height != self.config.image_height or width != self.config.image_width:
            raise ValueError(
                "images_rgb: H,W must match SMGTTinyConfig "
                f"{self.config.image_height},{self.config.image_width}"
            )
        hidden = None if memory_state is None else memory_state.hidden
        decoded_frames: list[Any] = []
        pose_vectors: list[Any] = []
        for index in range(frame_count):
            low, skips = self._encode_frame(images_rgb[:, index], K[:, index])
            hidden = low if not self.config.use_temporal_memory else self.memory(low, hidden)
            decoded_frames.append(self._decode_frame(hidden, skips))
            pose_vectors.append(self.pose_head(hidden.mean(dim=(-2, -1))))
        decoded = _TORCH.stack(decoded_frames, dim=1).reshape(
            batch_size * frame_count,
            self.config.hidden_dim,
            height,
            width,
        )
        depth = _F.softplus(self.depth_head(decoded)).reshape(
            batch_size, frame_count, height, width
        )
        depth = depth.clamp(min=self.config.min_depth_m, max=self.config.max_depth_m)
        sigma = (
            _F.softplus(self.sigma_head(decoded)).reshape(batch_size, frame_count, height, width)
            + 1e-4
        )
        confidence = _TORCH.sigmoid(self.confidence_head(decoded)).reshape(
            batch_size, frame_count, height, width
        )
        dynamic = _TORCH.sigmoid(self.dynamic_head(decoded)).reshape(
            batch_size, frame_count, height, width
        )
        pose_raw = _TORCH.stack(pose_vectors, dim=1)
        relative_translation = 0.25 * _TORCH.tanh(pose_raw[..., :3])
        relative_translation[:, 0, :] = 0.0
        relative_rotation_6d = pose_raw[..., 3:9]
        relative_rotation_6d[:, 0, :] = relative_rotation_6d.new_tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        T_world_camera = transforms_from_relative_deltas(relative_translation, relative_rotation_6d)
        pointmap = unproject_depth_camera(depth, K)
        prediction = {
            "depth_m": depth,
            "depth_sigma_m": sigma,
            "confidence": confidence,
            "dynamic_probability": dynamic,
            "normals_camera": normals_from_depth(depth),
            "pointmap_camera_m": pointmap,
            "T_world_camera": T_world_camera,
            "intrinsics": K,
            "relative_translation_m": relative_translation,
            "relative_rotation_6d": relative_rotation_6d,
        }
        if return_memory_state:
            prediction["memory_state"] = SMGTTinyMemoryState(hidden=hidden)
        return prediction

    def forward_step(
        self,
        image_rgb: Any,
        K: Any,
        memory_state: SMGTTinyMemoryState | None = None,
    ) -> tuple[dict[str, Any], SMGTTinyMemoryState]:
        prediction = self.forward(
            image_rgb.unsqueeze(1),
            K.unsqueeze(1),
            memory_state=memory_state,
            return_memory_state=True,
        )
        state = cast(SMGTTinyMemoryState, prediction.pop("memory_state"))
        return prediction, state

    def model_config(self) -> dict[str, object]:
        return {
            "model_name": "SMGTTiny",
            "model_role": "first_learned_diagnostic_smgt_tiny_student",
            "final_smgt": False,
            "object_aware_fusion_implemented": False,
            "input_shape": "B,T,3,H,W RGB plus B,T,3,3 intrinsics-derived ray channels",
            "outputs": [
                "depth_m B,T,H,W",
                "depth_sigma_m B,T,H,W",
                "confidence B,T,H,W",
                "dynamic_probability B,T,H,W",
                "normals_camera B,T,3,H,W",
                "pointmap_camera_m B,T,3,H,W",
                "relative_translation_m B,T,3",
                "relative_rotation_6d B,T,6",
                "T_world_camera B,T,4,4",
            ],
            "config": self.config.to_json(),
        }

    def _encode_frame(self, image_rgb: Any, K: Any) -> tuple[Any, tuple[Any, Any, Any]]:
        rays = camera_ray_channels(
            K,
            height=int(image_rgb.shape[-2]),
            width=int(image_rgb.shape[-1]),
            dtype=image_rgb.dtype,
            device=image_rgb.device,
        )
        x = _TORCH.cat([image_rgb, rays], dim=1)
        f1 = self.scale1(x)
        f2 = self.scale2(f1)
        f3 = self.scale3(f2)
        f4 = self.scale4(f3)
        return f4, (f1, f2, f3)

    def _decode_frame(self, memory: Any, skips: tuple[Any, Any, Any]) -> Any:
        f1, f2, f3 = skips
        x = _F.interpolate(memory, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(_TORCH.cat([x, f3], dim=1))
        x = _F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(_TORCH.cat([x, f2], dim=1))
        x = _F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(_TORCH.cat([x, f1], dim=1))

    def _initialize_pose_head(self) -> None:
        final_layer = self.pose_head[-1]
        with _TORCH.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            final_layer.bias[3:9].copy_(
                _TORCH.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=final_layer.bias.dtype)
            )


def smgt_prediction_to_student_output(
    prediction: Mapping[str, Any],
    *,
    frame_ids: tuple[int, ...],
    truth_boundary: Mapping[str, object],
) -> StudentForwardOutput:
    return StudentForwardOutput(
        frame_ids=frame_ids,
        depth_m=_numpy(prediction["depth_m"]),
        depth_sigma_m=_numpy(prediction["depth_sigma_m"]),
        confidence=_numpy(prediction["confidence"]),
        dynamic_probability=_numpy(prediction["dynamic_probability"]),
        normals_camera=_numpy(prediction["normals_camera"]),
        pointmap_camera_m=_numpy(prediction["pointmap_camera_m"]),
        T_world_camera=_numpy(prediction["T_world_camera"]),
        intrinsics=_numpy(prediction["intrinsics"]),
        truth_boundary=dict(truth_boundary),
    )


def _conv_block(in_channels: int, out_channels: int, *, stride: int) -> Any:
    return _NN.Sequential(
        _NN.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        _NN.GroupNorm(_group_count(out_channels), out_channels),
        _NN.SiLU(inplace=True),
        _NN.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        _NN.GroupNorm(_group_count(out_channels), out_channels),
        _NN.SiLU(inplace=True),
    )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_inputs(images_rgb: Any, K: Any) -> None:
    if images_rgb.ndim != 5 or int(images_rgb.shape[2]) != 3:
        raise ValueError("images_rgb: must have shape B,T,3,H,W")
    if K.ndim != 4 or tuple(K.shape[2:]) != (3, 3):
        raise ValueError("K: must have shape B,T,3,3")
    if int(K.shape[0]) != int(images_rgb.shape[0]) or int(K.shape[1]) != int(images_rgb.shape[1]):
        raise ValueError("K: B,T dimensions must match images_rgb")


def _numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = ["SMGTTiny", "smgt_prediction_to_student_output"]
